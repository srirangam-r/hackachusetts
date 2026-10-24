"""Skills: pick / place / push / actuate / home from motor memory, with a hand-written fallback (Req 4).

Inputs are only perceived ObjectEstimates, proprioception and the robot's own kinematics. Every motion
from the index is traceable to `segment_id`; primitive motions set `used_fallback`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from motormem import geometry as g
from motormem.skills.control import (GRIPPER_CLOSING_AXIS, GRIPPER_MAX_WIDTH, Controller,
                                     StepBudgetExceeded, densify)
from motormem.skills.feasibility import Feasibility
from motormem.types import ObjectEstimate, Segment, SegmentKind, SkillResult, Status

APPROACH_LIFT = 0.08      # pre-pose height above the first contact waypoint
LIFT = 0.10               # lift after grasp / retreat after release
CONTACT_RADIUS = 0.06     # segment poses within this distance of the gripper event = "contact part"
NEAR_TIE = 0.02           # registration hypotheses within this fitness of the best are near-ties
MIN_FITNESS = 0.3         # reject registrations worse than this
MAX_IK_WAYPOINTS = 60
RELATION_OFFSETS = {      # robot-base frame directions for spatial relations
    "left_of": (0, 1), "right_of": (0, -1), "front_of": (1, 0), "back_of": (-1, 0),
}


@dataclass
class Held:
    label: str
    name: str | None
    T_eef_obj: np.ndarray | None     # held object's memory frame in the gripper frame (index picks)
    T_eef_bottom: np.ndarray         # held object's bottom-centre point frame in the gripper frame
    extent: np.ndarray
    symmetric: bool = False


@dataclass
class Plan:
    targets: np.ndarray               # (N,4,4) full path, world
    gripper: np.ndarray               # (N,)
    event: int                        # index of the gripper event (close for grasp, open for place)
    segment_id: str | None
    T_world_obj: np.ndarray | None = None
    symmetric: bool = False
    info: dict = field(default_factory=dict)


class EpisodeOver(Exception):
    pass


def _contact_start(pos: np.ndarray, event: int, radius: float = CONTACT_RADIUS) -> int:
    """Earliest index s <= event such that every pose in [s, event] is within radius of the event pose."""
    s = event
    while s > 0 and np.linalg.norm(pos[s - 1] - pos[event]) <= radius:
        s -= 1
    return s


def topdown_rotation(closing_axis: np.ndarray, closing_col: int = 0) -> np.ndarray:
    """Grip-site rotation with z pointing down and the finger closing axis along `closing_axis`."""
    c = np.asarray(closing_axis, dtype=float).copy()
    c[2] = 0.0
    c /= np.linalg.norm(c) + 1e-12
    z = np.array([0.0, 0.0, -1.0])
    if closing_col == 0:
        x, y = c, np.cross(z, c)
    else:
        y = c
        x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def frontal_rotation(approach: np.ndarray, closing_axis: np.ndarray, closing_col: int = 0) -> np.ndarray:
    """Grip-site rotation with z along `approach` and the closing axis along `closing_axis`."""
    z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
    c = np.asarray(closing_axis, dtype=float)
    c = c - z * (c @ z)
    c /= np.linalg.norm(c) + 1e-12
    if closing_col == 0:
        x, y = c, np.cross(z, c)
    else:
        y = c
        x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)))


class Skills:
    def __init__(self, env, perception=None, memory=None, motion_source: str = "hybrid",
                 step_budget: int = 500, feasibility: Feasibility | None = None, controller: Controller | None = None):
        if motion_source not in ("index", "primitive", "hybrid"):
            raise ValueError(f"motion_source must be index|primitive|hybrid, got {motion_source!r}")
        self.env = env
        self.perception = perception
        self.memory = memory
        self.motion_source = motion_source
        self.step_budget = step_budget
        self.ctrl = controller or Controller(env)
        self.feas = feasibility or Feasibility()
        self.home_pose = self.ctrl.pose().copy()
        self.held: Held | None = None
        self._reg_cache: dict = {}

    # ================================================================ public skills
    def pick(self, obj: ObjectEstimate) -> SkillResult:
        return self._run("pick", self._pick, obj)

    def place(self, target: ObjectEstimate, relation: str = "on") -> SkillResult:
        return self._run("place", self._place, target, relation)

    def push(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult:
        return self._run("push", self._push, obj, dx, dy)

    def actuate(self, part: ObjectEstimate, action: str) -> SkillResult:
        from motormem.skills.articulation import actuate as _actuate
        return self._run("actuate", _actuate, self, part, action)

    def home(self) -> SkillResult:
        return self._run("home", self._home)

    def primitive_pick(self, obj: ObjectEstimate) -> SkillResult:
        return self._run("pick", self._primitive_pick, obj)

    def primitive_place(self, target: ObjectEstimate, relation: str = "on") -> SkillResult:
        return self._run("place", self._primitive_place, target, relation)

    # ================================================================ plumbing
    def _run(self, name: str, fn: Callable, *args) -> SkillResult:
        start = self.ctrl.steps
        self.ctrl.set_budget(self.step_budget)
        try:
            res = fn(*args)
        except StepBudgetExceeded:
            res = SkillResult(Status.TIMEOUT, f"{name}: step budget ({self.step_budget}) exhausted")
        except Exception as e:  # noqa: BLE001 - skills never raise to the caller
            if type(e).__name__ in ("EpisodeOver", "EpisodeOverError"):
                res = SkillResult(Status.EPISODE_OVER, "episode over")
            else:
                res = SkillResult(Status.ERROR, f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}")
        finally:
            self.ctrl.set_budget(None)
        res.steps = self.ctrl.steps - start
        return res

    def _use_index(self) -> bool:
        return self.motion_source in ("index", "hybrid") and self.memory is not None

    def _canonical(self, label: str) -> str | None:
        try:
            return self.memory.canonical(label)
        except Exception:
            return None

    def _query(self, kind: SegmentKind, obj: ObjectEstimate, strict: bool = False, **kw) -> list[Segment]:
        """Segments for obj. strict=True: only demos of exactly this object (and held object, if given)."""
        name = self._canonical(obj.label)
        if strict and name is None:
            return []
        ext = None if name else obj.extent
        try:
            segs = self.memory.query(kind, object_name=name, extent=ext, strict=True, **kw) if strict \
                else self.memory.query(kind, object_name=name, extent=ext, **kw)
        except TypeError:  # memory without a `strict` parameter: filter below
            segs = self.memory.query(kind, object_name=name, extent=ext, **kw)
        segs = list(segs or [])
        if strict:
            held = kw.get("held_name")
            segs = [s for s in segs if s.object_name == name and (held is None or s.held_name == held)]
        return segs

    def _register(self, src: np.ndarray, dst: np.ndarray) -> list[tuple[np.ndarray, float, float]]:
        key = (src.shape, src[:4].tobytes(), dst.shape, dst[:4].tobytes())
        if key not in self._reg_cache:
            s = src if len(src) <= 800 else src[np.linspace(0, len(src) - 1, 800).astype(int)]
            d = dst if len(dst) <= 1500 else dst[np.linspace(0, len(dst) - 1, 1500).astype(int)]
            self._reg_cache[key] = g.register_multi_yaw(s, d)
        return self._reg_cache[key]

    def _feasible(self, targets: np.ndarray) -> tuple[bool, str]:
        idx = np.unique(np.linspace(0, len(targets) - 1, min(len(targets), MAX_IK_WAYPOINTS)).astype(int))
        return self.feas.check(self.env.kinematics(), self.ctrl.to_site(targets[idx], 1.0))

    def _approach(self, T_contact: np.ndarray, lift: float = APPROACH_LIFT) -> np.ndarray:
        """Up (if needed) -> over to the pre-pose `lift` above T_contact -> descend to T_contact."""
        cur = self.ctrl.pose().copy()
        pre = T_contact.copy()
        pre[2, 3] += lift
        pts = [cur]
        if cur[2, 3] < pre[2, 3] - 0.01:
            up = cur.copy()
            up[2, 3] = pre[2, 3]
            pts.append(up)
        pts.append(pre)
        a = densify(pts, 0.02, 0.1)
        b = densify([pre, T_contact], 0.01, 0.1)
        return np.concatenate([a, b[1:]])

    @staticmethod
    def _lifted(T: np.ndarray, dz: float = LIFT) -> np.ndarray:
        up = T.copy()
        up[2, 3] += dz
        return densify([T, up], 0.02, 0.1)[1:]

    def _execute(self, plan: Plan, close: bool) -> None:
        """Track plan up to the event, actuate the gripper there, track the remainder."""
        e = plan.event
        g_before = -1.0 if close else 1.0
        self.ctrl.track(plan.targets[:e + 1], g_before)
        self.ctrl.gripper(1.0 if close else -1.0, 12 if close else 8)
        if e + 1 < len(plan.targets):
            self.ctrl.track(plan.targets[e + 1:], -g_before)

    def _ensure_open(self) -> None:
        if not self.ctrl.is_open():
            self.ctrl.gripper(-1.0, 10)

    # ================================================================ pick
    def _grasp_plans(self, obj: ObjectEstimate, segs: list[Segment]) -> Iterator[Plan]:
        for seg in segs:
            yield from self._segment_plans(seg, obj, close=True)

    def _segment_plans(self, seg: Segment, obj: ObjectEstimate, close: bool,
                       correct: Callable[[np.ndarray, Segment], list[tuple[str, np.ndarray]]] | None = None
                       ) -> Iterator[Plan]:
        hyps = self._register(seg.object_points, obj.points_world)
        if not hyps or hyps[0][2] < MIN_FITNESS:
            return
        best = hyps[0][2]
        near = [h for h in hyps if h[2] >= best - NEAR_TIE]
        yaws = [g.yaw_of(h[0]) for h in near]
        symmetric = len(near) > 1 and np.ptp(np.unwrap(yaws)) > np.radians(30)
        grip = np.asarray(seg.gripper, dtype=float)
        ev = np.flatnonzero(grip > 0) if close else np.flatnonzero((grip[:-1] > 0) & (grip[1:] <= 0)) + 1
        event = int(ev[0]) if len(ev) else len(grip) - 1
        R_cur = self.ctrl.pose()[:3, :3]
        cands = []
        for T_world_obj, rmse, fit in near:
            demo = g.compose(T_world_obj, seg.T_obj_eef)
            variants = correct(demo, seg) if correct else [("none", demo)]
            for tag, path in variants:
                s = _contact_start(path[:, :3, 3], event)
                cost = rot_angle(R_cur, path[s, :3, :3])
                cands.append((cost, tag, T_world_obj, path, s, fit))
        cands.sort(key=lambda c: c[0])
        for cost, tag, T_world_obj, path, s, fit in cands:
            app = self._approach(path[s])
            body = path[s:]
            tail = self._lifted(body[-1])
            targets = np.concatenate([app, body[1:], tail])
            ev_idx = len(app) - 1 + (event - s)
            gr = np.where(np.arange(len(targets)) < ev_idx, -1.0 if close else 1.0, 1.0 if close else -1.0)
            ok, reason = self._feasible(targets)
            if ok:
                yield Plan(targets, gr, ev_idx, seg.segment_id, T_world_obj, symmetric,
                           {"fitness": fit, "variant": tag})
            else:
                self._last_reason = reason

    def _pick(self, obj: ObjectEstimate) -> SkillResult:
        if obj.points_world is None or len(obj.points_world) < 20:
            return SkillResult(Status.PERCEPTION_FAILED, f"pick: too few points on mark {obj.mark}")
        self._last_reason = ""
        if self._use_index():
            segs = self._query(SegmentKind.GRASP, obj, strict=True)
            if not segs and self.motion_source == "index":
                return SkillResult(Status.NO_CANDIDATE, f"pick: no demo of {obj.label} in motor memory")
            n_tried, failed = 0, None
            for plan in self._grasp_plans(obj, segs):
                n_tried += 1
                res = self._do_grasp(plan, obj)
                if res.status == Status.OK or n_tried >= 2:
                    return res
                failed = res
                self._ensure_open()
                self.ctrl.track(self._lifted(self.ctrl.pose(), 0.05), -1.0)
            if failed is not None:
                return failed
            if self.motion_source == "index":
                st = Status.IK_FAILURE if self._last_reason else Status.NO_CANDIDATE
                return SkillResult(st, f"pick: no feasible grasp segment for '{obj.label}'"
                                   + (f" ({self._last_reason})" if self._last_reason else ""))
        res = self._primitive_pick(obj)
        res.used_fallback = True
        return res

    def _do_grasp(self, plan: Plan, obj: ObjectEstimate) -> SkillResult:
        self._ensure_open()
        e = plan.event
        self.ctrl.track(plan.targets[:e + 1], -1.0)
        self.ctrl.gripper(1.0, 12)
        T_close = self.ctrl.pose().copy()
        self.ctrl.track(plan.targets[e + 1:], 1.0)
        obs = self.ctrl.refresh()
        if not self.ctrl.holding(obs):
            self.held = None
            return SkillResult(Status.GRASP_FAILED, f"pick: gripper empty after grasp (opening {obs.proprio.gripper_opening:.2f})",
                               segment_id=plan.segment_id)
        self._set_held(obj, T_close, plan.T_world_obj, plan.symmetric)
        return SkillResult(Status.OK, f"picked {obj.label} (mark {obj.mark})", segment_id=plan.segment_id,
                           info=plan.info)

    def _set_held(self, obj: ObjectEstimate, T_close: np.ndarray, T_world_objmem: np.ndarray | None,
                  symmetric: bool) -> None:
        P = obj.points_world
        bottom = g.make_pose([obj.T_world_obj[0, 3], obj.T_world_obj[1, 3], P[:, 2].min()])
        name = self._canonical(obj.label) if self.memory is not None else None
        if T_world_objmem is None and name and self.memory is not None:
            try:  # primitive pick: still estimate the memory-frame pose for PLACE held-offset correction
                hyps = self._register(self.memory.object_points(name), P)
                if hyps and hyps[0][2] >= MIN_FITNESS:
                    T_world_objmem = hyps[0][0]
                    near = [h for h in hyps if h[2] >= hyps[0][2] - NEAR_TIE]
                    symmetric = len(near) > 1
            except Exception:
                T_world_objmem = None
        Tinv = g.inv(T_close)
        self.held = Held(label=obj.label, name=name,
                         T_eef_obj=None if T_world_objmem is None else Tinv @ T_world_objmem,
                         T_eef_bottom=Tinv @ bottom, extent=obj.extent.copy(), symmetric=symmetric)

    # ================================================================ primitive pick
    def _gripper_geometry(self) -> tuple[float, int]:
        k = self.ctrl.gripper_key()
        return GRIPPER_MAX_WIDTH.get(k, 0.08), GRIPPER_CLOSING_AXIS.get(k, 0)

    def _primitive_grasp_options(self, obj: ObjectEstimate) -> list[tuple[str, np.ndarray]]:
        """Top-down grasp poses: across the minor axis at the centroid if it fits, else on the edge."""
        width, col = self._gripper_geometry()
        P, T, ext = obj.points_world, obj.T_world_obj, obj.extent
        c = T[:3, 3]
        ax, minor = T[:3, 0], T[:3, 1]
        z_top, z_bot = P[:, 2].max(), P[:, 2].min()
        R_cur = self.ctrl.pose()[:3, :3]
        base_xy = self.ctrl.obs.T_world_base[:2, 3]
        opts = []
        if ext[1] <= width - 0.015:
            depth = float(np.clip(0.5 * ext[2], 0.01, 0.035))
            z = max(z_top - depth, z_bot + 0.012)
            for s in (1, -1):
                opts.append(("center", g.make_pose([c[0], c[1], z], topdown_rotation(s * minor, col))))
        else:  # too wide: grasp the rim/edge, fingers closing radially
            z = max(z_top - 0.02, z_bot + 0.012)
            for u, half in ((ax, ext[0] / 2), (-ax, ext[0] / 2), (minor, ext[1] / 2), (-minor, ext[1] / 2)):
                p = c + u * (half - 0.01)
                for s in (1, -1):
                    opts.append(("edge", g.make_pose([p[0], p[1], z], topdown_rotation(s * u, col))))
            # prefer edges nearest the robot (more reachable)
            opts.sort(key=lambda o: np.linalg.norm(o[1][:2, 3] - base_xy))
        opts.sort(key=lambda o: (o[0] != "center", rot_angle(R_cur, o[1][:3, :3]) + 2.0 * np.linalg.norm(o[1][:2, 3] - base_xy)))
        return opts

    def _primitive_pick(self, obj: ObjectEstimate) -> SkillResult:
        if obj.points_world is None or len(obj.points_world) < 20:
            return SkillResult(Status.PERCEPTION_FAILED, f"pick: too few points on mark {obj.mark}")
        reason = ""
        for kind, T_grasp in self._primitive_grasp_options(obj):
            app = self._approach(T_grasp, lift=LIFT)
            targets = np.concatenate([app, self._lifted(T_grasp)])
            ok, reason = self._feasible(targets)
            if not ok:
                continue
            self._ensure_open()
            self.ctrl.track(app, -1.0)
            self.ctrl.gripper(1.0, 12)
            T_close = self.ctrl.pose().copy()
            self.ctrl.track(self._lifted(T_close), 1.0)
            obs = self.ctrl.refresh()
            if not self.ctrl.holding(obs):
                self.held = None
                return SkillResult(Status.GRASP_FAILED, f"pick: primitive {kind} grasp missed {obj.label} "
                                   f"(opening {obs.proprio.gripper_opening:.2f})", used_fallback=True)
            self._set_held(obj, T_close, None, False)
            return SkillResult(Status.OK, f"picked {obj.label} (mark {obj.mark}) with primitive {kind} grasp",
                               used_fallback=True, info={"grasp": kind})
        return SkillResult(Status.IK_FAILURE, f"pick: no reachable top-down grasp for {obj.label} ({reason})",
                           used_fallback=True)

    # ================================================================ place
    def _held_corrections(self, demo: np.ndarray, seg: Segment) -> list[tuple[str, np.ndarray]]:
        h = self.held
        out = []
        if h is not None and h.T_eef_obj is not None and seg.T_eef_held is not None:
            full = g.compose(demo, seg.T_eef_held, g.inv(h.T_eef_obj))
            p_held = g.compose(demo, seg.T_eef_held)[:, :3, 3]            # demo held-object path
            pos = demo.copy()
            pos[:, :3, 3] = p_held - np.einsum("nij,j->ni", demo[:, :3, :3], h.T_eef_obj[:3, 3])
            out = [("pos", pos), ("full", full)] if h.symmetric else [("full", full), ("pos", pos)]
        out.append(("none", demo))
        return out

    def _place(self, target: ObjectEstimate, relation: str) -> SkillResult:
        if target.points_world is None or len(target.points_world) < 20:
            return SkillResult(Status.PERCEPTION_FAILED, f"place: too few points on mark {target.mark}")
        if not self.ctrl.holding():
            return SkillResult(Status.NOT_HOLDING, "place: gripper is empty")
        self._last_reason = ""
        if self._use_index():
            held_name = self.held.name if self.held else None
            segs = self._query(SegmentKind.PLACE, target, strict=True, relation=relation, held_name=held_name)
            if not segs and held_name is not None and self.motion_source == "hybrid":
                # cross-object reuse: any held object placed on/in this same target
                segs = self._query(SegmentKind.PLACE, target, strict=True, relation=relation)
            if not segs and self.motion_source == "index":
                return SkillResult(Status.NO_CANDIDATE, f"place: no demo of {target.label} in motor memory")
            for seg in segs:
                for plan in self._segment_plans(seg, target, close=False, correct=self._held_corrections):
                    self._execute(plan, close=False)
                    return self._finish_place(target, relation, plan.segment_id, False, plan.info)
            if self.motion_source == "index":
                st = Status.IK_FAILURE if self._last_reason else Status.NO_CANDIDATE
                return SkillResult(st, f"place: no feasible '{relation}' segment for '{target.label}'"
                                   + (f" ({self._last_reason})" if self._last_reason else ""))
        res = self._primitive_place(target, relation)
        res.used_fallback = True
        return res

    def _finish_place(self, target, relation, seg_id, fallback, info=None) -> SkillResult:
        self.ctrl.track(self._lifted(self.ctrl.pose(), 0.05), -1.0)
        obs = self.ctrl.refresh()
        if self.ctrl.holding(obs):
            return SkillResult(Status.GRASP_FAILED, "place: object still in gripper after release",
                               segment_id=seg_id, used_fallback=fallback)
        held = self.held.label if self.held else "object"
        self.held = None
        return SkillResult(Status.OK, f"placed {held} {relation} {target.label} (mark {target.mark})",
                           segment_id=seg_id, used_fallback=fallback, info=info or {})

    def _primitive_place(self, target: ObjectEstimate, relation: str) -> SkillResult:
        if not self.ctrl.holding():
            return SkillResult(Status.NOT_HOLDING, "place: gripper is empty", used_fallback=True)
        P, c = target.points_world, target.T_world_obj[:3, 3]
        top = P[:, 2].max()
        xy = c[:2].copy()
        h = self.held
        h_ext = h.extent if h is not None else np.zeros(3)
        if relation in RELATION_OFFSETS:
            u = self.ctrl.obs.T_world_base[:2, :2] @ np.asarray(RELATION_OFFSETS[relation], float)
            reach = float(np.abs(u @ target.T_world_obj[:2, :2]) @ target.extent[:2]) / 2 + max(h_ext[:2]) / 2 + 0.03
            xy = xy + u * reach
            top = float(np.percentile(P[:, 2], 5)) if relation != "top_of" else top
        elif relation not in ("on", "in", "top_of", "on_top_of", "inside"):
            return SkillResult(Status.BAD_ARGUMENT, f"place: unknown relation '{relation}'", used_fallback=True)
        R = self.ctrl.pose()[:3, :3].copy()
        off = h.T_eef_bottom[:3, 3] if h is not None else np.array([0, 0, 0.03])
        bottom = np.array([xy[0], xy[1], top + 0.015])
        T_rel = g.make_pose(bottom - R @ off, R)
        app = self._approach(T_rel, lift=APPROACH_LIFT)
        ok, reason = self._feasible(np.concatenate([app, self._lifted(T_rel)]))
        if not ok:
            return SkillResult(Status.IK_FAILURE, f"place: release pose over {target.label} unreachable ({reason})",
                               used_fallback=True)
        self.ctrl.track(app, 1.0)
        self.ctrl.gripper(-1.0, 10)
        return self._finish_place(target, relation, None, True, {"release_z": float(T_rel[2, 3])})

    # ================================================================ push
    def _push(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult:
        d = np.array([dx, dy], dtype=float)
        dist = float(np.linalg.norm(d))
        if not np.isfinite(dist) or dist < 0.005 or dist > 0.4:
            return SkillResult(Status.BAD_ARGUMENT, f"push: distance {dist:.3f} m outside 0.005..0.4")
        u = d / dist
        if self._use_index():
            for seg in self._query(SegmentKind.PUSH, obj):
                for plan in self._segment_plans(seg, obj, close=True):
                    net = plan.targets[-1, :2, 3] - plan.targets[plan.event, :2, 3]
                    if np.linalg.norm(net) > 0.01 and (net @ u) / np.linalg.norm(net) > np.cos(np.radians(45)):
                        self.ctrl.track(plan.targets, plan.gripper)
                        return SkillResult(Status.OK, f"pushed {obj.label} with demo motion", segment_id=plan.segment_id)
            if self.motion_source == "index":
                return SkillResult(Status.NO_CANDIDATE, f"push: no demo pushes {obj.label} in that direction")
        P, T, ext = obj.points_world, obj.T_world_obj, obj.extent
        c = T[:3, 3]
        half = float(np.abs(u @ T[:2, :2]) @ ext[:2]) / 2
        z = P[:, 2].min() + float(np.clip(0.5 * ext[2], 0.015, 0.03))
        _, col = self._gripper_geometry()
        R = topdown_rotation(np.array([-u[1], u[0], 0.0]), col)
        if rot_angle(self.ctrl.pose()[:3, :3], R) > rot_angle(self.ctrl.pose()[:3, :3], topdown_rotation(np.array([u[1], -u[0], 0.0]), col)):
            R = topdown_rotation(np.array([u[1], -u[0], 0.0]), col)
        start = g.make_pose([c[0] - u[0] * (half + 0.03), c[1] - u[1] * (half + 0.03), z], R)
        end = start.copy()
        end[:2, 3] += u * dist
        app = self._approach(start, lift=LIFT)
        stroke = densify([start, end], 0.01, 0.1)[1:]
        targets = np.concatenate([app, stroke, self._lifted(end)])
        ok, reason = self._feasible(targets)
        if not ok:
            return SkillResult(Status.IK_FAILURE, f"push: stroke unreachable ({reason})", used_fallback=True)
        self.ctrl.gripper(1.0, 8)
        self.ctrl.track(app, 1.0)
        self.ctrl.track(stroke, 1.0, max_steps_per_wp=6)
        self.ctrl.track(self._lifted(self.ctrl.pose()), 1.0)
        return SkillResult(Status.OK, f"pushed {obj.label} {dist * 100:.0f} cm", used_fallback=True)

    # ================================================================ home
    def _home(self) -> SkillResult:
        cur = self.ctrl.pose().copy()
        pts = [cur]
        if cur[2, 3] < self.home_pose[2, 3] - 0.01:
            up = cur.copy()
            up[2, 3] = self.home_pose[2, 3]
            pts.append(up)
        pts.append(self.home_pose)
        r = self.ctrl.track(densify(pts, 0.02, 0.1), 1.0 if self.ctrl.holding() else -1.0)
        return SkillResult(Status.OK if r.reached else Status.TIMEOUT,
                           "at home pose" if r.reached else f"home: stopped {r.pos_err * 100:.1f} cm short")
