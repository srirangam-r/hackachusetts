"""Skills: pick / place / push / actuate / home from motor memory, with a hand-written fallback (Req 4).

Inputs are only perceived ObjectEstimates, proprioception and the robot's own kinematics. Every motion
from the index is traceable to `segment_id`; primitive motions set `used_fallback`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator

import numpy as np

from motormem import geometry as g
from motormem.skills.control import (GRIPPER_CLOSING_AXIS, GRIPPER_MAX_WIDTH, GRIPPER_RIM_EXTRA_DEPTH, Controller,
                                     StepBudgetExceeded, densify)
from motormem.skills.feasibility import Feasibility
from motormem.types import ObjectEstimate, Segment, SegmentKind, SkillResult, Status

APPROACH_LIFT = 0.08      # pre-pose height above the first contact waypoint
LIFT = 0.10               # lift after grasp / retreat after release
CONTACT_RADIUS = 0.06     # segment poses within this distance of the gripper event = "contact part"
NEAR_TIE = 0.02           # registration hypotheses within this fitness of the best are near-ties
MIN_FITNESS = 0.3         # reject registrations worse than this
MAX_IK_WAYPOINTS = 60
BLOCKED_ERR = 0.015       # approach ending this far from the grasp pose = blocked; retry another grasp
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


def _same_rest_pose(seg, obj: ObjectEstimate) -> bool:
    """A demo grasp is only valid if the object rests the same way up as in the demo (an upright bottle vs a
    lying one). Compared by height: the live object's measured height vs the demo object's stored points.
    Live points are a partial view and can only under-estimate height, so only the clear case is rejected:
    the live object is much taller than the demo object (it stands where the demo one was lying)."""
    P = seg.object_points
    if P is None or len(P) < 10 or obj.extent is None:
        return True
    h_demo = float(P[:, 2].max() - P[:, 2].min())
    h_live = float(obj.points_world[:, 2].max() - obj.points_world[:, 2].min()) if len(obj.points_world) else 0.0
    if h_demo < 0.02 or h_live < 0.02:
        return True
    return h_live / h_demo <= 1.8


def _typical_first(segs: list) -> list:
    """Try first the grasp segments whose close point lies inside the object's own height range.

    Human demos vary; an occasional demo closes at or above the top of a flat object (e.g. a 2 cm box) and
    only worked thanks to finger compliance. A grasp must close where the object is, so segments whose
    close point (object frame) is within the object's z-extent go first; otherwise the query order is kept.
    Generic: uses only the demo's own stored object points.
    """
    def inside(s) -> bool:
        k = min(max(int(s.meta.get("key_index", len(s.T_obj_eef) - 1)), 0), len(s.T_obj_eef) - 1)
        z = float(s.T_obj_eef[k][2, 3])
        P = s.object_points
        if P is None or len(P) == 0:
            return True
        return float(P[:, 2].min()) - 0.002 <= z <= float(P[:, 2].max()) - 0.002

    return sorted(segs, key=lambda s: 0 if inside(s) else 1)


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
        self._placed_label: str | None = None
        self._placed_centroid: np.ndarray | None = None

    # ================================================================ public skills
    def pick(self, obj: ObjectEstimate, part: str | None = None) -> SkillResult:
        """pick(M) uses demos (else geometry); pick(M, part=...) grasps the named part, found inside M's mask
        by perception, with the geometric grasp (demos describe whole objects, not parts)."""
        if part:
            p = self._find_part(obj, part)
            if p is None:
                return SkillResult(Status.PERCEPTION_FAILED, f"pick: no '{part}' found on {obj.label}")
            res = self._run("pick", self._primitive_pick, p)
            res.used_fallback = True
            return res
        return self._run("pick", self._pick, obj)

    def _find_part(self, obj: ObjectEstimate, part: str) -> ObjectEstimate | None:
        parts_fn = getattr(self.perception, "parts", None) if self.perception is not None else None
        if parts_fn is None:
            return None
        try:
            cands = parts_fn(self.ctrl.refresh(), obj, [f"{obj.label} {part}", part])
        except Exception:
            return None
        cands = [c for c in cands if c.points_world is not None and len(c.points_world) >= 20]
        if not cands:
            return None
        best = max(cands, key=lambda c: c.score)
        best.label = f"{obj.label} {part}"
        return best

    def place(self, target: ObjectEstimate, relation: str = "on") -> SkillResult:
        from motormem.skills.targets import parent_of
        return self._run("place", self._place, parent_of(self, target), relation)

    def push(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult:
        return self._run("push", self._push, obj, dx, dy)

    def actuate(self, part: ObjectEstimate, action: str) -> SkillResult:
        from motormem.skills.articulation import actuate as _actuate
        if action in ("turn_on", "turn_off"):
            from motormem.skills.targets import knob_of
            part = knob_of(self, part)
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

    def _find_segments(self, kind: SegmentKind, obj: ObjectEstimate, **kw) -> list[Segment]:
        """Segments for obj by motion source.

        index: strict only (exact object; editability: deleting an object's demos removes the skill).
        hybrid: strict; else the best-token-overlap visible name ("wooden rack" -> wine_rack); else
        segments of the same kind (and relation, if given) ranked by extent. Excluded objects never appear.
        """
        segs = self._query(kind, obj, strict=True, **kw)
        if segs or self.motion_source != "hybrid":
            return segs
        rel = kw.get("relation")
        keep = (lambda ss: [s for s in ss if rel is None or s.relation == rel])
        name = None
        try:
            name = self.memory.lenient_name(obj.label)
        except Exception:
            name = None
        if name is not None:
            for kk in (kw, {k: v for k, v in kw.items() if k != "held_name"}):
                segs = keep([s for s in self.memory.query(kind, object_name=name, strict=True, **kk) or []
                             if s.object_name == name])
                if segs:
                    return segs
        try:
            segs = self.memory.query(kind, object_name=None, extent=obj.extent, **kw)
        except Exception:
            segs = []
        return keep(list(segs or []))

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

    def _clear_height(self, p0: np.ndarray, p1: np.ndarray, margin: float = 0.06) -> float:
        """Lowest safe transit height between p0 and p1: tallest depth point in a corridor along the path.

        Uses only the agentview depth of the current observation (no sim geometry). Points near the
        gripper (the robot itself) and near the destination (the thing we are descending onto) are ignored.
        """
        try:
            cam = self.ctrl.obs.cameras["agentview"]
            P = g.deproject(cam.depth, cam.K, cam.T_world_cam, stride=2)
        except Exception:
            return -np.inf
        if len(P) == 0:
            return -np.inf
        a, b = np.asarray(p0[:2]), np.asarray(p1[:2])
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((P[:, :2] - a) @ ab) / L2, 0.0, 1.0) if L2 > 1e-9 else np.zeros(len(P))
        d = np.linalg.norm(P[:, :2] - (a + t[:, None] * ab), axis=1)
        far_start = np.linalg.norm(P[:, :2] - a, axis=1) > 0.12
        far_end = np.linalg.norm(P[:, :2] - b, axis=1) > 0.08
        base_z = self.ctrl.obs.T_world_base[2, 3]
        sel = (d < 0.07) & far_start & far_end & (P[:, 2] < base_z + 0.6)
        if not sel.any():
            return -np.inf
        return float(np.percentile(P[sel, 2], 98)) + margin

    def _approach(self, T_contact: np.ndarray, lift: float = APPROACH_LIFT) -> np.ndarray:
        """Up -> over to the pre-pose `lift` above T_contact -> descend. The transit height clears every
        obstacle seen in depth along the way (no collision model of the scene needed)."""
        cur = self.ctrl.pose().copy()
        pre = T_contact.copy()
        pre[2, 3] += lift
        z_clear = max(pre[2, 3], self._clear_height(cur[:3, 3], pre[:3, 3]))
        pts = [cur]
        if cur[2, 3] < z_clear - 0.01:
            up = cur.copy()
            up[2, 3] = z_clear
            pts.append(up)
        if z_clear > pre[2, 3] + 0.01:
            over = pre.copy()
            over[2, 3] = z_clear
            pts.append(over)
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
            yield from self._segment_plans(seg, obj, close=True, correct=self._rim_depth(obj))

    def _rim_depth(self, obj: ObjectEstimate) -> Callable | None:
        """Grasp correction for this gripper: rim grasps (object wider than the jaw, closing near its top)
        go GRIPPER_RIM_EXTRA_DEPTH deeper. None (no-op) for the Panda."""
        extra = GRIPPER_RIM_EXTRA_DEPTH.get(self.ctrl.gripper_key(), 0.0)
        width, _ = self._gripper_geometry()
        if extra <= 0 or obj.extent[1] <= width - 0.015:
            return None
        z_top, z_bot = obj.points_world[:, 2].max(), obj.points_world[:, 2].min()

        def correct(demo: np.ndarray, seg: Segment) -> list[tuple[str, np.ndarray]]:
            grip = np.asarray(seg.gripper, dtype=float)
            ev = np.flatnonzero(grip > 0)
            z_ev = demo[int(ev[0]) if len(ev) else -1, 2, 3]
            dz = min(extra, max(0.0, z_ev - (z_bot + 0.012)))
            if z_ev < z_top - 0.035 or dz <= 0:
                return [("none", demo)]
            deeper = demo.copy()
            deeper[:, 2, 3] -= dz
            return [("rim_deeper", deeper)]
        return correct

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
            segs = _typical_first([s for s in self._find_segments(SegmentKind.GRASP, obj) if _same_rest_pose(s, obj)])
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
        r = self.ctrl.track(plan.targets[:e + 1], -1.0)
        if r.pos_err > BLOCKED_ERR and plan.info.get("variant") == "rim_deeper":
            # stopped short of a rim grasp (e.g. the forearm touches the scene): do not pinch with the fingertips
            return SkillResult(Status.GRASP_FAILED, f"pick: approach blocked {r.pos_err * 100:.1f} cm short",
                               segment_id=plan.segment_id, info={"blocked": True})
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
        if ext[1] > ext[0]:  # keep `minor` the SHORT horizontal axis
            ax, minor = minor, ax
            ext = np.array([ext[1], ext[0], ext[2]])
        z_top, z_bot = P[:, 2].max(), P[:, 2].min()
        R_cur = self.ctrl.pose()[:3, :3]
        base_xy = self.ctrl.obs.T_world_base[:2, 3]
        opts = []
        if ext[1] <= width - 0.015:
            # across the short horizontal axis at ~40% of the height (at most 3.5 cm below the top), with
            # the fingertips clear of the table
            h = max(float(ext[2]), float(z_top - z_bot), 0.005)
            z = max(z_top - min(0.6 * h, 0.035), z_top - h + 0.008)
            for s in (1, -1):
                opts.append(("center", g.make_pose([c[0], c[1], z], topdown_rotation(s * minor, col))))
        else:  # too wide: grasp the rim/edge, fingers closing radially
            z = max(z_top - 0.02 - GRIPPER_RIM_EXTRA_DEPTH.get(self.ctrl.gripper_key(), 0.0), z_bot + 0.012)
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
        n_blocked = 0
        opts = self._primitive_grasp_options(obj)
        for i_opt, (kind, T_grasp) in enumerate(opts):
            app = self._approach(T_grasp, lift=LIFT)
            targets = np.concatenate([app, self._lifted(T_grasp)])
            ok, reason = self._feasible(targets)
            if not ok:
                continue
            self._ensure_open()
            r = self.ctrl.track(app, -1.0)
            rim_retry = kind == "edge" and GRIPPER_RIM_EXTRA_DEPTH.get(self.ctrl.gripper_key(), 0.0) > 0
            if rim_retry and r.pos_err > BLOCKED_ERR and n_blocked < 2 and i_opt < len(opts) - 1:
                # the arm stopped short (e.g. a link touches the scene): back off and try the next grasp
                n_blocked += 1
                reason = f"approach blocked {r.pos_err * 100:.1f} cm short"
                self.ctrl.track(self._lifted(self.ctrl.pose(), 0.05), -1.0)
                continue
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
            if not segs and self.motion_source == "hybrid":
                segs = self._find_segments(SegmentKind.PLACE, target, relation=relation, held_name=held_name)
            if not segs and self.motion_source == "index":
                return SkillResult(Status.NO_CANDIDATE, f"place: no demo of {target.label} in motor memory")
            # gather feasible place plans (lazily); keep at most two so we can retry once if the first misses
            plans = []
            for seg in segs:
                for plan in self._segment_plans(seg, target, close=False, correct=self._held_corrections):
                    plans.append(plan)
                    if len(plans) >= 2:
                        break
                if len(plans) >= 2:
                    break
            if plans:
                return self._place_with_retry(target, relation,
                                              [(p.segment_id, False, p.info, p) for p in plans])
            if self.motion_source == "index":
                st = Status.IK_FAILURE if self._last_reason else Status.NO_CANDIDATE
                return SkillResult(st, f"place: no feasible '{relation}' segment for '{target.label}'"
                                   + (f" ({self._last_reason})" if self._last_reason else ""))
        return self._place_with_retry(target, relation, [(None, True, None, None)])

    def _place_with_retry(self, target, relation, attempts) -> SkillResult:
        """Execute place attempts in order; verify each; retry once if a verified failure is still
        graspable and another attempt remains. `attempts` items are (seg_id, fallback, info, plan)."""
        last = None
        for i, (seg_id, fallback, info, plan) in enumerate(attempts):
            if plan is not None:
                self._execute(plan, close=False)
                res = self._finish_place(target, relation, seg_id, fallback, info)
            else:
                res = self._primitive_place(target, relation)
                res.used_fallback = True
            last = res
            if res.status != Status.OK:                 # release/IK failure: let the brain handle it
                return res
            # OK means released; now check the spatial outcome with perception
            ok, reason, still_graspable = self._verify_place(target, relation)
            if ok:
                return res
            # Only retry when the object is still in the gripper (a re-place can help); once it has
            # been released a second place would just disturb the scene, so report STALL and let the
            # brain re-pick and retry.
            if self.ctrl.holding() and i + 1 < len(attempts):
                continue
            res.status = Status.STALL
            res.message = f"placed but not {relation} {target.label}: {reason}"
            return res
        return last

    def _verify_place(self, target: ObjectEstimate, relation: str) -> tuple[bool, str, bool]:
        """(holds, reason, still_graspable). Uses perception + proprioception only (fairness)."""
        from motormem.skills.verify import locate, relation_holds
        if self.perception is None:
            return True, "", False                       # no perception (unit tests): trust the release
        held_label = self._placed_label or (self.held.label if self.held else None)
        t_near = np.asarray(target.T_world_obj, float)[:3, 3]
        held_near = self._placed_centroid if self._placed_centroid is not None else t_near
        held_now = locate(self, held_label, held_near, radius=0.20) if held_label else None
        tgt_now = locate(self, target.label, t_near, radius=0.20) or target
        if held_now is None:
            return True, "could not re-locate the placed object", False
        base = self.ctrl.obs.T_world_base
        holds, reason = relation_holds(held_now, tgt_now, relation, T_world_base=base)
        still = bool(held_now.extent[:2].max() <= self._gripper_geometry()[0] + 0.02)
        return holds, reason, still

    def _finish_place(self, target, relation, seg_id, fallback, info=None) -> SkillResult:
        held = self.held.label if self.held else "object"
        # remember what/where we released so verification can re-locate it after the gripper clears
        self._placed_label = held
        self._placed_centroid = self.ctrl.pose()[:3, 3].copy()
        self.ctrl.track(self._lifted(self.ctrl.pose(), 0.05), -1.0)
        obs = self.ctrl.refresh()
        if self.ctrl.holding(obs):
            return SkillResult(Status.GRASP_FAILED, "place: object still in gripper after release",
                               segment_id=seg_id, used_fallback=fallback)
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
    def _push_candidates(self, obj: ObjectEstimate, u: np.ndarray) -> list[tuple]:
        """(angle, seg, T_world_obj) for index PUSH segments whose demo displacement, mapped onto the live
        object, points within 90 deg of u. Symmetric objects (yaw-ambiguous registration) are yawed about
        their own vertical axis so the demo displacement aligns with u exactly. Best-aligned first."""
        segs = [s for s in self._find_segments(SegmentKind.PUSH, obj, k=500)
                if s.meta.get("displacement") is not None and s.meta.get("T_world_obj") is not None
                and s.meta.get("motion_t") is not None]
        hyps_by_name: dict[str, list] = {}
        out = []
        for seg in segs:
            if seg.object_name not in hyps_by_name:
                src = self.memory.object_points(seg.object_name)
                if len(src) < 20:
                    src = np.asarray(seg.object_points, float)
                hyps = self._register(src, obj.points_world) if len(src) >= 20 else []
                if hyps and hyps[0][2] >= MIN_FITNESS:
                    near = [h for h in hyps if h[2] >= hyps[0][2] - NEAR_TIE]
                    yaws = [g.yaw_of(h[0]) for h in near]
                    sym = len(near) > 1 and np.ptp(np.unwrap(yaws)) > np.radians(30)
                    hyps_by_name[seg.object_name] = [(h[0], sym) for h in near]
                else:
                    hyps_by_name[seg.object_name] = []
            d_obj = np.asarray(seg.meta["T_world_obj"], float)[:3, :3].T @ np.asarray(seg.meta["displacement"], float)
            for T_wo, sym in hyps_by_name[seg.object_name]:
                d = (T_wo[:3, :3] @ d_obj)[:2]
                if np.linalg.norm(d) < 0.01:
                    continue
                ang = float(np.arctan2(d[0] * u[1] - d[1] * u[0], d @ u))   # signed angle d -> u
                T = T_wo
                if sym:  # any yaw is an equally good registration: align the demo push with the request
                    T = T_wo.copy()
                    c, s_ = np.cos(ang), np.sin(ang)
                    T[:3, :3] = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]]) @ T_wo[:3, :3]
                    ang = 0.0
                if abs(ang) < np.radians(90):
                    out.append((abs(ang), seg, T))
        out.sort(key=lambda c: (round(c[0], 3), c[1].segment_id))
        return out

    def _push_plan(self, seg: Segment, T_world_obj: np.ndarray, max_travel: float | None = None
                   ) -> tuple[np.ndarray, np.ndarray, int]:
        """Approach -> demo contact path until the object stops moving (or the gripper has travelled
        `max_travel` in contact: the demo's stroke length is the demonstrator's, not ours) -> lift."""
        path = g.compose(T_world_obj, seg.T_obj_eef)
        n = len(path)
        t0 = int(seg.meta.get("t_range", [0])[0])
        a, b = (int(t) - t0 for t in seg.meta["motion_t"])
        ia = int(np.clip(a, 0, n - 1))
        ib = int(np.clip(b + 2, ia, n - 1))
        s = _contact_start(path[:, :3, 3], ia)
        grip = np.asarray(seg.gripper, float)
        app = self._approach(path[s], lift=APPROACH_LIFT)
        if max_travel is not None and ib > s + 1:
            xy = path[s:ib + 1, :2, 3]
            travel = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
            cut = int(np.searchsorted(travel, max_travel))
            ib = int(np.clip(s + cut, s + 1, ib))
        body = path[s + 1:ib + 1]
        tail = self._lifted(path[ib], 0.06)
        targets = np.concatenate([app, body, tail])
        gr = np.concatenate([np.full(len(app), grip[s]), grip[s + 1:ib + 1], np.full(len(tail), grip[ib])])
        return targets, np.where(gr > 0, 1.0, -1.0), len(app)

    PUSH_TOL = 0.03          # stop when the remaining displacement is below this (metres)
    PUSH_MAX_STROKES = 3

    def _push(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult:
        """Closed loop: push, re-measure the object, and push again toward the remaining displacement
        (at most PUSH_MAX_STROKES strokes), then report the measured final displacement."""
        from motormem.skills.verify import locate
        d0 = np.array([dx, dy], dtype=float)
        dist0 = float(np.linalg.norm(d0))
        if not np.isfinite(dist0) or dist0 < 0.005 or dist0 > 0.4:
            return SkillResult(Status.BAD_ARGUMENT, f"push: distance {dist0:.3f} m outside 0.005..0.4")
        if obj.points_world is None or len(obj.points_world) < 20:
            return SkillResult(Status.PERCEPTION_FAILED, f"push: too few points on mark {obj.mark}")
        start_xy = np.asarray(obj.T_world_obj, float)[:2, 3].copy()
        base_R = self.ctrl.obs.T_world_base[:3, :3]
        target_xy = start_xy + (base_R[:2, :2] @ d0)          # world-frame goal position
        # Cap each stroke at the skill budget so a single stalled stroke cannot starve the loop; the
        # loop re-measures between strokes and reports measured progress rather than a bare TIMEOUT
        # (this is the UR5e task-5 fix: one long stroke used to exhaust the 500-step budget).
        stroke_cap = self.step_budget
        cur_obj = obj
        last = None
        for stroke in range(self.PUSH_MAX_STROKES):
            cur_xy = np.asarray(cur_obj.T_world_obj, float)[:2, 3]
            rem_w = target_xy - cur_xy                        # remaining displacement, world xy
            rem = base_R[:2, :2].T @ rem_w                    # into base frame (dx, dy the skill wants)
            if float(np.linalg.norm(rem)) <= self.PUSH_TOL and stroke > 0:
                break
            self.ctrl.set_budget(stroke_cap)                  # fresh per-stroke cap
            try:
                res = self._push_once(cur_obj, float(rem[0]), float(rem[1]))
            except StepBudgetExceeded:
                res = SkillResult(Status.STALL, "push: stroke hit its step cap", used_fallback=True)
            last = res
            if res.status not in (Status.OK, Status.STALL):
                self.ctrl.set_budget(None)
                return res
            # re-measure where the object ended up (perception + proprioception only)
            relocated = locate(self, obj.label, np.append(cur_xy, obj.T_world_obj[2, 3]), radius=0.25)
            if relocated is None:
                break
            cur_obj = relocated
        self.ctrl.set_budget(None)
        final_xy = np.asarray(cur_obj.T_world_obj, float)[:2, 3]
        moved_w = final_xy - start_xy
        moved_b = base_R[:2, :2].T @ moved_w
        rem_b = base_R[:2, :2].T @ (target_xy - final_xy)
        rem_cm = float(np.linalg.norm(rem_b) * 100)
        st = Status.OK if rem_cm <= self.PUSH_TOL * 100 else Status.STALL
        msg = (f"pushed {obj.label}: moved dx={moved_b[0]:.2f} dy={moved_b[1]:.2f} m "
               f"({rem_cm:.1f} cm short of the request)")
        src = (last.segment_id if last else None)
        return SkillResult(st, msg, segment_id=src, used_fallback=bool(last and last.used_fallback),
                           info={"moved_base": moved_b.tolist(), "remaining_cm": rem_cm})

    def _push_once(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult:
        """One push stroke toward (dx, dy) in the base frame (metres)."""
        d = np.array([dx, dy], dtype=float)
        dist = float(np.linalg.norm(d))
        if not np.isfinite(dist) or dist < 0.005:
            return SkillResult(Status.OK, "push: already within tolerance")
        dist = min(dist, 0.4)
        if obj.points_world is None or len(obj.points_world) < 20:
            return SkillResult(Status.PERCEPTION_FAILED, f"push: too few points on mark {obj.mark}")
        u = d / np.linalg.norm(d)
        self._last_reason = ""
        if self._use_index():
            cands = self._push_candidates(obj, u)
            for ang, seg, T_wo in cands[:12]:
                targets, gr, n_app = self._push_plan(seg, T_wo, max_travel=dist)
                ok, reason = self._feasible(targets)
                if not ok:
                    self._last_reason = reason
                    continue
                if gr[0] > 0:
                    self.ctrl.gripper(1.0, 8)
                else:
                    self._ensure_open()
                self.ctrl.track(targets[:n_app], float(gr[0]))
                self.ctrl.track(targets[n_app:], gr[n_app:], max_steps_per_wp=6)
                disp_w = np.zeros(3)
                disp_w[:2] = (T_wo[:3, :3] @ np.asarray(seg.meta["T_world_obj"], float)[:3, :3].T
                              @ np.asarray(seg.meta["displacement"], float))[:2]
                db = self.ctrl.obs.T_world_base[:3, :3].T @ disp_w
                return SkillResult(Status.OK, f"pushed {obj.label} with demo motion: demo displacement "
                                   f"dx={db[0]:.2f} dy={db[1]:.2f} m ({np.degrees(ang):.0f} deg from request)",
                                   segment_id=seg.segment_id,
                                   info={"demo_displacement": db[:2].tolist(), "angle_deg": float(np.degrees(ang))})
            if self.motion_source == "index":
                st = Status.IK_FAILURE if (cands and self._last_reason) else Status.NO_CANDIDATE
                return SkillResult(st, f"push: no demo pushes {obj.label} in that direction"
                                   + (f" ({self._last_reason})" if self._last_reason else ""))
        P, T, ext = obj.points_world, obj.T_world_obj, obj.extent
        c = T[:3, 3]
        half = float(np.abs(u @ T[:2, :2]) @ ext[:2]) / 2
        # contact height from the measured top surface and height: thin objects (plates) are pushed with
        # the closed fingertips at most 1 cm above the table
        z_top = float(P[:, 2].max())
        h = max(float(ext[2]), z_top - float(P[:, 2].min()), 0.005)
        table = z_top - h
        z = table + (min(0.01, 0.5 * h + 0.005) if h < 0.03 else float(np.clip(0.5 * h, 0.015, 0.03)))
        _, col = self._gripper_geometry()
        R = topdown_rotation(np.array([-u[1], u[0], 0.0]), col)
        if rot_angle(self.ctrl.pose()[:3, :3], R) > rot_angle(self.ctrl.pose()[:3, :3], topdown_rotation(np.array([u[1], -u[0], 0.0]), col)):
            R = topdown_rotation(np.array([u[1], -u[0], 0.0]), col)
        start = g.make_pose([c[0] - u[0] * (half + 0.03), c[1] - u[1] * (half + 0.03), z], R)
        end = start.copy()
        end[:2, 3] += u * (dist + 0.02)   # the closed fingers first cover the ~2 cm gap to the near edge
        app = self._approach(start, lift=LIFT)
        stroke = densify([start, end], 0.01, 0.1)[1:]
        targets = np.concatenate([app, stroke, self._lifted(end)])
        ok, reason = self._feasible(targets)
        if not ok:
            return SkillResult(Status.IK_FAILURE, f"push: stroke unreachable ({reason})", used_fallback=True)
        self.ctrl.gripper(1.0, 8)
        # turn to the push orientation in place first: a large yaw change during the descent stalls the
        # translation (the tracker sees no progress) and the stroke then runs above the object
        cur = self.ctrl.pose().copy()
        turned = cur.copy()
        turned[:3, :3] = R
        self.ctrl.track(densify([cur, turned], 0.02, 0.1), 1.0)
        app = self._approach(start, lift=LIFT)
        self.ctrl.track(app, 1.0)
        self.ctrl.track(stroke, 1.0, max_steps_per_wp=6)
        self.ctrl.track(self._lifted(self.ctrl.pose()), 1.0)
        return SkillResult(Status.OK, f"pushed {obj.label} {dist * 100:.0f} cm (contact {100 * (z - table):.1f} cm "
                           f"above the table)", used_fallback=True)

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
