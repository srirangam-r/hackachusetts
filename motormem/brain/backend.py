"""What the robot MCP server drives: a `Backend` owns one episode (sim, perception, skills).

`FakeBackend` is a self-contained 2-D tabletop toy for tests and headless Kiro runs.
`RealBackend` wires the env/perception/memory/skills streams (imports are lazy so this module
loads without them).

Conventions shared by every backend (and promised to the brain in robot-protocol.md):
- Marks are integers assigned by the last `observe`; `run` resolves marks against it.
- Positions in text are in the robot base frame, centimetres: x forward, y left, z up.
- `push(M, dx, dy)` offsets are metres in the same base frame.
- Nothing returned by `observe` / `run` may contain simulator state or success information.
  `runner_result()` is the only place `success` appears, and it is runner-only.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from motormem.brain.program import Call, format_call
from motormem.types import EpisodeSpec, Status


# ============================================================================ interface
@dataclass
class MarkInfo:
    """What the brain is told about one mark (all from perception, never from sim state)."""
    mark: int
    label: str
    xyz: np.ndarray            # (3,) metres, robot base frame
    size: np.ndarray           # (3,) metres


@runtime_checkable
class Backend(Protocol):
    instruction: str

    def observe(self, find: list[str]) -> tuple[list[Image.Image], str]:
        """Perceive `find` nouns. Returns [agentview overlay, wrist overlay] and a text report.
        Assigns fresh marks that the next `run` resolves against."""

    def run(self, calls: list[Call]) -> tuple[list[str], list[Image.Image], str]:
        """Execute calls in order, stopping at the first non-OK status. Returns one status
        line per call (attempted or skipped), fresh overlay images and a fresh text report
        (marks re-assigned by re-perceiving the last `find`)."""

    def done(self, summary: str) -> None:
        """The brain declares the episode finished."""

    def episode_over(self) -> bool:
        """True after done(), after the runner-only success check fired, or when the step
        budget is spent. The server then answers every tool with EPISODE_OVER."""

    def last_marks(self) -> dict[int, MarkInfo]:
        """Marks from the last observation (used by the scripted rung-0 runner)."""

    def runner_result(self) -> dict:
        """RUNNER-ONLY: {"success": bool, "ended_by": str|None, "sim_steps": int,
        "segment_ids": [...], "fallbacks": int, ...}. Never sent to the brain."""

    def close(self) -> None:
        ...


def status_line(i: int, call: Call, status: Status | str, message: str) -> str:
    s = status.value if isinstance(status, Status) else str(status)
    return f"{i}. {format_call(call)} -> {s}" + (f": {message}" if message else "")


def describe_marks(marks: dict[int, MarkInfo], gripper: str, missing: list[str] = ()) -> str:
    lines = ["Marks (robot base frame, cm; x = forward, y = left, z = up):"]
    if not marks:
        lines.append("  (none)")
    for m in sorted(marks):
        mi = marks[m]
        x, y, z = (100 * np.asarray(mi.xyz, float)).tolist()
        sx, sy, sz = (100 * np.asarray(mi.size, float)).tolist()
        lines.append(f"  [{m}] {mi.label}: x={x:.1f} y={y:.1f} z={z:.1f}, size {sx:.0f}x{sy:.0f}x{sz:.0f}")
    if missing:
        lines.append("Not found: " + ", ".join(missing))
    lines.append(f"Gripper: {gripper}")
    return "\n".join(lines)


# ============================================================================ fake world
FAKE_LIBERO_GOAL = [
    "open the middle drawer of the cabinet",
    "put the bowl on the stove",
    "put the wine bottle on top of the cabinet",
    "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet",
    "push the plate to the front of the stove",
    "put the cream cheese in the bowl",
    "turn on the stove",
    "put the bowl on the plate",
    "put the wine bottle on the rack",
]
_STOP = {"the", "a", "an", "of", "to", "on", "in", "and", "inside", "top", "front"}
_STOP_MATCH = {"the", "a", "an", "of", "to", "and"}


@dataclass
class _Obj:
    label: str
    xyz: np.ndarray
    size: np.ndarray
    color: tuple
    kind: str = "rigid"            # rigid | fixture | drawer | knob
    support: tuple | None = None   # (relation, label)
    is_open: bool = False          # drawer
    is_on: bool = False            # knob


def _words(s: str, stop=_STOP_MATCH) -> set[str]:
    return {w for w in s.lower().replace("_", " ").split() if w not in stop}


def _matches(label: str, noun: str) -> bool:
    a, b = _words(label), _words(noun)
    return bool(a) and bool(b) and (a <= b or b <= a)


class FakeBackend:
    """Toy LIBERO-Goal-like tabletop. Deterministic per (seed, init_index); optional injected
    grasp failures (`fail_prob`) exercise the brain's retry protocol."""

    STEP_COST = {"pick": 80, "place": 80, "push": 60, "actuate": 100, "look": 10, "home": 40}
    VIEW = (0.15, 0.85, -0.40, 0.40)    # x_min, x_max, y_min, y_max (metres, base frame)

    def __init__(self, spec: EpisodeSpec, fail_prob: float | None = None, img_size: int = 384):
        self.spec = spec
        tid = int(spec.task_id) if str(spec.task_id).isdigit() else 0
        self.task = tid % len(FAKE_LIBERO_GOAL)
        self.instruction = FAKE_LIBERO_GOAL[self.task]
        self.rng = random.Random(1000 * spec.seed + spec.init_index)
        self.fail_prob = float(os.environ.get("MOTORMEM_FAKE_FAIL", "0")) if fail_prob is None else fail_prob
        self.img_size = img_size
        self.objs = self._layout()
        self.gripper = np.array([0.35, 0.0, 0.30])
        self.holding: str | None = None
        self.sim_steps = 0
        self.over = False
        self.ended_by: str | None = None
        self.success = False
        self.marks: dict[int, MarkInfo] = {}
        self._mark_label: dict[int, str] = {}
        self.last_find: list[str] = []
        self.statuses: list[str] = []
        self.summary: str | None = None

    # ------------------------------------------------------------------ layout
    def _layout(self) -> dict[str, _Obj]:
        j = lambda: self.rng.uniform(-0.03, 0.03)
        O = {}
        def add(label, x, y, size, color, kind="rigid", z=None):
            O[label] = _Obj(label, np.array([x + j(), y + j(), size[2] / 2 if z is None else z]),
                            np.array(size, float), color, kind)
        add("cabinet", 0.72, 0.28, (0.22, 0.20, 0.30), (150, 105, 60), "fixture")
        cab = O["cabinet"].xyz
        for k, (name, z) in enumerate([("top drawer", 0.25), ("middle drawer", 0.15), ("bottom drawer", 0.05)]):
            O[name] = _Obj(name, np.array([cab[0] - 0.13, cab[1] - 0.06 + 0.06 * k, z]),
                           np.array([0.03, 0.05, 0.02]), (205, 160, 90), "drawer")
        add("stove", 0.62, -0.22, (0.20, 0.20, 0.03), (70, 70, 70), "knob")
        add("wine rack", 0.78, -0.02, (0.10, 0.16, 0.12), (120, 70, 35), "fixture")
        add("plate", 0.40, -0.12, (0.18, 0.18, 0.02), (230, 230, 230))
        add("black bowl", 0.48, 0.08, (0.12, 0.12, 0.06), (25, 25, 25))
        add("cream cheese", 0.38, 0.20, (0.07, 0.04, 0.04), (235, 220, 120))
        add("wine bottle", 0.55, -0.02, (0.07, 0.07, 0.25), (110, 20, 45))
        return O

    # ------------------------------------------------------------------ runner-only goal
    def _goal(self) -> bool:
        o, t = self.objs, self.task
        sup = lambda name: o[name].support if self.holding != name else None
        on = lambda a, b, rels=("on", "top_of"): (s := sup(a)) is not None and s[1] == b and s[0] in rels
        if t == 0:
            return o["middle drawer"].is_open
        if t == 1:
            return on("black bowl", "stove")
        if t == 2:
            return on("wine bottle", "cabinet")
        if t == 3:
            return o["top drawer"].is_open and on("black bowl", "top drawer", ("in",))
        if t == 4:
            return on("black bowl", "cabinet")
        if t == 5:
            st = o["stove"].xyz
            target = np.array([st[0] - 0.17, st[1]])
            return self.holding != "plate" and np.linalg.norm(o["plate"].xyz[:2] - target) < 0.06
        if t == 6:
            return on("cream cheese", "black bowl", ("in", "on"))
        if t == 7:
            return o["stove"].is_on
        if t == 8:
            return on("black bowl", "plate")
        return on("wine bottle", "wine rack", ("on", "in"))

    # ------------------------------------------------------------------ Backend API
    def observe(self, find: list[str]) -> tuple[list[Image.Image], str]:
        self.last_find = [f for f in (find or []) if isinstance(f, str) and f.strip()][:12]
        visible = list(self.objs.values())
        if self.last_find:
            hits = [o for o in visible if any(_matches(o.label, f) for f in self.last_find)]
            missing = [f for f in self.last_find if not any(_matches(o.label, f) for o in visible)]
        else:
            hits, missing = visible, []
        hits.sort(key=lambda o: -o.xyz[1])                 # left-to-right in the image
        self.marks = {i + 1: MarkInfo(i + 1, o.label, o.xyz.copy(), o.size.copy()) for i, o in enumerate(hits)}
        self._mark_label = {m: mi.label for m, mi in self.marks.items()}
        return self._images(), describe_marks(self.marks, self._gripper_text(), missing)

    def run(self, calls: list[Call]) -> tuple[list[str], list[Image.Image], str]:
        lines: list[str] = []
        for i, call in enumerate(calls, start=1):
            if self.over:
                lines.append(status_line(i, call, Status.EPISODE_OVER, ""))
                continue
            status, msg = self._exec(call)
            lines.append(status_line(i, call, status, msg))
            self.statuses.append(status.value)
            if self.sim_steps >= self.spec.max_steps and not self.over:
                self.over, self.ended_by = True, "budget"
            if self._goal() and not self.over:              # runner-only check, standard protocol
                self.over, self.ended_by, self.success = True, "success", True
            if status != Status.OK:
                for k, rest in enumerate(calls[i:], start=i + 1):
                    lines.append(status_line(k, rest, "SKIPPED", "earlier step failed"))
                break
        imgs, text = self.observe(self.last_find)
        return lines, imgs, text

    def done(self, summary: str) -> None:
        self.summary = summary
        if not self.over:
            self.over, self.ended_by = True, "done"

    def episode_over(self) -> bool:
        return self.over

    def last_marks(self) -> dict[int, MarkInfo]:
        return dict(self.marks)

    def runner_result(self) -> dict:
        return {"success": bool(self.success or self._goal()), "ended_by": self.ended_by,
                "sim_steps": self.sim_steps, "segment_ids": [], "fallbacks": 0,
                "skill_statuses": list(self.statuses), "backend": "fake"}

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ skills (toy)
    def _exec(self, call: Call) -> tuple[Status, str]:
        self.sim_steps += self.STEP_COST.get(call.name, 50)
        if call.name == "home":
            self.gripper = np.array([0.35, 0.0, 0.30])
            return Status.OK, "at home pose"
        label = self._mark_label.get(call.mark)
        if label is None:
            known = ", ".join(map(str, sorted(self._mark_label))) or "none; call observe first"
            return Status.BAD_ARGUMENT, f"unknown mark {call.mark} (current marks: {known})"
        o = self.objs[label]
        if call.name == "look":
            x, y, z = (100 * o.xyz).tolist()
            return Status.OK, f"{label} at x={x:.1f} y={y:.1f} z={z:.1f} cm"
        if call.name == "pick":
            if self.holding:
                return Status.BAD_ARGUMENT, f"already holding the {self.holding}; place it first"
            if o.kind != "rigid":
                return Status.GRASP_FAILED, f"{label} cannot be lifted"
            self.gripper = o.xyz + np.array([0, 0, 0.1])
            if self.rng.random() < self.fail_prob:
                o.xyz[:2] += [self.rng.uniform(-0.03, 0.03), self.rng.uniform(-0.03, 0.03)]
                return Status.GRASP_FAILED, "gripper closed on nothing"
            self.holding, o.support = label, None
            o.xyz[2] += 0.10
            return Status.OK, f"holding {label}"
        if call.name == "place":
            if not self.holding:
                return Status.NOT_HOLDING, "nothing in the gripper"
            if label == self.holding:
                return Status.BAD_ARGUMENT, "target is the held object"
            rel = call.args[1]
            if o.kind == "drawer" and rel == "in" and not o.is_open:
                return Status.COLLISION, f"{label} is closed"
            held = self.objs[self.holding]
            off = {"left_of": (0, 0.12), "right_of": (0, -0.12), "front_of": (-0.12, 0),
                   "back_of": (0.12, 0)}.get(rel, (0, 0))
            held.xyz[:2] = o.xyz[:2] + np.array(off)
            held.xyz[2] = (o.xyz[2] + o.size[2] / 2 + held.size[2] / 2) if rel in ("on", "top_of") else o.xyz[2]
            held.support = (rel, label)
            self.holding = None
            self.gripper = held.xyz + np.array([0, 0, 0.12])
            return Status.OK, f"released above {label}"
        if call.name == "push":
            if self.holding:
                return Status.BAD_ARGUMENT, f"holding the {self.holding}; place it first"
            if o.kind != "rigid":
                return Status.STALL, f"{label} did not move"
            dx, dy = call.args[1], call.args[2]
            o.xyz[:2] += [dx, dy]
            o.support = None
            self.gripper = o.xyz + np.array([0, 0, 0.08])
            return Status.OK, f"moved {label} by {100 * math.hypot(dx, dy):.0f} cm"
        if call.name == "actuate":
            if self.holding:
                return Status.BAD_ARGUMENT, f"holding the {self.holding}; place it first"
            action = call.args[1]
            self.gripper = o.xyz + np.array([-0.05, 0, 0.02])
            if o.kind == "drawer" and action in ("open", "close"):
                o.is_open = action == "open"
                return Status.OK, f"{label} {'opened' if o.is_open else 'closed'}"
            if o.kind == "knob" and action in ("turn_on", "turn_off"):
                o.is_on = action == "turn_on"
                return Status.OK, f"{label} knob turned {'on' if o.is_on else 'off'}"
            if o.kind in ("drawer", "knob"):
                return Status.BAD_ARGUMENT, f"{label} does not support {action}"
            return Status.NO_CANDIDATE, f"{label} has no articulated part"
        return Status.BAD_ARGUMENT, f"unsupported skill {call.name}"

    # ------------------------------------------------------------------ rendering
    def _gripper_text(self) -> str:
        x, y, z = (100 * self.gripper).tolist()
        state = f"closed, holding the {self.holding}" if self.holding else "open, empty"
        return f"{state}; pinch point at x={x:.1f} y={y:.1f} z={z:.1f}"

    def _px(self, xy, W, H):
        x0, x1, y0, y1 = self.VIEW
        return ((y1 - xy[1]) / (y1 - y0) * W, (x1 - xy[0]) / (x1 - x0) * H)

    def _render(self, W: int) -> Image.Image:
        H = W
        img = Image.new("RGB", (W, H), (196, 170, 130))
        d = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        x0, x1, y0, y1 = self.VIEW
        for gx in np.arange(math.ceil(x0 * 10) / 10, x1, 0.1):          # 10 cm grid
            _, r = self._px((gx, 0), W, H)
            d.line([(0, r), (W, r)], fill=(170, 145, 110))
            d.text((2, r + 1), f"x{gx * 100:.0f}", fill=(90, 70, 50), font=font)
        for gy in np.arange(math.ceil(y0 * 10) / 10, y1, 0.1):
            c, _ = self._px((0, gy), W, H)
            d.line([(c, 0), (c, H)], fill=(170, 145, 110))
            d.text((c + 1, H - 12), f"y{round(gy * 100):d}", fill=(90, 70, 50), font=font)
        order = sorted(self.objs.values(), key=lambda o: o.xyz[2])
        for o in order:
            c, r = self._px(o.xyz, W, H)
            hw = o.size[1] / (y1 - y0) * W / 2
            hh = o.size[0] / (x1 - x0) * H / 2
            fill = o.color
            if o.kind == "drawer" and o.is_open:
                r += 0.08 / (x1 - x0) * H        # drawn pulled toward the robot
            if o.kind == "knob" and o.is_on:
                d.ellipse([c - hw * 0.5, r - hh * 0.5, c + hw * 0.5, r + hh * 0.5], fill=(230, 60, 20))
            if o.label in ("black bowl", "plate", "wine bottle"):
                d.ellipse([c - hw, r - hh, c + hw, r + hh], fill=fill, outline=(0, 0, 0))
            else:
                d.rectangle([c - hw, r - hh, c + hw, r + hh], fill=fill, outline=(0, 0, 0))
        for m, mi in self.marks.items():
            c, r = self._px(self.objs[mi.label].xyz, W, H)
            rad = max(7, W // 40)
            d.ellipse([c - rad, r - rad, c + rad, r + rad], fill=(255, 255, 255), outline=(0, 0, 0), width=2)
            d.text((c, r), str(m), fill=(0, 0, 0), font=font, anchor="mm")
        gc, gr = self._px(self.gripper, W, H)
        d.line([(gc - 8, gr), (gc + 8, gr)], fill=(220, 0, 0), width=3)
        d.line([(gc, gr - 8), (gc, gr + 8)], fill=(220, 0, 0), width=3)
        return img

    def _images(self) -> list[Image.Image]:
        W = self.img_size
        agent = self._render(W)
        big = self._render(2 * W)
        gc, gr = self._px(self.gripper, 2 * W, 2 * W)
        h = W // 3
        box = (int(gc - h), int(gr - h), int(gc + h), int(gr + h))
        wrist = big.crop(box).resize((256, 256))
        return [agent, wrist]


# ============================================================================ real backend
class RealBackend:
    """Drives the real streams. Calls exactly (design.md interfaces):

    - `motormem.env.make_env(spec) -> RobotEnv`; `env.reset()`, `env.observe()`,
      `env._runner_success()` (runner-only, after every skill), `obs.step` for the budget.
    - `motormem.memory.MotorMemory.load(spec.index_path)`, `.without(objects, demo_ids)`.
    - `motormem.perception.make_segmenter(spec.perception)`, `Perception(seg).scan(obs, find)`,
      `Perception.refine(obs, est, camera="wrist")` for look(M), `render_overlay(obs, ests, camera)`,
      and `describe(...)` if stream B provides it (else `describe_marks` here);
      `OraclePerception(env, memory)` when spec.perception == "oracle" (rung-0 only).
    - `motormem.skills.Skills(env, perception, memory, motion_source)`:
      `.pick(est)`, `.place(est, relation)`, `.push(est, dx_world, dy_world)`,
      `.actuate(est, action)`, `.home()` -> `SkillResult`.
    Marks resolve to the `ObjectEstimate`s of the last observe. Untested until integration.
    """

    def __init__(self, spec: EpisodeSpec):
        from motormem.env import make_env                      # lazy: other streams
        from motormem.memory import MotorMemory
        from motormem.skills import Skills
        from motormem import perception as P
        from motormem import config

        self.spec = spec
        self.P = P
        self.env = make_env(spec)
        self.instruction = self.env.instruction
        self.obs = self.env.reset()
        index_path = spec.index_path or str(config.INDEX_DIR / f"{spec.suite.split('_pro')[0]}.pkl")
        memory = MotorMemory.load(index_path)
        ex = spec.index_exclude or {}
        if ex.get("objects") or ex.get("demo_ids"):
            memory = memory.without(objects=ex.get("objects", ()), demo_ids=ex.get("demo_ids", ()))
        self.memory = memory
        if spec.perception == "oracle":                       # rung-0 diagnostic only
            names = {seg.object_name for seg in memory.segments}
            self.perception = P.OraclePerception(self.env.privileged_object_poses,
                                                 {n: memory.object_points(n) for n in names})
        else:
            self.perception = P.Perception(P.make_segmenter(spec.perception))
        self.skills = Skills(self.env, self.perception, memory, motion_source=spec.motion_source)
        self.estimates: dict = {}
        self.marks: dict[int, MarkInfo] = {}
        self.last_find: list[str] = []
        self.over = False
        self.ended_by: str | None = None
        self.success = False
        self.segment_ids: list[str] = []
        self.fallbacks = 0
        self.statuses: list[str] = []
        self.summary: str | None = None

    def _to_base(self, p_world: np.ndarray) -> np.ndarray:
        T = self.obs.T_world_base
        return T[:3, :3].T @ (np.asarray(p_world)[:3] - T[:3, 3])

    def _gripper_text(self) -> str:
        pr = self.obs.proprio
        x, y, z = (100 * self._to_base(pr.T_world_eef[:3, 3])).tolist()
        state = "open" if pr.gripper_opening > 0.5 else "closed"
        return f"{state} (opening {pr.gripper_opening:.2f}); pinch point at x={x:.1f} y={y:.1f} z={z:.1f}"

    def observe(self, find: list[str]) -> tuple[list[Image.Image], str]:
        self.last_find = [f for f in (find or []) if isinstance(f, str) and f.strip()][:12]
        self.obs = self.env.observe()
        ests = self.perception.scan(self.obs, self.last_find, camera="agentview")
        self.estimates = {e.mark: e for e in ests}
        self.marks = {e.mark: MarkInfo(e.mark, e.label, self._to_base(e.T_world_obj[:3, 3]), np.asarray(e.extent))
                      for e in ests}
        found = {e.label for e in ests}
        missing = [f for f in self.last_find if f not in found]
        imgs = [self.P.render_overlay(self.obs, ests, camera="agentview"),
                self.P.render_overlay(self.obs, ests, camera="wrist")]
        describe = getattr(self.P, "describe", None)
        text = describe(ests, self.obs) if describe else describe_marks(self.marks, self._gripper_text(), missing)
        return imgs, text

    def _exec(self, call: Call):
        if call.name == "home":
            return self.skills.home()
        est = self.estimates.get(call.mark)
        if est is None:
            from motormem.types import SkillResult
            known = ", ".join(map(str, sorted(self.estimates))) or "none; call observe first"
            return SkillResult(Status.BAD_ARGUMENT, f"unknown mark {call.mark} (current marks: {known})")
        if call.name == "pick":
            return self.skills.pick(est)
        if call.name == "place":
            return self.skills.place(est, call.args[1])
        if call.name == "actuate":
            return self.skills.actuate(est, call.args[1])
        if call.name == "push":
            d = self.obs.T_world_base[:3, :3] @ np.array([call.args[1], call.args[2], 0.0])
            return self.skills.push(est, float(d[0]), float(d[1]))
        if call.name == "look":
            from motormem.types import SkillResult
            self.obs = self.env.observe()
            new = self.perception.refine(self.obs, est, camera="wrist")
            self.estimates[call.mark] = new
            x, y, z = (100 * self._to_base(new.T_world_obj[:3, 3])).tolist()
            return SkillResult(Status.OK, f"{new.label} at x={x:.1f} y={y:.1f} z={z:.1f} cm")
        raise ValueError(call.name)

    def run(self, calls: list[Call]) -> tuple[list[str], list[Image.Image], str]:
        lines: list[str] = []
        for i, call in enumerate(calls, start=1):
            if self.over:
                lines.append(status_line(i, call, Status.EPISODE_OVER, ""))
                continue
            try:
                res = self._exec(call)
            except Exception as e:                            # never crash the episode
                from motormem.types import SkillResult
                res = SkillResult(Status.ERROR, f"{type(e).__name__}: {str(e)[:120]}")
            if res.segment_id:
                self.segment_ids.append(res.segment_id)
            self.fallbacks += int(res.used_fallback)
            self.statuses.append(res.status.value)
            src = " [fallback]" if res.used_fallback else ""
            lines.append(status_line(i, call, res.status, res.message + src))
            self.obs = self.env.observe()
            if self.env._runner_success() and not self.over:   # runner-only, standard protocol
                self.over, self.ended_by, self.success = True, "success", True
            elif self.obs.step >= self.spec.max_steps and not self.over:
                self.over, self.ended_by = True, "budget"
            if res.status != Status.OK:
                for k, rest in enumerate(calls[i:], start=i + 1):
                    lines.append(status_line(k, rest, "SKIPPED", "earlier step failed"))
                break
        imgs, text = self.observe(self.last_find)
        return lines, imgs, text

    def done(self, summary: str) -> None:
        self.summary = summary
        if not self.over:
            self.over, self.ended_by = True, "done"

    def episode_over(self) -> bool:
        return self.over

    def last_marks(self) -> dict[int, MarkInfo]:
        return dict(self.marks)

    def runner_result(self) -> dict:
        success = self.success or bool(self.env._runner_success())
        return {"success": success, "ended_by": self.ended_by, "sim_steps": int(self.obs.step),
                "segment_ids": list(self.segment_ids), "fallbacks": self.fallbacks,
                "skill_statuses": list(self.statuses), "backend": "real"}

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close:
            close()


# ============================================================================ factory
def make_backend(spec: EpisodeSpec, kind: str | None = None) -> Backend:
    kind = kind or os.environ.get("MOTORMEM_BACKEND", "fake")
    if kind == "fake":
        return FakeBackend(spec)
    if kind == "real":
        return RealBackend(spec)
    raise ValueError(f"unknown backend {kind!r} (fake|real)")


def instruction_for(spec: EpisodeSpec, kind: str = "fake") -> str:
    """The language instruction the runner passes to the brain (not privileged)."""
    if kind == "fake":
        return FakeBackend(spec).instruction
    try:                                                      # stream A may provide this
        from motormem.env import instruction_for as env_instruction
        return env_instruction(spec)
    except ImportError:
        pass
    try:                                                      # LIBERO / LIBERO-Pro: read the task's BDDL file
        from motormem.env.libero import PRO_SUITES, STANDARD_SUITES, resolve_task
        if spec.suite in STANDARD_SUITES or spec.suite in PRO_SUITES:
            from libero.libero.envs.bddl_utils import get_problem_info
            instr = get_problem_info(str(resolve_task(spec.suite, spec.task_id).bddl_file))["language_instruction"]
            return " ".join(instr) if isinstance(instr, (list, tuple)) else str(instr)
    except ImportError:
        pass
    if spec.suite in ("desk_sort", "bin_sort"):              # custom scenes: build the env once for its text
        from motormem.env import make_env
        env = make_env(spec)
        try:
            return env.instruction
        finally:
            env.close()
    from libero.libero import benchmark                       # standard LIBERO suites
    return benchmark.get_benchmark_dict()[spec.suite]().get_task(int(spec.task_id)).language
