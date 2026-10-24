"""Rung-0 skills smoke test (DIAGNOSTIC ONLY: ground-truth object estimates, fake segments, env success).

usage: python scripts/skills_smoke.py [--arms Panda UR5e] [--out runs/skills_smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT))

from skills_sim_helpers import make_adapter  # noqa: E402

from motormem import geometry as g  # noqa: E402
from motormem.skills import Skills  # noqa: E402
from motormem.skills.control import densify  # noqa: E402
from motormem.types import Segment, SegmentKind  # noqa: E402


class FakeMemory:
    """Minimal MotorMemory stand-in: canonical names + segments built from ground truth."""

    def __init__(self, names: dict[str, str], segments: list[Segment], points: dict[str, np.ndarray]):
        self.names, self.segments, self.points = names, segments, points

    def canonical(self, label):
        return self.names.get(label)

    def object_points(self, name):
        return self.points[name]

    def query(self, kind, object_name=None, relation=None, held_name=None, extent=None, k=8):
        out = [s for s in self.segments if s.kind == kind and s.object_name == object_name
               and (relation is None or s.relation == relation)]
        return out[:k]


def fake_bowl_plate_segments(env) -> FakeMemory:
    """Synthesise a GRASP (bowl) and PLACE (on plate) segment in object frames from this env's GT poses."""
    bowl = env.gt_estimate("akita_black_bowl_1_main", "black bowl")
    plate = env.gt_estimate("plate_1_main", "plate")
    T_bowl = env.gt_body_pose("akita_black_bowl_1_main")
    T_plate = env.gt_body_pose("plate_1_main")
    pts_bowl = g.transform_points(g.inv(T_bowl), bowl.points_world)
    pts_plate = g.transform_points(g.inv(T_plate), plate.points_world)
    sk = Skills(env, memory=None, motion_source="primitive")
    _, T_grasp = sk._primitive_grasp_options(bowl)[0]
    above = T_grasp.copy(); above[2, 3] += 0.12
    lift = T_grasp.copy(); lift[2, 3] += 0.08
    app = densify([above, T_grasp], 0.01, 0.1)
    path = np.concatenate([app, np.repeat(T_grasp[None], 15, 0), densify([T_grasp, lift], 0.01)[1:]])
    grip = np.r_[-np.ones(len(app)), np.ones(len(path) - len(app))]
    seg_g = Segment("fake/grasp/bowl", "fake/demo0", "fake", SegmentKind.GRASP, "akita_black_bowl", None, None,
                    g.compose(g.inv(T_bowl), path), grip, None, pts_bowl, "pick up the black bowl")
    # place: bowl bottom-centre 1.5 cm above plate top, gripper keeps its grasp orientation
    T_eef_held = g.inv(T_grasp) @ T_bowl
    bottom_off = np.array([bowl.T_world_obj[0, 3], bowl.T_world_obj[1, 3], bowl.points_world[:, 2].min()]) - T_bowl[:3, 3]
    held_rel = T_bowl.copy()
    held_rel[:3, 3] = [plate.T_world_obj[0, 3], plate.T_world_obj[1, 3], plate.points_world[:, 2].max() + 0.015] - bottom_off
    T_rel = held_rel @ g.inv(T_eef_held)
    carry = T_rel.copy(); carry[2, 3] += 0.12
    up = T_rel.copy(); up[2, 3] += 0.06
    a = densify([carry, T_rel], 0.01)
    p_path = np.concatenate([a, np.repeat(T_rel[None], 10, 0), densify([T_rel, up], 0.01)[1:]])
    p_grip = np.r_[np.ones(len(a)), -np.ones(len(p_path) - len(a))]
    seg_p = Segment("fake/place/bowl_on_plate", "fake/demo0", "fake", SegmentKind.PLACE, "plate", "on",
                    "akita_black_bowl", g.compose(g.inv(T_plate), p_path), p_grip, T_eef_held, pts_plate,
                    "put the bowl on the plate")
    return FakeMemory({"black bowl": "akita_black_bowl", "plate": "plate"}, [seg_g, seg_p],
                      {"akita_black_bowl": pts_bowl, "plate": pts_plate})


def run_pick_place(arm: str, source: str, init_index: int, memory=None) -> dict:
    env = make_adapter("put_the_bowl_on_the_plate", arm=arm, init_index=init_index)
    sk = Skills(env, memory=memory, motion_source=source)
    bowl = env.gt_estimate("akita_black_bowl_1_main", "black bowl", mark=1)
    plate = env.gt_estimate("plate_1_main", "plate", mark=2)
    r1 = sk.pick(bowl)
    r2 = sk.place(plate, "on") if r1.status.value == "OK" else None
    for _ in range(10):  # let the bowl settle before checking
        env.step(np.r_[np.zeros(6), -1.0])
    out = {"results": [r1, r2], "success": env.success(), "steps": env.step_count}
    env.close()
    return out


def run_actuate(arm: str, task: str, body: str, label: str, action: str, mesh_idx: int | None,
                keep_above: float | None, joint: str, init_index: int = 0) -> dict:
    env = make_adapter(task, arm=arm, init_index=init_index)
    geoms = [env.mesh_geoms(body)[mesh_idx]] if mesh_idx is not None else None
    part = env.gt_estimate(body, label, mark=1, geoms=geoms, keep_above=keep_above)
    q0 = env.joint_qpos(joint)
    sk = Skills(env, memory=None, motion_source="primitive")
    r = sk.actuate(part, action)
    for _ in range(5):
        env.step(np.r_[np.zeros(6), -1.0])
    out = {"results": [r], "success": env.success(), "steps": env.step_count,
           "joint": (q0, env.joint_qpos(joint))}
    env.close()
    return out


def fmt(r) -> str:
    if r is None:
        return "-"
    s = f"{r.status.value}: {r.message} [{r.steps} steps"
    s += ", fallback" if r.used_fallback else ""
    s += f", seg={r.segment_id}" if r.segment_id else ""
    return s + "]"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["Panda", "UR5e"])
    ap.add_argument("--inits", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "skills_smoke"))
    args = ap.parse_args()
    rows = []
    for arm in args.arms:
        jobs = []
        for i in args.inits:
            jobs.append((f"bowl->plate primitive (init {i})", lambda i=i: run_pick_place(arm, "primitive", i)))

        def index_job(i):
            src = make_adapter("put_the_bowl_on_the_plate", arm=arm, init_index=(i + 1) % 50)
            mem = fake_bowl_plate_segments(src)  # segments recorded in another layout
            src.close()
            return run_pick_place(arm, "index", i, memory=mem)
        for i in args.inits:
            jobs.append((f"bowl->plate index, fake segs (init {i})", lambda i=i: index_job(i)))
        for i in args.inits:
            jobs.append((f"open middle drawer (init {i})", lambda i=i: run_actuate(
                arm, "open_the_middle_drawer_of_the_cabinet", "wooden_cabinet_1_cabinet_middle",
                "middle drawer handle", "open", 1, None, "wooden_cabinet_1_middle_level", i)))
            jobs.append((f"turn on stove (init {i})", lambda i=i: run_actuate(
                arm, "turn_on_the_stove", "flat_stove_1_button", "stove knob", "turn_on", 0, 0.015,
                "flat_stove_1_button", i)))
        for name, job in jobs:
            if args.only and not any(k in name for k in args.only):
                continue
            t = time.time()
            try:
                out = job()
                row = (arm, name, out["success"], out["steps"], time.time() - t,
                       " / ".join(fmt(r) for r in out["results"] if r is not None)
                       + (f" joint {out['joint'][0]:.3f}->{out['joint'][1]:.3f}" if "joint" in out else ""))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                row = (arm, name, False, 0, time.time() - t, f"CRASH {type(e).__name__}: {e}")
            print(row, flush=True)
            rows.append(row)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# Skills rung-0 smoke (diagnostic: GT estimates, fake segments, env success)", "",
             "| arm | task | success | steps | wall s | skill results |", "|---|---|---|---|---|---|"]
    lines += [f"| {a} | {n} | {'yes' if s else 'no'} | {st} | {w:.0f} | {m} |" for a, n, s, st, w, m in rows]
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'results.md'}")


if __name__ == "__main__":
    main()
