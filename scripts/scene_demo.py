"""Run a showcase scene end to end through the real backend with a scripted program over observed marks.

    scene_demo.py desk_sort --layout 0 [--arm Panda] [--motion hybrid] [--record out.mp4]
    scene_demo.py desk_sort --only black_book --exclude-objects black_book --motion index     # editability
    scene_demo.py desk_sort --only black_book --keep-one-demo black_book --motion index      # add one demo back
    scene_demo.py bin_sort --layout 0 --arm UR5e --motion hybrid

The program is fixed per scene (no per-layout logic): observe every object, then for each object pick its mark and
place it on/in its target mark, re-reading marks after every call. Per-object success is a runner-only diagnostic
(sim predicates), never fed back into the program.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from motormem import config
from motormem.types import EpisodeSpec

INDEX_DIR = config.INDEX_DIR
VIDEO_DIR = config.RUNS_DIR / "videos"

# scene -> (index, [(object name, find label, target label, relation)])
SCENES = {
    "desk_sort": ("libero90_study_living.pkl", [
        ("black_book", "black book", "desk organizer", "in"),
        ("yellow_book", "book", "desk organizer", "in"),             # SAM 3 misses "yellow book"
        ("red_coffee_mug", "red coffee mug", "wooden tray", "on"),
        ("porcelain_mug", "white porcelain mug", "wooden tray", "on"),
        ("white_yellow_mug", "white and yellow mug", "wooden tray", "on"),
    ]),
    "bin_sort": ("mimicgen_pick_place.pkl", [              # SAM 3 labels the cereal box "loaf of bread" too
        ("Can", "soda can", "tray", "in"),
        ("Bread", "loaf of bread", "tray", "in"),
        ("Cereal", "loaf of bread", "tray", "in"),
        ("Milk", "milk", "tray", "in"),
    ]),
}


def object_status(env, scene: str) -> dict[str, bool]:
    """Runner-only per-object success."""
    if scene == "desk_sort":
        from motormem.env.desk_sort import object_status as st
        return st(env)
    return env.privileged_objects_in_bins()


def in_target_bin(env) -> dict[str, bool]:
    """Runner-only, looser than robosuite's per-section check: object anywhere inside the target bin (bin2)."""
    rs = env._rs
    out = {}
    for i, obj in enumerate(rs.objects):
        p = env.sim.data.body_xpos[rs.obj_body_id[obj.name]]
        d = np.abs(p[:2] - rs.bin2_pos[:2])
        out[obj.name] = bool(d[0] < rs.bin_size[0] / 2 and d[1] < rs.bin_size[1] / 2 and p[2] < rs.bin2_pos[2] + 0.1)
    return out


def _ok(line: str) -> bool:
    """A backend status line '1. pick(3) -> OK: ...' reports success."""
    return line.split("-> ", 1)[-1].split(":", 1)[0].strip() == "OK"


def target_mark(b, target: str, home_xyz, far_from=None) -> int | None:
    """Targets (caddy, tray, bins) do not move: take the `target` mark nearest where it was first seen. SAM 3
    sometimes swaps labels between similar wooden containers, so if no such mark is near, re-ask for the target
    alone and finally accept any mark at the remembered spot."""
    def near(marks, label):
        c = [m for m in marks.values() if (label is None or m.label == label)
             and np.linalg.norm(np.asarray(m.xyz)[:2] - home_xyz[:2]) < 0.08]
        return min(c, key=lambda m: np.linalg.norm(np.asarray(m.xyz)[:2] - home_xyz[:2])).mark if c else None
    if home_xyz is None:
        return pick_mark(b.marks, target, far_from=far_from)
    m = near(b.marks, target)
    if m is None:
        b.observe([target])
        m = near(b.marks, target) or near(b.marks, None)
    return m


def pick_mark(marks, label: str, far_from=None, avoid=None) -> int | None:
    """The mark for `label`: only marks inside the reachable workspace (SAM 3 also fires on background shelves),
    minus marks that sit on top of a mark with a different label (generic prompts like "book" also hit the
    black book). With `far_from` and several candidates (two bins), take the one farthest from that point."""
    def in_ws(m):
        x, y, z = np.asarray(m.xyz)
        return 0.15 < x < 0.95 and abs(y) < 0.6 and -0.15 < z < 0.4
    others = [np.asarray(m.xyz) for m in marks.values()
              if m.label != label and set(label.split()) < set(m.label.split()) and in_ws(m)]
    cands = [m for m in marks.values() if m.label == label and in_ws(m)
             and all(np.linalg.norm(np.asarray(m.xyz) - o) > 0.04 for o in others)
             and (avoid is None or np.linalg.norm(np.asarray(m.xyz)[:2] - avoid[:2]) > 0.2)]
    if not cands:
        return None
    if far_from is not None and len(cands) > 1:
        cands.sort(key=lambda m: -np.linalg.norm(np.asarray(m.xyz)[:2] - far_from[:2]))
    return cands[0].mark


def keep_one_demo_exclusions(index_path: Path, obj: str) -> list[str]:
    from motormem.memory import MotorMemory
    mem = MotorMemory.load(str(index_path))
    ids = sorted({s.demo_id for s in mem.segments if obj in (s.object_name, s.held_name)})
    return ids[1:]


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=sorted(SCENES))
    ap.add_argument("--layout", type=int, default=0)
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--arm", default="Panda")
    ap.add_argument("--motion", default="hybrid", choices=["index", "hybrid", "primitive"])
    ap.add_argument("--perception", default="sam3")
    ap.add_argument("--index", default=None)
    ap.add_argument("--exclude-objects", default="")
    ap.add_argument("--keep-one-demo", default="", help="exclude every demo of this object but one")
    ap.add_argument("--only", default="", help="comma list of object names to sort")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--record", default="", help="mp4 name or path (bare names go to runs/videos/)")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--every", type=int, default=2, help="record every n-th sim step")
    ap.add_argument("--log", default="", help="append the result as a JSON line")
    a = ap.parse_args(argv)

    from motormem.brain.backend import RealBackend
    from motormem.brain.program import parse_program

    index_name, plan = SCENES[a.scene]
    index_path = Path(a.index) if a.index else INDEX_DIR / index_name
    if a.only:
        keep = set(a.only.split(","))
        plan = [p for p in plan if p[0] in keep]
    exclude = {}
    if a.exclude_objects:
        exclude["objects"] = a.exclude_objects.split(",")
    if a.keep_one_demo:
        exclude["demo_ids"] = keep_one_demo_exclusions(index_path, a.keep_one_demo)
    spec = EpisodeSpec(suite=a.scene, task_id=a.task, init_index=a.layout, arm=a.arm, perception=a.perception,
                       motion_source=a.motion, index_path=str(index_path), index_exclude=exclude,
                       max_steps=a.max_steps)
    t0 = time.time()
    b = RealBackend(spec)
    env = b.env
    frames: list[np.ndarray] = []
    if a.record:
        frames.append(env.render("agentview", 512))
        orig_step = env.step

        def step(action, *args, **kw):
            o = orig_step(action, *args, **kw)
            if o.step % a.every == 0:
                frames.append(env.render("agentview", 512))
            return o
        env.step = step

    print(f"[{a.scene} L{a.layout} {a.arm} {a.motion}] {b.instruction} | segments={len(b.memory.segments)}")
    finds = list(dict.fromkeys([p[1] for p in plan] + [p[2] for p in plan]))
    _, text = b.observe(finds)
    found = {m.label for m in b.marks.values()}
    print("found:", sorted(found), "missing:", [f for f in finds if f not in found])
    lines = []
    homes = {}                                          # where each (static) target was first seen
    for t in dict.fromkeys(p[2] for p in plan):
        tm = pick_mark(b.marks, t)
        if a.scene == "bin_sort":                       # two bins: the target is the one away from the items
            items = [np.asarray(m.xyz) for m in b.marks.values() if m.label not in {p[2] for p in plan}]
            c = np.mean(items, 0) if items else np.zeros(3)
            tm = pick_mark(b.marks, t, far_from=c)
        homes[t] = None if tm is None else np.asarray(b.marks[tm].xyz)
    for i, (name, label, target, rel) in enumerate(plan):
        if b.episode_over():
            break
        if i:
            b.observe(finds)
        avoid = homes.get(target) if a.scene == "bin_sort" else None   # skip items already in the target bin
        m = pick_mark(b.marks, label, avoid=avoid)
        if m is None:
            lines.append(f"{name}: not found")
            continue
        start = np.asarray(b.marks[m].xyz)
        out, _, _ = b.run(parse_program(f"pick({m})"))
        lines.append(f"{name}: {out[0]}")
        if not _ok(out[0]):
            b.run(parse_program("home()"))
            continue
        tm = target_mark(b, target, homes.get(target), far_from=start)
        if tm is None:
            lines.append(f"{name}: target '{target}' not found; marks: " + ", ".join(
                f"{m.label}@{(100 * np.asarray(m.xyz)).round().astype(int).tolist()}" for m in b.marks.values()))
            b.run(parse_program("home()"))
            continue
        out, _, _ = b.run(parse_program(f"place({tm}, \"{rel}\")"))
        lines.append(f"{name}: {out[0]}")
        if not _ok(out[0]):
            b.run(parse_program("home()"))
    print("\n".join(lines))
    res = b.runner_result()
    status = object_status(env, a.scene)
    if a.only:
        status = {k: v for k, v in status.items() if k in set(a.only.split(","))}
    result = {"scene": a.scene, "layout": a.layout, "arm": a.arm, "motion": a.motion, "exclude": a.exclude_objects,
              "keep_one_demo": a.keep_one_demo, "objects": status, "n_ok": sum(status.values()), "n": len(status),
              "success": res["success"], "sim_steps": res["sim_steps"], "fallbacks": res["fallbacks"],
              "statuses": res["skill_statuses"], "segment_ids": res["segment_ids"],
              "wall_s": round(time.time() - t0, 1), "lines": lines}
    if a.scene == "bin_sort":
        result["in_target_bin"] = in_target_bin(env)
    if a.record:
        import imageio.v2 as imageio
        path = Path(a.record)
        if not path.is_absolute() and path.parent == Path("."):
            VIDEO_DIR.mkdir(parents=True, exist_ok=True)
            path = VIDEO_DIR / path
        imageio.mimwrite(str(path), frames, fps=a.fps, macro_block_size=1)
        result["video"] = str(path)
    print(json.dumps({k: v for k, v in result.items() if k not in ("lines", "statuses", "segment_ids")}))
    if a.log:
        with open(a.log, "a") as f:
            f.write(json.dumps(result) + "\n")
    b.close()
    return result


if __name__ == "__main__":
    main(sys.argv[1:])
