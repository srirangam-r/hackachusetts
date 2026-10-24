#!/usr/bin/env python
"""Quick offline check: detections per requested noun, with vs without the retry/knob fixes.

For LIBERO-Goal init frames of tasks 1, 6, 7, 8, 9 it builds the agentview frame, scans the exact
nouns the scripted runner asks for, and counts how many frames found each noun. It runs the scan
twice: `before` (retry + knob localization off) and `after` (on). Ground truth is NOT used here;
this only counts whether SAM 3 returned a mark with the requested label.

    env PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/percep_retry_check.py [--segmenter sam3] [--inits 2]
"""
from __future__ import annotations

import argparse
import logging
from collections import defaultdict

from motormem.brain.scripted import LIBERO_GOAL
from motormem.perception import Perception, make_segmenter

# reuse the frame builder from the accuracy check
from scripts.perception_check import build_obs, init_states, libero_root, make_env

log = logging.getLogger("percep_retry_check")

TASKS = [1, 6, 7, 8, 9]


def task_name(task_id: int) -> str:
    names = sorted(p.stem for p in (libero_root() / "bddl_files" / "libero_goal").glob("*.bddl"))
    return names[task_id]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default="sam3")
    ap.add_argument("--inits", type=int, default=2)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    seg = make_segmenter(args.segmenter)
    before = Perception(seg, retry_variants=False, localize_knobs=False)
    after = Perception(seg, retry_variants=True, localize_knobs=True)

    hits_before: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # noun -> [found, total]
    hits_after: dict[str, list[int]] = defaultdict(lambda: [0, 0])

    for tid in TASKS:
        task = task_name(tid)
        find, _steps = LIBERO_GOAL[tid]
        env = make_env(task)
        inits = init_states(task)
        for k in range(args.inits):
            env.reset()
            env.set_init_state(inits[k])
            for _ in range(10):
                raw, *_ = env.step([0, 0, 0, 0, 0, 0, -1.0])
            obs = build_obs(env, raw)
            eb = {e.label for e in before.scan(obs, find, "agentview")}
            ea = {e.label for e in after.scan(obs, find, "agentview")}
            for noun in find:
                hits_before[noun][1] += 1
                hits_after[noun][1] += 1
                hits_before[noun][0] += int(noun in eb)
                hits_after[noun][0] += int(noun in ea)
            log.warning("task %d (%s) init %d: before=%s after=%s", tid, task, k, sorted(eb), sorted(ea))
        env.close()

    print("\nDetections per requested noun (found / frames):")
    print(f"{'noun':<16}{'before':>10}{'after':>10}")
    for noun in sorted(set(hits_before) | set(hits_after)):
        b = hits_before[noun]
        a = hits_after[noun]
        print(f"{noun:<16}{b[0]:>4}/{b[1]:<5}{a[0]:>4}/{a[1]:<5}")


if __name__ == "__main__":
    main()
