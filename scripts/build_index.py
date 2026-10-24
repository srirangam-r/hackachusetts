#!/usr/bin/env python
"""Build a motor-memory index from recorded demos.

  env PYTHONPATH=$PWD MUJOCO_GL=egl python scripts/build_index.py --suites libero_goal \
      --out /home/nail/stuff/sundai/hackachusetts/data/index/libero_goal.pkl [--max-demos N]

Sources: libero_goal | libero_spatial | libero_object | libero_10 | libero_90[:SCENE_PREFIX,...] | mimicgen/<task>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motormem import config  # noqa: E402
from motormem.memory.build import build_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-demos", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else config.DATA_DIR.parent / a.out
    if config.RAW_DIR in out.resolve().parents:
        raise SystemExit("refusing to write into data/raw")
    build_index(a.suites, out, a.max_demos, workers=a.workers)


if __name__ == "__main__":
    main()
