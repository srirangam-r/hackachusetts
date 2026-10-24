#!/usr/bin/env python3
"""PostFileSave hook: run the fast tests that belong to the saved module.

motormem/<pkg>/<mod>.py -> tests/test_<pkg>*.py and tests/test_<mod>*.py (excluding @pytest.mark.sim).
Exit 1 (warn) with a short failure summary on stderr; exit 0 otherwise.
"""
import glob
import json
import os
import re
import subprocess
import sys

PY = "/home/nail/stuff/sundai/hackachusetts/.venv/bin/python"


def find_path(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "path" in k.lower() and v.endswith(".py"):
                return v
            r = find_path(v)
            if r:
                return r
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    path = find_path(payload) or os.environ.get("KIRO_FILE_PATH", "")
    m = re.search(r"motormem/(?:(\w+)/)?(\w+)\.py$", path or "")
    if not m:
        return 0
    root = path.split("/motormem/")[0] if os.path.isabs(path) and "/motormem/" in path else (payload.get("cwd") or os.getcwd())
    names = {n for n in m.groups() if n and n != "__init__"}
    tests = sorted({t for n in names for t in glob.glob(os.path.join(root, "tests", f"test_{n}*.py"))})
    if not tests:
        return 0
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = root  # import the worktree's own motormem
    env["MUJOCO_GL"] = "egl"
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-m", "not sim", *tests], cwd=root, env=env,
                       capture_output=True, text=True, timeout=110)
    if r.returncode not in (0, 5):
        tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
        print(f"Tests failing after editing {path}:\n{tail}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
