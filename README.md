# Motor Memory

A robot harness where a frontier model running inside **Kiro** decides *what* to do, SAM 3 + depth decide *where*, and an index of recorded demos decides *how* the gripper moves. No generated code in the loop, no fine-tuned VLA, no practice on the test tasks. The same demos run on two different arms (Franka Panda, UR5e).

Track: **Robotics & Hardware**. Work in progress — see `.kiro/specs/motor-memory-harness/` for the spec.

## Setup
```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev,mcp]"
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python -m pytest -q
```
