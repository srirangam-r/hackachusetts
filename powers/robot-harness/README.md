# robot-harness power

Packages the Motor Memory robot MCP server (`observe` / `run` / `done`) and a skill that teaches an agent
the robot protocol (turn format, skill grammar, status words, retry rules). Agent Plugins 1.0.0 layout:

```
plugin.json                       manifest
mcp.json                          robot MCP server: python -m motormem.brain.mcp_server
skills/robot-harness/SKILL.md     the robot protocol
```

## Requirements
- `python` on `PATH` must import `motormem` and `mcp`: start Kiro from the activated repo venv (or
  `pip install -e .`). Otherwise the `robot` server connects with no tools; set `command` in `mcp.json`
  to the venv's absolute python path.
- Real simulator backend: LIBERO/robosuite installed, a GPU with EGL, and a built index (`data/index/*.pkl`).

## Install
```
kiro-cli powers install ./powers/robot-harness      # from the repo root
# or, inside a Kiro chat:  /powers install /abs/path/to/powers/robot-harness
kiro-cli powers uninstall robot-harness
```

## Configure an episode
The server reads its episode from `ROBOT_EPISODE`, which defaults to `${PLUGIN_DATA}/episode.json`
(Kiro's per-power data directory). Without that file every tool answers `ERROR: no episode configured`.
Write a spec there, e.g. the toy backend (no simulator needed):

```json
{"suite": "libero_goal", "task_id": 8, "init_index": 0, "out_dir": "/tmp/robot_ep", "max_steps": 600}
```

For the real simulator change `MOTORMEM_BACKEND` in `mcp.json` to `real` (it overrides the spec's
`backend` key) and add `"index_path"` to the spec. Add `"record": true` to save video frames to `out_dir`, then render
with `python scripts/make_video.py <out_dir>`.

Fairness: the server strips privileged simulator state from every observation and never reports task
success to the agent (see `.kiro/steering/fairness.md`).
