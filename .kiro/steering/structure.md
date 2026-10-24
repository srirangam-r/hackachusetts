---
inclusion: always
---
# Structure and ownership

```
motormem/
  types.py        # shared dataclasses (contract, coordinator-owned)
  geometry.py     # SE(3) + point-cloud helpers (contract, coordinator-owned)
  config.py       # paths and constants (contract, coordinator-owned)
  env/            # stream A: LIBERO/robosuite adapters, arms, privileged stripping, scenes
  perception/     # stream B: SAM 3 segmenter, deprojection, object estimates, overlays
  memory/         # stream C: demo replay, segmentation into skills, object-frame index
  skills/         # stream D: OSC tracking, IK feasibility, pick/place/push/actuate, primitive fallback
  brain/          # stream E: program parser, robot MCP server, episode runner, metrics
scripts/          # CLI entry points (build_index, run_eval, render_video)
tests/            # one test file per module; property tests named test_*_props.py
.kiro/specs/motor-memory-harness/   # the single spec; tasks grouped by stream
```

Rules for parallel work (git worktrees, one branch per stream: `stream/env`, `stream/perception`, `stream/memory`, `stream/skills`, `stream/brain`):
- Only edit files inside your stream's package, its tests, and its scripts. Contract files (`types.py`, `geometry.py`, `config.py`) change only via the coordinator; if you need a change, write it in your final report.
- Depend on other streams only through the interfaces in `design.md`; use small fakes in tests until the real module lands.
- Mark your tasks done in `tasks.md` (`- [x]`) in your worktree.
- Commit early on your branch with clear messages. Never push, never force, never touch `data/raw/`.
