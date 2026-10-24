# Ro/WIP — working notes

Branch for our work + tests on top of `main` (`origin/main` @ 11cdf2a).
Nothing here is pushed to remote; `main` is never pushed to without explicit request.

## Environment (this machine)
- venv: `/home/nail/workspaces/hackachusetts/.venv` (Python 3.12, torch 2.11.0+cu128, RTX 4060 8 GB).
- Data symlinks: `data/raw/libero_hdf5/{libero_goal,libero_10,libero_90,libero_object,libero_spatial}`
  -> `/home/nail/workspaces/gen_manipulation/data/raw/libero/*`.
- Indexes built: `data/index/libero_goal.pkl` (500 demos -> 937 segments),
  `data/index/libero_spatial.pkl` (500 -> 1020).
- Always run with: `env -u PYTHONPATH MUJOCO_GL=egl MOTORMEM_DATA=$PWD/data MOTORMEM_RUNS=$PWD/runs`.

## Local changes vs main (why)
- `motormem/brain/runner.py`: `make_workspace` now writes a JSON agent (`robot-brain.json`)
  via `_agent_md_to_json`, because kiro-cli 2.34.0 discovers JSON agents, not the old `.md` format.
- `.kiro/agents/robot-brain.md`: MCP server python path -> this machine's venv.
- `tests/test_brain_runner.py`: isolation test updated to assert the JSON agent (same guarantees).

## Reproducing the author's main-branch runs
Exact author config (from `results/*/summary.json`): suite `libero_goal`, tasks 0-9,
episodes 5, model auto, backend real, motion hybrid, perception **sam3**.

ONE unavoidable difference: SAM 3 weights won't load under pinned transformers 5.5.4 here, and
`--perception sam3` hard-crashes (no fallback). Runnable equivalent: `--perception auto`
(falls back to Grounding DINO + SAM 2.1). Fallback can't detect cream cheese / wine rack / cabinet,
so those tasks under-perform the author's SAM 3 numbers.

```bash
env -u PYTHONPATH MUJOCO_GL=egl MOTORMEM_DATA=$PWD/data MOTORMEM_RUNS=$PWD/runs \
  .venv/bin/python -m motormem.brain.runner --suite libero_goal --tasks 0-9 --episodes 5 \
  --arm Panda --backend real --motion-source hybrid --perception auto \
  --index-path data/index/libero_goal.pkl --model auto --parallel 2 --name goal_panda
```

## Failing / weak test cases to focus on (author's main-branch runs)
| task | instruction | Panda | UR5e | documented cause |
|---|---|---|---|---|
| 3 | open top drawer AND put bowl inside | 0/5 | 0/5 | drawer-handle grasp + multi-step (worst cell) |
| 9 | put wine bottle on the rack | 1/5 | 5/5 | tight-tolerance insertion; no force-guided insertion |
| 6 | put cream cheese in the bowl | 2/5 | 1/5 | cream cheese undetected by fallback; box grasp weak |
| 0 | open middle drawer | 5/5 | 0/5 | UR5e drawer handle/articulation |
| 5 | push plate to front of stove | 3/5 | 4/5 | push distance calibration |
| 7 | turn on the stove | 3/5 | 4/5 | knob localization/turn |
| 2 | wine bottle on top of cabinet | 5/5 | 3/5 | UR5e grasp reachability |
| 8 | bowl on plate | 5/5 | 3/5 | UR5e place accuracy |

Themes (README + docs/architecture.md): contact skills (drawer handle, box grasp, push
distance, bottle insertion); fallback-perception misses (cream cheese, wine rack, cabinet);
tight-tolerance insertion not implemented.

## Status log
- (setup) venv + deps installed, tests green (101 non-sim + runner tests), libero_goal/spatial indexes built.
- (next) run author-matching goal_panda eval on current code to get our baseline numbers.
