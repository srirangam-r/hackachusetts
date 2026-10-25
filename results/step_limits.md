# Final Panda runs at two step limits

Same 150 episodes (Kiro `auto` brain, SAM 3, 900-step limit). Episodes end the moment the task is done, so an episode counts at 300 steps (LIBERO-Pro's limit, the one the VLA baselines are scored at) when it succeeded within its first 300 sim steps (`success and sim_steps <= 300` in each `result.json`).

| condition | run | 300 steps | 900 steps | mean Kiro tool calls |
|---|---|---|---|---|
| LIBERO-Goal, familiar tasks | [`fix_goal_panda`](fix_goal_panda/summary.md) | 31/50 (62%) | 35/50 (70%) | 3.2 |
| LIBERO-Pro, moved objects | [`fix_pro_swap_panda`](fix_pro_swap_panda/summary.md) | 20/50 (40%) | 25/50 (50%) | 3.8 |
| LIBERO-Pro, new instructions | [`fix_pro_task_panda`](fix_pro_task_panda/summary.md) | 12/50 (24%) | 13/50 (26%) | 4.3 |
