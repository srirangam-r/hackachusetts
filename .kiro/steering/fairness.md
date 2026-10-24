---
inclusion: always
---
# Fairness rules (non-negotiable)

These keep our numbers comparable to published VLA and agent baselines. Violating any of them invalidates a result.

**At test time the robot brain and skills must never use:**
- Simulator object state: LIBERO obs keys `<obj>_pos`, `<obj>_quat`, `<obj>_to_robot0_eef_*`, `object-state`, `env.sim` body/joint reads of objects, `obj_of_interest`, BDDL/init files, asset XML sizes.
- The success signal (`check_success`, `_check_success`, `done` from success) for decisions, retries or resets. The runner may end the episode on success (standard LIBERO protocol), nothing else.
- Simulator segmentation masks. Masks come from SAM 3 (or the documented fallback) only.
- Per-task prompts, hints, or code written for specific test tasks or seeds.

**Allowed:**
- RGB, depth, camera intrinsics/extrinsics (nominal calibration), proprioception (EEF pose, gripper qpos, joint pos), wrist force/torque. Report depth use.
- Training demos (the same LIBERO/MimicGen demos VLAs train on). Offline index building MAY read the demos' recorded sim states to label object poses (as MimicGen does); this never happens on test episodes.
- Ground-truth object poses ONLY in the rung-0 diagnostic mode, which is runner-only, never reachable from the brain, and never reported as a result.

**Enforcement:** the `robot-brain` agent's tools are limited to the robot MCP server (no shell, files or web). Every observation passes through `strip_privileged()`; a property test asserts no privileged key ever reaches the agent.
