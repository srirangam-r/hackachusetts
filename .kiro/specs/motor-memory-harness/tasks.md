# Tasks: Motor Memory harness

Streams run in parallel git worktrees (`../wt-<stream>`, branch `stream/<stream>`). Contracts (`types.py`, `geometry.py`, `config.py`) are done.

- [x] 0. Contracts and shared geometry
  - [x] 0.1 `motormem/types.py`, `motormem/config.py`, `motormem/geometry.py`
  - [x] 0.2 Property tests for geometry (Properties 1-3) — `tests/test_geometry_props.py`

- [x] 1. Stream A: env
  - [x] 1.1 `env/libero.py`: build standard LIBERO suites (goal/spatial/object/10) from the installed package's bddl/init files with depth on, 256 px; return stripped `Observation` (OpenCV extrinsics verified by projecting the EEF into the image) _Req 1.1, 1.5_
  - [x] 1.2 `strip_privileged` + property test (Property 4) _Req 1.2_
  - [x] 1.3 `env/arms.py`: `MountedUR5e` registration and init-state object transfer; smoke test that objects match Panda positions _Req 1.3_
  - [x] 1.4 `kinematics()`, `_runner_success()`, `privileged_object_poses()`, `privileged_segmentation()`, `render()` _Req 1.4_
  - [x] 1.5 LIBERO-Pro goal swap/task episodes from `data/raw/libero_pro` _Req 1.5_
  - [x] 1.6 `env/bin_sort.py` (robosuite PickPlace, Panda/UR5e) and `env/desk_sort.py` (custom study scene, 6-10 objects, 5 init layouts) _Req 1.5_

- [x] 2. Stream B: perception
  - [x] 2.1 `perception/segmenters.py`: SAM 3 (transformers `Sam3Model`, text prompts) and Grounding DINO + SAM 2.1 fallback; `make_segmenter("auto")` _Req 2.2_
  - [x] 2.2 `perception/core.py`: `Perception.scan/refine`, mask dedupe, depth filtering, marks _Req 2.1, 2.4_
  - [x] 2.3 `perception/overlay.py`: numbered marks, metric grid, gripper pinch point _Req 2.3_
  - [x] 2.4 `perception/oracle.py`: rung-0 oracle perception from privileged poses + memory object points
  - [x] 2.5 Accuracy check on 20 LIBERO-Goal frames vs privileged poses (centroid error report) — script only, results to `runs/perception_check/`

- [x] 3. Stream C: memory
  - [x] 3.1 `memory/replay.py`: read LIBERO HDF5 (states, actions), replay states, extract grip-site pose, gripper command, object poses, articulated joint qpos
  - [x] 3.2 `memory/segmenting.py`: gripper-event segmentation and labelling (GRASP/PLACE/PUSH/ACTUATE) + property test (Property 9) _Req 3.1_
  - [x] 3.3 `memory/index.py`: `MotorMemory` (query, canonical names, object points, without, save/load) + property test (Property 5) _Req 3.2-3.5_
  - [x] 3.4 `scripts/build_index.py`: build `data/index/libero_goal.pkl` first (all 10 Goal tasks), then spatial/object/10 and study/living LIBERO-90 scenes; MimicGen `pick_place` for bin sort
  - [x] 3.5 Index report: segments per kind/object, failure counts — `runs/index_report.md`

- [ ] 4. Stream D: skills
  - [x] 4.1 `skills/control.py`: `Controller.track` (absolute targets via clipped OSC deltas), gripper, holding _Req 4.3, 4.6_
  - [x] 4.2 `skills/feasibility.py`: damped-least-squares IK on `ArmKinematics` copy + joint limits _Req 4.1_
  - [x] 4.3 `skills/core.py`: `pick`, `place` (index retarget + held-offset correction), `push`, `home`, primitive fallback, `motion_source` switch _Req 4.1, 4.2, 4.5_
  - [x] 4.4 `skills/articulation.py`: `actuate` with equilibrium-point lead, online line/circle fit, stall/slip detection + property test (Property 8) _Req 4.4_
  - [ ] 4.5 Rung-0 smoke: with oracle perception, scripted pick/place/actuate on 3 LIBERO-Goal tasks, Panda and UR5e (Panda done 8/8 index+actuate, primitive place off-target; UR5e grasps fail: Robotiq calibration open)

- [x] 5. Stream E: brain
  - [x] 5.1 `brain/program.py` grammar + property test (Property 6) _Req 5.2_
  - [x] 5.2 `brain/mcp_server.py` with a fake backend for tests; real backend wiring behind an interface _Req 5.1, 5.3, 5.4_
  - [x] 5.3 Property test: tool outputs never leak privileged/success fields (Property 7) _Req 5.4_
  - [x] 5.4 `.kiro/agents/robot-brain.md` + `.kiro/steering/robot-protocol.md`; end-to-end headless run against the fake backend _Req 5.1_
  - [x] 5.5 `brain/runner.py`: parallel episodes, results, summary; `brain/scripted.py` rung-0 programs for LIBERO-Goal _Req 6.1-6.4_

- [ ] 6. Integration (coordinator)
  - [ ] 6.1 Merge streams; wire real backend into MCP server
  - [ ] 6.2 Rung 0 / rung 1 / rung 2 on LIBERO-Goal (Panda); checkpoint against kill criteria
  - [ ] 6.3 UR5e, LIBERO-Pro goal, desk sort (incl. editability), bin sort
  - [ ] 6.4 Videos, README, Kiro write-up
