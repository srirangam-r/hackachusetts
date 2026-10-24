# Architecture

The authoritative interfaces are in [`.kiro/specs/motor-memory-harness/design.md`](../.kiro/specs/motor-memory-harness/design.md). This page explains how the pieces fit together.

## Layers

| layer | decides | module | inputs |
|---|---|---|---|
| Brain | *what*: which objects, in which order, which skill | Kiro `robot-brain` agent via the robot MCP server (`motormem/brain/`) | instruction, two annotated images (agentview + wrist), mark list in cm, skill status lines |
| Perception | *where*: object position, frame and surface points | `motormem/perception/` | RGB, depth, nominal camera calibration |
| Motor memory | *how*: gripper trajectory for a contact | `motormem/memory/` | an index built offline from training demos |
| Skills | executes and checks | `motormem/skills/` | the above plus proprioception; OSC_POSE delta actions at 20 Hz |
| Env | simulator adapter | `motormem/env/` | LIBERO / robosuite; every observation goes through `strip_privileged()` |

Shared contracts (`types.py`, `geometry.py`, `config.py`) fix the data types: poses are 4x4 `T_a_b` matrices, and all SE(3) and point-cloud maths goes through `motormem.geometry`.

## Skill grammar

The brain never writes code. `run(program)` accepts one call per line, from this grammar (`motormem/brain/program.py`):

```
pick(M)
place(M, "on" | "in" | "left_of" | "right_of" | "front_of" | "back_of" | "top_of")
push(M, dx=<m>, dy=<m>)
actuate(M, "open" | "close" | "turn_on" | "turn_off")
look(M)
home()
```

`M` is the integer mark that `observe(find=[...])` drew on the images. Anything else raises `ProgramError` before the robot moves. A property test checks that arbitrary text either parses into this grammar or is rejected, and that nothing is ever evaluated. Each step returns a status (`OK`, `IK_FAILURE`, `NO_CANDIDATE`, `GRASP_FAILED`, `NOT_HOLDING`, `TIMEOUT`, ...). The program stops at the first failure, so the brain can look again and re-plan.

## How `pick` works

1. **Retrieve.** `MotorMemory.query(GRASP, object_name)` returns up to 8 grasp segments: exact canonical-name matches first ("black bowl" → `akita_black_bowl`), then other objects ranked by bounding-box extent similarity. A grasp recorded on one object or task can therefore serve another.
2. **Register.** The segment's stored object surface points are aligned to the live masked depth points with ICP from 8 initial yaw headings (`geometry.register_multi_yaw`). Hypotheses with fitness below 0.3 are rejected. Among near-ties (within 0.02 of the best fitness, e.g. for symmetric objects), the most reachable one wins.
3. **Transform.** `T_world_eef = T_world_obj · T_obj_eef` for every waypoint of the segment. Segments store gripper poses in the object's frame, so the same segment fits any layout.
4. **IK check.** `Feasibility.check` runs damped least-squares IK with joint limits over the waypoints (at most 60), on a copy of the arm's own MuJoCo model. Infeasible candidates are skipped and the next segment is tried.
5. **OSC tracking.** The controller approaches a pre-pose 8 cm above the first contact waypoint, then tracks the segment closed-loop. Each step commands the clipped error between the *measured* end-effector pose and the target, and the gripper closes at the recorded event. Open-loop replay of the demo deltas lifted the bowl 0/25 times; tracking lifted it 25/25 (`runs/index_report.md`, rung-0 diagnostic).
6. **Grasp check.** After a 10 cm lift, `Controller.holding` checks that the gripper opening is inside the per-gripper "holding" band. If not, the skill returns `GRASP_FAILED`.

If no segment is feasible and `motion_source` is `hybrid`, the hand-written primitive takes over: a top-down grasp at the mask centroid, with yaw from the principal axis. The result is flagged `used_fallback`. `index` disables the fallback; `primitive` uses only the fallback (this is the ablation). `place` works the same way. It queries PLACE segments by target and relation, and corrects for the difference between the live held-object offset and the demo's `T_eef_held`.

Every `SkillResult` carries the `segment_id` (demo id + event) it executed, so every motion traces back to a demo.

## Articulation skill

Drawers, doors and knobs share one joint-agnostic skill (`motormem/skills/articulation.py`). Demos only supply the handle grasp.

- **Lead, don't plan.** After grasping, the controller moves an equilibrium point 1.5 cm per step ahead of the gripper along the current motion direction. The initial direction is the mask's face normal, as in Jain & Kemp's equilibrium point control.
- **Estimate the joint online.** After 3 cm of travel, the gripper's own measured path is fit with both a line (prismatic) and a circle (revolute), using least squares with a BIC-style penalty and a noise floor. Circles with a radius over 1 m count as lines. The lead then follows the fitted tangent (in the spirit of Sturm et al. 2011 and Martín-Martín & Brock).
- **Stop conditions.** The skill stops on a stall (less than 2 mm of progress in 10 steps while the lead is saturated), a slip (the gripper opening collapses) or 30 cm of travel.
- **Knobs** turn in 0.1 rad wrist increments, up to 1.6 rad. A slip after 0.9 rad still counts as a completed turn.

No joint axis, range or type is read from the simulator. A property test checks that the line and circle fits recover the generating model from noisy samples.

## Two arms

Segments are gripper poses (grip site) in object frames, and OSC takes end-effector pose deltas. Nothing in the segment depends on the arm. To run the same Panda demos on a UR5e, `motormem/env/arms.py`:

- mounts the robosuite UR5e like LIBERO's Panda (`MountedUR5e`, registered as a `SingleArm`);
- builds the UR5e start state by copying only the object joint positions out of the Panda-sized LIBERO init state, keeping the UR5e's default joints and zeroing velocities. Loading the Panda state directly corrupts the sim and fakes successes.

Gripper commands mean the same thing on both grippers. The holding band is normalised per gripper from its joint range. IK feasibility uses each arm's own kinematics, so a grasp that is reachable for the Panda but not the UR5e is skipped on the UR5e.

## Index build (offline)

`scripts/build_index.py` → `motormem.memory.build_index`:

1. Replay each training HDF5 demo by setting its recorded sim states step by step. This is the one place sim state is read, offline, as in MimicGen.
2. Record the grip-site pose, the gripper command, object body poses and articulated joint positions.
3. Cut segments at gripper events (a close is the command crossing 0 upward, an open is the reverse):
   - GRASP: from the previous open to close + 15 steps. The object is whichever body rises with the gripper.
   - PLACE: from close + 15 to open + 10. The target is the object or region under the held object, and the relation comes from the task language.
   - ACTUATE: an articulated joint moves while the gripper is near it.
   - PUSH: the object moves more than 3 cm with no grasp.
4. Express each segment's poses in the acted-on object's frame. Store object surface points (from offline masks + depth) in the same frame for registration.
5. Save one pickle per index under `data/index/` with a report (`runs/index_report.md`). Example: LIBERO-90 study + living-room scenes gave 2200 demos → 4501 segments in 224 s.

`MotorMemory.without(objects=..., demo_ids=...)` returns a filtered view. That's how the editability experiment removes an object's demos, and a property test checks that removed demos never come back from a query.
