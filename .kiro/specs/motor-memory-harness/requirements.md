# Requirements: Motor Memory harness

## Introduction
A robot harness where a frontier model running in Kiro plans and grounds, and a fixed execution layer acts by replaying recorded demo segments in the frame of the live object. It must run on two arms, stay fair (no privileged state, no success oracle), and be fast (few model turns per episode).

## Glossary
- **Brain**: the `robot-brain` Kiro agent (headless `kiro-cli`), restricted to the robot MCP tools.
- **Segment**: a demo motion (grasp/place/push/actuate) stored as gripper poses in its object's frame.
- **Motor memory (index)**: the searchable collection of segments.
- **Primitive**: a hand-written fallback motion (top-down grasp, place-above).
- **Mark**: the number drawn on a segmented object in the brain's images.

## Requirement 1: Fair environment adapter
**User story:** As an evaluator, I want every observation scrubbed of simulator state so results are comparable to published baselines.
1. WHEN an observation is built THE SYSTEM SHALL include only RGB, depth, intrinsics, extrinsics, proprioception and wrist force/torque.
2. THE SYSTEM SHALL remove every key for which `config.is_privileged_key` is true before any skill or the brain receives it.
3. WHEN an episode uses the UR5e THE SYSTEM SHALL build the LIBERO scene with `MountedUR5e` and transfer only object state from the Panda-sized init state.
4. THE SYSTEM SHALL expose the success signal and ground-truth poses only through runner-only methods (`_runner_success`, `privileged_object_poses`).
5. THE SYSTEM SHALL support LIBERO goal/spatial/object/10, LIBERO-Pro goal (swap, task), a custom desk-sort scene and a robosuite bin-sort scene.

## Requirement 2: Perception
**User story:** As the brain, I want numbered object masks and metric overlays so I can point at things by number.
1. WHEN `scan(obs, find)` is called THE SYSTEM SHALL return one `ObjectEstimate` per detected instance with a unique mark, mask, world points and frame.
2. THE SYSTEM SHALL use SAM 3 when its weights are available and otherwise Grounding DINO + SAM 2.1, behind one interface.
3. THE SYSTEM SHALL render overlays with numbered marks, a metric grid on the table plane, and the gripper pinch point.
4. WHEN a mask has fewer than 20 valid depth points THE SYSTEM SHALL drop it and report it.

## Requirement 3: Motor memory
**User story:** As a developer, I want demos turned into reusable object-frame segments so one demo serves every layout.
1. WHEN building THE SYSTEM SHALL replay each demo's recorded states and cut it at gripper close/open events into GRASP, PLACE, PUSH and ACTUATE segments.
2. THE SYSTEM SHALL store each segment's gripper poses in the frame of the object it acts on, plus that object's surface points in the same frame.
3. WHEN queried by kind and canonical object name THE SYSTEM SHALL return ranked candidates; WHEN no exact name exists THE SYSTEM SHALL fall back to shape similarity.
4. WHEN a demo or object is removed THE SYSTEM SHALL never return any segment derived from it.
5. THE SYSTEM SHALL support excluding objects or demo ids per episode (editability experiments) without rebuilding.

## Requirement 4: Skills and execution
**User story:** As the brain, I want a few reliable skills that report short, typed failures.
1. WHEN `pick(obj)` runs THE SYSTEM SHALL register the candidate segment's object points to the live object, transform the segment, check reachability for the current arm, and execute it with OSC tracking.
2. WHEN no index candidate is feasible AND motion_source permits THE SYSTEM SHALL use the primitive and set `used_fallback`.
3. THE SYSTEM SHALL verify grasps with the normalised gripper opening and report `GRASP_FAILED` or `NOT_HOLDING`.
4. WHEN `actuate(part, action)` runs THE SYSTEM SHALL grasp/contact the part, move with an equilibrium-point lead, fit a line or circle to the measured gripper path online, and stop on target, stall or slip.
5. THE SYSTEM SHALL produce identical behaviour on Panda and UR5e apart from kinematics (same segments, same code).
6. THE SYSTEM SHALL complete every skill within a step budget and return `TIMEOUT` otherwise.

## Requirement 5: Brain interface
**User story:** As a Kiro agent, I want a tiny, safe tool surface.
1. THE SYSTEM SHALL expose MCP tools `observe(find)`, `run(program)`, `done(summary)` and nothing else.
2. WHEN a program contains anything outside the grammar THE SYSTEM SHALL reject it without moving the robot.
3. WHEN `run` executes THE SYSTEM SHALL return each step's status and a fresh observation.
4. THE SYSTEM SHALL never include privileged keys or success information in tool output (except `EPISODE_OVER` when the runner ends the episode).

## Requirement 6: Evaluation
**User story:** As the team, I want reproducible, parallel evaluations and a diagnostic ladder.
1. THE SYSTEM SHALL run N episodes in parallel, each a headless `kiro-cli` session with its own `EpisodeSpec`.
2. THE SYSTEM SHALL record success, wall time, model turns, skill statuses and the segment ids used.
3. THE SYSTEM SHALL support a scripted rung-0 mode (oracle perception, scripted plans, no brain) for diagnostics only.
4. THE SYSTEM SHALL support motion_source ablations (index vs primitive vs hybrid).

## Requirement 7: Geometric correctness
1. Pose inversion, composition and object-frame storage SHALL be lossless to 1e-9.
2. Deprojection followed by projection SHALL return the same pixel.
