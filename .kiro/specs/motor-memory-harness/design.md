# Design: Motor Memory harness

## Overview
```
runner (motormem.brain.runner)  -- per episode: EpisodeSpec json -> ROBOT_EPISODE env var
 └─ kiro-cli chat --no-interactive -a --agent robot-brain --model M "<instruction>"
     robot-brain agent: tools = @robot only; prompt = .kiro/steering/robot-protocol.md
      └─ robot MCP server (motormem.brain.mcp_server, stdio) owns the sim for one episode
           observe(find) / run(program) / done(summary)
            └─ Skills(env, perception, memory, motion_source)       [motormem.skills]
                 ├─ Perception.scan / refine                         [motormem.perception]
                 ├─ MotorMemory.query                                [motormem.memory]
                 └─ RobotEnv.step / observe / kinematics             [motormem.env]
```
Contracts shared by all streams: `motormem/types.py`, `motormem/geometry.py`, `motormem/config.py`.

## Stream A: env (`motormem/env/`)
```python
def make_env(spec: EpisodeSpec) -> RobotEnv
class RobotEnv:
    instruction: str; arm: str; suite: str
    def reset(self) -> Observation
    def observe(self) -> Observation                    # stripped, see strip_privileged
    def step(self, action: np.ndarray) -> Observation   # 7-D OSC_POSE delta, gripper last
    def kinematics(self) -> ArmKinematics               # robot self-model for IK (not privileged)
    def render(self, camera="agentview", size=512) -> np.ndarray   # for videos only
    # runner-only (never passed to skills/brain):
    def _runner_success(self) -> bool
    def privileged_object_poses(self) -> dict[str, np.ndarray]     # rung-0 + offline only
    def privileged_segmentation(self, camera) -> dict[str, np.ndarray]  # offline index build only
def strip_privileged(raw: dict) -> dict
@dataclass class ArmKinematics:
    model: mujoco.MjModel; data: mujoco.MjData   # a COPY; IK may mutate it
    qpos_addr: np.ndarray; dof_addr: np.ndarray; site_id: int
    q_lower: np.ndarray; q_upper: np.ndarray; q_current: np.ndarray
```
- Scenes: `libero.py` (standard suites + LIBERO-Pro from `config.LIBERO_PRO_DIR`), `arms.py` (MountedUR5e + init-state remap), `desk_sort.py` (custom study-scene BDDL via `libero.libero.utils.task_generation_utils`), `bin_sort.py` (robosuite PickPlace, any arm).
- `gripper_opening` normalised per gripper from its qpos range.

## Stream B: perception (`motormem/perception/`)
```python
class Segmenter(Protocol):
    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[MaskCandidate]   # mask, label, score, box
def make_segmenter(name: str = "auto") -> Segmenter   # sam3 if weights load, else gdino_sam2
class Perception:
    def __init__(self, segmenter: Segmenter, object_points_by_name: dict[str, np.ndarray] | None = None)
    def scan(self, obs: Observation, find: list[str], camera="agentview") -> list[ObjectEstimate]
    def refine(self, obs: Observation, est: ObjectEstimate, camera="wrist") -> ObjectEstimate
def render_overlay(obs: Observation, objects: list[ObjectEstimate], camera="agentview") -> PIL.Image
class OraclePerception(Perception)   # rung-0 only: masks/frames from privileged poses
```
- Masks deprojected with `geometry.deproject`; frames from `geometry.pca_frame`; dedupe overlapping masks (IoU > 0.7); marks numbered left-to-right.

## Stream C: memory (`motormem/memory/`)
```python
def build_index(sources: list[str], out: Path, max_demos_per_task: int | None = None) -> None
class MotorMemory:
    @classmethod
    def load(cls, path) -> MotorMemory
    def save(self, path) -> None
    def query(self, kind: SegmentKind, object_name: str | None = None, relation: str | None = None,
              held_name: str | None = None, extent: np.ndarray | None = None, k: int = 8) -> list[Segment]
    def canonical(self, label: str) -> str | None      # "black bowl" -> "akita_black_bowl"
    def object_points(self, name: str) -> np.ndarray   # merged surface points, object frame
    def without(self, objects=(), demo_ids=()) -> MotorMemory   # filtered view
    def add_segments(self, segs: list[Segment]) -> None
```
- Build: for each HDF5 demo set states step by step in a LIBERO env, read grip-site pose, gripper command (`actions[:, -1]`), object body poses and articulated joint qpos. Close event = command crosses 0 upward; open = downward.
  - GRASP: from the previous open (or start) to close + 15 steps; object = the body that rises with the gripper after close.
  - PLACE: from close + 15 to open + 10; target = object/region under/containing the held object at release; relation parsed from task language (on/in/left_of/right_of/front_of/back_of/top_of).
  - ACTUATE: segments where an articulated joint moves while the gripper is near it; relation from the sign of the joint change and the joint's open/close range.
  - PUSH: object displacement > 3 cm with gripper open/no grasp.
- Object points: offline from `privileged_segmentation` + depth over a few frames, expressed in the object body frame.
- Storage: one `.npz`/pickle per index under `config.INDEX_DIR`.

## Stream D: skills (`motormem/skills/`)
```python
class Controller:
    def __init__(self, env: RobotEnv)
    def track(self, targets: np.ndarray, gripper: np.ndarray, pos_tol=0.005, rot_tol=0.05,
              max_steps_per_wp=15) -> TrackResult          # absolute pose targets via OSC deltas
    def gripper(self, cmd: float, steps=10) -> Observation
    def holding(self, obs) -> bool
class Feasibility:
    def check(self, kin: ArmKinematics, targets: np.ndarray) -> tuple[bool, str]   # damped-LS IK + joint limits
class Skills:
    def __init__(self, env, perception, memory, motion_source="hybrid")
    def pick(self, obj: ObjectEstimate) -> SkillResult
    def place(self, target: ObjectEstimate, relation: str) -> SkillResult
    def push(self, obj: ObjectEstimate, dx: float, dy: float) -> SkillResult
    def actuate(self, part: ObjectEstimate, action: str) -> SkillResult    # open|close|turn_on|turn_off
    def home(self) -> SkillResult
```
- Retarget: `register_multi_yaw(segment.object_points, live.points_world)` -> `T_world_obj`; targets = `compose(T_world_obj, seg.T_obj_eef)`; for PLACE also correct for the current held offset vs `seg.T_eef_held`.
- Approach: free-space interpolation to the first segment pose (lifted 8 cm), then the contact part of the segment.
- Actuate: lead 1.5 cm/step along the current direction (initial = mask face normal), refit line vs circle (least squares, BIC-style penalty) after 3 cm, stall = < 2 mm progress over 10 steps with saturated lead, slip = gripper opening collapses.
- Primitive: top-down grasp at the mask centroid, yaw = principal axis, then lift; place = above the target centroid + extent, open.

## Stream E: brain (`motormem/brain/`)
- `program.py`: grammar, one call per line:
  `pick(M)`, `place(M, "on"|"in"|...)`, `push(M, dx=..., dy=...)`, `actuate(M, "open"|"close"|"turn_on"|"turn_off")`, `look(M)`, `home()`. Integers are marks; metres for dx/dy. Anything else -> `ProgramError`.
- `mcp_server.py` (mcp 2.3.0 FastMCP, stdio): reads `ROBOT_EPISODE`, builds env/perception/memory/skills, writes `trace.jsonl` and `result.json` to `spec.out_dir`; ends the episode on `_runner_success()` or step budget and then answers `EPISODE_OVER`.
  - `observe(find: list[str])` -> [image(agentview overlay), image(wrist overlay), text(marks, labels, positions in cm, gripper state)].
  - `run(program: str)` -> text(per-step status lines) + fresh images.
  - `done(summary: str)` -> ends the episode.
- `runner.py`: builds specs, launches parallel `kiro-cli` sessions, collects results into `runs/<name>/summary.json`.
- `scripted.py`: rung-0 hand-written programs for LIBERO-Goal (diagnostic only).
- `.kiro/agents/robot-brain.md` + `.kiro/steering/robot-protocol.md` (inclusion: manual).

## Correctness properties (property-based tests)
1. Pose inverse round-trip (Req 7.1) — `tests/test_geometry_props.py`.
2. Object-frame storage round-trip and rigid re-targeting (Req 7.1, 3.2).
3. Deproject/project round-trip (Req 7.2).
4. For any raw observation dict, `strip_privileged` output has no key with `is_privileged_key` true and keeps every allowed proprio key (Req 1.2).
5. For any memory and any removed demo/object, no query result derives from it (Req 3.4).
6. For any string, the parser either returns calls that all belong to the grammar or raises `ProgramError`; it never executes (Req 5.2).
7. For any MCP tool output, no privileged key names or success fields appear (Req 5.4).
8. Line/circle fitting recovers the generating model from noisy samples of a line or an arc (Req 4.4).
9. Gripper-event segmentation: for any synthetic command sequence, segments cover each close/open event exactly once (Req 3.1).

## Error handling
Every skill returns `SkillResult`; exceptions inside skills become `Status.ERROR` with a one-line message. The MCP server never crashes the episode on a bad program.
