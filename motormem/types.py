"""Shared data contracts between streams. Coordinator-owned: change only via the coordinator."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Arm(str, Enum):
    PANDA = "Panda"
    UR5E = "UR5e"


@dataclass
class CameraFrame:
    """One camera at one instant. Images are in image convention (row 0 = top, already un-flipped)."""
    name: str                    # "agentview" | "wrist"
    rgb: np.ndarray              # (H, W, 3) uint8
    depth: np.ndarray            # (H, W) float32, metres along the optical axis
    K: np.ndarray                # (3, 3) intrinsics
    T_world_cam: np.ndarray      # (4, 4) camera pose in world, OpenCV axes (x right, y down, z forward)


@dataclass
class Proprio:
    T_world_eef: np.ndarray      # (4, 4) grip-site pose (the OSC control frame)
    gripper_qpos: np.ndarray     # raw finger joint positions (size differs per gripper)
    gripper_opening: float       # normalised 0 = fully closed .. 1 = fully open (comparable across grippers)
    joint_pos: np.ndarray        # arm joint positions
    ee_force: np.ndarray         # (3,) wrist force, world frame
    ee_torque: np.ndarray        # (3,) wrist torque, world frame


@dataclass
class Observation:
    """Everything the skills and the brain may see. Never contains privileged simulator state."""
    cameras: dict[str, CameraFrame]
    proprio: Proprio
    instruction: str
    step: int
    T_world_base: np.ndarray     # (4, 4) robot base pose (robot self-knowledge, not privileged)


@dataclass
class ObjectEstimate:
    """A perceived object or part. `mark` is the number drawn on the overlay the brain sees."""
    mark: int
    label: str                   # text label the segmenter used (e.g. "black bowl")
    camera: str                  # camera the mask is in
    mask: np.ndarray             # (H, W) bool
    points_world: np.ndarray     # (N, 3) observed surface points
    T_world_obj: np.ndarray      # (4, 4) estimated frame: origin = centroid, z = world up, x = main horizontal axis
    extent: np.ndarray           # (3,) size along the frame axes, metres
    score: float = 1.0


class SegmentKind(str, Enum):
    GRASP = "grasp"      # approach -> close -> lift, relative to the grasped object
    PLACE = "place"      # carry -> open -> retreat, relative to the target object/region
    PUSH = "push"        # contact motion without grasp, relative to the pushed object
    ACTUATE = "actuate"  # grasp/contact of an articulated part (handle, knob) + its motion


@dataclass
class Segment:
    """One reusable demo motion, stored in the frame of the object it acts on."""
    segment_id: str
    demo_id: str                 # e.g. "libero_goal/open_the_middle_drawer_of_the_cabinet/demo_12"
    source: str                  # dataset name, e.g. "libero_goal", "mimicgen/pick_place"
    kind: SegmentKind
    object_name: str             # canonical name of the frame object (grasped / target / pushed / articulated)
    relation: str | None         # PLACE: "on" | "in" | "left_of" | ...; ACTUATE: "open" | "close" | "turn_on" | "turn_off"
    held_name: str | None        # PLACE: canonical name of the object being carried
    T_obj_eef: np.ndarray        # (N, 4, 4) gripper poses in the object frame
    gripper: np.ndarray          # (N,) gripper commands in [-1, 1]
    T_eef_held: np.ndarray | None  # PLACE: (4, 4) held object pose in gripper frame at segment start
    object_points: np.ndarray    # (M, 3) surface points of the frame object, in the object frame
    task_language: str
    arm: str = Arm.PANDA.value
    meta: dict[str, Any] = field(default_factory=dict)


class Status(str, Enum):
    OK = "OK"
    IK_FAILURE = "IK_FAILURE"
    COLLISION = "COLLISION"
    NO_CANDIDATE = "NO_CANDIDATE"
    PERCEPTION_FAILED = "PERCEPTION_FAILED"
    GRASP_FAILED = "GRASP_FAILED"
    NOT_HOLDING = "NOT_HOLDING"
    STALL = "STALL"
    TIMEOUT = "TIMEOUT"
    BAD_ARGUMENT = "BAD_ARGUMENT"
    EPISODE_OVER = "EPISODE_OVER"
    ERROR = "ERROR"


@dataclass
class SkillResult:
    status: Status
    message: str                 # short, model-readable reason
    segment_id: str | None = None
    used_fallback: bool = False  # True when the hand-written primitive produced the motion
    steps: int = 0
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeSpec:
    """Everything needed to reproduce one evaluation episode."""
    suite: str                   # "libero_goal" | "libero_spatial" | "libero_object" | "libero_10" |
                                 # "libero_pro_goal_swap" | "libero_pro_goal_task" | "desk_sort" | "bin_sort"
    task_id: int | str
    init_index: int
    arm: str = Arm.PANDA.value
    motion_source: str = "hybrid"    # "index" | "primitive" | "hybrid" (index, else primitive)
    perception: str = "sam3"          # "sam3" | "gdino_sam2" | "oracle" (rung-0 diagnostic only)
    index_path: str = ""
    index_exclude: dict[str, list[str]] = field(default_factory=dict)  # {"objects": [...], "demo_ids": [...]}
    max_steps: int = 600
    seed: int = 0
    out_dir: str = ""
    model: str = "claude-opus-5.5"
