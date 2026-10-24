"""Paths and constants shared by all streams. Coordinator-owned."""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_CHECKOUT = Path("/home/nail/stuff/sundai/hackachusetts")
DATA_DIR = Path(os.environ.get("MOTORMEM_DATA", MAIN_CHECKOUT / "data"))
RAW_DIR = DATA_DIR / "raw"
LIBERO_HDF5_DIR = RAW_DIR / "libero_hdf5"
MIMICGEN_DIR = RAW_DIR / "mimicgen" / "source"
LIBERO_PRO_DIR = RAW_DIR / "libero_pro"
INDEX_DIR = DATA_DIR / "index"          # built motor-memory indexes (generated, not raw)
RUNS_DIR = Path(os.environ.get("MOTORMEM_RUNS", MAIN_CHECKOUT / "runs"))

CONTROL_HZ = 20
ACTION_POS_SCALE = 0.05     # metres per unit OSC delta action per step
ACTION_ROT_SCALE = 0.5      # radians per unit OSC delta action per step
IMAGE_SIZE = 256
CAMERAS = ("agentview", "wrist")

# Observation keys that must never reach the brain or the skills at test time.
PRIVILEGED_KEY_PATTERNS = [
    re.compile(r".*_pos$"),
    re.compile(r".*_quat$"),
    re.compile(r".*_to_robot0_eef_(pos|quat)$"),
    re.compile(r"^object-state$"),
    re.compile(r".*segmentation.*"),
]
# Proprioceptive keys that are allowed even though they match the patterns above.
ALLOWED_PROPRIO_KEYS = {
    "robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos",
    "robot0_joint_pos_cos", "robot0_joint_pos_sin",
}


def is_privileged_key(key: str) -> bool:
    if key in ALLOWED_PROPRIO_KEYS:
        return False
    return any(p.match(key) for p in PRIVILEGED_KEY_PATTERNS)
