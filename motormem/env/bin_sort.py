"""Industrial bin sorting: robosuite ``PickPlace`` (milk, bread, cereal, can -> their matching bin sections).

Works for any robosuite arm (``Panda``, ``UR5e``). Layouts are reproducible: ``init_index`` and ``seed`` seed the
placement sampler (robosuite's samplers draw from the global numpy RNG).
"""
from __future__ import annotations

import numpy as np
import robosuite as suite

from motormem import config, geometry
from motormem.env.robot_env import CAMERA_MAP, RobotEnv
from motormem.types import Arm

INSTRUCTION = "sort each item into its matching bin"


class BinSortEnv(RobotEnv):
    def __init__(self, arm: str = Arm.UR5E.value, init_index: int = 0, seed: int = 0,
                 cameras=config.CAMERAS, image_size: int = config.IMAGE_SIZE, settle_steps: int = 10):
        self.init_index = int(init_index)
        self.seed = int(seed)
        rs = suite.make(
            "PickPlace",
            robots=arm,
            controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
            single_object_mode=0,                     # all four objects
            initialization_noise=None,                # same layout for every arm given (seed, init_index)
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            use_object_obs=True,                      # raw keys exist but are stripped by strip_privileged
            camera_names=[CAMERA_MAP.get(c, c) for c in cameras],
            camera_heights=image_size,
            camera_widths=image_size,
            camera_depths=True,
            control_freq=config.CONTROL_HZ,
            ignore_done=True,
            horizon=100_000,
            reward_shaping=False,
        )
        super().__init__(rs, instruction=INSTRUCTION, arm=arm, suite="bin_sort", cameras=cameras,
                         settle_steps=settle_steps, image_size=image_size)

    def _reset_sim(self) -> dict:
        state = np.random.get_state()
        np.random.seed((self.seed * 100_003 + self.init_index) % (2**32))
        try:
            raw = self._rs.reset()
        finally:
            np.random.set_state(state)
        return raw

    def _object_body_ids(self) -> dict[str, int]:
        return dict(self._rs.obj_body_id)

    def privileged_object_poses(self) -> dict[str, np.ndarray]:
        out = super().privileged_object_poses()
        out["bin1"] = geometry.make_pose(self._rs.bin1_pos)
        out["bin2"] = geometry.make_pose(self._rs.bin2_pos)
        for i, obj in enumerate(self._rs.objects):           # target section of each object in bin2
            out[f"{obj.name}_target"] = geometry.make_pose(self._rs.target_bin_placements[i])
        return out

    def privileged_objects_in_bins(self) -> dict[str, bool]:
        """Per-object success (runner/diagnostics only)."""
        return {obj.name: bool(self._rs.objects_in_bins[i]) for i, obj in enumerate(self._rs.objects)}


def make_bin_sort_env(arm: str = Arm.UR5E.value, init_index: int = 0, seed: int = 0, **kwargs) -> BinSortEnv:
    return BinSortEnv(arm=arm, init_index=init_index, seed=seed, **kwargs)
