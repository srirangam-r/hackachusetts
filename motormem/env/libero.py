"""LIBERO and LIBERO-Pro scenes behind ``RobotEnv``, plus the ``make_env`` dispatcher.

BDDL/init files are resolved explicitly (never through ``~/.libero/config.yaml``, which points elsewhere):
- standard suites: the INSTALLED ``libero`` package's ``bddl_files/`` and ``init_files/``;
- LIBERO-Pro: ``config.LIBERO_PRO_DIR/{bddl_files,init_files}/<suite>_<perturbation>/`` (read-only), task order
  from LIBERO-Pro's ``libero_suite_task_map.py``.
Assets resolve through ``libero.libero.get_assets_path()`` (``~/.cache/libero/assets``).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import libero.libero as _libero_pkg
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_utils import get_problem_info

from motormem import config
from motormem.env import arms  # noqa: F401  (registers MountedUR5e)
from motormem.env.robot_env import CAMERA_MAP, RobotEnv
from motormem.types import Arm, EpisodeSpec

LIBERO_ROOT = Path(os.path.dirname(_libero_pkg.__file__))
LIBERO_BDDL_DIR = LIBERO_ROOT / "bddl_files"
LIBERO_INIT_DIR = LIBERO_ROOT / "init_files"
STANDARD_SUITES = ("libero_goal", "libero_spatial", "libero_object", "libero_10", "libero_90")
PRO_SUITES = {  # our suite name -> LIBERO-Pro folder
    "libero_pro_goal_swap": "libero_goal_swap",
    "libero_pro_goal_task": "libero_goal_task",
}
PRO_TASK_MAP_FILE = config.RAW_DIR / "LIBERO-PRO-src" / "libero" / "libero" / "benchmark" / "libero_suite_task_map.py"


@dataclass
class LiberoTask:
    suite: str
    task_id: int
    name: str
    bddl_file: Path
    init_file: Path


def _pro_task_map() -> dict[str, list[str]]:
    tree = ast.parse(PRO_TASK_MAP_FILE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            return ast.literal_eval(node.value)
    raise RuntimeError(f"no task map dict in {PRO_TASK_MAP_FILE}")


def suite_task_names(suite: str) -> list[str]:
    if suite in STANDARD_SUITES:
        from libero.libero.benchmark import get_benchmark_dict
        bench = get_benchmark_dict()[suite]()
        return [bench.get_task(i).name for i in range(bench.n_tasks)]
    if suite in PRO_SUITES:
        return list(_pro_task_map()[PRO_SUITES[suite]])
    raise KeyError(f"unknown LIBERO suite {suite!r}")


def resolve_task(suite: str, task_id: int | str) -> LiberoTask:
    names = suite_task_names(suite)
    if isinstance(task_id, str) and not task_id.isdigit():
        if task_id not in names:
            raise KeyError(f"task {task_id!r} not in {suite}")
        idx = names.index(task_id)
    else:
        idx = int(task_id)
    name = names[idx]
    if suite in STANDARD_SUITES:
        from libero.libero.benchmark import get_benchmark_dict
        t = get_benchmark_dict()[suite]().get_task(idx)
        bddl = LIBERO_BDDL_DIR / t.problem_folder / t.bddl_file
        init = LIBERO_INIT_DIR / t.problem_folder / t.init_states_file
    else:
        folder = PRO_SUITES[suite]
        bddl = config.LIBERO_PRO_DIR / "bddl_files" / folder / f"{name}.bddl"
        init = config.LIBERO_PRO_DIR / "init_files" / folder / f"{name}.pruned_init"
    for p in (bddl, init):
        if not p.exists():
            raise FileNotFoundError(p)
    return LiberoTask(suite=suite, task_id=idx, name=name, bddl_file=bddl, init_file=init)


def load_init_states(path: Path) -> np.ndarray:
    import torch
    states = torch.load(str(path), weights_only=False)
    return np.asarray(states, dtype=np.float64)


def make_libero_rs_env(bddl_file: Path | str, arm: str = Arm.PANDA.value, image_size: int = config.IMAGE_SIZE,
                       cameras=config.CAMERAS, seed: int = 0, **kwargs) -> OffScreenRenderEnv:
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        robots=[arm],
        camera_names=[CAMERA_MAP.get(c, c) for c in cameras],
        camera_heights=image_size,
        camera_widths=image_size,
        camera_depths=True,
        ignore_done=True,
        horizon=100_000,
        **kwargs,
    )
    env.seed(seed)
    return env


class LiberoEnv(RobotEnv):
    """A LIBERO / LIBERO-Pro / custom-BDDL scene, reset to one recorded init state."""

    def __init__(self, bddl_file: Path | str, init_states: np.ndarray | None, init_index: int = 0,
                 arm: str = Arm.PANDA.value, suite: str = "libero", seed: int = 0,
                 cameras=config.CAMERAS, image_size: int = config.IMAGE_SIZE, settle_steps: int = 10,
                 task_name: str = ""):
        self.bddl_file = Path(bddl_file)
        self.task_name = task_name or self.bddl_file.stem
        self.init_states = init_states
        self.init_index = int(init_index)
        self._wrapper = make_libero_rs_env(bddl_file, arm=arm, image_size=image_size, cameras=cameras, seed=seed)
        instruction = get_problem_info(str(bddl_file))["language_instruction"]
        if isinstance(instruction, (list, tuple)):
            instruction = " ".join(instruction)
        super().__init__(self._wrapper.env, instruction=str(instruction), arm=arm, suite=suite,
                         cameras=cameras, settle_steps=settle_steps, image_size=image_size)

    def _reset_sim(self) -> dict:
        raw = self._wrapper.reset()
        if self.init_states is None:
            return raw
        state = self.init_states[self.init_index % len(self.init_states)]
        if self.arm == Arm.PANDA.value and len(state) == self.sim.get_state().flatten().shape[0]:
            return self._wrapper.set_init_state(state)
        # Other arm: NEVER set_init_state (Panda-sized). Copy object joints only, keep default arm joints.
        arms.transfer_object_state(self.sim, state)
        self._rs._post_process()
        return self._refresh_raw()

    def close(self) -> None:
        try:
            self._wrapper.close()
        except Exception:
            pass


def make_libero_env(suite: str, task_id: int | str, init_index: int = 0, arm: str = Arm.PANDA.value,
                    seed: int = 0, **kwargs) -> LiberoEnv:
    task = resolve_task(suite, task_id)
    return LiberoEnv(task.bddl_file, load_init_states(task.init_file), init_index=init_index, arm=arm,
                     suite=suite, seed=seed, task_name=task.name, **kwargs)


def make_env(spec: EpisodeSpec, **kwargs) -> RobotEnv:
    """Build the episode's environment (not yet reset)."""
    if spec.suite in STANDARD_SUITES or spec.suite in PRO_SUITES:
        return make_libero_env(spec.suite, spec.task_id, spec.init_index, arm=spec.arm, seed=spec.seed, **kwargs)
    if spec.suite == "grocery_sort":
        from motormem.env.grocery_sort import make_grocery_env
        return make_grocery_env(init_index=spec.init_index, arm=spec.arm, seed=spec.seed, **kwargs)
    if spec.suite == "bin_sort":
        from motormem.env.bin_sort import make_bin_sort_env
        return make_bin_sort_env(arm=spec.arm, init_index=spec.init_index, seed=spec.seed, **kwargs)
    if spec.suite == "desk_sort":
        from motormem.env.desk_sort import make_desk_sort_env
        return make_desk_sort_env(task_id=spec.task_id, init_index=spec.init_index, arm=spec.arm,
                                  seed=spec.seed, **kwargs)
    raise KeyError(f"unknown suite {spec.suite!r}")
