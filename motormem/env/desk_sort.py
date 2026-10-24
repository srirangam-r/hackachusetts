"""Cluttered desk sorting: a custom LIBERO study-table scene with 7 objects and 5 layouts.

Scenes are registered with LIBERO's ``@register_mu`` (one ``DeskSortLayout<k>`` class per layout) and BDDL files
are generated with ``task_generation_utils.register_task_info`` / ``generate_bddl_from_task_info`` into
``motormem/env/scenes/`` (committed; regenerated if missing). ``init_index`` selects the layout (mod 5); the
episode seed drives LIBERO's small in-region placement jitter.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from motormem.types import Arm

SCENES_DIR = Path(__file__).resolve().parent / "scenes"
FIXTURES = ["desk_caddy"]
MOVABLE = ["wooden_tray", "black_book", "yellow_book", "porcelain_mug", "red_coffee_mug", "white_yellow_mug"]
OBJECTS = FIXTURES + MOVABLE
N_LAYOUTS = 5

# Region centres (x, y) relative to the study-table workspace offset (-0.2, 0, 0.867); the robot base sits at
# world x = -0.75, so x in [-0.25, 0.15] is 0.3-0.7 m in front of it. Footprints (measured): caddy 0.43 x 0.15
# (long axis along x at yaw +-pi/2), tray 0.30 x 0.17, mugs/books <= 0.14. The caddy and the tray take the two
# side bands (|y| = 0.30); the five small objects take 5 of 6 slots in the middle band (|y| <= 0.10).
SMALL_SLOTS = [(-0.22, -0.10), (-0.22, 0.10), (-0.04, -0.10), (-0.04, 0.10), (0.13, -0.08), (0.13, 0.08)]
LAYOUTS = [  # (caddy xy, caddy yaw, tray xy, SMALL_SLOTS index for each of MOVABLE[1:])
    ((-0.05, 0.30), np.pi / 2, (-0.05, -0.30), [0, 1, 2, 3, 4]),
    ((-0.05, -0.30), -np.pi / 2, (-0.05, 0.30), [3, 2, 5, 4, 0]),
    ((-0.03, 0.30), -np.pi / 2, (-0.08, -0.30), [5, 0, 4, 1, 2]),
    ((-0.03, -0.30), np.pi / 2, (-0.08, 0.30), [1, 4, 0, 5, 3]),
    ((-0.07, 0.30), np.pi / 2, (-0.02, -0.30), [4, 3, 1, 0, 5]),
]

TASKS = [
    ("put both books in the desk caddy and all the mugs on the wooden tray",
     ["black_book_1", "yellow_book_1", "porcelain_mug_1", "red_coffee_mug_1", "white_yellow_mug_1"],
     [("In", "black_book_1", "desk_caddy_1_back_contain_region"),
      ("In", "yellow_book_1", "desk_caddy_1_front_contain_region"),
      ("In", "porcelain_mug_1", "wooden_tray_1_contain_region"),
      ("In", "red_coffee_mug_1", "wooden_tray_1_contain_region"),
      ("In", "white_yellow_mug_1", "wooden_tray_1_contain_region")]),
    ("put all the mugs on the wooden tray",
     ["porcelain_mug_1", "red_coffee_mug_1", "white_yellow_mug_1"],
     [("In", "porcelain_mug_1", "wooden_tray_1_contain_region"),
      ("In", "red_coffee_mug_1", "wooden_tray_1_contain_region"),
      ("In", "white_yellow_mug_1", "wooden_tray_1_contain_region")]),
    ("put the black book in the back compartment of the caddy",
     ["black_book_1"], [("In", "black_book_1", "desk_caddy_1_back_contain_region")]),
    ("put the red coffee mug on the wooden tray",
     ["red_coffee_mug_1"], [("In", "red_coffee_mug_1", "wooden_tray_1_contain_region")]),
]


def layout_positions(layout: int) -> dict[str, tuple[float, float]]:
    caddy, _, tray, small = LAYOUTS[layout % N_LAYOUTS]
    pos = {"desk_caddy": caddy, "wooden_tray": tray}
    for name, si in zip(MOVABLE[1:], small):
        pos[name] = SMALL_SLOTS[si]
    return pos


def _scene_class_name(layout: int) -> str:
    return f"DeskSortLayout{layout}"


def _register_scenes() -> None:
    from libero.libero.utils.mu_utils import MU_DICT, InitialSceneTemplates, register_mu
    from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info

    for k in range(N_LAYOUTS):
        if f"desk_sort_layout{k}" in MU_DICT:
            continue
        pos = layout_positions(k)
        caddy_yaw = LAYOUTS[k][1]

        def __init__(self):
            InitialSceneTemplates.__init__(
                self, workspace_name="study_table",
                fixture_num_info={"study_table": 1, "desk_caddy": 1},
                object_num_info={name: 1 for name in MOVABLE},
            )

        def define_regions(self, _pos=pos, _yaw=caddy_yaw):
            for name, xy in _pos.items():
                yaw = (_yaw, _yaw) if name == "desk_caddy" else (0.0, 0.0) if name == "wooden_tray" else \
                    (-np.pi / 8, np.pi / 8)
                half = 0.005 if name in ("desk_caddy", "wooden_tray") else 0.015
                self.regions.update(self.get_region_dict(region_centroid_xy=list(xy),
                                                         region_name=f"{name}_init_region",
                                                         target_name=self.workspace_name,
                                                         region_half_len=half, yaw_rotation=yaw))
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

        def init_states(self, _names=tuple(pos)):
            return [("On", f"{n}_1", f"study_table_{n}_init_region") for n in _names]

        cls = type(_scene_class_name(k), (InitialSceneTemplates,),
                   {"__init__": __init__, "define_regions": define_regions, "init_states": property(init_states)})
        register_mu(scene_type="study")(cls)


def bddl_path(task_id: int, layout: int) -> Path:
    return SCENES_DIR / f"desk_sort_t{task_id}_l{layout}.bddl"


def generate_bddls(force: bool = False) -> list[Path]:
    """(Re)generate every task x layout BDDL into ``SCENES_DIR``."""
    from libero.libero.utils import task_generation_utils as tgu

    _register_scenes()
    wanted = [(t, k) for t in range(len(TASKS)) for k in range(N_LAYOUTS)]
    if not force and all(bddl_path(t, k).exists() for t, k in wanted):
        return [bddl_path(t, k) for t, k in wanted]
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    saved = dict(tgu.TASK_INFO)
    out = []
    try:
        for t, k in wanted:
            language, ooi, goals = TASKS[t]
            tgu.TASK_INFO.clear()
            tgu.register_task_info(language, scene_name=f"desk_sort_layout{k}", objects_of_interest=ooi,
                                   goal_states=goals)
            with tempfile.TemporaryDirectory() as tmp:
                names, failures = tgu.generate_bddl_from_task_info(folder=tmp)
                if failures or len(names) != 1:
                    raise RuntimeError(f"BDDL generation failed for task {t} layout {k}: {failures}")
                src = names[0] if os.path.isabs(names[0]) else os.path.join(tmp, names[0])
                shutil.copy(src, bddl_path(t, k))
            out.append(bddl_path(t, k))
    finally:
        tgu.TASK_INFO.clear()
        tgu.TASK_INFO.update(saved)
    return out


def make_desk_sort_env(task_id: int | str = 0, init_index: int = 0, arm: str = Arm.PANDA.value, seed: int = 0,
                       **kwargs):
    from motormem.env.libero import LiberoEnv

    if isinstance(task_id, str):
        langs = [t[0] for t in TASKS]
        task_id = langs.index(task_id) if task_id in langs else int(task_id)
    layout = int(init_index) % N_LAYOUTS
    path = bddl_path(int(task_id), layout)
    if not path.exists():
        generate_bddls()
    return LiberoEnv(path, init_states=None, init_index=layout, arm=arm, suite="desk_sort", seed=seed,
                     task_name=f"desk_sort_t{task_id}_l{layout}", **kwargs)
