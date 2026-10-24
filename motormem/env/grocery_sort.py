"""Cluttered grocery sorting (the live "bin sorting" demo): one LIBERO living-room table layout with a basket,
a wooden tray and nine groceries, three of them duplicated (ketchup, butter, milk).

The task is free text given at run time (e.g. "put all the ketchup in the basket and the butter on the tray"),
so the scene has no fixed goal: `_runner_success()` is always False and the episode ends when the brain
calls done() or the step budget runs out. `object_status()` (runner/diagnostics only) reports where each
grocery ended up.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from motormem.types import Arm

SCENES_DIR = Path(__file__).resolve().parent / "scenes"
CONTAINERS = {"basket": (0.02, 0.30), "wooden_tray": (0.02, -0.31)}
# (object, instance count); every category has pick and place demos in the LIBERO living-room scenes
GROCERIES = [("ketchup", 2), ("butter", 2), ("milk", 2), ("alphabet_soup", 1), ("tomato_sauce", 1),
             ("orange_juice", 1)]
SLOTS = [(-0.10, -0.13), (-0.10, 0.0), (-0.10, 0.13), (0.02, -0.13), (0.02, 0.0), (0.02, 0.13),
         (0.14, -0.13), (0.14, 0.0), (0.14, 0.13)]
INSTANCES = [f"{n}_{i + 1}" for n, k in GROCERIES for i in range(k)]
# interleave categories so duplicates are not next to each other
PLACEMENT = dict(zip(["ketchup_1", "butter_1", "orange_juice_1", "alphabet_soup_1", "ketchup_2", "milk_1",
                      "milk_2", "tomato_sauce_1", "butter_2"], SLOTS))
DEFAULT_INSTRUCTION = "put all the ketchup bottles in the basket and the butter boxes on the tray"
SCENE = "grocery_sort_scene"


def _register_scene() -> None:
    from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
    from libero.libero.utils.mu_utils import MU_DICT, InitialSceneTemplates, register_mu

    if SCENE in MU_DICT:
        return

    def __init__(self):
        InitialSceneTemplates.__init__(
            self, workspace_name="living_room_table",
            fixture_num_info={"living_room_table": 1},
            object_num_info={**{n: k for n, k in GROCERIES}, "basket": 1, "wooden_tray": 1})

    def define_regions(self):
        for name, xy in {**{f"{c}_1": p for c, p in CONTAINERS.items()}, **PLACEMENT}.items():
            self.regions.update(self.get_region_dict(region_centroid_xy=list(xy), region_name=f"{name}_init_region",
                                                     target_name=self.workspace_name,
                                                     region_half_len=0.005 if name[:-2] in CONTAINERS else 0.015))
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    def init_states(self):
        names = [f"{c}_1" for c in CONTAINERS] + list(PLACEMENT)
        return [("On", n, f"living_room_table_{n}_init_region") for n in names]

    cls = type("GrocerySortScene", (InitialSceneTemplates,),
               {"__init__": __init__, "define_regions": define_regions, "init_states": property(init_states)})
    register_mu(scene_type="living_room")(cls)


def bddl_path() -> Path:
    return SCENES_DIR / "grocery_sort.bddl"


def generate_bddl(force: bool = False) -> Path:
    from libero.libero.utils import task_generation_utils as tgu

    _register_scene()
    if bddl_path().exists() and not force:
        return bddl_path()
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    saved = dict(tgu.TASK_INFO)
    try:
        tgu.TASK_INFO.clear()
        # placeholder goal (BDDL needs one); the live task is free text and success is not used
        tgu.register_task_info(DEFAULT_INSTRUCTION, scene_name=SCENE, objects_of_interest=["ketchup_1"],
                               goal_states=[("In", "ketchup_1", "basket_1_contain_region")])
        with tempfile.TemporaryDirectory() as tmp:
            names, failures = tgu.generate_bddl_from_task_info(folder=tmp)
            if failures or len(names) != 1:
                raise RuntimeError(f"grocery scene BDDL generation failed: {failures}")
            src = names[0] if os.path.isabs(names[0]) else os.path.join(tmp, names[0])
            shutil.copy(src, bddl_path())
    finally:
        tgu.TASK_INFO.clear()
        tgu.TASK_INFO.update(saved)
    return bddl_path()


def object_status(env) -> dict[str, str]:
    """Runner/diagnostics only: 'basket' | 'tray' | 'table' for each grocery instance."""
    from libero.libero.envs.predicates import eval_predicate_fn

    st = env._rs.object_states_dict
    out = {}
    for n in INSTANCES:
        if eval_predicate_fn("in", st[n], st["basket_1_contain_region"]):
            out[n] = "basket"
        elif eval_predicate_fn("in", st[n], st["wooden_tray_1_contain_region"]):
            out[n] = "tray"
        else:
            out[n] = "table"
    return out


def make_grocery_env(init_index: int = 0, arm: str = Arm.PANDA.value, seed: int = 0, **kwargs):
    from motormem.env.libero import LiberoEnv

    path = generate_bddl()
    env = LiberoEnv(path, init_states=None, init_index=0, arm=arm, suite="grocery_sort", seed=seed,
                    task_name="grocery_sort", **{"settle_steps": 100, **kwargs})
    env._runner_success = lambda: False           # free-text task: the brain decides when it is done
    return env
