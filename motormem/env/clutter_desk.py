"""Cluttered desk sorting: the live demo scene.

A wide LIBERO living-room table. Containers sit at the edges (desk caddy, wooden tray, basket) and a cabinet with
three drawers stands at the side with free space in front of it, so an open drawer never blocks the clutter.
Fourteen everyday items (books, mugs, snack boxes / packaging, cans, bottles, cartons) are dropped at random
positions and headings into one dense zone and settled with physics, so they touch, lean on and lie on top of
each other. The clutter is reproducible per `seed`.

The agentview camera is moved up and back with a wider field of view so the whole desk is visible (nominal
calibration is still read from the simulator's camera model, as for every scene).

The task is free text given at run time, so `_runner_success()` is always False; the brain ends the episode with
done(). `object_status()` (runner/diagnostics only) reports where each item ended up.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from motormem.types import Arm

SCENES_DIR = Path(__file__).resolve().parent / "scenes"
SCENE = "clutter_desk_scene"
# table frame (living_room_table, origin at the table centre): +x away from the robot, +y to the robot's left
CABINET = ("wooden_cabinet", (0.22, -0.46), np.pi)
CONTAINERS = {"desk_caddy": ((0.20, 0.44), np.pi / 2), "wooden_tray": ((-0.14, 0.46), 0.0),
              "basket": ((-0.16, -0.40), 0.0)}
ITEMS = ["black_book", "yellow_book", "porcelain_mug", "red_coffee_mug", "white_yellow_mug",
         "butter", "cream_cheese", "chocolate_pudding", "ketchup", "alphabet_soup", "tomato_sauce",
         "milk", "orange_juice", "salad_dressing"]
CLUTTER_ZONE = ((-0.17, 0.15), (-0.20, 0.22))     # x range, y range (table frame)
N_STACKED = 4                                      # items dropped on top of others
DEFAULT_INSTRUCTION = ("tidy the desk: books in the caddy, mugs on the tray, food packaging in the basket, "
                       "and the soup can in the top drawer")


def _register_scene() -> None:
    from libero.libero.utils.bddl_generation_utils import get_xy_region_kwargs_list_from_regions_info
    from libero.libero.utils.mu_utils import MU_DICT, InitialSceneTemplates, register_mu

    if SCENE in MU_DICT:
        return

    def __init__(self):
        InitialSceneTemplates.__init__(
            self, workspace_name="living_room_table",
            fixture_num_info={"living_room_table": 1, CABINET[0]: 1},
            object_num_info={**{n: 1 for n in ITEMS}, **{c: 1 for c in CONTAINERS}})

    def define_regions(self):
        def add(name, xy, half, yaw=0.0):
            self.regions.update(self.get_region_dict(region_centroid_xy=list(xy), region_name=f"{name}_init_region",
                                                     target_name=self.workspace_name, region_half_len=half,
                                                     yaw_rotation=(yaw, yaw)))
        add(f"{CABINET[0]}_1", CABINET[1], 0.005, CABINET[2])
        for c, (xy, yaw) in CONTAINERS.items():
            add(f"{c}_1", xy, 0.005, yaw)
        # parking spots far apart (LIBERO's sampler needs a valid start); the clutter generator moves them
        for i, n in enumerate(ITEMS):
            x = -0.30 + 0.6 * (i % 7) / 6
            y = -0.75 if i < 7 else 0.75
            add(f"{n}_1", (x, y), 0.002)
        self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    def init_states(self):
        names = [f"{CABINET[0]}_1"] + [f"{c}_1" for c in CONTAINERS] + [f"{n}_1" for n in ITEMS]
        return [("On", n, f"living_room_table_{n}_init_region") for n in names]

    cls = type("ClutterDeskScene", (InitialSceneTemplates,),
               {"__init__": __init__, "define_regions": define_regions, "init_states": property(init_states)})
    register_mu(scene_type="living_room")(cls)


def bddl_path() -> Path:
    return SCENES_DIR / "clutter_desk.bddl"


def generate_bddl(force: bool = False) -> Path:
    from libero.libero.utils import task_generation_utils as tgu

    _register_scene()
    if bddl_path().exists() and not force:
        return bddl_path()
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    saved = dict(tgu.TASK_INFO)
    try:
        tgu.TASK_INFO.clear()
        tgu.register_task_info(DEFAULT_INSTRUCTION, scene_name=SCENE, objects_of_interest=["black_book_1"],
                               goal_states=[("In", "black_book_1", "desk_caddy_1_back_contain_region")])
        with tempfile.TemporaryDirectory() as tmp:
            names, failures = tgu.generate_bddl_from_task_info(folder=tmp)
            if failures or len(names) != 1:
                raise RuntimeError(f"clutter desk BDDL generation failed: {failures}")
            src = names[0] if os.path.isabs(names[0]) else os.path.join(tmp, names[0])
            shutil.copy(src, bddl_path())
    finally:
        tgu.TASK_INFO.clear()
        tgu.TASK_INFO.update(saved)
    return bddl_path()


def _quat_wxyz_yaw(yaw: float, tilt: float = 0.0, tilt_axis: float = 0.0) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    r = Rotation.from_euler("z", yaw) * Rotation.from_rotvec([np.cos(tilt_axis) * tilt, np.sin(tilt_axis) * tilt, 0])
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z])


def scatter(env, seed: int) -> None:
    """Drop the items one at a time from above random spots in the clutter zone (random heading, some tilted)
    and let each settle before the next: they land on the table or on each other, touching, leaning, stacked."""
    rs, sim = env._rs, env.sim
    rng = np.random.default_rng(10_007 + int(seed))
    T = np.array(rs.workspace_offset if hasattr(rs, "workspace_offset") else (0.0, 0.0, 0.41), dtype=float)
    (x0, x1), (y0, y1) = CLUTTER_ZONE
    order = list(ITEMS)
    rng.shuffle(order)
    robot = rs.robots[0]
    qi = list(robot._ref_joint_pos_indexes) + list(getattr(robot, "_ref_gripper_joint_pos_indexes", []))
    vi = list(robot._ref_joint_vel_indexes) + list(getattr(robot, "_ref_gripper_joint_vel_indexes", []))
    q_hold = sim.data.qpos[qi].copy()

    def settle(n_steps: int) -> None:          # raw physics with the arm held at its start pose
        for _ in range(n_steps):
            sim.step()
            sim.data.qpos[qi] = q_hold
            sim.data.qvel[vi] = 0.0

    for i, n in enumerate(order):
        p = np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)])
        tilt = rng.uniform(0.2, 1.0) if i >= len(order) - N_STACKED else rng.uniform(0.0, 0.25)
        _set_free(sim, rs, n, [T[0] + p[0], T[1] + p[1], T[2] + 0.30],
                  _quat_wxyz_yaw(rng.uniform(-np.pi, np.pi), tilt, rng.uniform(-np.pi, np.pi)))
        sim.forward()
        settle(120)
    settle(300)
    # anything that bounced off the desk goes back onto the middle of the pile
    for n in order:
        jn = rs.objects_dict[f"{n}_1"].joints[0]
        q = sim.data.get_joint_qpos(jn)
        if q[2] < T[2] - 0.05 or not (x0 - 0.12 < q[0] - T[0] < x1 + 0.12 and y0 - 0.15 < q[1] - T[1] < y1 + 0.15):
            _set_free(sim, rs, n, [T[0] + (x0 + x1) / 2, T[1] + (y0 + y1) / 2, T[2] + 0.35],
                      _quat_wxyz_yaw(rng.uniform(-np.pi, np.pi)))
            sim.forward()
            settle(150)
    sim.data.qvel[vi] = 0.0
    sim.forward()


def _set_free(sim, rs, name: str, pos, quat_wxyz) -> None:
    obj = rs.objects_dict[f"{name}_1"]
    jn = obj.joints[0]
    sim.data.set_joint_qpos(jn, np.r_[pos, quat_wxyz])
    sim.data.set_joint_qvel(jn, np.zeros(6))


def widen_camera(env) -> None:
    """Move agentview up/back with a wider FOV so the whole desk is in view."""
    m = env.sim.model
    cid = m.camera_name2id("agentview")
    m.cam_fovy[cid] = 62.0
    p = np.array(m.cam_pos[cid], dtype=float)
    m.cam_pos[cid] = p + np.array([0.10, 0.0, 0.18])


def object_status(env) -> dict[str, str]:
    """Runner/diagnostics only: where each item ended up."""
    from libero.libero.envs.predicates import eval_predicate_fn

    st = env._rs.object_states_dict
    regions = {"caddy": [r for r in st if r.startswith("desk_caddy_1_") and r.endswith("contain_region")],
               "tray": ["wooden_tray_1_contain_region"], "basket": ["basket_1_contain_region"],
               "drawer": [f"wooden_cabinet_1_{d}_region" for d in ("top", "middle", "bottom")]}
    out = {}
    for n in ITEMS:
        where = "table"
        for k, rs_ in regions.items():
            if any(r in st and eval_predicate_fn("in", st[f"{n}_1"], st[r]) for r in rs_):
                where = k
                break
        out[n] = where
    return out


def make_clutter_desk_env(init_index: int = 0, arm: str = Arm.PANDA.value, seed: int = 0, **kwargs):
    from motormem.env.libero import LiberoEnv

    path = generate_bddl()
    env = LiberoEnv(path, init_states=None, init_index=0, arm=arm, suite="clutter_desk", seed=seed,
                    task_name="clutter_desk", **{"settle_steps": 60, "image_size": 512, **kwargs})
    env._runner_success = lambda: False
    orig_reset_sim = env._reset_sim

    def _reset_sim():
        raw = orig_reset_sim()
        widen_camera(env)
        scatter(env, seed + 1000 * int(init_index))
        return env._refresh_raw()

    env._reset_sim = _reset_sim
    return env
