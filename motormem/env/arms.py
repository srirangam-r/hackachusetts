"""Arm models and the Panda -> UR5e init-state transfer.

LIBERO's problem classes prefix the robot name with "Mounted" (``robots=["UR5e"]`` -> "MountedUR5e"), so a
UR5e LIBERO scene needs a ``MountedUR5e`` model (robosuite registers ManipulatorModel subclasses by class
name) and a ``ROBOT_CLASS_MAPPING`` entry.

LIBERO init states are flattened Panda-sized MuJoCo states ``[time, qpos, qvel]``. They must never be loaded
into a UR5e sim (different nq/nv -> silently corrupted state). ``transfer_object_state`` copies only the
object/fixture joints, matched by joint name, and leaves the UR5e at its default joints with zero velocity.
"""
from __future__ import annotations

import numpy as np
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm

import libero.libero.envs.robots  # noqa: F401  (registers MountedPanda; keeps mapping order stable)

# Robot-owned joints in a Panda LIBERO scene: 7 arm joints + 2 finger joints (verified by joint names in
# tests/test_env_arms.py). Object/fixture joints follow in the same order as in any other arm's scene.
PANDA_ROBOT_JOINTS = (
    [f"robot0_joint{i}" for i in range(1, 8)] + ["gripper0_finger_joint1", "gripper0_finger_joint2"]
)
PANDA_ROBOT_NQ = 9
PANDA_ROBOT_NV = 9


class MountedUR5e(UR5e):
    """robosuite UR5e with LIBERO's mounted-arm base offsets (copied from ``MountedPanda``)."""

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }


ROBOT_CLASS_MAPPING.setdefault("MountedUR5e", SingleArm)


def is_robot_joint(name: str) -> bool:
    return name.startswith("robot0_") or name.startswith("gripper0_") or name.startswith("mount0_")


def object_joint_layout(model) -> list[tuple[str, int, int, int, int]]:
    """[(joint name, qpos addr, qpos size, dof addr, dof size)] for every non-robot joint, in model order.

    ``model`` is a raw ``mujoco.MjModel``.
    """
    import mujoco

    qsize = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
             mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
    vsize = {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3,
             mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
    out = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if is_robot_joint(name):
            continue
        t = mujoco.mjtJoint(model.jnt_type[j])
        out.append((name, int(model.jnt_qposadr[j]), qsize[t], int(model.jnt_dofadr[j]), vsize[t]))
    return out


def panda_object_slices(layout, panda_state_len: int) -> list[tuple[str, slice, slice]]:
    """Where each object joint lives in a flattened Panda LIBERO state, given the target scene's layout.

    The Panda scene has the same object joints in the same order, preceded by ``PANDA_ROBOT_NQ`` qpos and
    ``PANDA_ROBOT_NV`` dofs. Raises if the state length is inconsistent with that assumption.
    """
    obj_nq = sum(l[2] for l in layout)
    obj_nv = sum(l[4] for l in layout)
    nq, nv = PANDA_ROBOT_NQ + obj_nq, PANDA_ROBOT_NV + obj_nv
    if panda_state_len != 1 + nq + nv:
        raise ValueError(f"Panda state has {panda_state_len} entries, expected 1 + {nq} + {nv} for this scene")
    out, qa, va = [], 1 + PANDA_ROBOT_NQ, 1 + nq + PANDA_ROBOT_NV
    for name, _, qs, _, vs in layout:
        out.append((name, slice(qa, qa + qs), slice(va, va + vs)))
        qa += qs
        va += vs
    return out


def transfer_object_state(sim, panda_state: np.ndarray) -> None:
    """Copy object/fixture qpos from a Panda-sized LIBERO state into ``sim`` (any arm); robot joints untouched.

    All velocities are zeroed and ``sim.forward()`` is called. ``sim`` is a robosuite ``MjSim``.
    """
    model, data = sim.model._model, sim.data._data
    layout = object_joint_layout(model)
    panda_state = np.asarray(panda_state, dtype=np.float64).ravel()
    for (name, qadr, qs, _, _), (_, qsl, _) in zip(layout, panda_object_slices(layout, len(panda_state))):
        data.qpos[qadr:qadr + qs] = panda_state[qsl]
    data.qvel[:] = 0.0
    if data.act is not None and len(data.act):
        data.act[:] = 0.0
    sim.forward()
