"""Test-only LIBERO adapter for the skills stream (stands in for Stream A's RobotEnv).

Exposes `observe()`, `step(a7)`, `kinematics()` with the shapes from design.md, plus TEST-ONLY
ground-truth access (`gt_estimate`, `gt_body_pose`, `success`) used to build fake ObjectEstimates and
fake Segments for rung-0 diagnostics. Skills never receive this object's privileged methods: they
only call observe/step/kinematics.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

from motormem import geometry as g
from motormem.types import ObjectEstimate, Observation, Proprio


# ---------------------------------------------------------------- UR5e support (test-only copy)
def _register_mounted_ur5e() -> None:
    from robosuite.models.robots.manipulators.ur5e_robot import UR5e
    from robosuite.robots import ROBOT_CLASS_MAPPING
    from robosuite.robots.single_arm import SingleArm
    from libero.libero.envs.robots.mounted_panda import MountedPanda

    if "MountedUR5e" in ROBOT_CLASS_MAPPING:
        return

    class MountedUR5e(UR5e):  # noqa: D401 - registered by robosuite's model metaclass on definition
        @property
        def base_xpos_offset(self):
            return MountedPanda.base_xpos_offset.fget(self)

    ROBOT_CLASS_MAPPING["MountedUR5e"] = SingleArm
    globals()["MountedUR5e"] = MountedUR5e


@dataclass
class ArmKinematics:
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_addr: np.ndarray
    dof_addr: np.ndarray
    site_id: int
    q_lower: np.ndarray
    q_upper: np.ndarray
    q_current: np.ndarray


class LiberoSimAdapter:
    def __init__(self, suite: str, task: str, arm: str = "Panda", init_index: int = 0,
                 cameras: bool = False, size: int = 128):
        import libero.libero as L
        from libero.libero.envs import OffScreenRenderEnv

        if arm == "UR5e":
            _register_mounted_ur5e()
        base = os.path.dirname(L.__file__)
        bddl = os.path.join(base, "bddl_files", suite, f"{task}.bddl")
        self.arm = arm
        self.env = OffScreenRenderEnv(bddl_file_name=bddl, robots=[arm], use_camera_obs=cameras,
                                      camera_heights=size, camera_widths=size)
        self.env.seed(init_index)
        self._raw = self.env.reset()
        self.instruction = self.env.language_instruction
        init_file = os.path.join(base, "init_files", suite, f"{task}.pruned_init")
        if os.path.exists(init_file):
            import torch
            states = torch.load(init_file, weights_only=False)
            self._load_init(np.asarray(states[init_index % len(states)]))
        self.inner = self.env.env
        self.sim = self.inner.sim
        self.m = self.sim.model._model
        self.d = self.sim.data._data
        self.robot = self.inner.robots[0]
        self.site_id = int(self.robot.eef_site_id)
        self.arm_qpos = np.array(self.robot._ref_joint_pos_indexes, dtype=int)
        self.arm_dof = np.array(self.robot._ref_joint_vel_indexes, dtype=int)
        self.grip_qpos = np.array(self.robot._ref_gripper_joint_pos_indexes, dtype=int)
        self.step_count = 0
        self._calibrate_gripper()
        # settle
        for _ in range(5):
            self.step(np.r_[np.zeros(6), -1.0])
        self.step_count = 0

    # ---------------------------------------------------------- init states
    def _load_init(self, state: np.ndarray) -> None:
        if self.arm == "Panda":
            self._raw = self.env.set_init_state(state)
            return
        # Panda-sized state: [time, qpos(nq_p), qvel(nv_p)]; copy only object qpos (panda 9: -> ur5e 12:)
        sim = self.env.env.sim
        nq = sim.model.nq
        n_arm_p, n_arm_u = 9, 12
        n_obj = nq - n_arm_u
        panda_obj = state[1 + n_arm_p: 1 + n_arm_p + n_obj]
        sim.data.qpos[n_arm_u:] = panda_obj
        sim.data.qvel[:] = 0.0
        sim.forward()
        self.env.env._post_process()
        self.env.env._update_observables(force=True)
        self._raw = self.env.env._get_observations()

    def _calibrate_gripper(self) -> None:
        """Measure open/closed finger qpos by simulation, then restore the state (test helper only)."""
        state = self.sim.get_state().flatten().copy()
        hold = np.zeros(7)
        for cmd, attr in ((-1.0, "_q_open"), (1.0, "_q_closed")):
            hold[6] = cmd
            for _ in range(15):
                self.env.step(hold)
            setattr(self, attr, self.d.qpos[self.grip_qpos].copy())
        self.sim.set_state_from_flattened(state)
        self.sim.forward()

    # ---------------------------------------------------------- RobotEnv-like API
    def observe(self) -> Observation:
        R = self.d.site_xmat[self.site_id].reshape(3, 3).copy()
        T = g.make_pose(self.d.site_xpos[self.site_id].copy(), R)
        qg = self.d.qpos[self.grip_qpos].copy()
        span = self._q_open - self._q_closed
        opening = float(np.clip(np.mean((qg - self._q_closed) / np.where(np.abs(span) > 1e-6, span, 1.0)), 0, 1))
        base_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
        T_base = g.make_pose(self.d.xpos[base_id].copy(), self.d.xmat[base_id].reshape(3, 3).copy())
        prop = Proprio(T_world_eef=T, gripper_qpos=qg, gripper_opening=opening,
                       joint_pos=self.d.qpos[self.arm_qpos].copy(),
                       ee_force=np.asarray(self.robot.ee_force, dtype=float).copy(),
                       ee_torque=np.asarray(self.robot.ee_torque, dtype=float).copy())
        return Observation(cameras={}, proprio=prop, instruction=self.instruction, step=self.step_count,
                           T_world_base=T_base)

    def step(self, action: np.ndarray) -> Observation:
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self._raw, _, _, _ = self.env.step(a)
        self.step_count += 1
        return self.observe()

    def kinematics(self) -> ArmKinematics:
        data = mujoco.MjData(self.m)
        data.qpos[:] = self.d.qpos
        mujoco.mj_kinematics(self.m, data)
        jids = [self.m.dof_jntid[i] for i in self.arm_dof]
        lo = np.array([self.m.jnt_range[j, 0] if self.m.jnt_limited[j] else -2 * np.pi for j in jids])
        hi = np.array([self.m.jnt_range[j, 1] if self.m.jnt_limited[j] else 2 * np.pi for j in jids])
        return ArmKinematics(model=self.m, data=data, qpos_addr=self.arm_qpos.copy(), dof_addr=self.arm_dof.copy(),
                             site_id=self.site_id, q_lower=lo, q_upper=hi,
                             q_current=self.d.qpos[self.arm_qpos].copy())

    # ---------------------------------------------------------- TEST-ONLY privileged access
    def success(self) -> bool:
        return bool(self.inner._check_success())

    def body_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)

    def gt_body_pose(self, body: str) -> np.ndarray:
        b = self.body_id(body)
        return g.make_pose(self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy())

    def joint_qpos(self, joint: str) -> float:
        j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint)
        return float(self.d.qpos[self.m.jnt_qposadr[j]])

    def gt_points(self, body: str, geoms: list[int] | None = None, subtree: bool = True,
                  max_points: int = 1500, seed: int = 0) -> np.ndarray:
        """Surface points of a body's visual meshes (or collision boxes) in world coordinates."""
        root = self.body_id(body)
        bodies = {root}
        if subtree:
            for b in range(self.m.nbody):
                p = b
                while p > 0:
                    if p == root:
                        bodies.add(b)
                        break
                    p = self.m.body_parentid[p]
        ids = geoms if geoms is not None else [i for i in range(self.m.ngeom) if self.m.geom_bodyid[i] in bodies]
        pts = []
        rng = np.random.default_rng(seed)
        for i in ids:
            R = self.d.geom_xmat[i].reshape(3, 3)
            if self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = self.m.geom_dataid[i]
                a, n = self.m.mesh_vertadr[mid], self.m.mesh_vertnum[mid]
                pts.append(self.m.mesh_vert[a:a + n] @ R.T + self.d.geom_xpos[i])
        if not pts:  # fall back to sampling collision boxes
            for i in ids:
                if self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                    h = self.m.geom_size[i]
                    local = rng.uniform(-1, 1, size=(80, 3)) * h
                    pts.append(local @ self.d.geom_xmat[i].reshape(3, 3).T + self.d.geom_xpos[i])
        P = np.concatenate(pts, axis=0).astype(np.float64)
        if len(P) > max_points:
            P = P[rng.choice(len(P), max_points, replace=False)]
        return P

    def gt_estimate(self, body: str, label: str, mark: int = 0, geoms: list[int] | None = None,
                    keep_above: float | None = None) -> ObjectEstimate:
        P = self.gt_points(body, geoms=geoms)
        if keep_above is not None:
            P = P[P[:, 2] > P[:, 2].min() + keep_above]
        T, ext = g.pca_frame(P)
        return ObjectEstimate(mark=mark, label=label, camera="agentview", mask=np.zeros((1, 1), bool),
                              points_world=P, T_world_obj=T, extent=ext)

    def mesh_geoms(self, body: str) -> list[int]:
        b = self.body_id(body)
        return [i for i in range(self.m.ngeom)
                if self.m.geom_bodyid[i] == b and self.m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_adapter(task: str, arm: str = "Panda", init_index: int = 0, suite: str = "libero_goal") -> LiberoSimAdapter:
    return LiberoSimAdapter(suite, task, arm=arm, init_index=init_index)
