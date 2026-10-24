"""Arm- and scene-agnostic wrapper around a robosuite environment.

``RobotEnv`` turns raw robosuite observations into the fair ``motormem.types.Observation`` (RGB, metric depth,
intrinsics, OpenCV extrinsics, proprioception, wrist F/T) after ``strip_privileged``. Methods prefixed with
``_runner`` / ``privileged_`` are for the episode runner and offline index building only.

Camera convention (verified in tests/test_env_sim.py): robosuite renders OpenGL images (row 0 = bottom). We
flip images and depth vertically; then ``K = get_camera_intrinsic_matrix`` and
``T_world_cam = get_camera_extrinsic_matrix`` (already OpenCV axes: x right, y down, z forward) satisfy
``geometry.project(p_world, K, T_world_cam) -> (u, v)`` with u = column, v = row of the flipped image.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np
from robosuite.utils.camera_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_real_depth_map,
)

from motormem import config, geometry
from motormem.env.privileged import strip_privileged
from motormem.types import CameraFrame, Observation, Proprio

# our camera name -> robosuite camera name
CAMERA_MAP = {"agentview": "agentview", "wrist": "robot0_eye_in_hand"}

# gripper action sign that CLOSES each gripper actuator (from robosuite's gripper.format_action(+1)).
_CLOSE_SIGNS = {"PandaGripper": (-1.0, 1.0), "Robotiq85Gripper": (1.0, 1.0), "Robotiq85LinkedGripper": (1.0, 1.0)}


@dataclass
class ArmKinematics:
    """Robot self-model for IK. ``data`` is a COPY of the live MjData; IK may mutate it freely."""
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_addr: np.ndarray        # (n,) qpos indices of the arm joints
    dof_addr: np.ndarray         # (n,) dof indices of the arm joints
    site_id: int                 # grip site (OSC control frame)
    q_lower: np.ndarray
    q_upper: np.ndarray
    q_current: np.ndarray


class RobotEnv:
    """One episode's simulator. Subclasses implement ``_reset_sim() -> raw obs dict``."""

    def __init__(self, rs_env, instruction: str, arm: str, suite: str, cameras=config.CAMERAS,
                 settle_steps: int = 10, image_size: int = config.IMAGE_SIZE):
        self._rs = rs_env                      # the robosuite environment (inner env for LIBERO)
        self.instruction = instruction
        self.arm = arm
        self.suite = suite
        self.cameras = tuple(cameras)
        self.settle_steps = int(settle_steps)
        self.image_size = int(image_size)
        self._step = 0
        self._obs: Observation | None = None
        self._init_gripper_constants()

    # ------------------------------------------------------------------ public API
    def reset(self) -> Observation:
        self._make_current()
        raw = self._reset_sim()
        self._make_current()               # a hard reset may have created a new sim + GL context
        for _ in range(self.settle_steps):     # let objects settle (not counted as episode steps)
            raw, *_ = self._rs.step(self._hold_action())
        self._step = 0
        self._obs = self._build_obs(raw)
        return self._obs

    def observe(self) -> Observation:
        if self._obs is None:
            raise RuntimeError("call reset() first")
        return self._obs

    def step(self, action) -> Observation:
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        if a.shape != (7,):
            raise ValueError(f"expected a 7-D OSC_POSE delta action, got shape {a.shape}")
        self._make_current()
        raw, *_ = self._rs.step(a)
        self._step += 1
        self._obs = self._build_obs(raw)
        return self._obs

    def kinematics(self) -> ArmKinematics:
        robot = self._rs.robots[0]
        model = self.sim.model._model
        jids = [model.joint(n).id for n in robot.robot_joints]
        qadr = model.jnt_qposadr[jids].astype(int)
        return ArmKinematics(
            model=model,
            data=copy.copy(self.sim.data._data),
            qpos_addr=qadr,
            dof_addr=model.jnt_dofadr[jids].astype(int),
            site_id=int(robot.eef_site_id),
            q_lower=model.jnt_range[jids, 0].copy(),
            q_upper=model.jnt_range[jids, 1].copy(),
            q_current=self.sim.data._data.qpos[qadr].copy(),
        )

    def render(self, camera: str = "agentview", size: int = 512) -> np.ndarray:
        """RGB frame for videos only, image convention."""
        cam = CAMERA_MAP.get(camera, camera)
        self._make_current()
        return np.ascontiguousarray(self.sim.render(camera_name=cam, height=size, width=size)[::-1])

    def close(self) -> None:
        try:
            self._rs.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ runner-only
    def _runner_success(self) -> bool:
        return bool(self._rs._check_success())

    def privileged_object_poses(self) -> dict[str, np.ndarray]:
        """Ground-truth world poses of objects and fixtures (rung-0 diagnostics and offline only)."""
        data = self.sim.data._data
        out = {}
        for name, bid in self._object_body_ids().items():
            out[name] = geometry.make_pose(data.xpos[bid], data.xmat[bid].reshape(3, 3))
        return out

    def privileged_joint_positions(self) -> dict[str, float]:
        """Articulated (hinge/slide) object joint positions (offline only)."""
        from motormem.env.arms import object_joint_layout
        model, data = self.sim.model._model, self.sim.data._data
        out = {}
        for name, qadr, qs, _, _ in object_joint_layout(model):
            if qs == 1:
                out[name] = float(data.qpos[qadr])
        return out

    def privileged_segmentation(self, camera: str = "agentview", size: int | None = None) -> dict[str, np.ndarray]:
        """Simulator instance masks {object name: (H, W) bool}, image convention. Offline index build only.

        Also returns "robot" (arm + gripper + mount) when visible.
        """
        size = size or self.image_size
        cam = CAMERA_MAP.get(camera, camera)
        seg = self._render_geom_ids(cam, size)
        model = self.sim.model._model
        geom_root = model.body_rootid[model.geom_bodyid]
        root_ids = np.full(model.ngeom + 1, -1, dtype=np.int64)
        root_ids[1:] = geom_root
        root_img = root_ids[seg + 1]
        out = {}
        for name, bid in self._object_body_ids().items():
            m = root_img == model.body_rootid[bid]
            if m.any():
                out[name] = m
        robot_roots = {model.body_rootid[model.body(b).id] for b in [self._rs.robots[0].robot_model.root_body]}
        m = np.isin(root_img, list(robot_roots))
        if m.any():
            out["robot"] = m
        return out

    # ------------------------------------------------------------------ internals
    @property
    def sim(self):
        return self._rs.sim

    def _reset_sim(self) -> dict:
        raise NotImplementedError

    def _object_body_ids(self) -> dict[str, int]:
        return dict(getattr(self._rs, "obj_body_id", {}))

    def _hold_action(self) -> np.ndarray:
        a = np.zeros(7)
        a[-1] = -1.0
        return a

    def _make_current(self) -> None:
        """robosuite never re-binds its EGL context before rendering; with several envs in one process the
        last-created context would be used. Bind ours before every render."""
        ctx = getattr(self.sim, "_render_context_offscreen", None)
        if ctx is not None:
            ctx.gl_ctx.make_current()

    def _refresh_raw(self) -> dict:
        self._make_current()
        self._rs._update_observables(force=True)
        return self._rs._get_observations(force_update=True)

    def _init_gripper_constants(self) -> None:
        robot = self._rs.robots[0]
        model = self.sim.model._model
        acts = np.asarray(robot._ref_joint_gripper_actuator_indexes, dtype=int)
        signs = _CLOSE_SIGNS.get(type(robot.gripper).__name__, (1.0,) * len(acts))
        signs = np.resize(np.asarray(signs, dtype=float), len(acts))
        lo, hi = model.actuator_ctrlrange[acts, 0], model.actuator_ctrlrange[acts, 1]
        jids = model.actuator_trnid[acts, 0]
        self._grip_qadr = model.jnt_qposadr[jids].astype(int)
        self._arm_qadr = np.asarray(robot._ref_joint_pos_indexes, dtype=int)
        self._gripper_qadr_all = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=int)
        self._grip_closed = np.where(signs > 0, hi, lo)
        self._grip_open = np.where(signs > 0, lo, hi)
        self._eef_site = int(robot.eef_site_id)
        self._base_body = model.body(robot.robot_model.root_body).id
        fs = robot.gripper.important_sensors["force_ee"]
        ts = robot.gripper.important_sensors["torque_ee"]
        self._force_site = int(model.sensor_objid[model.sensor(fs).id])
        self._torque_site = int(model.sensor_objid[model.sensor(ts).id])

    def gripper_opening(self) -> float:
        q = self.sim.data._data.qpos[self._grip_qadr]
        r = (q - self._grip_closed) / (self._grip_open - self._grip_closed)
        return float(np.clip(np.mean(r), 0.0, 1.0))

    def _render_geom_ids(self, cam: str, size: int) -> np.ndarray:
        """(H, W) geom id per pixel (-1 = none), image convention."""
        sim = self.sim
        self._make_current()
        ctx = sim._render_context_offscreen
        cam_id = sim.model.camera_name2id(cam)
        ctx.render(width=size, height=size, camera_id=cam_id, segmentation=True)
        rgb = np.empty((size, size, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, size, size), con=ctx.con)
        rgb = rgb.astype(np.int64)
        segid = rgb[..., 0] + rgb[..., 1] * 256 + rgb[..., 2] * 65536   # 0 = background, else segid + 1
        scn = ctx.scn
        lut = np.full(scn.ngeom + 2, -1, dtype=np.int64)
        for i in range(scn.ngeom):
            g = scn.geoms[i]
            if g.segid != -1 and g.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                lut[g.segid + 1] = g.objid
        segid[(segid < 0) | (segid > scn.ngeom)] = 0
        return lut[segid][::-1].copy()

    def _build_obs(self, raw: dict) -> Observation:
        obs = strip_privileged(raw)
        sim = self.sim
        data = sim.data._data
        frames = {}
        for name in self.cameras:
            cam = CAMERA_MAP.get(name, name)
            rgb = np.ascontiguousarray(obs[f"{cam}_image"][::-1])
            H, W = rgb.shape[:2]
            depth = get_real_depth_map(sim, obs[f"{cam}_depth"])
            depth = np.ascontiguousarray(np.asarray(depth, dtype=np.float32).reshape(H, W)[::-1])
            frames[name] = CameraFrame(
                name=name, rgb=rgb, depth=depth,
                K=np.asarray(get_camera_intrinsic_matrix(sim, cam, H, W), dtype=np.float64),
                T_world_cam=np.asarray(get_camera_extrinsic_matrix(sim, cam), dtype=np.float64),
            )
        # Proprioception is read from the robot's own joints/sites at the current physics step (robosuite's
        # observables may be sampled one substep earlier). Grip site = OSC control frame; its position equals
        # robot0_eef_pos, but robot0_eef_quat is the arm's last-link body, rotated 90 deg about z from the site.
        T_world_eef = geometry.make_pose(data.site_xpos[self._eef_site],
                                         data.site_xmat[self._eef_site].reshape(3, 3))
        R_f = data.site_xmat[self._force_site].reshape(3, 3)
        R_t = data.site_xmat[self._torque_site].reshape(3, 3)
        robot = self._rs.robots[0]
        proprio = Proprio(
            T_world_eef=T_world_eef,
            gripper_qpos=data.qpos[self._gripper_qadr_all].copy(),
            gripper_opening=self.gripper_opening(),
            joint_pos=data.qpos[self._arm_qadr].copy(),
            ee_force=R_f @ np.asarray(robot.ee_force, dtype=np.float64),
            ee_torque=R_t @ np.asarray(robot.ee_torque, dtype=np.float64),
        )
        T_world_base = geometry.make_pose(data.xpos[self._base_body], data.xmat[self._base_body].reshape(3, 3))
        return Observation(cameras=frames, proprio=proprio, instruction=self.instruction, step=self._step,
                           T_world_base=T_world_base)
