"""Offline replay of recorded demos (LIBERO HDF5) into per-step traces.

Offline index building only: this module reads recorded simulator states and privileged
segmentation, which the fairness rules allow for training demos (never on test episodes).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from motormem import geometry as g

WRIST_CAMERA = "robot0_eye_in_hand"
AGENT_CAMERA = "agentview"
RENDER_SIZE = 256
MAX_POINTS = 800


# ---------------------------------------------------------------- data
@dataclass
class JointInfo:
    name: str                 # full mujoco joint name, e.g. "wooden_cabinet_1_top_level"
    short: str                # name without the owner prefix, e.g. "top_level"
    owner: str                # instance owning the joint, e.g. "wooden_cabinet_1"
    jtype: str                # "slide" | "hinge"
    body: str                 # body moved by the joint
    axis_obj: np.ndarray      # (3,) joint axis in the owner's root-body frame
    anchor_obj: np.ndarray    # (3,) joint anchor in the owner's root-body frame
    articulation: dict[str, Any] = field(default_factory=dict)   # open/close/turnon ranges from the model


@dataclass
class DemoTrace:
    demo_id: str
    source: str
    task_language: str
    T_world_eef: np.ndarray            # (T, 4, 4) grip-site pose
    gripper: np.ndarray                # (T,) gripper command actions[:, -1]
    gripper_qpos: np.ndarray           # (T, k)
    bodies: dict[str, np.ndarray]      # instance -> (T, 4, 4) root-body pose
    movable: set[str]                  # instances that are free objects (vs fixtures)
    categories: dict[str, str]         # instance -> canonical category name
    joints: dict[str, np.ndarray]      # joint name -> (T,) qpos
    joint_info: dict[str, JointInfo]
    part_bodies: dict[str, np.ndarray] # joint name -> (T, 3) world position of the moved body
    points: dict[str, np.ndarray]      # instance -> (M, 3) surface points in the root-body frame
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.gripper)


# ---------------------------------------------------------------- paths
def libero_root() -> Path:
    import libero.libero as L
    return Path(os.path.dirname(L.__file__))


def bddl_for(suite: str, hdf5_path: str | Path) -> Path:
    """Installed-package bddl file matching an HDF5 demo file (by task name)."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        rel = f["data"].attrs.get("bddl_file_name", "")
    name = Path(str(rel)).name if rel else re.sub(r"_demo$", "", Path(hdf5_path).stem) + ".bddl"
    p = libero_root() / "bddl_files" / suite / name
    if not p.exists():
        raise FileNotFoundError(f"no bddl for {hdf5_path} at {p}")
    return p


def canonical_instance(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


# ---------------------------------------------------------------- env
def make_libero_env(bddl: str | Path, size: int = RENDER_SIZE):
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=size, camera_widths=size,
                             camera_depths=False, use_camera_obs=False,
                             camera_names=[AGENT_CAMERA, WRIST_CAMERA])
    env.reset()
    return env


def _pose_of_body(sim, bid: int) -> np.ndarray:
    return g.make_pose(sim.data.body_xpos[bid].copy(), sim.data.body_xmat[bid].reshape(3, 3).copy())


def grip_site_pose(sim, site_id: int) -> np.ndarray:
    return g.make_pose(sim.data.site_xpos[site_id].copy(), sim.data.site_xmat[site_id].reshape(3, 3).copy())


# ---------------------------------------------------------------- rendering (offline, privileged)
def render_depth_seg(sim, camera: str, size: int = RENDER_SIZE):
    """Return (depth metres (H,W), geom-id image (H,W) int, K, T_world_cam OpenCV) in image convention.

    robosuite 1.4's segmentation path overflows under numpy 2, so the id image is decoded here.
    """
    import mujoco
    from robosuite.utils.camera_utils import (get_camera_extrinsic_matrix, get_camera_intrinsic_matrix,
                                              get_real_depth_map)
    ctx = sim._render_context_offscreen
    cam_id = sim.model.camera_name2id(camera)
    vp = mujoco.MjrRect(0, 0, size, size)
    # depth
    ctx.render(width=size, height=size, camera_id=cam_id, segmentation=False)
    rgb = np.empty((size, size, 3), dtype=np.uint8)
    depth = np.empty((size, size), dtype=np.float32)
    mujoco.mjr_readPixels(rgb=rgb, depth=depth, viewport=vp, con=ctx.con)
    # segmentation
    ctx.render(width=size, height=size, camera_id=cam_id, segmentation=True)
    seg_rgb = np.empty((size, size, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb=seg_rgb, depth=None, viewport=vp, con=ctx.con)
    seg = seg_rgb[..., 0].astype(np.int64) + seg_rgb[..., 1].astype(np.int64) * 256 \
        + seg_rgb[..., 2].astype(np.int64) * 65536
    ngeom = ctx.scn.ngeom
    seg[seg >= ngeom + 1] = 0
    lut = np.full(ngeom + 1, -1, dtype=np.int64)
    for i in range(ngeom):
        gm = ctx.scn.geoms[i]
        if gm.segid != -1 and gm.objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
            lut[gm.segid + 1] = gm.objid
    geom_ids = lut[seg]
    # multisampling blends id colours at silhouettes: keep only pixels whose 4-neighbours agree
    core = np.ones_like(geom_ids, dtype=bool)
    core[1:, :] &= geom_ids[1:, :] == geom_ids[:-1, :]
    core[:-1, :] &= geom_ids[:-1, :] == geom_ids[1:, :]
    core[:, 1:] &= geom_ids[:, 1:] == geom_ids[:, :-1]
    core[:, :-1] &= geom_ids[:, :-1] == geom_ids[:, 1:]
    geom_ids = np.where(core, geom_ids, -1)
    depth = get_real_depth_map(sim, depth)
    depth = np.flipud(np.asarray(depth).reshape(size, size)).astype(np.float32)
    geom_ids = np.flipud(geom_ids)
    K = get_camera_intrinsic_matrix(sim, camera, size, size)
    T_world_cam = get_camera_extrinsic_matrix(sim, camera)
    return depth, geom_ids, K, T_world_cam


def voxel_downsample(P: np.ndarray, max_points: int = MAX_POINTS, voxel: float = 0.004) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    if len(P) <= max_points:
        return P
    while True:
        keys = np.floor(P / voxel).astype(np.int64)
        _, inv_idx = np.unique(keys, axis=0, return_inverse=True)
        n = inv_idx.max() + 1
        if n <= max_points:
            sums = np.zeros((n, 3))
            np.add.at(sums, inv_idx, P)
            cnt = np.bincount(inv_idx, minlength=n)[:, None]
            return sums / cnt
        voxel *= 1.3


# ---------------------------------------------------------------- replay
class LiberoReplayer:
    """Holds one LIBERO env per bddl file and replays demos of that task."""

    def __init__(self, suite: str, hdf5_path: str | Path):
        self.suite = suite
        self.hdf5_path = Path(hdf5_path)
        self.task_name = re.sub(r"_demo$", "", self.hdf5_path.stem)
        self.env = make_libero_env(bddl_for(suite, hdf5_path))
        e = self.env.env
        self.sim = e.sim
        m = self.sim.model
        self.site_id = e.robots[0].eef_site_id
        self.instances: dict[str, Any] = {}
        self.movable: set[str] = set()
        for k, v in e.objects_dict.items():
            self.instances[k] = v
            self.movable.add(k)
        for k, v in e.fixtures_dict.items():
            self.instances[k] = v
        self.root_bid = {k: m.body_name2id(v.root_body) for k, v in self.instances.items()}
        bid_to_inst = {b: k for k, b in self.root_bid.items()}
        # map every geom to the instance whose root body is its top-level ancestor
        self.geom_inst = np.full(m.ngeom, -1, dtype=np.int64)
        self.inst_index = {k: i for i, k in enumerate(self.instances)}
        for gi in range(m.ngeom):
            b = int(m.geom_bodyid[gi])
            while b != 0 and b not in bid_to_inst:
                b = int(m.body_parentid[b])
            if b in bid_to_inst:
                self.geom_inst[gi] = self.inst_index[bid_to_inst[b]]
        self.categories = {k: (getattr(v, "category_name", None) or canonical_instance(k)) for k, v in
                           self.instances.items()}
        # articulated joints
        self.joint_info: dict[str, JointInfo] = {}
        for k, v in self.instances.items():
            for jn in getattr(v, "joints", []) or []:
                try:
                    jid = m.joint_name2id(jn)
                except Exception:
                    continue
                jt = int(m.jnt_type[jid])
                if jt not in (2, 3):  # slide, hinge
                    continue
                art = dict(getattr(v, "object_properties", {}).get("articulation", {}) or {})
                self.joint_info[jn] = JointInfo(
                    name=jn, short=jn[len(k) + 1:] if jn.startswith(k + "_") else jn, owner=k,
                    jtype="slide" if jt == 2 else "hinge", body=m.body_id2name(int(m.jnt_bodyid[jid])),
                    axis_obj=np.zeros(3), anchor_obj=np.zeros(3),
                    articulation={a: list(b) for a, b in art.items() if isinstance(b, (list, tuple))})
        self.jnt_ids = {jn: m.joint_name2id(jn) for jn in self.joint_info}
        self.jnt_qadr = {jn: int(m.jnt_qposadr[j]) for jn, j in self.jnt_ids.items()}
        self.jnt_bid = {jn: int(m.jnt_bodyid[j]) for jn, j in self.jnt_ids.items()}

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass

    def _set(self, state):
        self.sim.set_state_from_flattened(state)
        self.sim.forward()

    def _points_at(self, cams=(AGENT_CAMERA, WRIST_CAMERA)) -> dict[str, list[np.ndarray]]:
        """Surface points per instance (in its root-body frame) visible at the current state."""
        out: dict[str, list[np.ndarray]] = {}
        names = list(self.instances)
        for cam in cams:
            depth, gids, K, Twc = render_depth_seg(self.sim, cam)
            inst = np.where(gids >= 0, self.geom_inst[np.clip(gids, 0, None)], -1)
            for i, k in enumerate(names):
                mask = inst == i
                if mask.sum() < 5:
                    continue
                Pw = g.deproject(depth, K, Twc, mask=mask)
                T = _pose_of_body(self.sim, self.root_bid[k])
                out.setdefault(k, []).append(g.transform_points(g.inv(T), Pw))
        return out

    def _contacts(self) -> dict[str, set[str]]:
        """Instance -> set of instances it touches (offline only)."""
        names = list(self.instances)
        d, m = self.sim.data, self.sim.model
        out: dict[str, set[str]] = {k: set() for k in names}
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = self.geom_inst[c.geom1], self.geom_inst[c.geom2]
            if a >= 0 and b >= 0 and a != b:
                out[names[a]].add(names[b])
                out[names[b]].add(names[a])
        return out

    def _landing(self, state, steps: int = 15) -> dict[str, Any]:
        """Let released objects settle (arm holds still, gripper open); return poses and contacts."""
        self._set(state)
        e = self.env.env
        try:
            e.robots[0].controller.reset_goal()
        except Exception:
            pass
        act = np.zeros(7)
        act[-1] = -1.0
        for _ in range(steps):
            e.robots[0].control(act, policy_step=True)
            for _ in range(int(e.control_timestep / e.model_timestep)):
                e.sim.step()
        poses = {k: _pose_of_body(self.sim, b) for k, b in self.root_bid.items()}
        return {"poses": poses, "contacts": self._contacts()}

    def replay(self, demo_key: str, max_points: int = MAX_POINTS, render: bool = True) -> DemoTrace:
        import h5py
        with h5py.File(self.hdf5_path, "r") as f:
            d = f["data"]
            lang = json.loads(d.attrs["problem_info"])["language_instruction"]
            states = d[demo_key]["states"][()]
            actions = d[demo_key]["actions"][()]
        T = len(states)
        Teef = np.zeros((T, 4, 4))
        gq = []
        bodies = {k: np.zeros((T, 4, 4)) for k in self.instances}
        joints = {jn: np.zeros(T) for jn in self.joint_info}
        parts = {jn: np.zeros((T, 3)) for jn in self.joint_info}
        robot = self.env.env.robots[0]
        gidx = list(robot._ref_gripper_joint_pos_indexes)
        for t in range(T):
            self._set(states[t])
            d = self.sim.data
            Teef[t] = grip_site_pose(self.sim, self.site_id)
            gq.append(d.qpos[gidx].copy())
            for k, b in self.root_bid.items():
                bodies[k][t] = _pose_of_body(self.sim, b)
            for jn in self.joint_info:
                joints[jn][t] = d.qpos[self.jnt_qadr[jn]]
                parts[jn][t] = d.body_xpos[self.jnt_bid[jn]]
        # joint axes / anchors in owner frame (frame 0)
        self._set(states[0])
        info = {}
        for jn, ji in self.joint_info.items():
            j = self.jnt_ids[jn]
            Towner = _pose_of_body(self.sim, self.root_bid[ji.owner])
            Ti = g.inv(Towner)
            axis = Ti[:3, :3] @ self.sim.data.xaxis[j]
            anchor = g.transform_points(Ti, self.sim.data.xanchor[j][None])[0]
            info[jn] = JointInfo(ji.name, ji.short, ji.owner, ji.jtype, ji.body, axis, anchor, ji.articulation)
        # surface points: frame 0, gripper close events (+10), last frame
        cmd = actions[:, -1]
        closed = cmd > 0
        points: dict[str, np.ndarray] = {}
        if render:
            frames = {0, T - 1}
            for t in np.nonzero(closed[1:] & ~closed[:-1])[0] + 1:
                frames.add(int(min(t + 10, T - 1)))
            acc: dict[str, list[np.ndarray]] = {}
            owners = {ji.owner for ji in self.joint_info.values()}
            for t in sorted(frames):
                self._set(states[t])
                for k, lst in self._points_at().items():
                    # articulated fixtures: only frame 0 (parts in their initial configuration)
                    if t != 0 and k in owners:
                        continue
                    acc.setdefault(k, []).extend(lst)
            for k, lst in acc.items():
                P = np.concatenate(lst, axis=0)
                points[k] = voxel_downsample(P, max_points).astype(np.float32)
        # where released objects land (for PLACE targets): settle after each open event
        landing = {}
        for t in np.nonzero(~closed[1:] & closed[:-1])[0] + 1:
            landing[int(t)] = self._landing(states[int(min(t + 10, T - 1))])
        return DemoTrace(
            demo_id=f"{self.suite}/{self.task_name}/{demo_key}", source=self.suite, task_language=lang,
            T_world_eef=Teef, gripper=actions[:, -1].astype(np.float64), gripper_qpos=np.array(gq),
            bodies=bodies, movable=set(self.movable), categories=dict(self.categories), joints=joints,
            joint_info=info, part_bodies=parts, points=points,
            meta={"hdf5": str(self.hdf5_path), "demo_key": demo_key, "task": self.task_name, "landing": landing})
