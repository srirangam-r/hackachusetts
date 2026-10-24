"""SE(3) and point-cloud helpers. Coordinator-owned contract; all streams use these.

Conventions: T_a_b is a 4x4 float64 pose of frame b in frame a. Quaternions are xyzw (robosuite).
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------- poses
def make_pose(pos, rot=None) -> np.ndarray:
    """4x4 pose from position and a 3x3 rotation matrix (identity if None)."""
    T = np.eye(4)
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    if rot is not None:
        T[:3, :3] = np.asarray(rot, dtype=np.float64)
    return T


def quat_xyzw_to_mat(q) -> np.ndarray:
    return Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix()


def mat_to_quat_xyzw(R) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_quat()


def pose_from_pos_quat(pos, quat_xyzw) -> np.ndarray:
    return make_pose(pos, quat_xyzw_to_mat(quat_xyzw))


def inv(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform (or a stack (..., 4, 4))."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def compose(*Ts: np.ndarray) -> np.ndarray:
    """compose(A, B, C) = A @ B @ C (broadcasts over stacks)."""
    out = np.asarray(Ts[0], dtype=np.float64)
    for T in Ts[1:]:
        out = np.matmul(out, np.asarray(T, dtype=np.float64))
    return out


def transform_points(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
    return P @ T[:3, :3].T + T[:3, 3]


def rot_z(theta: float) -> np.ndarray:
    return Rotation.from_euler("z", theta).as_matrix()


def yaw_of(T: np.ndarray) -> float:
    """Heading of the frame's x-axis projected on the world xy-plane."""
    x = T[:3, 0]
    return float(np.arctan2(x[1], x[0]))


def pose_error(T_current: np.ndarray, T_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(dpos, drot) in WORLD frame that move T_current to T_target. drot is an axis-angle vector."""
    dpos = T_target[:3, 3] - T_current[:3, 3]
    dR = T_target[:3, :3] @ T_current[:3, :3].T
    drot = Rotation.from_matrix(dR).as_rotvec()
    return dpos, drot


def interpolate(T0: np.ndarray, T1: np.ndarray, n: int) -> np.ndarray:
    """n poses from T0 to T1 inclusive (linear position, slerp rotation)."""
    n = max(int(n), 2)
    s = np.linspace(0.0, 1.0, n)
    slerp = Slerp([0.0, 1.0], Rotation.from_matrix(np.stack([T0[:3, :3], T1[:3, :3]])))
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = slerp(s).as_matrix()
    out[:, :3, 3] = (1 - s)[:, None] * T0[:3, 3] + s[:, None] * T1[:3, 3]
    return out


# ---------------------------------------------------------------- cameras
def deproject(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray,
              mask: np.ndarray | None = None, stride: int = 1,
              max_depth: float = 5.0) -> np.ndarray:
    """Pixels (optionally masked) -> world points. Camera uses OpenCV axes."""
    H, W = depth.shape[:2]
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride].astype(np.float64)
    keep = (z > 1e-6) & (z < max_depth)
    if mask is not None:
        keep &= mask[::stride, ::stride].astype(bool)
    u, v, z = u[keep], v[keep], z[keep]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return transform_points(T_world_cam, np.stack([x, y, z], axis=-1))


def project(points_world: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World points -> (pixels (N,2) as (u, v), depth (N,))."""
    Pc = transform_points(inv(T_world_cam), points_world)
    z = Pc[:, 2]
    uv = np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], axis=-1)
    return uv, z


# ---------------------------------------------------------------- object frames and registration
def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Frame at the centroid, z = world up, x = principal horizontal axis. Returns (T_world_obj, extent)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    c = P.mean(axis=0)
    xy = P[:, :2] - c[:2]
    if len(P) >= 3 and np.linalg.norm(xy) > 1e-9:
        w, V = np.linalg.eigh(np.cov(xy.T))
        ax = V[:, np.argmax(w)]
        theta = float(np.arctan2(ax[1], ax[0]))
    else:
        theta = 0.0
    T = make_pose(c, rot_z(theta))
    local = transform_points(inv(T), P)
    extent = local.max(axis=0) - local.min(axis=0) if len(P) else np.zeros(3)
    return T, extent


def icp(src: np.ndarray, dst: np.ndarray, T_init: np.ndarray, iters: int = 30,
        max_corr: float = 0.03, planar: bool = True) -> tuple[np.ndarray, float, float]:
    """Point-to-point ICP aligning src (object frame) into dst (world). Returns (T, rmse, fitness).

    planar=True restricts the update to translation + yaw (objects resting on a table).
    fitness = fraction of transformed src points with a dst neighbour within max_corr.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if len(src) < 3 or len(dst) < 3:
        return T_init.copy(), float("inf"), 0.0
    tree = cKDTree(dst)
    T = T_init.copy()
    for _ in range(iters):
        S = transform_points(T, src)
        d, idx = tree.query(S, distance_upper_bound=max_corr)
        ok = np.isfinite(d)
        if ok.sum() < 3:
            break
        A, B = S[ok], dst[idx[ok]]
        ca, cb = A.mean(0), B.mean(0)
        if planar:
            a, b = A[:, :2] - ca[:2], B[:, :2] - cb[:2]
            ang = np.arctan2((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum(), (a * b).sum())
            R = rot_z(ang)
        else:
            U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
            D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
            R = Vt.T @ D @ U.T
        step = make_pose(cb - R @ ca, R)
        T = step @ T
        if np.linalg.norm(step[:3, 3]) < 1e-5 and abs(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))) < 1e-5:
            break
    S = transform_points(T, src)
    d, _ = tree.query(S)
    fitness = float((d < max_corr).mean())
    rmse = float(np.sqrt(np.mean(np.minimum(d, max_corr) ** 2)))
    return T, rmse, fitness


def register_multi_yaw(src_obj: np.ndarray, dst_world: np.ndarray, n_yaw: int = 8,
                       max_corr: float = 0.03) -> list[tuple[np.ndarray, float, float]]:
    """Register object-frame points to observed world points from n_yaw initial headings.

    Returns all hypotheses sorted best-first as (T_world_obj, rmse, fitness). Callers can pick the
    best, or the most reachable among near-ties (symmetric objects).
    """
    src_obj = np.asarray(src_obj, dtype=np.float64).reshape(-1, 3)
    dst_world = np.asarray(dst_world, dtype=np.float64).reshape(-1, 3)
    c_src = src_obj.mean(0)
    c_dst = dst_world.mean(0)
    out = []
    for k in range(n_yaw):
        R = rot_z(2 * np.pi * k / n_yaw)
        T0 = make_pose(c_dst - R @ c_src, R)
        out.append(icp(src_obj, dst_world, T0, max_corr=max_corr))
    out.sort(key=lambda h: (-h[2], h[1]))
    return out
