"""Rung-0 diagnostic perception: ObjectEstimates from privileged world poses.

NEVER use for reported results. The runner passes the poses in (a dict, or a callable such as
`env.privileged_object_poses`); this module imports nothing from the env and the brain never sees it.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
from scipy import ndimage

from motormem import geometry as G
from motormem.types import CameraFrame, ObjectEstimate, Observation

from .core import Perception, get_camera

PoseSource = dict[str, np.ndarray] | Callable[[], dict[str, np.ndarray]]


def _tokens(s: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and not t.isdigit()]


def match_name(label: str, names: list[str]) -> list[str]:
    """Sim names a text label refers to. "black bowl" -> akita_black_bowl_1, akita_black_bowl_2.

    A name matches if the label contains its head (last) word; the names sharing the most words with
    the label win. An exact name always wins.
    """
    if label in names:
        return [label]
    L = set(_tokens(label))
    best, out = 0, []
    for n in names:
        N = _tokens(n)
        if not N or N[-1] not in L:
            continue
        k = len(L & set(N))
        if k > best:
            best, out = k, [n]
        elif k == best:
            out.append(n)
    return sorted(out)


def _canonical(name: str) -> str:
    return re.sub(r"_\d+$", "", name)


class OraclePerception(Perception):
    def __init__(self, poses: PoseSource, object_points_by_name: dict[str, np.ndarray] | None = None,
                 dilate_px: int = 2):
        super().__init__(segmenter=None, object_points_by_name=object_points_by_name)
        self._poses = poses
        self.dilate_px = dilate_px

    def poses(self) -> dict[str, np.ndarray]:
        return dict(self._poses() if callable(self._poses) else self._poses)

    def set_poses(self, poses: PoseSource) -> None:
        self._poses = poses

    def _points_for(self, name: str) -> np.ndarray | None:
        for key in (name, _canonical(name)):
            if key in self.object_points_by_name:
                return np.asarray(self.object_points_by_name[key], dtype=np.float64).reshape(-1, 3)
        return None

    def _build(self, cam: CameraFrame, name: str, label: str, T_pose: np.ndarray) -> ObjectEstimate:
        pts_obj = self._points_for(name)
        if pts_obj is not None and len(pts_obj):
            pts = G.transform_points(T_pose, pts_obj)
            T, ext = G.pca_frame(pts)
        else:
            pts = T_pose[:3, 3][None].copy()
            T = G.make_pose(T_pose[:3, 3], G.rot_z(G.yaw_of(T_pose)))
            ext = np.zeros(3)
        H, W = cam.depth.shape[:2]
        mask = np.zeros((H, W), dtype=bool)
        uv, z = G.project(pts, cam.K, cam.T_world_cam)
        ok = (z > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        if ok.any():
            mask[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
            if self.dilate_px:
                mask = ndimage.binary_dilation(mask, iterations=self.dilate_px)
        return ObjectEstimate(mark=0, label=label, camera=cam.name, mask=mask, points_world=pts,
                              T_world_obj=T, extent=ext, score=1.0)

    def scan(self, obs: Observation, find: list[str] | None = None, camera: str = "agentview") -> list[ObjectEstimate]:
        cam = get_camera(obs, camera)
        poses = self.poses()
        self.dropped = []
        if find is None:
            pairs = [(n, _canonical(n).replace("_", " ")) for n in poses]
        else:
            seen, pairs = set(), []
            for label in find:
                for n in match_name(label, list(poses)):
                    if n not in seen:
                        seen.add(n)
                        pairs.append((n, label))
        ests = [self._build(cam, n, label, poses[n]) for n, label in pairs]

        def u_of(e):
            uv, z = G.project(e.T_world_obj[:3, 3][None], cam.K, cam.T_world_cam)
            return (uv[0, 0] if z[0] > 0 else np.inf, uv[0, 1])
        ests.sort(key=u_of)
        for i, e in enumerate(ests, start=1):
            e.mark = i
        return ests

    def refine(self, obs: Observation, est: ObjectEstimate, camera: str = "wrist") -> ObjectEstimate:
        try:
            cam = get_camera(obs, camera)
        except KeyError:
            return est
        poses = self.poses()
        names = match_name(est.label, list(poses)) or list(poses)
        if not names:
            return est
        c0 = est.T_world_obj[:3, 3]
        n = min(names, key=lambda n: np.linalg.norm(poses[n][:3, 3] - c0))
        out = self._build(cam, n, est.label, poses[n])
        out.mark = est.mark
        return out
