"""Perception: text-prompted masks -> depth-filtered world points -> numbered ObjectEstimates."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from motormem import geometry as G
from motormem.types import CameraFrame, ObjectEstimate, Observation

from .segmenters import MaskCandidate, Segmenter

log = logging.getLogger(__name__)

# Used when scan(find=None): a generic vocabulary for tabletop scenes (no task-specific words).
DEFAULT_PROMPTS = [
    "bowl", "plate", "mug", "cup", "bottle", "can", "box", "basket", "tray", "book", "caddy",
    "pan", "pot", "jar", "fruit", "vegetable", "cabinet", "drawer", "stove", "rack",
    "microwave", "knob", "handle",
]

CAMERA_ALIASES = {"wrist": ("wrist", "robot0_eye_in_hand", "eye_in_hand"),
                  "agentview": ("agentview", "frontview")}


def get_camera(obs: Observation, camera: str) -> CameraFrame:
    for name in (camera,) + CAMERA_ALIASES.get(camera, ()):
        if name in obs.cameras:
            return obs.cameras[name]
    raise KeyError(f"camera {camera!r} not in observation (have {sorted(obs.cameras)})")


@dataclass
class Dropped:
    label: str
    reason: str
    score: float


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def dedupe(cands: list[MaskCandidate], iou: float = 0.7) -> list[MaskCandidate]:
    """Greedy: keep the higher-score mask of any pair with IoU > iou."""
    kept: list[MaskCandidate] = []
    for c in sorted(cands, key=lambda c: -c.score):
        if all(mask_iou(c.mask, k.mask) <= iou for k in kept):
            kept.append(c)
    return kept


def valid_depth(depth: np.ndarray, max_depth: float = 5.0) -> np.ndarray:
    return np.isfinite(depth) & (depth > 1e-6) & (depth < max_depth)


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    return ndimage.binary_erosion(mask, iterations=px, border_value=0)


def remove_outliers(points: np.ndarray, cam_z: np.ndarray | None = None, band_k: float = 3.0,
                    band_min: float = 0.03, sor_k: int = 8, sor_std: float = 2.0) -> np.ndarray:
    """Boolean keep-mask: depth-median band (kills background bleed) then statistical outlier removal."""
    keep = np.ones(len(points), dtype=bool)
    if len(points) < 5:
        return keep
    if cam_z is not None:
        med = np.median(cam_z)
        mad = 1.4826 * np.median(np.abs(cam_z - med))
        keep &= np.abs(cam_z - med) <= max(band_k * mad, band_min)
    P = points[keep]
    if len(P) > sor_k + 1:
        d, _ = cKDTree(P).query(P, k=sor_k + 1)
        md = d[:, 1:].mean(1)
        ok = md <= md.mean() + sor_std * md.std()
        idx = np.nonzero(keep)[0]
        keep[idx[~ok]] = False
    return keep


def label_anchor(mask: np.ndarray) -> tuple[float, float]:
    """(u, v) pixel deepest inside the mask (good place for a label, always inside the mask)."""
    if not mask.any():
        return (0.0, 0.0)
    dt = ndimage.distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
    v, u = np.unravel_index(np.argmax(dt), dt.shape)
    return float(u), float(v)


class Perception:
    """`scan` returns one ObjectEstimate per detected instance, marks numbered left to right from 1."""

    def __init__(self, segmenter: Segmenter, object_points_by_name: dict[str, np.ndarray] | None = None,
                 min_points: int = 20, dedupe_iou: float = 0.7, erode_px: int = 2,
                 max_area_frac: float = 0.6, refine_gate: float = 0.15):
        self.segmenter = segmenter
        self.object_points_by_name = dict(object_points_by_name or {})
        self.min_points = min_points
        self.dedupe_iou = dedupe_iou
        self.erode_px = erode_px
        self.max_area_frac = max_area_frac
        self.refine_gate = refine_gate
        self.dropped: list[Dropped] = []      # what the last scan/refine dropped and why (Req 2.4)

    # ------------------------------------------------------------ building blocks
    def estimate(self, cam: CameraFrame, cand: MaskCandidate) -> ObjectEstimate | str:
        """One candidate -> ObjectEstimate (mark 0), or a reason string if it must be dropped."""
        H, W = cam.depth.shape[:2]
        mask = cand.mask.astype(bool)
        if mask.sum() > self.max_area_frac * H * W:
            return f"mask covers {mask.mean():.0%} of the image"
        ok = valid_depth(cam.depth)
        # Erode to avoid depth bleeding at edges; back off for thin masks (handles, bottle necks).
        core = mask
        for px in range(self.erode_px, 0, -1):
            m = erode(mask, px)
            if (m & ok).sum() >= self.min_points:
                core = m
                break
        n_valid = int((core & ok).sum())
        if n_valid < self.min_points:
            return f"only {n_valid} valid depth points (< {self.min_points})"
        pts = G.deproject(np.where(ok, cam.depth, 0.0), cam.K, cam.T_world_cam, mask=core)
        cam_z = G.transform_points(G.inv(cam.T_world_cam), pts)[:, 2]
        keep = remove_outliers(pts, cam_z)
        pts = pts[keep]
        if len(pts) < self.min_points:
            return f"only {len(pts)} points after outlier removal (< {self.min_points})"
        T, ext = G.pca_frame(pts)
        return ObjectEstimate(mark=0, label=cand.label, camera=cam.name, mask=mask,
                              points_world=pts, T_world_obj=T, extent=ext, score=float(cand.score))

    def _estimates(self, cam: CameraFrame, prompts: list[str]) -> list[ObjectEstimate]:
        cands = self.segmenter.segment(cam.rgb, prompts)
        good: list[tuple[MaskCandidate, ObjectEstimate]] = []
        for c in cands:
            e = self.estimate(cam, c)
            if isinstance(e, str):
                self.dropped.append(Dropped(c.label, e, c.score))
                log.info("dropped %s (%.2f): %s", c.label, c.score, e)
            else:
                good.append((c, e))
        kept = {id(c) for c in dedupe([c for c, _ in good], self.dedupe_iou)}
        for c, _ in good:
            if id(c) not in kept:
                self.dropped.append(Dropped(c.label, "duplicate of a higher-score mask", c.score))
        return [e for c, e in good if id(c) in kept]

    # ------------------------------------------------------------ public API
    def scan(self, obs: Observation, find: list[str] | None = None, camera: str = "agentview") -> list[ObjectEstimate]:
        cam = get_camera(obs, camera)
        prompts = [p for p in (find or DEFAULT_PROMPTS) if p and p.strip()]
        self.dropped = []
        ests = self._estimates(cam, prompts)
        ests.sort(key=lambda e: label_anchor(e.mask))   # left to right (u), then top to bottom
        for i, e in enumerate(ests, start=1):
            e.mark = i
        return ests

    def refine(self, obs: Observation, est: ObjectEstimate, camera: str = "wrist") -> ObjectEstimate:
        """Re-segment est.label in `camera`; return the candidate nearest est's centroid (same mark).

        Returns `est` unchanged if nothing within `refine_gate` metres is found.
        """
        try:
            cam = get_camera(obs, camera)
        except KeyError:
            return est
        self.dropped = []
        c0 = est.T_world_obj[:3, 3]
        ests = self._estimates(cam, [est.label])
        if not ests:
            return est
        d = [np.linalg.norm(e.T_world_obj[:3, 3] - c0) for e in ests]
        j = int(np.argmin(d))
        if d[j] > self.refine_gate:
            log.info("refine %s: nearest candidate %.3f m away (> gate), keeping estimate", est.label, d[j])
            return est
        best = ests[j]
        best.mark = est.mark
        return best
