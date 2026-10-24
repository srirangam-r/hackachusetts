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

# Generic words only (no task-specific nouns). Used to build retry prompts when a noun finds nothing.
_COLOR_MATERIAL = {
    "black", "white", "red", "green", "blue", "yellow", "orange", "brown", "grey", "gray",
    "pink", "purple", "silver", "gold", "dark", "light", "wooden", "wood", "metal", "metallic",
    "plastic", "glass", "ceramic", "steel", "small", "large", "big", "little", "tall", "short",
}
# Generic container/category words to append as alternative prompts (one noun -> generic shape word).
_CATEGORY_WORDS = ["box", "bottle", "container", "object"]

# Generic knob prompts for localizing a knob inside a flat control surface (no task-specific words).
KNOB_WORDS = ["burner knob", "knob", "dial"]
KNOB_MIN_SIZE = 0.03   # m: a real knob is at least this wide in its smaller horizontal dimension


def prompt_variants(noun: str) -> list[str]:
    """Generic retry prompts for a noun that found nothing, most-specific first, no task knowledge.

    Order: last noun only, drop colour/material adjectives, then add generic category words.
    Duplicates and the original are removed; every variant is still resolved back to `noun`.
    """
    noun = noun.strip()
    words = noun.lower().split()
    out: list[str] = []

    def add(p: str) -> None:
        p = p.strip()
        if p and p != noun.lower() and p not in out:
            out.append(p)

    if len(words) >= 2:
        add(words[-1])                                   # last noun only ("cream cheese" -> "cheese")
    kept = [w for w in words if w not in _COLOR_MATERIAL]
    if kept and len(kept) < len(words):
        add(" ".join(kept))                              # drop colour/material ("black bowl" -> "bowl")
        if len(kept) >= 2:
            add(kept[-1])
    head = (kept or words)[-1] if (kept or words) else ""
    for cat in _CATEGORY_WORDS:                          # generic shape/category words
        if cat != head:
            add(f"{head} {cat}" if head else cat)
    return out


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
                 max_area_frac: float = 0.6, refine_gate: float = 0.15,
                 retry_variants: bool = True, retry_threshold: float = 0.12, localize_knobs: bool = False):
        self.segmenter = segmenter
        self.object_points_by_name = dict(object_points_by_name or {})
        self.min_points = min_points
        self.dedupe_iou = dedupe_iou
        self.erode_px = erode_px
        self.max_area_frac = max_area_frac
        self.refine_gate = refine_gate
        self.retry_variants = retry_variants   # retry a missing noun with generic prompt variants
        self.retry_threshold = retry_threshold  # lowered detection threshold for the retry pass
        # Off by default: at scan time we can't know whether a stove will be actuated (knob) or
        # used as a surface (place on it), and localizing the knob would break placing on the stove.
        # The capability + `parts` helper exist for an actuation-aware caller (needs a skills change).
        self.localize_knobs = localize_knobs
        self.rel_score = 0.6                     # keep instances >= 0.6 x the best score of their noun
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

    def _segment(self, cam: CameraFrame, prompts: list[str], threshold: float | None = None) -> list[MaskCandidate]:
        """Call the segmenter, optionally with a temporarily lowered detection threshold."""
        if threshold is None:
            return self.segmenter.segment(cam.rgb, prompts)
        attr = next((a for a in ("threshold", "box_threshold") if hasattr(self.segmenter, a)), None)
        if attr is None:
            return self.segmenter.segment(cam.rgb, prompts)
        old = getattr(self.segmenter, attr)
        try:
            setattr(self.segmenter, attr, min(old, threshold))
            return self.segmenter.segment(cam.rgb, prompts)
        finally:
            setattr(self.segmenter, attr, old)

    def _estimates(self, cam: CameraFrame, prompts: list[str],
                   threshold: float | None = None) -> list[ObjectEstimate]:
        cands = self._segment(cam, prompts, threshold)
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
        requested = [p.strip() for p in (find or DEFAULT_PROMPTS) if p and p.strip()]
        self.dropped = []
        ests = self._estimates(cam, requested)
        if find:                                         # only retry explicit nouns, not the generic scan
            # open-vocabulary false positives (a juice carton answering "butter box") score well below the
            # real instance: per noun, keep instances scoring at least REL_SCORE x that noun's best score
            best: dict[str, float] = {}
            for e in ests:
                best[e.label] = max(best.get(e.label, 0.0), e.score)
            for e in ests:
                if e.score < self.rel_score * best[e.label]:
                    self.dropped.append(Dropped(e.label, f"score {e.score:.2f} < {self.rel_score} x best", e.score))
            ests = [e for e in ests if e.score >= self.rel_score * best[e.label]]
            found = {e.label for e in ests}
            ests += self._retry_missing(cam, [n for n in requested if n not in found])
            if self.localize_knobs:                      # point a flat/wide object's frame at its knob
                ests = [self._localize_knob(obs, e, camera) for e in ests]
        ests.sort(key=lambda e: label_anchor(e.mask))   # left to right (u), then top to bottom
        for i, e in enumerate(ests, start=1):
            e.mark = i
        return ests

    def _retry_missing(self, cam: CameraFrame, missing: list[str]) -> list[ObjectEstimate]:
        """For each noun that found nothing, try generic prompt variants at a lowered threshold.

        Every surviving estimate is relabelled to the noun the caller asked for so the brain and the
        scripted resolver match it. Only the best (highest-score) estimate per noun is returned.
        """
        out: list[ObjectEstimate] = []
        if not (self.retry_variants and missing):
            return out
        for noun in missing:
            variants = prompt_variants(noun)
            if not variants:
                continue
            cands = self._estimates(cam, variants, threshold=self.retry_threshold)
            if not cands:
                log.info("no detection for %r even after variants %s", noun, variants)
                continue
            best = max(cands, key=lambda e: e.score)
            best.label = noun                            # keep the label the brain asked for
            log.info("recovered %r via variant (score %.2f)", noun, best.score)
            out.append(best)
        return out

    # ------------------------------------------------------------ parts
    def parts(self, obs: Observation, est: ObjectEstimate, words: list[str],
              camera: str | None = None, margin: float = 0.04) -> list[ObjectEstimate]:
        """Re-segment `words` and keep candidates spatially associated with `est`.

        Generic (no task knowledge): used to locate a small graspable part (a knob, a handle) that
        belongs to a larger object so skills contact the part, not the object centroid. A candidate
        is kept when its 3-D centroid lies within the object's horizontal footprint grown by
        `margin` (the knob can sit on a control panel just off the segmented surface) and it is
        smaller than the object. Returned estimates carry `est`'s label and mark.
        """
        if self.segmenter is None:
            return []
        try:
            cam = get_camera(obs, camera or est.camera)
        except KeyError:
            return []
        T_inv = G.inv(est.T_world_obj)
        half = est.extent[:2] / 2 + margin
        area = max(int(est.mask.astype(bool).sum()), 1)
        out: list[ObjectEstimate] = []
        for e in self._estimates(cam, [w for w in words if w and w.strip()],
                                 threshold=self.retry_threshold):
            if int(e.mask.astype(bool).sum()) > 0.6 * area:
                continue
            p = G.transform_points(T_inv, e.T_world_obj[:3, 3][None])[0]
            if np.all(np.abs(p[:2]) <= half):
                e.label, e.mark = est.label, est.mark
                out.append(e)
        return out

    def _localize_knob(self, obs: Observation, est: ObjectEstimate, camera: str) -> ObjectEstimate:
        """If a distinct knob sits inside this object's (flat, wide) mask, return it relabelled.

        Fires generically for any wide, low object (a stovetop, a control panel); the skill then
        contacts the knob via `part.T_world_obj` / `part.points_world`. No task-specific words.
        """
        flat = est.extent[2] < 0.06 and max(est.extent[0], est.extent[1]) > 0.12
        if not flat:
            return est
        # Knobs sit on the control panel just off the front edge: allow ~half an object-width.
        margin = max(0.05, 0.5 * float(est.extent[:2].max()))
        knobs = self.parts(obs, est, KNOB_WORDS, camera=camera, margin=margin)
        # A knob is a compact blob, clearly smaller than the surface; drop thin slivers (edge noise).
        knobs = [k for k in knobs if k.extent.max() <= 0.5 * est.extent.max()
                 and min(k.extent[0], k.extent[1]) > 0.5 * KNOB_MIN_SIZE]
        if not knobs:
            return est
        knob = max(knobs, key=lambda e: e.score)
        log.info("localized knob inside %r (score %.2f, extent %s)", est.label, knob.score,
                 np.round(knob.extent, 3))
        return knob

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
