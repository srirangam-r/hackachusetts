"""Outcome checks for skills (Req 4, rule 3): measure the result with perception + proprioception
only, never with simulator state or the success signal.

Two helpers:

- `locate(skills, label, near_xyz, radius)`: re-scan one label with `skills.perception` on a fresh
  observation and return the `ObjectEstimate` whose centroid is nearest `near_xyz` (world frame,
  metres) within `radius`; `None` if nothing is found in range.
- `relation_holds(held_est, target_est, relation)`: whether the held object now satisfies the
  spatial relation to the target, from their estimated centroids / extents / surface points, plus a
  short human-readable reason. Used after a place to decide OK vs a retry/failure message.

Both work in the world frame. For the left/right/front/back relations the displacement sign is taken
in the robot base frame, so the caller passes `T_world_base` from the live observation.
"""
from __future__ import annotations

import numpy as np

from motormem.types import ObjectEstimate

# tolerances (metres)
XY_MARGIN = 0.01      # held centroid may sit this far outside the target footprint for on/top_of/in
ON_GAP = 0.04         # held bottom at most this far above the target top for on/top_of
IN_DROP = 0.02        # for "in", held bottom must be below the target top by at least -this


def _centroid(est: ObjectEstimate) -> np.ndarray:
    return np.asarray(est.T_world_obj, float)[:3, 3]


def _top_z(est: ObjectEstimate) -> float:
    P = getattr(est, "points_world", None)
    if P is not None and len(P):
        return float(np.asarray(P, float)[:, 2].max())
    c = _centroid(est)
    return float(c[2] + np.asarray(est.extent, float)[2] / 2)


def _bottom_z(est: ObjectEstimate) -> float:
    P = getattr(est, "points_world", None)
    if P is not None and len(P):
        return float(np.asarray(P, float)[:, 2].min())
    c = _centroid(est)
    return float(c[2] - np.asarray(est.extent, float)[2] / 2)


def _footprint_halfspan(est: ObjectEstimate, axis_world: np.ndarray) -> float:
    """Half the object's footprint projected onto a world xy direction, from its PCA frame extent."""
    R = np.asarray(est.T_world_obj, float)[:3, :3]
    ext = np.asarray(est.extent, float)
    u = np.asarray(axis_world, float)[:2]
    n = np.linalg.norm(u)
    if n < 1e-9:
        return 0.0
    u = u / n
    return float(np.abs(u @ R[:2, :2]) @ ext[:2]) / 2


def _xy_inside(held: ObjectEstimate, target: ObjectEstimate, margin: float) -> tuple[bool, float]:
    """True if the held centroid xy lies within the target footprint grown by `margin`.
    Returns (inside, overshoot_cm) where overshoot is how far outside along the worst axis."""
    ch, ct = _centroid(held)[:2], _centroid(target)[:2]
    R = np.asarray(target.T_world_obj, float)[:2, :2]
    ext = np.asarray(target.extent, float)[:2]
    local = R.T @ (ch - ct)                     # target-frame xy offset
    half = ext / 2 + margin
    over = np.maximum(np.abs(local) - half, 0.0)
    return bool(np.all(over <= 1e-9)), float(np.max(over) * 100)


def locate(skills, label: str, near_xyz, radius: float = 0.15) -> ObjectEstimate | None:
    """Re-scan `label` and return the estimate nearest `near_xyz` within `radius` (world, metres)."""
    perception = getattr(skills, "perception", None)
    if perception is None:
        return None
    near = np.asarray(near_xyz, float).reshape(3)
    obs = skills.ctrl.refresh()
    try:
        ests = perception.scan(obs, [label], camera="agentview")
    except Exception:
        return None
    if not ests:
        return None
    d = [float(np.linalg.norm(_centroid(e) - near)) for e in ests]
    j = int(np.argmin(d))
    if d[j] > radius:
        return None
    return ests[j]


def relation_holds(held_est: ObjectEstimate, target_est: ObjectEstimate, relation: str,
                   T_world_base: np.ndarray | None = None) -> tuple[bool, str]:
    """Does the held object satisfy `relation` to the target now? Returns (holds, reason).

    on/top_of: held centroid xy inside the target footprint (+1 cm) and the held bottom within 4 cm
               above the target top.
    in:        held centroid xy inside the target footprint (+1 cm) and the held bottom below the
               target top (+2 cm tolerance).
    left_of/right_of/front_of/back_of: sign of the base-frame displacement of the held centroid from
               the target centroid (y+ = left, x+ = front/toward the robot in base frame).
    """
    rel = relation
    ch, ct = _centroid(held_est), _centroid(target_est)
    if rel in ("on", "top_of", "on_top_of"):
        inside, over = _xy_inside(held_est, target_est, XY_MARGIN)
        gap = _bottom_z(held_est) - _top_z(target_est)
        if not inside:
            return False, f"off the target by {over:.1f} cm horizontally"
        if gap > ON_GAP:
            return False, f"resting {gap * 100:.1f} cm above the target top"
        if gap < -0.05:
            # Resting inside an open container (tray, basket) counts as "on" it: the object is within the
            # footprint and its bottom is at or above the container's floor rather than its rim.
            floor = float(np.asarray(target_est.points_world, float)[:, 2].min())
            if rel == "on" and _bottom_z(held_est) >= floor - 0.02:
                return True, "inside the target's footprint, resting on its floor"
            return False, f"below the target top by {-gap * 100:.1f} cm"
        return True, f"on target (xy inside, bottom {gap * 100:+.1f} cm vs top)"
    if rel in ("in", "inside"):
        inside, over = _xy_inside(held_est, target_est, XY_MARGIN)
        drop = _top_z(target_est) - _bottom_z(held_est)   # how far the held bottom is below the top
        if not inside:
            return False, f"off the target by {over:.1f} cm horizontally"
        if drop < -IN_DROP:
            return False, f"sitting {-drop * 100:.1f} cm above the opening"
        return True, f"inside target (bottom {drop * 100:+.1f} cm below top)"
    if rel in ("left_of", "right_of", "front_of", "back_of"):
        d = ch - ct
        if T_world_base is not None:
            d = np.asarray(T_world_base, float)[:3, :3].T @ d   # into base frame
        dx, dy = float(d[0]), float(d[1])                       # base: x forward, y left
        want = {"left_of": ("y", +1), "right_of": ("y", -1), "front_of": ("x", +1),
                "back_of": ("x", -1)}[rel]
        axis, sign = want
        val = dy if axis == "y" else dx
        ok = sign * val > 0.01
        word = {"left_of": "left", "right_of": "right", "front_of": "front", "back_of": "back"}[rel]
        return ok, (f"{word} by {abs(val) * 100:.1f} cm" if ok
                    else f"not {word}: displacement {axis}={val * 100:+.1f} cm")
    return False, f"unknown relation {relation!r}"
