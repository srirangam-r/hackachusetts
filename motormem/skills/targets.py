"""Generic target resolution between parts and their parent objects (perception only, no task knowledge).

- `parent_of`: a placement aimed at a small part (a drawer handle, a knob) really means the object the part
  belongs to (the cabinet, the stove). Demo PLACE segments and registration are defined on the whole object.
- `knob_of`: an actuation aimed at a wide, flat object (a stovetop) for turn_on/turn_off means its knob.
"""
from __future__ import annotations

import numpy as np

from motormem import geometry as g
from motormem.types import ObjectEstimate

PART_WORDS = {"handle", "knob", "button", "lid", "door", "drawer", "switch"}
PART_MAX_WIDTH = 0.12      # parts are at most this wide (m); wider marks are whole objects or panels


def _is_part(est: ObjectEstimate) -> bool:
    return max(est.extent[0], est.extent[1]) <= PART_MAX_WIDTH and bool(set(est.label.split()) & PART_WORDS)


def parent_of(sk, part: ObjectEstimate) -> ObjectEstimate:
    """The largest perceived object, wider than a part, whose footprint contains `part`'s centroid.

    Largest, because demo PLACE segments and their registration are defined on whole objects (the cabinet,
    not the drawer front). Searched with the label minus part words ("drawer handle" -> "drawer"), plus
    generic furniture words. Returns `part` unchanged when it is not a part or no parent is found.
    """
    if sk.perception is None or getattr(sk.perception, "segmenter", None) is None or not _is_part(part):
        return part
    rest = [w for w in part.label.split() if w not in {"handle", "knob", "button"}]
    prompts = [p for p in dict.fromkeys([" ".join(rest), "cabinet"]) if p]
    try:
        cands = sk.perception.scan(sk.ctrl.refresh(), prompts, camera=part.camera)
    except Exception:
        return part
    c = part.T_world_obj[:3, 3]
    best, best_area = None, -1.0
    for e in cands:
        if max(e.extent[0], e.extent[1]) <= PART_MAX_WIDTH:
            continue
        p = g.transform_points(g.inv(e.T_world_obj), c[None])[0]
        if np.all(np.abs(p[:2]) <= e.extent[:2] / 2 + 0.05):
            area = float(e.extent[0] * e.extent[1])
            if area > best_area:
                best, best_area = e, area
    if best is None:
        return part
    best.mark = part.mark
    return best


def knob_of(sk, obj: ObjectEstimate) -> ObjectEstimate:
    """For a wide, flat object (stovetop, control panel), the knob perceived on or just off its footprint."""
    if sk.perception is None or getattr(sk.perception, "segmenter", None) is None:
        return obj
    if max(obj.extent[0], obj.extent[1]) <= PART_MAX_WIDTH:
        return obj
    loc = getattr(sk.perception, "_localize_knob", None)
    if loc is None:
        return obj
    try:
        k = loc(sk.ctrl.refresh(), obj, obj.camera)
    except Exception:
        return obj
    return k if k is not None else obj
