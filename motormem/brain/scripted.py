"""RUNG-0 DIAGNOSTIC ONLY - NOT A RESULT.

Hand-written programs for the 10 LIBERO-Goal tasks, written against object labels and resolved
to marks by label at run time. They bypass the brain entirely and exist to debug the execution
layer (usually with oracle perception). Never report numbers from this mode as results.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motormem.brain.backend import MarkInfo, _matches
from motormem.brain.program import PUSH_LIMIT, Call, format_call


@dataclass(frozen=True)
class Step:
    skill: str                      # pick | place | actuate | push_to
    label: str                      # object the skill acts on
    arg: str | None = None          # relation / action / reference label (push_to)
    offset: tuple = (0.0, 0.0)      # push_to: target = reference xy + offset (m, base frame)
    by_height: str | None = None    # among marks matching `label`, pick "top" | "middle" | "bottom" by z


# task_id -> (find nouns, steps)
LIBERO_GOAL: dict[int, tuple[list[str], list[Step]]] = {
    0: (["drawer handle"], [Step("actuate", "drawer handle", "open", by_height="middle")]),
    1: (["black bowl", "stove"], [Step("pick", "black bowl"), Step("place", "stove", "on")]),
    2: (["wine bottle", "cabinet"], [Step("pick", "wine bottle"), Step("place", "cabinet", "top_of")]),
    3: (["drawer handle", "black bowl"], [Step("actuate", "drawer handle", "open", by_height="top"),
                                          Step("pick", "black bowl"),
                                          Step("place", "drawer handle", "in", by_height="top")]),
    4: (["black bowl", "cabinet"], [Step("pick", "black bowl"), Step("place", "cabinet", "top_of")]),
    # "front of the stove" = the stove_front_region, which sits ~18 cm in +x (forward, away from the
    # robot) of the segmented stove centroid in the base frame; same y as the stove.
    5: (["plate", "stove"], [Step("push_to", "plate", "stove", (0.18, 0.0))]),
    6: (["cream cheese", "black bowl"], [Step("pick", "cream cheese"), Step("place", "black bowl", "in")]),
    7: (["stove"], [Step("actuate", "stove", "turn_on")]),
    8: (["black bowl", "plate"], [Step("pick", "black bowl"), Step("place", "plate", "on")]),
    9: (["wine bottle", "wine rack"], [Step("pick", "wine bottle"), Step("place", "wine rack", "on")]),
}


def resolve(label: str, marks: dict[int, MarkInfo], by_height: str | None = None) -> int | None:
    """Mark whose label equals `label`, else the first (left-most) whose label matches it.

    `by_height` ("top" | "middle" | "bottom") selects among all marks matching `label` by their z
    (robot base frame, up): used to pick one drawer handle out of the stack the segmenter returns,
    without any task- or sim-specific knowledge.
    """
    cands = [m for m in sorted(marks) if marks[m].label == label]
    if not cands:
        cands = [m for m in sorted(marks) if _matches(marks[m].label, label)]
    if not cands:  # e.g. "drawer handle" vs a backend that labels whole drawers: match on the shared noun
        words = set(label.split())
        cands = [m for m in sorted(marks) if words & set(marks[m].label.split())]
        if by_height and cands:
            hit = [m for m in cands if by_height in marks[m].label.split()]
            cands = hit or cands
    if not cands:
        return None
    if by_height is None or len(cands) == 1:
        return cands[0]
    ordered = sorted(cands, key=lambda m: float(marks[m].xyz[2]))  # low -> high z
    if by_height == "bottom":
        return ordered[0]
    if by_height == "top":
        return ordered[-1]
    return ordered[len(ordered) // 2]  # "middle"


def to_call(step: Step, marks: dict[int, MarkInfo]) -> Call | str:
    """A `Call`, or a short reason string when a label cannot be resolved."""
    m = resolve(step.label, marks, step.by_height)
    if m is None:
        return f"no mark for {step.label!r}"
    if step.skill == "pick":
        return Call("pick", (m,))
    if step.skill == "place":
        return Call("place", (m, step.arg))
    if step.skill == "actuate":
        return Call("actuate", (m, step.arg))
    if step.skill == "push_to":
        r = resolve(step.arg, marks)
        if r is None:
            return f"no mark for {step.arg!r}"
        d = marks[r].xyz[:2] + np.asarray(step.offset) - marks[m].xyz[:2]
        dx, dy = (float(np.clip(v, -PUSH_LIMIT, PUSH_LIMIT)) for v in d)
        return Call("push", (m, dx, dy))
    raise ValueError(step.skill)


def run_scripted(session, task_id: int) -> list[str]:
    """Drive a `RobotSession` through the task's script (observe before every step). Returns a log."""
    find, steps = LIBERO_GOAL[int(task_id) % len(LIBERO_GOAL)]
    log: list[str] = []
    for step in steps:
        session.call("observe", {"find": find})
        if session._over():
            break
        call = to_call(step, session.backend.last_marks())
        if isinstance(call, str):
            log.append(f"{step}: {call}")
            break
        out = session.call("run", {"program": format_call(call)})
        log.append(out[0])
        if session._over() or "-> OK" not in out[0]:
            break
    if not session._over():
        session.call("done", {"summary": "scripted"})
    return log
