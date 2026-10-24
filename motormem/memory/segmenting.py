"""Cut replayed demos into object-frame segments (GRASP / PLACE / PUSH / ACTUATE).

Event rules (design.md, Stream C):
- close = gripper command crosses 0 upward, open = downward, debounced (>= DEBOUNCE steps).
- GRASP: previous open (or start) -> close + 15; frame object = body that rises with the gripper.
- PLACE: close + 15 -> open + 10; frame object = body under/containing the held object at release.
- ACTUATE: an articulated joint moves while the gripper is near it; relation from the joint change.
- PUSH: > 3 cm object displacement while it is not grasped.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from motormem import geometry as g
from motormem.types import Segment, SegmentKind

DEBOUNCE = 3
GRASP_TAIL = 15
PLACE_TAIL = 10
ACTUATE_TAIL = 10
PUSH_MIN_DISP = 0.03
LIFT_MIN = 0.02
SLIDE_MIN = 0.02
HINGE_MIN = 0.2
NEAR_DIST = 0.08
MAX_APPROACH = 40     # steps of approach kept before the key event (contact / close / motion start)


# ---------------------------------------------------------------- gripper events (pure)
@dataclass(frozen=True)
class Event:
    t: int
    kind: str   # "close" | "open"


@dataclass(frozen=True)
class Window:
    kind: str   # "grasp" (keyed by a close event) | "place" (keyed by an open event)
    start: int
    end: int    # exclusive
    event: Event


def gripper_events(cmd, debounce: int = DEBOUNCE) -> list[Event]:
    """Debounced close/open events. A change counts only if the new state lasts >= debounce steps."""
    closed = np.asarray(cmd, dtype=np.float64).reshape(-1) > 0
    T = len(closed)
    if T == 0:
        return []
    state = bool(closed[0])
    events: list[Event] = []
    t = 1
    while t < T:
        if bool(closed[t]) != state:
            run = closed[t:t + debounce]
            if len(run) == debounce and np.all(run == closed[t]):
                state = bool(closed[t])
                events.append(Event(t, "close" if state else "open"))
                t += debounce
                continue
        t += 1
    return events


def event_windows(events: list[Event], T: int, grasp_tail: int = GRASP_TAIL,
                  place_tail: int = PLACE_TAIL) -> list[Window]:
    """One window per event: GRASP for a close event, PLACE for an open event."""
    out: list[Window] = []
    prev_open, prev_close = 0, None
    for ev in events:
        if ev.kind == "close":
            start = prev_open
            end = min(ev.t + grasp_tail, T)
            out.append(Window("grasp", start, max(end, ev.t + 1) if ev.t < T else T, ev))
            prev_close = ev.t
        else:
            start = min(prev_close + grasp_tail, ev.t) if prev_close is not None else 0
            end = min(ev.t + place_tail, T)
            out.append(Window("place", start, max(end, min(ev.t + 1, T)), ev))
            prev_open = ev.t
            prev_close = None
    return out


# ---------------------------------------------------------------- language
RELATION_PATTERNS = [
    ("top_of", r"\bon top of\b|\bon the top of\b"),
    ("left_of", r"\bleft of\b"),
    ("right_of", r"\bright of\b"),
    ("front_of", r"\bfront of\b|\bin front\b"),
    ("back_of", r"\bback of\b|\bbehind\b"),
    ("in", r"\binside\b|\binto\b|\bin the\b|\bin it\b|\bin\b"),
    ("on", r"\bon\b|\bonto\b"),
]


def parse_relation(text: str) -> str | None:
    s = " " + text.lower() + " "
    # "back compartment of the caddy" means "in", not "back_of"
    if re.search(r"\b(in|into|inside)\b.*\bcompartment\b", s):
        return "in"
    for rel, pat in RELATION_PATTERNS:
        if re.search(pat, s):
            return rel
    return None


def _tokens(s: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", s.lower()) if t and t not in {"the", "a", "of", "and", "it"}}


def clause_for(language: str, held_category: str | None) -> str:
    """The placing clause of a compound instruction ("pick up X ... and place it on Y" -> "place it on Y")."""
    parts = [p for p in re.split(r"\band then\b|\bthen\b|,|\band (?=put|place|pick|open|close|turn|push|stack)",
                                 language) if p.strip()]
    if len(parts) == 1:
        return language
    placing = [p for p in parts if re.search(r"\b(put|place|stack)\b", p)] or parts
    if held_category is None:
        return placing[-1]
    want = _tokens(held_category.replace("_", " "))
    best = max(placing, key=lambda p: len(_tokens(p) & want))
    return best if _tokens(best) & want else placing[-1]


# ---------------------------------------------------------------- labelling helpers
def _min_dist(points_obj: np.ndarray | None, T_world_obj: np.ndarray, p: np.ndarray) -> float:
    if points_obj is None or len(points_obj) == 0:
        return float(np.linalg.norm(T_world_obj[:3, 3] - p))
    Pw = g.transform_points(T_world_obj, points_obj)
    return float(np.min(np.linalg.norm(Pw - p, axis=1)))


def _extent(points_obj: np.ndarray | None, T_world_obj: np.ndarray) -> np.ndarray:
    if points_obj is None or len(points_obj) < 3:
        return np.zeros(3)
    _, ext = g.pca_frame(g.transform_points(T_world_obj, points_obj))
    return ext


def _actuate_relation(ji, q0: float, q1: float) -> str:
    art = ji.articulation or {}
    on = art.get("default_turnon_ranges") or []
    off = art.get("default_turnoff_ranges") or []
    if on or off:
        return "turn_on" if q1 > q0 else "turn_off"
    op = art.get("default_open_ranges") or []
    cl = art.get("default_close_ranges") or []
    if op and cl:
        open_mid = float(np.mean(op))
        close_mid = float(np.mean(cl))
        return "open" if abs(q1 - open_mid) < abs(q0 - open_mid) or \
            (np.sign(q1 - q0) == np.sign(open_mid - close_mid)) else "close"
    return "open" if abs(q1) > abs(q0) else "close"


def _active_intervals(x: np.ndarray, eps: float, gap: int = 5) -> list[tuple[int, int]]:
    """Intervals [a, b] where consecutive changes exceed eps (merged over short gaps)."""
    moving = np.abs(np.diff(x, axis=0))
    if moving.ndim > 1:
        moving = np.linalg.norm(moving, axis=1)
    idx = np.nonzero(moving > eps)[0]
    if len(idx) == 0:
        return []
    out = [[idx[0], idx[0] + 1]]
    for i in idx[1:]:
        if i - out[-1][1] <= gap:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [(int(a), int(b)) for a, b in out]


# ---------------------------------------------------------------- main
def segment_demo(tr, arm: str = "Panda") -> tuple[list[Segment], list[str]]:
    """Segments from a DemoTrace plus a list of human-readable labelling failures."""
    T = tr.T
    Teef = tr.T_world_eef
    eef_p = Teef[:, :3, 3]
    events = gripper_events(tr.gripper)
    windows = event_windows(events, T)
    segs: list[Segment] = []
    issues: list[str] = []
    cat = tr.categories
    held_intervals: list[tuple[int, int, str]] = []   # (close t, open t, instance)
    used_close: set[int] = set()

    def floor(t, opens_only=False):
        """Earliest start for a segment whose key event is at t: after earlier events and motions."""
        c = [0, t - MAX_APPROACH] + [ev.t for ev in events if ev.t < t and (ev.kind == "open" or not opens_only)] + \
            [b + 1 for a, b, _ in actuations if b < t]
        return max(c)

    def mk(kind, inst, t0, t1, T_ref, relation=None, held=None, T_eef_held=None, meta=None, frame_t=None,
           key_t=None):
        if key_t is not None and kind != SegmentKind.PLACE:
            t0 = max(t0, floor(key_t, opens_only=kind == SegmentKind.ACTUATE))
        sl = slice(t0, t1)
        if meta and "joint_full" in meta:
            meta["joint_qpos"] = tr.joints[meta["joint_full"]][sl].copy()
        T_obj_eef = g.compose(g.inv(T_ref)[None], Teef[sl])
        pts = tr.points.get(inst, np.zeros((0, 3), np.float32))
        m = {"instance": inst, "t_range": [int(t0), int(t1)], "frame_t": frame_t,
             "key_index": int(key_t - t0) if key_t is not None else 0,
             "T_world_obj": T_ref.copy(), "extent": _extent(pts, T_ref), "task": tr.meta.get("task")}
        if meta:
            m.update(meta)
        segs.append(Segment(
            segment_id=f"{tr.demo_id}/{kind.value}{len(segs)}", demo_id=tr.demo_id, source=tr.source,
            kind=kind, object_name=cat.get(inst, inst), relation=relation,
            held_name=cat.get(held, held) if held else None, T_obj_eef=T_obj_eef,
            gripper=tr.gripper[sl].copy(), T_eef_held=T_eef_held, object_points=np.asarray(pts, np.float32),
            task_language=tr.task_language, arm=arm, meta=m))

    # ---- articulated joints (motion-based, independent of gripper events)
    actuations = []
    for jn, q in tr.joints.items():
        ji = tr.joint_info[jn]
        thr = SLIDE_MIN if ji.jtype == "slide" else HINGE_MIN
        for a, b in _active_intervals(q, 1e-4 if ji.jtype == "slide" else 2e-3):
            if abs(q[b] - q[a]) < thr:
                continue
            near = [np.linalg.norm(eef_p[t] - tr.part_bodies[jn][t]) for t in range(a, b + 1)]
            Towner = tr.bodies[ji.owner][a]
            dmin = min(_min_dist(tr.points.get(ji.owner), Towner, eef_p[t]) for t in range(a, b + 1, 3))
            if dmin > 0.12 and min(near) > 0.3:
                issues.append(f"{jn} moved without gripper nearby")
                continue
            actuations.append((a, b, jn))
    for a, b, jn in actuations:
        ji = tr.joint_info[jn]
        prior = [ev.t for ev in events if ev.t <= a and ev.kind == "open"]
        start = prior[-1] if prior else 0
        # skip back past any earlier segment's end so approach starts after previous interaction
        end = min(b + ACTUATE_TAIL, T)
        q = tr.joints[jn]
        rel = _actuate_relation(ji, float(q[a]), float(q[b]))
        closes = [ev.t for ev in events if ev.kind == "close" and start <= ev.t <= b]
        used_close.update(closes)
        mk(SegmentKind.ACTUATE, ji.owner, start, end, tr.bodies[ji.owner][start], relation=rel,
           meta={"joint": ji.short, "joint_full": jn, "joint_type": ji.jtype,
                 "joint_axis_obj": ji.axis_obj, "joint_anchor_obj": ji.anchor_obj, "part_body": ji.body,
                 "motion_t": [a, b], "grasped": bool(closes)}, frame_t=start, key_t=a)

    # ---- grasp / place pairs from gripper events
    for i, w in enumerate(windows):
        if w.kind != "grasp":
            continue
        c = w.event.t
        nxt = windows[i + 1] if i + 1 < len(windows) else None
        o = nxt.event.t if nxt is not None else T
        lo, hi = c, max(c + 1, o)
        best, best_rise = None, LIFT_MIN
        for inst in tr.movable:
            z = tr.bodies[inst][lo:hi, 2, 3]
            rise_t = int(np.argmax(z))
            rise = float(z[rise_t] - z[0])
            if rise < best_rise:
                continue
            t1 = lo + rise_t
            ta = min(c + 10, t1)
            rel0 = g.compose(g.inv(Teef[ta]), tr.bodies[inst][ta])[:3, 3]
            rel1 = g.compose(g.inv(Teef[t1]), tr.bodies[inst][t1])[:3, 3]
            eef_rise = eef_p[t1, 2] - eef_p[c, 2]
            if np.linalg.norm(rel1 - rel0) > 0.05 or eef_rise < 0.5 * rise:
                continue
            best, best_rise = inst, rise
        if best is None:
            if c not in used_close:
                issues.append(f"close@{c}: nothing lifted")
            continue
        held_intervals.append((c, o, best))
        used_close.add(c)
        T_obj = tr.bodies[best][c]
        mk(SegmentKind.GRASP, best, w.start, w.end, T_obj, meta={"lift": best_rise, "close_t": c}, frame_t=c,
           key_t=c)
        if nxt is None:   # demo ends while holding (success fired before release): release at the end
            o = T - 1
            ps, pe = min(c + GRASP_TAIL, o), T
        else:
            ps, pe = nxt.start, nxt.end
        # PLACE: target under/containing the held object once it has settled after release
        land = (tr.meta.get("landing") or {}).get(o)
        held_p = (land["poses"][best] if land else tr.bodies[best][min(o + PLACE_TAIL, T - 1)])[:3, 3]
        touching = land["contacts"].get(best, set()) if land else set()
        held_cat = cat.get(best, best)
        clause = clause_for(tr.task_language, held_cat)
        rel = parse_relation(clause)
        lang_tokens = _tokens(tr.task_language) - _tokens(held_cat.replace("_", " "))
        cands = []
        for inst, Tb in tr.bodies.items():
            if inst == best:
                continue
            Tt = land["poses"][inst] if land else Tb[o]
            pts = tr.points.get(inst)
            if pts is None or len(pts) < 10:
                continue
            Pw = g.transform_points(Tt, pts)
            dxy = np.linalg.norm(Pw[:, :2] - held_p[:2], axis=1)
            d = float(dxy.min())
            name_tok = _tokens(cat.get(inst, inst).replace("_", " "))
            lang = len(name_tok & lang_tokens) > 0
            score = (0 if inst in touching else 1, 0 if lang else 1, d)
            cands.append((score, d, inst, lang))
        if not cands:
            issues.append(f"open@{o}: no place target")
            continue
        cands.sort()
        _, d, target, lang = cands[0]
        lang_named = [c_[2] for c_ in cands if c_[3]]
        if lang_named and not lang:
            issues.append(f"open@{o}: landed on {target}, language names {lang_named[0]}")
            continue
        if not lang_named and held_p[2] < tr.bodies[best][c, 2, 3] + 0.01 and \
                np.linalg.norm(held_p[:2] - tr.bodies[best][c, :2, 3]) < 0.10:
            issues.append(f"open@{o}: dropped near start height (near {target})")
            continue
        if d > 0.15 and target not in touching:
            issues.append(f"open@{o}: place target far ({d:.2f} m)")
            continue
        part = None
        for jn, ji in tr.joint_info.items():
            if ji.owner == target:
                dd = np.linalg.norm(tr.part_bodies[jn][o] - held_p)
                if part is None or dd < part[1]:
                    part = (ji.short, dd)
        T_eef_held = g.compose(g.inv(Teef[ps]), tr.bodies[best][ps])
        mk(SegmentKind.PLACE, target, ps, pe, tr.bodies[target][o], relation=rel or "on", held=best,
           T_eef_held=T_eef_held,
           meta={"open_t": o, "part": part[0] if part else None, "lang_match": lang,
                 "contact": target in touching,
                 "T_world_held_release": tr.bodies[best][o].copy()}, frame_t=o, key_t=o)

    # ---- pushes: movable objects displaced while not held and not riding on a joint
    for inst in tr.movable:
        P = tr.bodies[inst][:, :3, 3]
        for a, b in _active_intervals(P, 2e-4, gap=8):
            if np.linalg.norm(P[b, :2] - P[a, :2]) < PUSH_MIN_DISP:
                continue
            if any(c - 2 <= b and a <= o + 2 and h == inst for c, o, h in held_intervals):
                continue
            if any(not (b < aa or a > bb) for aa, bb, _ in actuations) and \
                    min(_min_dist(tr.points.get(inst), tr.bodies[inst][t], eef_p[t]) for t in range(a, b + 1)) > NEAR_DIST:
                continue
            dmin = min(_min_dist(tr.points.get(inst), tr.bodies[inst][t], eef_p[t]) for t in range(a, b + 1, 2))
            if dmin > NEAR_DIST:
                continue
            holding = any(c <= a <= o for c, o, _ in held_intervals)
            if holding:
                continue  # knocked by a carried object, not a push
            prior = [ev.t for ev in events if ev.t <= a]
            start = prior[-1] if prior else 0
            end = min(b + ACTUATE_TAIL, T)
            rel = parse_relation(tr.task_language)
            mk(SegmentKind.PUSH, inst, start, end, tr.bodies[inst][a], relation=rel,
               meta={"displacement": (P[b] - P[a]).tolist(), "motion_t": [a, b]}, frame_t=a, key_t=a)
    return segs, issues
