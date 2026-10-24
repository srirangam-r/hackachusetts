"""Joint-agnostic articulation skill (Req 4.4): grasp a handle/knob, move with an equilibrium-point
lead, estimate the joint online (line vs circle) from the gripper's own measured path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motormem import geometry as g
from motormem.skills.control import densify
from motormem.types import ObjectEstimate, SegmentKind, SkillResult, Status

LEAD = 0.015          # equilibrium-point lead per control period (m)
FIT_AFTER = 0.03      # start fitting after this much travel (m)
STALL_WINDOW = 10     # steps
STALL_DIST = 0.002    # m of progress within the window
MAX_TRAVEL = 0.30     # m
KNOB_STEP = 0.1       # rad per wrist increment
KNOB_MAX = 1.6        # rad
NOISE_FLOOR = 5e-4    # m; residual floor for the BIC (sensor/compliance noise)
LINE_RADIUS = 1.0     # circles with larger radius are treated as lines
ACTIONS = ("open", "close", "turn_on", "turn_off")


# ---------------------------------------------------------------- model fitting (Property 8)
@dataclass
class JointFit:
    kind: str                 # "line" | "circle"
    rss: float
    bic: float
    point: np.ndarray         # line: centroid; circle: centre
    axis: np.ndarray          # line: direction; circle: plane normal (rotation axis)
    radius: float = np.inf

    def tangent(self, p: np.ndarray, prev_dir: np.ndarray) -> np.ndarray:
        if self.kind == "line":
            t = self.axis
        else:
            r = p - self.point
            r -= self.axis * (r @ self.axis)
            t = np.cross(self.axis, r)
        t = t / (np.linalg.norm(t) + 1e-12)
        return t if t @ prev_dir >= 0 else -t


def fit_line(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    u = Vt[0]
    r = (P - c) - np.outer((P - c) @ u, u)
    return c, u, float((r ** 2).sum())


def fit_circle(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    c0 = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c0, full_matrices=False)
    e1, e2, n = Vt[0], Vt[1], Vt[2]
    Q = np.stack([(P - c0) @ e1, (P - c0) @ e2], 1)
    A = np.column_stack([2 * Q, np.ones(len(Q))])                 # Kasa algebraic fit
    b = (Q ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    ctr2 = sol[:2]
    rad = float(np.sqrt(max(sol[2] + ctr2 @ ctr2, 1e-12)))
    for _ in range(5):                                            # Gauss-Newton refinement (geometric)
        dv = Q - ctr2
        dist = np.linalg.norm(dv, axis=1) + 1e-12
        res = dist - rad
        J = np.column_stack([-dv / dist[:, None], -np.ones(len(Q))])
        step, *_ = np.linalg.lstsq(J, -res, rcond=None)
        ctr2, rad = ctr2 + step[:2], float(abs(rad + step[2]))
    out = (P - c0) @ n
    res = np.linalg.norm(Q - ctr2, axis=1) - rad
    centre = c0 + ctr2[0] * e1 + ctr2[1] * e2
    return centre, n, rad, float((res ** 2).sum() + (out ** 2).sum())


def select_model(P: np.ndarray, noise_floor: float = NOISE_FLOOR) -> JointFit:
    """Line vs circle by a BIC-style criterion: n log(RSS/n) + k log(n), k_line=4, k_circle=7."""
    P = np.asarray(P, dtype=np.float64)
    n = len(P)
    floor = noise_floor ** 2
    c, u, rss_l = fit_line(P)
    bic_l = n * np.log(rss_l / n + floor) + 4 * np.log(n)
    line = JointFit("line", rss_l, bic_l, c, u)
    if n < 6:
        return line
    ctr, nrm, rad, rss_c = fit_circle(P)
    bic_c = n * np.log(rss_c / n + floor) + 7 * np.log(n)
    if not np.isfinite(rad) or rad > LINE_RADIUS or bic_c >= bic_l:
        return line
    return JointFit("circle", rss_c, bic_c, ctr, nrm, rad)


# ---------------------------------------------------------------- the skill
def _face_normal(sk, part: ObjectEstimate) -> np.ndarray:
    """Horizontal outward normal of the part: frame y (perpendicular to the main axis), signed toward
    the robot base (the side the robot and cameras see)."""
    y = part.T_world_obj[:3, 1].copy()
    y[2] = 0
    y /= np.linalg.norm(y) + 1e-12
    to_base = sk.ctrl.obs.T_world_base[:3, 3] - part.T_world_obj[:3, 3]
    return y if y @ to_base >= 0 else -y


def _contact_from_index(sk, part: ObjectEstimate, action: str):
    if not sk._use_index():
        return None
    for seg in sk._query(SegmentKind.ACTUATE, part, relation=action):
        for plan in sk._segment_plans(seg, part, close=True):
            # keep approach + grasp only: drop the lift appended by _segment_plans and the demo motion
            body = plan.targets[: plan.event + 1]
            return plan, body
    return None


def _primitive_contact(sk, part: ObjectEstimate, action: str) -> tuple[np.ndarray, np.ndarray]:
    """(T_contact, approach direction). Knobs: top-down at the centroid; handles: frontal grasp."""
    from motormem.skills.core import frontal_rotation, topdown_rotation, rot_angle
    _, col = sk._gripper_geometry()
    P, T = part.points_world, part.T_world_obj
    c = T[:3, 3]
    R_cur = sk.ctrl.pose()[:3, :3]
    if action in ("turn_on", "turn_off"):
        z = max(P[:, 2].max() - 0.02, P[:, 2].min() + 0.01)
        opts = [topdown_rotation(s * T[:3, 1], col) for s in (1, -1)]
        R = min(opts, key=lambda r: rot_angle(R_cur, r))
        return g.make_pose([c[0], c[1], z], R), np.array([0, 0, -1.0])
    n = _face_normal(sk, part)
    bar = T[:3, 0]
    closing = np.cross(n, bar)               # perpendicular to the bar and to the approach
    if np.linalg.norm(closing) < 1e-6:
        closing = np.array([0, 0, 1.0])
    opts = [frontal_rotation(-n, s * closing, col) for s in (1, -1)]
    R = min(opts, key=lambda r: rot_angle(R_cur, r))
    if action == "close":
        p = c + n * 0.01                      # fingertips just outside the handle face
    else:
        p = c - n * 0.005
    return g.make_pose(p, R), -n


def _approach_frontal(sk, T_contact: np.ndarray, approach: np.ndarray, standoff: float = 0.08) -> np.ndarray:
    pre = T_contact.copy()
    pre[:3, 3] -= approach * standoff
    cur = sk.ctrl.pose().copy()
    pts = [cur]
    if cur[2, 3] < pre[2, 3] - 0.01:
        up = cur.copy()
        up[2, 3] = pre[2, 3]
        pts.append(up)
    pts.append(pre)
    a = densify(pts, 0.02, 0.1)
    b = densify([pre, T_contact], 0.01, 0.1)
    return np.concatenate([a, b[1:]])


def _retreat(sk, away: np.ndarray) -> None:
    T = sk.ctrl.pose().copy()
    back = T.copy()
    back[:3, 3] += away * 0.06
    up = back.copy()
    up[2, 3] += 0.06
    sk.ctrl.track(densify([T, back, up], 0.02, 0.1), -1.0)


def actuate(sk, part: ObjectEstimate, action: str) -> SkillResult:
    if action not in ACTIONS:
        return SkillResult(Status.BAD_ARGUMENT, f"actuate: action must be one of {ACTIONS}")
    if part.points_world is None or len(part.points_world) < 20:
        return SkillResult(Status.PERCEPTION_FAILED, f"actuate: too few points on mark {part.mark}")
    knob = action in ("turn_on", "turn_off")
    seg_id, fallback = None, True
    idx = _contact_from_index(sk, part, action)
    grasp = action != "close"
    if idx is not None:
        plan, path = idx
        seg_id, fallback = plan.segment_id, False
        approach = path[-1, :3, 2]
    else:
        if sk.motion_source == "index":
            return SkillResult(Status.NO_CANDIDATE, f"actuate: no '{action}' segment for '{part.label}'")
        T_c, approach = _primitive_contact(sk, part, action)
        path = sk._approach(T_c, lift=0.08) if knob else _approach_frontal(sk, T_c, approach)
    ok, reason = sk._feasible(path)
    if not ok:
        return SkillResult(Status.IK_FAILURE, f"actuate: contact pose unreachable ({reason})",
                           segment_id=seg_id, used_fallback=fallback)
    if grasp:
        sk._ensure_open()
    else:
        sk.ctrl.gripper(1.0, 8)
    sk.ctrl.track(path, -1.0 if grasp else 1.0)
    if grasp:
        sk.ctrl.gripper(1.0, 12)
        if not sk.ctrl.holding():
            sk._ensure_open()
            _retreat(sk, -approach)
            return SkillResult(Status.GRASP_FAILED, f"actuate: could not grasp {part.label}",
                               segment_id=seg_id, used_fallback=fallback)
    if knob:
        out = _turn(sk, part, +1.0 if action == "turn_on" else -1.0)
    else:
        n = _face_normal(sk, part)
        out = _ep_motion(sk, n if action == "open" else -n, grasp)
    sk.ctrl.gripper(-1.0, 8)
    _retreat(sk, -approach if not knob else np.zeros(3))
    status, msg, info = out
    return SkillResult(status, f"actuate {action} {part.label}: {msg}", segment_id=seg_id,
                       used_fallback=fallback, info=info)


def _ep_motion(sk, d0: np.ndarray, grasped: bool):
    """Equilibrium-point motion with online line/circle estimation."""
    ctrl = sk.ctrl
    d = d0 / np.linalg.norm(d0)
    T0 = ctrl.pose().copy()
    path = [T0[:3, 3].copy()]
    fit: JointFit | None = None
    travel, stop = 0.0, "max_travel"
    grip = 1.0
    lo = ctrl.thresholds(ctrl.obs)[0]
    for k in range(int(MAX_TRAVEL / 0.004)):
        p = ctrl.pose()[:3, 3]
        R = T0[:3, :3]
        if fit is not None and fit.kind == "circle":   # rotate the wrist with the arc
            a0 = path[0] - fit.point
            a1 = p - fit.point
            a0 -= fit.axis * (a0 @ fit.axis)
            a1 -= fit.axis * (a1 @ fit.axis)
            ang = np.arctan2(np.cross(a0, a1) @ fit.axis, a0 @ a1)
            R = g.Rotation.from_rotvec(fit.axis * ang).as_matrix() @ T0[:3, :3]
        target = g.make_pose(p + LEAD * d, R)
        ctrl.step(ctrl.action_to(target, grip))
        q = ctrl.pose()[:3, 3]
        travel += float(np.linalg.norm(q - path[-1]))
        path.append(q.copy())
        if grasped and ctrl.obs.proprio.gripper_opening < lo:
            return Status.GRASP_FAILED, f"slipped off after {travel * 100:.1f} cm", {"travel": travel}
        if travel >= FIT_AFTER and len(path) >= 6:
            fit = select_model(np.array(path))
            d = fit.tangent(q, d)
        if len(path) > STALL_WINDOW:
            lag = np.linalg.norm(target[:3, 3] - q)
            if np.linalg.norm(q - path[-1 - STALL_WINDOW]) < STALL_DIST and lag > 0.75 * LEAD:
                stop = "stall"
                break
        if travel >= MAX_TRAVEL:
            break
    info = {"travel": travel, "model": fit.kind if fit else "none", "stop": stop,
            "radius": None if fit is None or fit.kind == "line" else fit.radius}
    return Status.OK, f"moved {travel * 100:.1f} cm ({info['model']}), stopped on {stop}", info


def _turn(sk, part: ObjectEstimate, sign: float):
    """Rotate the grasped knob about the part's vertical axis in KNOB_STEP increments."""
    ctrl = sk.ctrl
    T0 = ctrl.pose().copy()
    axis_p = part.T_world_obj[:3, 3]
    yaw0 = g.yaw_of(T0)
    yaws, stop, total = [0.0], "max_angle", 0.0
    lo = ctrl.thresholds(ctrl.obs)[0]
    n_inc = int(KNOB_MAX / KNOB_STEP)
    for i in range(1, n_inc + 1):
        Rz = g.rot_z(sign * KNOB_STEP * i)
        T = g.make_pose(axis_p + Rz @ (T0[:3, 3] - axis_p), Rz @ T0[:3, :3])
        T[2, 3] = T0[2, 3]
        for _ in range(4):
            ctrl.step(ctrl.action_to(T, 1.0))
            yaws.append(float(np.unwrap([yaw0, g.yaw_of(ctrl.pose())])[1] - yaw0) * sign)
        total = yaws[-1]
        if ctrl.obs.proprio.gripper_opening < lo:
            return Status.GRASP_FAILED, f"slipped off after {np.degrees(total):.0f} deg", {"angle": total}
        if len(yaws) > STALL_WINDOW and yaws[-1] - yaws[-1 - STALL_WINDOW] < 0.02:
            stop = "stall"
            break
    return Status.OK, f"turned {np.degrees(total):.0f} deg, stopped on {stop}", {"angle": total, "stop": stop}
