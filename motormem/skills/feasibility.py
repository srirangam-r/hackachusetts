"""Reachability check: damped-least-squares IK on the robot's own kinematic model copy (Req 4.1).

Only the arm joints of the MjData copy are written; nothing about objects is read.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from motormem import geometry as g
from motormem.types import Status


@dataclass
class IKResult:
    ok: bool
    fail_index: int            # first failing waypoint (-1 if ok)
    q_path: np.ndarray         # (n_solved, n_joints)
    pos_err: float
    rot_err: float
    joint_travel: float        # sum of |dq| along the path (reachability cost)


class Feasibility:
    def __init__(self, pos_tol: float = 0.01, rot_tol: float = 0.08, damping: float = 0.05,
                 max_iters_first: int = 60, max_iters: int = 20, rot_weight: float = 0.3,
                 max_dq: float = 0.3, limit_margin: float = 0.01):
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self.damping = damping
        self.max_iters_first, self.max_iters = max_iters_first, max_iters
        self.rot_weight = rot_weight
        self.max_dq = max_dq
        self.limit_margin = limit_margin

    def solve(self, kin, targets: np.ndarray) -> IKResult:
        m, d = kin.model, kin.data
        qa, da, site = np.asarray(kin.qpos_addr), np.asarray(kin.dof_addr), int(kin.site_id)
        lo = np.asarray(kin.q_lower) + self.limit_margin
        hi = np.asarray(kin.q_upper) - self.limit_margin
        q = np.clip(np.asarray(kin.q_current, dtype=np.float64).copy(), lo, hi)
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4, 4)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        lam2 = self.damping ** 2
        w = self.rot_weight
        qs, travel = [], 0.0
        pe = re = 0.0
        for i, T in enumerate(targets):
            q_prev = q.copy()
            iters = self.max_iters_first if i == 0 else self.max_iters
            ok = False
            for _ in range(iters + 1):
                d.qpos[qa] = q
                mujoco.mj_kinematics(m, d)
                cur = g.make_pose(d.site_xpos[site], d.site_xmat[site].reshape(3, 3))
                dpos, drot = g.pose_error(cur, T)
                pe, re = float(np.linalg.norm(dpos)), float(np.linalg.norm(drot))
                if pe < self.pos_tol and re < self.rot_tol:
                    ok = True
                    break
                mujoco.mj_comPos(m, d)
                mujoco.mj_jacSite(m, d, jacp, jacr, site)
                J = np.vstack([jacp[:, da], w * jacr[:, da]])
                e = np.concatenate([dpos, w * drot])
                dq = J.T @ np.linalg.solve(J @ J.T + lam2 * np.eye(6), e)
                mx = np.max(np.abs(dq))
                if mx > self.max_dq:
                    dq *= self.max_dq / mx
                q = np.clip(q + dq, lo, hi)
            if not ok:
                return IKResult(False, i, np.array(qs), pe, re, travel)
            travel += float(np.abs(q - q_prev).sum())
            qs.append(q.copy())
        return IKResult(True, -1, np.array(qs), pe, re, travel)

    def check(self, kin, targets: np.ndarray) -> tuple[bool, str]:
        r = self.solve(kin, targets)
        if r.ok:
            return True, ""
        return False, (f"{Status.IK_FAILURE.value}: waypoint {r.fail_index} unreachable "
                       f"(err {r.pos_err * 100:.1f} cm, {np.degrees(r.rot_err):.0f} deg)")
