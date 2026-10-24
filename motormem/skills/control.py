"""OSC pose tracking on top of RobotEnv's 7-D delta actions (Req 4.3, 4.6).

Every action is computed from proprioception only: the grip-site pose (the OSC control frame) and the
normalised gripper opening.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from motormem import config, geometry as g
from motormem.types import Observation

POS_SCALE = config.ACTION_POS_SCALE   # metres per unit action per step
ROT_SCALE = config.ACTION_ROT_SCALE   # radians per unit action per step

# Grasp-state thresholds on the normalised opening, keyed by gripper qpos size.
# (closed_below, open_above): opening < closed_below = closed on nothing; > open_above = open/empty.
GRIPPER_THRESHOLDS: dict[int, tuple[float, float]] = {
    2: (0.03, 0.80),    # PandaGripper (calibrated in LIBERO: empty close ~0.00, bowl rim ~0.05+)
    6: (0.008, 0.80),   # Robotiq85LinkedGripper: empty close reads 0.000; a thin bowl rim pinched at the tips 0.012+
}
# Usable finger opening (metres) per gripper, keyed the same way.
GRIPPER_MAX_WIDTH: dict[int, float] = {2: 0.08, 6: 0.085}
# Closing axis of the fingers in the grip-site frame, keyed the same way (verified: both close along site x).
GRIPPER_CLOSING_AXIS: dict[int, int] = {2: 0, 6: 0}
# Extra pad-centre depth for rim/edge grasps (object wider than the jaw). The Robotiq85 pinching a bowl rim at the
# Panda depth holds it only with the fingertips: the rim pivots on lift and the jaw reads as nearly closed
# (opening 0.03 -> 0.015 < threshold). 1 cm deeper the pads take the wall (opening 0.15 -> 0.08 after lift).
GRIPPER_RIM_EXTRA_DEPTH: dict[int, float] = {2: 0.0, 6: 0.01}

# Panda-equivalent tool frame. Segments are recorded as PandaGripper grip-site poses; what must transfer is the
# pinch geometry, i.e. the finger-pad centre. Panda pads (prismatic) sit at a constant z = -4 mm in the site
# frame. The Robotiq85 pads swing on a four-bar arc: their centre moves from z = -14.7 mm (open) to -0.7 mm
# (closed) and the tips sweep 14 mm forward while closing. T_site_tcp(opening) = translate(0, 0, z_pad - z_panda)
# so the controller tracks the pad centre where the Panda's would be, and keeps it fixed while the jaw closes.
PANDA_PAD_Z = -0.004
# (opening samples, pad-centre z in the grip-site frame), measured by forward kinematics of the linked model.
GRIPPER_PAD_Z: dict[int, tuple[tuple[float, ...], tuple[float, ...]]] = {
    6: ((0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0),
        (-0.0007, -0.0006, -0.0010, -0.0019, -0.0034, -0.0055, -0.0081, -0.0111, -0.0147)),
}


def tcp_offset(key: int, opening: float) -> np.ndarray:
    """T_site_tcp: the Panda-equivalent tool frame in the gripper's grip-site frame (identity for the Panda)."""
    T = np.eye(4)
    tab = GRIPPER_PAD_Z.get(key)
    if tab is not None:
        T[2, 3] = float(np.interp(np.clip(opening, 0.0, 1.0), tab[0], tab[1])) - PANDA_PAD_Z
    return T


class StepBudgetExceeded(Exception):
    """Raised by Controller.step when the active skill's step budget is used up."""


@dataclass
class TrackResult:
    reached: bool
    steps: int
    pos_err: float
    rot_err: float
    waypoints_done: int
    obs: Observation | None = None
    blocked: bool = False


class Controller:
    def __init__(self, env, gain: float = 2.0, ki: float = 0.3, pass_scale: float = 2.5, final_scale: int = 3):
        self.env = env
        self.gain = gain
        self.ki = ki                      # leaky position integrator: removes static sag (UR5e ~1 cm)
        self.pass_scale = pass_scale
        self.final_scale = final_scale
        self.obs: Observation = env.observe()
        self.steps = 0
        self._budget_end: int | None = None
        self._i_pos = np.zeros(3)

    # ---------------------------------------------------------- basics
    def set_budget(self, n: int | None) -> None:
        self._budget_end = None if n is None else self.steps + int(n)

    def tcp(self, obs: Observation | None = None) -> np.ndarray:
        """T_site_tcp for the current gripper opening (see tcp_offset)."""
        obs = obs or self.obs
        return tcp_offset(len(obs.proprio.gripper_qpos), obs.proprio.gripper_opening)

    def pose(self) -> np.ndarray:
        """Panda-equivalent tool pose (= grip site for the Panda). All skill targets are in this frame."""
        return self.obs.proprio.T_world_eef @ self.tcp()

    def to_site(self, T_tcp: np.ndarray, opening: float | None = None) -> np.ndarray:
        """Grip-site pose that puts the tool frame at T_tcp (at `opening`, default: current)."""
        op = self.obs.proprio.gripper_opening if opening is None else opening
        return T_tcp @ np.linalg.inv(tcp_offset(len(self.obs.proprio.gripper_qpos), op))

    def refresh(self) -> Observation:
        self.obs = self.env.observe()
        return self.obs

    def step(self, action: np.ndarray) -> Observation:
        if self._budget_end is not None and self.steps >= self._budget_end:
            raise StepBudgetExceeded(f"step budget exhausted after {self.steps} steps")
        out = self.env.step(np.asarray(action, dtype=np.float64))
        self.obs = out if isinstance(out, Observation) else self.env.observe()
        self.steps += 1
        return self.obs

    def action_to(self, T_target: np.ndarray, grip: float, max_action: float = 1.0,
                  integrate: bool = False) -> np.ndarray:
        """Clipped OSC delta action that moves the current tool pose (see `pose`) toward T_target.
        `integrate` enables the leaky position integrator (only while settling on a fixed target)."""
        dpos, drot = g.pose_error(self.obs.proprio.T_world_eef, self.to_site(T_target))
        if integrate and np.linalg.norm(dpos) < 0.02:   # anti-windup: integrate only near the target
            self._i_pos = np.clip(0.95 * self._i_pos + dpos, -0.03, 0.03)
        else:
            self._i_pos *= 0.5
        a_pos = self.gain * (dpos + self.ki * self._i_pos) / POS_SCALE
        a_rot = self.gain * drot / ROT_SCALE
        for a in (a_pos, a_rot):           # direction-preserving clip to [-max, max]
            m = np.max(np.abs(a))
            if m > max_action:
                a *= max_action / m
        return np.concatenate([a_pos, a_rot, [float(np.clip(grip, -1, 1))]])

    # ---------------------------------------------------------- tracking
    def track(self, targets: np.ndarray, gripper: np.ndarray | float, pos_tol: float = 0.005,
              rot_tol: float = 0.05, max_steps_per_wp: int = 15, max_action: float = 0.6) -> TrackResult:
        """Track absolute grip-site poses. Intermediate waypoints are passed at `pass_scale` x tolerance
        (except right before a gripper change); the final waypoint is always tracked at full tolerance
        with `final_scale` x the step allowance."""
        targets = np.asarray(targets, dtype=np.float64).reshape(-1, 4, 4)
        n = len(targets)
        grip = np.broadcast_to(np.asarray(gripper, dtype=np.float64), (n,))
        start = self.steps
        done = 0
        blocked = False
        for i in range(n):
            last = i == n - 1
            exact = last or (grip[i + 1] > 0) != (grip[i] > 0)
            pt, rt = (pos_tol, rot_tol) if exact else (pos_tol * self.pass_scale, rot_tol * self.pass_scale)
            limit = max_steps_per_wp * (self.final_scale if last else 1)
            self._i_pos[:] = 0.0
            hist = []
            for _ in range(limit):
                dpos, drot = g.pose_error(self.pose(), targets[i])
                if np.linalg.norm(dpos) < pt and np.linalg.norm(drot) < rt:
                    break
                hist.append(self.pose()[:3, 3].copy())
                if len(hist) > 6 and np.linalg.norm(hist[-1] - hist[-7]) < 0.0015 and np.linalg.norm(dpos) > 2 * pt:
                    blocked = True            # contact / limit: no progress, stop pushing on this waypoint
                    break
                self.step(self.action_to(targets[i], grip[i], max_action, integrate=exact))
            done = i + 1
            if blocked and last:
                break
        dpos, drot = g.pose_error(self.pose(), targets[-1])
        pe, re = float(np.linalg.norm(dpos)), float(np.linalg.norm(drot))
        return TrackResult(reached=pe < pos_tol * 2 and re < rot_tol * 2, steps=self.steps - start,
                           pos_err=pe, rot_err=re, waypoints_done=done, obs=self.obs, blocked=blocked)

    def gripper(self, cmd: float, steps: int = 10) -> Observation:
        """Command the gripper while holding the current pose."""
        hold = self.pose().copy()
        self._i_pos[:] = 0.0
        for _ in range(steps):
            self.step(self.action_to(hold, cmd, integrate=True))
        return self.obs

    # ---------------------------------------------------------- grasp state
    @staticmethod
    def thresholds(obs: Observation) -> tuple[float, float]:
        return GRIPPER_THRESHOLDS.get(len(obs.proprio.gripper_qpos), (0.03, 0.80))

    def holding(self, obs: Observation | None = None) -> bool:
        obs = obs or self.obs
        lo, hi = self.thresholds(obs)
        return lo < obs.proprio.gripper_opening < hi

    def is_open(self, obs: Observation | None = None) -> bool:
        obs = obs or self.obs
        return obs.proprio.gripper_opening >= self.thresholds(obs)[1]

    def gripper_key(self) -> int:
        return len(self.obs.proprio.gripper_qpos)


def densify(poses: list[np.ndarray] | np.ndarray, pos_step: float = 0.02, rot_step: float = 0.1) -> np.ndarray:
    """Insert interpolated poses so consecutive waypoints differ by <= pos_step / rot_step."""
    poses = [np.asarray(p, dtype=np.float64) for p in poses]
    out = [poses[0]]
    for a, b in zip(poses[:-1], poses[1:]):
        dpos, drot = g.pose_error(a, b)
        n = int(np.ceil(max(np.linalg.norm(dpos) / pos_step, np.linalg.norm(drot) / rot_step))) + 1
        if n <= 1:
            continue
        out.extend(g.interpolate(a, b, n)[1:])
    return np.stack(out)
