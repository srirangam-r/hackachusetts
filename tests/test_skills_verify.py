"""Unit tests for motormem.skills.verify (perception/proprioception-only outcome checks)."""
from __future__ import annotations

import numpy as np

from motormem import geometry as g
from motormem.skills.verify import locate, relation_holds
from motormem.types import ObjectEstimate


def _est(mark, label, center, extent, top=None, bottom=None):
    c = np.asarray(center, float)
    ext = np.asarray(extent, float)
    T = g.make_pose(c)
    zt = c[2] + ext[2] / 2 if top is None else top
    zb = c[2] - ext[2] / 2 if bottom is None else bottom
    # a few surface points spanning the extent, with explicit top/bottom z
    pts = np.array([[c[0], c[1], zt], [c[0], c[1], zb],
                    [c[0] + ext[0] / 2, c[1], c[2]], [c[0] - ext[0] / 2, c[1], c[2]]], float)
    return ObjectEstimate(mark=mark, label=label, camera="agentview", mask=np.ones((2, 2), bool),
                          points_world=pts, T_world_obj=T, extent=ext)


def test_on_holds_when_centered_and_resting():
    plate = _est(1, "plate", [0.4, 0.0, 0.01], [0.18, 0.18, 0.02])
    top = plate.points_world[:, 2].max()
    bowl = _est(2, "bowl", [0.4, 0.0, top + 0.03], [0.12, 0.12, 0.06], bottom=top + 0.01)
    ok, reason = relation_holds(bowl, plate, "on")
    assert ok, reason


def test_on_fails_when_offset():
    plate = _est(1, "plate", [0.4, 0.0, 0.01], [0.18, 0.18, 0.02])
    bowl = _est(2, "bowl", [0.4, 0.3, 0.04], [0.12, 0.12, 0.06])   # 30 cm off in y
    ok, reason = relation_holds(bowl, plate, "on")
    assert not ok and "horizontal" in reason


def test_on_fails_when_hovering():
    plate = _est(1, "plate", [0.4, 0.0, 0.01], [0.18, 0.18, 0.02])
    top = plate.points_world[:, 2].max()
    bowl = _est(2, "bowl", [0.4, 0.0, top + 0.20], [0.12, 0.12, 0.06], bottom=top + 0.15)
    ok, reason = relation_holds(bowl, plate, "on")
    assert not ok and "above" in reason


def test_in_requires_bottom_below_top():
    drawer = _est(1, "drawer", [0.6, 0.0, 0.15], [0.2, 0.2, 0.1])
    top = drawer.points_world[:, 2].max()
    inside = _est(2, "bowl", [0.6, 0.0, top - 0.03], [0.1, 0.1, 0.05], bottom=top - 0.05)
    ok, _ = relation_holds(inside, drawer, "in")
    assert ok
    above = _est(3, "bowl", [0.6, 0.0, top + 0.10], [0.1, 0.1, 0.05], bottom=top + 0.08)
    ok2, reason = relation_holds(above, drawer, "in")
    assert not ok2 and "above" in reason


def test_spatial_relations_use_base_frame():
    # identity base frame: base x forward, y left
    target = _est(1, "stove", [0.5, 0.0, 0.03], [0.2, 0.2, 0.03])
    left = _est(2, "bowl", [0.5, 0.2, 0.05], [0.1, 0.1, 0.06])
    assert relation_holds(left, target, "left_of", T_world_base=np.eye(4))[0]
    assert not relation_holds(left, target, "right_of", T_world_base=np.eye(4))[0]
    front = _est(3, "bowl", [0.65, 0.0, 0.05], [0.1, 0.1, 0.06])   # +x in base = front
    assert relation_holds(front, target, "front_of", T_world_base=np.eye(4))[0]
    assert relation_holds(front, target, "back_of", T_world_base=np.eye(4))[0] is False


class _FakePerception:
    def __init__(self, ests):
        self._ests = ests

    def scan(self, obs, find, camera="agentview"):
        return [e for e in self._ests if e.label in find]


class _FakeCtrl:
    def __init__(self):
        self.obs = type("O", (), {"T_world_base": np.eye(4)})()

    def refresh(self):
        return self.obs


class _FakeSkills:
    def __init__(self, ests):
        self.perception = _FakePerception(ests)
        self.ctrl = _FakeCtrl()


def test_locate_returns_nearest_within_radius():
    a = _est(1, "bowl", [0.4, 0.0, 0.05], [0.1, 0.1, 0.06])
    b = _est(2, "bowl", [0.4, 0.3, 0.05], [0.1, 0.1, 0.06])
    s = _FakeSkills([a, b])
    got = locate(s, "bowl", [0.41, 0.01, 0.05], radius=0.1)
    assert got is a
    assert locate(s, "bowl", [1.0, 1.0, 1.0], radius=0.1) is None
    assert locate(s, "missing", [0.4, 0.0, 0.05]) is None
