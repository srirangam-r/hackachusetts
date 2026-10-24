"""Fast (no-sim) tests for Controller and Skills plumbing using a kinematic fake env."""
import numpy as np

from motormem import geometry as g
from motormem.skills import Controller, Skills
from motormem.skills.control import POS_SCALE
from motormem.types import ObjectEstimate, Observation, Proprio, Status


class FakeEnv:
    """Integrates OSC deltas exactly (perfect tracking) — enough to test the plumbing."""

    def __init__(self):
        self.T = g.make_pose([0.0, 0.0, 1.2], np.diag([1.0, -1.0, -1.0]))
        self.open = 1.0

    def observe(self):
        p = Proprio(self.T.copy(), np.array([0.04, -0.04]) * self.open, self.open, np.zeros(7), np.zeros(3), np.zeros(3))
        return Observation({}, p, "", 0, np.eye(4))

    def step(self, a):
        a = np.clip(a, -1, 1)
        self.T[:3, 3] += a[:3] * POS_SCALE * 0.5
        self.open = float(np.clip(self.open - 0.2 * a[6], 0, 1))
        return self.observe()

    def kinematics(self):
        raise RuntimeError("no kinematics in fake")


def test_track_reaches_final_waypoint():
    env = FakeEnv()
    c = Controller(env)
    T1 = env.T.copy()
    T1[:3, 3] += [0.1, -0.05, -0.1]
    r = c.track(g.interpolate(env.T.copy(), T1, 10), -1.0)
    assert r.reached and r.waypoints_done == 10 and r.pos_err < 0.01


def test_exceptions_become_error_and_bad_arguments():
    sk = Skills(FakeEnv(), motion_source="primitive")
    P = np.random.default_rng(0).normal([0.3, 0, 0.9], 0.02, (200, 3))
    T, ext = g.pca_frame(P)
    obj = ObjectEstimate(1, "cube", "agentview", np.zeros((1, 1), bool), P, T, ext)
    r = sk.pick(obj)                       # kinematics() raises -> ERROR, never an exception
    assert r.status == Status.ERROR and "RuntimeError" in r.message
    assert sk.actuate(obj, "wiggle").status == Status.BAD_ARGUMENT
    assert sk.push(obj, 0.0, 0.0).status == Status.BAD_ARGUMENT
    assert sk.place(obj, "on").status == Status.NOT_HOLDING


def _book_memory():
    from motormem.memory.index import MotorMemory
    from motormem.types import Segment, SegmentKind

    def seg(i, kind, obj, held=None, rel=None):
        return Segment(f"s{i}", f"d{i}", "synthetic", kind, obj, rel, held, np.tile(np.eye(4), (3, 1, 1)),
                       np.zeros(3), None, np.zeros((20, 3)), "")
    return MotorMemory([seg(0, SegmentKind.GRASP, "black_book"), seg(1, SegmentKind.GRASP, "yellow_book"),
                        seg(2, SegmentKind.PLACE, "desk_caddy", "black_book", "in")])


def _est(label):
    P = np.random.default_rng(0).normal([0.3, 0, 0.9], 0.02, (200, 3))
    T, ext = g.pca_frame(P)
    return ObjectEstimate(1, label, "agentview", np.zeros((1, 1), bool), P, T, ext)


def test_excluded_object_gives_no_candidate_or_fallback():
    from motormem.skills.core import Held
    from motormem.types import SkillResult
    view = _book_memory().without(objects=["black book"])
    sk = Skills(FakeEnv(), memory=view, motion_source="index")
    r = sk.pick(_est("black book"))
    assert r.status == Status.NO_CANDIDATE and "no demo of black book" in r.message
    sk = Skills(FakeEnv(), memory=view, motion_source="hybrid")
    sk._primitive_pick = lambda o: SkillResult(Status.OK, "primitive")
    r = sk.pick(_est("black book"))
    assert r.status == Status.OK and r.used_fallback


def test_place_cross_object_reuse_only_in_hybrid():
    from motormem.skills.core import Held
    from motormem.types import SkillResult
    mem = _book_memory()
    for source in ("index", "hybrid"):
        sk = Skills(FakeEnv(), memory=mem, motion_source=source)
        sk.ctrl.holding = lambda *a, **k: True
        sk.held = Held("yellow book", "yellow_book", None, np.eye(4), np.full(3, 0.05), False)
        seen = []
        sk._segment_plans = lambda seg, *a, **k: (seen.append(seg.segment_id), iter(()))[1]
        sk._primitive_place = lambda t, rel: SkillResult(Status.OK, "primitive")
        r = sk.place(_est("desk caddy"), "in")
        if source == "index":
            assert r.status == Status.NO_CANDIDATE and seen == []
        else:
            assert r.status == Status.OK and r.used_fallback and seen == ["s2"]
