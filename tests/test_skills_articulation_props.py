"""Property 8: line/circle fitting recovers the generating model from noisy samples (Req 4.4)."""
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation

from motormem.skills.articulation import select_model

unit = st.tuples(*[st.floats(-1, 1, allow_nan=False)] * 3).filter(lambda v: np.linalg.norm(v) > 0.2).map(
    lambda v: np.array(v) / np.linalg.norm(v))
centre = st.tuples(*[st.floats(-0.5, 0.5, allow_nan=False)] * 3).map(np.array)


@settings(max_examples=150, deadline=None)
@given(centre, unit, st.floats(0.04, 0.25), st.integers(15, 60), st.floats(0.0, 0.001), st.integers(0, 10_000))
def test_line_recovered(c, u, length, n, sigma, seed):
    rng = np.random.default_rng(seed)
    s = np.sort(rng.uniform(0, length, n))
    P = c + np.outer(s, u) + rng.normal(0, sigma, (n, 3))
    fit = select_model(P)
    assert fit.kind == "line"
    assert abs(fit.axis @ u) > np.cos(np.radians(3))


@settings(max_examples=150, deadline=None)
@given(centre, unit, st.floats(0.08, 0.5), st.floats(0.6, 1.8), st.integers(20, 60),
       st.floats(0.0, 0.001), st.integers(0, 10_000))
def test_circle_recovered(c, axis, r, span, n, sigma, seed):
    sagitta = r * (1 - np.cos(span / 2))
    if sagitta < max(0.008, 10 * sigma):     # arcs that are geometrically indistinguishable from lines
        return
    rng = np.random.default_rng(seed)
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    th = np.sort(rng.uniform(0, span, n))
    P = c + r * (np.outer(np.cos(th), e1) + np.outer(np.sin(th), e2)) + rng.normal(0, sigma, (n, 3))
    fit = select_model(P)
    assert fit.kind == "circle"
    assert abs(fit.radius - r) < 0.1 * r + 0.005
    assert np.linalg.norm(fit.point - c) < 0.1 * r + 0.005
    assert abs(fit.axis @ axis) > np.cos(np.radians(8))


def test_tangent_follows_travel_direction():
    P = np.array([[0.0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.03, 0, 0], [0.04, 0, 0], [0.05, 0, 0]])
    fit = select_model(P)
    assert fit.tangent(P[-1], np.array([1.0, 0, 0])) @ np.array([1.0, 0, 0]) > 0.99
    R = Rotation.from_euler("z", 0.3).as_matrix()
    assert fit.tangent(P[-1], R @ np.array([-1.0, 0, 0]))[0] < -0.99
