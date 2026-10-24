"""Property-based tests for the shared geometry contract (Requirement 7: correctness of frames)."""
import numpy as np
from hypothesis import given, settings, strategies as st
from scipy.spatial.transform import Rotation

from motormem import geometry as g

finite = st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False)
vec3 = st.tuples(finite, finite, finite).map(np.array)
quats = st.tuples(*[st.floats(-1, 1, allow_nan=False)] * 4).filter(lambda q: np.linalg.norm(q) > 0.1).map(
    lambda q: np.array(q) / np.linalg.norm(q))


def poses():
    return st.tuples(vec3, quats).map(lambda pq: g.pose_from_pos_quat(*pq))


@settings(max_examples=200, deadline=None)
@given(poses())
def test_inverse_round_trip(T):
    """Property 1: T @ inv(T) == I for every rigid transform."""
    assert np.allclose(g.compose(T, g.inv(T)), np.eye(4), atol=1e-9)


@settings(max_examples=200, deadline=None)
@given(poses(), poses())
def test_object_frame_round_trip(T_world_obj, T_world_eef):
    """Property 2: storing a gripper pose in an object frame and transforming back is lossless."""
    T_obj_eef = g.compose(g.inv(T_world_obj), T_world_eef)
    assert np.allclose(g.compose(T_world_obj, T_obj_eef), T_world_eef, atol=1e-9)


@settings(max_examples=100, deadline=None)
@given(poses(), poses(), poses())
def test_segment_moves_rigidly_with_object(T_demo_obj, T_live_obj, T_world_eef):
    """Property 3: re-targeting a segment preserves the gripper-object relative pose."""
    T_obj_eef = g.compose(g.inv(T_demo_obj), T_world_eef)
    T_new = g.compose(T_live_obj, T_obj_eef)
    assert np.allclose(g.compose(g.inv(T_live_obj), T_new), T_obj_eef, atol=1e-9)


@settings(max_examples=100, deadline=None)
@given(poses(), poses())
def test_pose_error_reaches_target(T0, T1):
    dpos, drot = g.pose_error(T0, T1)
    R = Rotation.from_rotvec(drot).as_matrix() @ T0[:3, :3]
    assert np.allclose(T0[:3, 3] + dpos, T1[:3, 3], atol=1e-9)
    assert np.allclose(R, T1[:3, :3], atol=1e-6)


@settings(max_examples=50, deadline=None)
@given(st.floats(0.3, 1.5), st.floats(-0.2, 0.2), st.floats(-0.2, 0.2))
def test_deproject_project_round_trip(z, x, y):
    K = np.array([[300.0, 0, 128], [0, 300.0, 128], [0, 0, 1]])
    T_world_cam = g.make_pose([0.5, 0.0, 1.5], Rotation.from_euler("x", 180, degrees=True).as_matrix())
    p_world = g.transform_points(T_world_cam, np.array([[x, y, z]]))
    uv, d = g.project(p_world, K, T_world_cam)
    assert np.isclose(d[0], z)
    depth = np.zeros((256, 256), np.float32)
    u, v = int(round(uv[0, 0])), int(round(uv[0, 1]))
    if 0 <= u < 256 and 0 <= v < 256:
        depth[v, u] = z
        pts = g.deproject(depth, K, T_world_cam)
        assert len(pts) == 1
        assert np.linalg.norm(pts[0] - p_world[0]) < z / 300.0 + 1e-6  # within one pixel


def test_register_recovers_planar_pose():
    rng = np.random.default_rng(0)
    box = rng.uniform([-0.05, -0.02, 0.0], [0.05, 0.02, 0.06], size=(600, 3))
    T_true = g.make_pose([0.1, -0.2, 0.9], g.rot_z(0.7))
    observed = g.transform_points(T_true, box) + rng.normal(0, 0.001, (600, 3))
    hyps = g.register_multi_yaw(box, observed)
    T, rmse, fit = hyps[0]
    assert fit > 0.95
    assert np.linalg.norm(T[:3, 3] - T_true[:3, 3]) < 0.01
    # boxes are 180-degree symmetric: accept either heading
    dyaw = (g.yaw_of(T) - 0.7 + np.pi / 2) % np.pi - np.pi / 2
    assert abs(dyaw) < 0.05
