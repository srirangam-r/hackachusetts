"""Simulator tests for the LIBERO adapter (Panda + UR5e): observation contents, camera convention,
kinematics copy, runner-only methods, and the Panda -> UR5e init-state transfer."""
import numpy as np
import pytest

pytestmark = pytest.mark.sim

mujoco = pytest.importorskip("mujoco")

from motormem import config, geometry  # noqa: E402
from motormem.types import EpisodeSpec  # noqa: E402

SUITE, TASK, INIT = "libero_goal", 8, 0     # "put the bowl on the plate"


@pytest.fixture(scope="module")
def panda():
    from motormem.env import make_env
    env = make_env(EpisodeSpec(suite=SUITE, task_id=TASK, init_index=INIT, arm="Panda"))
    obs = env.reset()
    yield env, obs
    env.close()


@pytest.fixture(scope="module")
def ur5e():
    from motormem.env import make_env
    env = make_env(EpisodeSpec(suite=SUITE, task_id=TASK, init_index=INIT, arm="UR5e"))
    obs = env.reset()
    yield env, obs
    env.close()


def _is_rigid(T):
    R = T[:3, :3]
    return np.allclose(R @ R.T, np.eye(3), atol=1e-6) and np.isclose(np.linalg.det(R), 1.0, atol=1e-6)


def test_observation_contents(panda):
    env, obs = panda
    assert env.instruction == "put the bowl on the plate"
    assert set(obs.cameras) == {"agentview", "wrist"}
    for cam in obs.cameras.values():
        assert cam.rgb.shape == (256, 256, 3) and cam.rgb.dtype == np.uint8
        assert cam.depth.shape == (256, 256) and cam.depth.dtype == np.float32
        assert 0.05 < float(np.median(cam.depth)) < 5.0
        assert cam.K.shape == (3, 3) and _is_rigid(cam.T_world_cam)
    p = obs.proprio
    assert _is_rigid(p.T_world_eef) and p.joint_pos.shape == (7,) and p.gripper_qpos.shape == (2,)
    assert 0.0 <= p.gripper_opening <= 1.0 and p.gripper_opening > 0.8      # starts open
    assert p.ee_force.shape == (3,) and p.ee_torque.shape == (3,)
    assert _is_rigid(obs.T_world_base) and obs.step == 0
    # grip-site z axis points down at the table at the start
    assert p.T_world_eef[2, 2] < -0.9
    # no privileged values hidden anywhere in the observation's attributes
    raw_keys = env._rs._get_observations().keys()
    assert any(config.is_privileged_key(k) for k in raw_keys)          # raw has them ...
    from motormem.env import strip_privileged
    assert not any(config.is_privileged_key(k) for k in strip_privileged(dict.fromkeys(raw_keys)))


@pytest.mark.parametrize("arm_fixture", ["panda", "ur5e"])
def test_camera_convention_projects_gripper_onto_gripper(arm_fixture, request):
    """project(EEF) lands on the gripper: rendered depth at the projected finger pixels equals the projected
    depth and the pixels belong to the robot's simulator mask (vertical flip + OpenCV extrinsics correct)."""
    import imageio

    env, _ = request.getfixturevalue(arm_fixture)
    obs = env.observe()
    cam = obs.cameras["agentview"]
    data, model = env.sim.data._data, env.sim.model._model
    robot_mask = env.privileged_segmentation("agentview")["robot"]
    img = cam.rgb.copy()
    site_uv, site_z = geometry.project(obs.proprio.T_world_eef[:3, 3][None], cam.K, cam.T_world_cam)
    u, v = np.round(site_uv[0]).astype(int)
    assert 0 <= u < 256 and 0 <= v < 256 and site_z[0] > 0
    win = robot_mask[max(v - 12, 0):v + 13, max(u - 12, 0):u + 13]   # site sits between open fingertips
    assert win.any(), "grip site does not project next to the gripper"
    # every gripper body origin projects inside the image; at least two (unoccluded ones) have rendered
    # depth equal to their projected depth and lie on the robot mask. A missing flip or wrong axis convention
    # puts them on unrelated pixels and fails this.
    bodies = [model.body(i).name for i in range(model.nbody) if model.body(i).name.startswith("gripper0_")]
    hits = 0
    for b in bodies:
        p = data.xpos[model.body(b).id]
        uv, z = geometry.project(p[None], cam.K, cam.T_world_cam)
        bu, bv = np.round(uv[0]).astype(int)
        assert 0 <= bu < 256 and 0 <= bv < 256, b
        if robot_mask[bv, bu] and abs(float(cam.depth[bv, bu]) - z[0]) < 0.03:
            hits += 1
            img[max(bv - 1, 0):bv + 2, max(bu - 1, 0):bu + 2] = (0, 255, 0)
    assert len(bodies) >= 2 and hits >= 2, (hits, len(bodies))
    img[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = (255, 0, 0)
    out = config.RUNS_DIR / "env_check"
    out.mkdir(parents=True, exist_ok=True)
    big = np.kron(img, np.ones((3, 3, 1), dtype=np.uint8))
    imageio.imwrite(out / f"eef_projection_{env.arm}.png", big)


def test_deprojection_matches_world_geometry(panda):
    """Deprojected depth of the plate mask is centred on the plate body (camera model consistent)."""
    env, obs = panda
    cam = obs.cameras["agentview"]
    masks = env.privileged_segmentation("agentview")
    poses = env.privileged_object_poses()
    pts = geometry.deproject(cam.depth, cam.K, cam.T_world_cam, mask=masks["plate_1"])
    assert len(pts) > 100
    assert np.linalg.norm(pts.mean(0)[:2] - poses["plate_1"][:2, 3]) < 0.03


def test_step_and_gripper(panda):
    env, obs0 = panda
    a = np.zeros(7)
    a[2], a[-1] = -0.5, 1.0
    for _ in range(8):
        obs = env.step(a)
    assert obs.step == 8
    assert obs.proprio.T_world_eef[2, 3] < obs0.proprio.T_world_eef[2, 3] - 0.04
    assert obs.proprio.gripper_opening < 0.3
    a[:] = 0.0
    a[-1] = -1.0
    for _ in range(10):
        obs = env.step(a)
    assert obs.proprio.gripper_opening > 0.8
    assert env.observe() is obs
    with pytest.raises(ValueError):
        env.step(np.zeros(6))


@pytest.mark.parametrize("arm_fixture", ["panda", "ur5e"])
def test_kinematics_is_a_copy(arm_fixture, request):
    env, _ = request.getfixturevalue(arm_fixture)
    obs = env.observe()
    kin = env.kinematics()
    n = 7 if env.arm == "Panda" else 6
    assert len(kin.qpos_addr) == n == len(kin.dof_addr) == len(kin.q_current)
    assert np.allclose(kin.q_current, obs.proprio.joint_pos, atol=1e-9)
    assert np.all(kin.q_lower < kin.q_upper)
    assert mujoco.mj_id2name(kin.model, mujoco.mjtObj.mjOBJ_SITE, kin.site_id) == "gripper0_grip_site"
    mujoco.mj_kinematics(kin.model, kin.data)
    # MuJoCo's site poses after mj_step lag qpos by one physics substep, hence 1e-4
    assert np.allclose(kin.data.site_xpos[kin.site_id], obs.proprio.T_world_eef[:3, 3], atol=1e-4)
    assert np.allclose(kin.data.site_xmat[kin.site_id].reshape(3, 3), obs.proprio.T_world_eef[:3, :3], atol=1e-4)
    live = env.sim.data._data.qpos.copy()
    kin.data.qpos[kin.qpos_addr] += 0.3
    mujoco.mj_kinematics(kin.model, kin.data)
    assert np.array_equal(env.sim.data._data.qpos, live)


def test_runner_only_methods(panda):
    env, _ = panda
    assert env._runner_success() is False
    poses = env.privileged_object_poses()
    assert {"akita_black_bowl_1", "plate_1", "wooden_cabinet_1", "flat_stove_1"} <= set(poses)
    assert all(_is_rigid(T) for T in poses.values())
    masks = env.privileged_segmentation("agentview")
    assert masks["akita_black_bowl_1"].shape == (256, 256) and masks["akita_black_bowl_1"].sum() > 50
    joints = env.privileged_joint_positions()
    assert "flat_stove_1_button" in joints
    frame = env.render("agentview", 128)
    assert frame.shape == (128, 128, 3)


def test_ur5e_objects_match_panda():
    """Same init state on both arms: identical object state, UR5e keeps its own default joints."""
    from motormem.env import arms
    from motormem.env.libero import make_libero_env

    p = make_libero_env(SUITE, TASK, INIT, arm="Panda", settle_steps=0)
    u = make_libero_env(SUITE, TASK, INIT, arm="UR5e", settle_steps=0)
    try:
        p.reset()
        u.reset()
        pm, um = p.sim.model._model, u.sim.model._model
        p_names = [mujoco.mj_id2name(pm, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(pm.njnt)]
        robot = [n for n in p_names if arms.is_robot_joint(n)]
        assert robot == arms.PANDA_ROBOT_JOINTS
        assert sum(pm.jnt_type[j] in (2, 3) for j in range(len(robot))) == arms.PANDA_ROBOT_NQ  # all 1-dof
        pl, ul = arms.object_joint_layout(pm), arms.object_joint_layout(um)
        assert [l[0] for l in pl] == [l[0] for l in ul]                     # same object joints, same order
        assert pl[0][1] == arms.PANDA_ROBOT_NQ and ul[0][1] == 12            # Panda 7+2, UR5e 6+6 robot qpos
        for (name, pq, qs, _, _), (_, uq, _, _, _) in zip(pl, ul):
            assert np.allclose(p.sim.data._data.qpos[pq:pq + qs], u.sim.data._data.qpos[uq:uq + qs], atol=1e-9), name
        pp, up = p.privileged_object_poses(), u.privileged_object_poses()
        free = {l[0].rsplit("_joint", 1)[0] for l in pl if l[2] == 7}
        for name in free:
            assert np.allclose(pp[name], up[name], atol=1e-6), name
        state = p.init_states[INIT]
        assert len(state) != u.sim.get_state().flatten().shape[0]           # dims differ: never set_init_state
        assert np.allclose(u.sim.data._data.qvel, 0.0)
        assert np.allclose(u.observe().proprio.joint_pos, u._rs.robots[0].init_qpos, atol=1e-6)
    finally:
        p.close()
        u.close()
