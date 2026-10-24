"""LIBERO-Pro goal swap/task episodes load from data/raw/libero_pro (read-only) on both arms."""
import numpy as np
import pytest

pytestmark = pytest.mark.sim

from motormem.types import EpisodeSpec  # noqa: E402


@pytest.mark.parametrize("suite", ["libero_pro_goal_swap", "libero_pro_goal_task"])
def test_pro_suite_has_ten_tasks_with_init_states(suite):
    from motormem.env.libero import load_init_states, resolve_task, suite_task_names
    names = suite_task_names(suite)
    assert len(names) == 10 and len(set(names)) == 10
    for i in range(10):
        t = resolve_task(suite, i)
        assert t.bddl_file.exists() and t.init_file.exists()
        assert "libero_pro" in str(t.bddl_file)
    assert load_init_states(resolve_task(suite, 0).init_file).shape[0] >= 10


def test_pro_swap_moves_objects_vs_standard_goal():
    from motormem.env import make_env
    std = make_env(EpisodeSpec(suite="libero_goal", task_id="put_the_bowl_on_the_plate", init_index=0), settle_steps=0)
    pro = make_env(EpisodeSpec(suite="libero_pro_goal_swap", task_id="put_the_bowl_on_the_plate", init_index=0),
                   settle_steps=0)
    try:
        std.reset()
        pro.reset()
        assert pro.instruction == "put the bowl on the plate"
        a, b = std.privileged_object_poses(), pro.privileged_object_poses()
        moved = [k for k in a if k in b and np.linalg.norm(a[k][:2, 3] - b[k][:2, 3]) > 0.03]
        assert moved, "swap perturbation should relocate objects"
    finally:
        std.close()
        pro.close()


def test_pro_task_ur5e_episode():
    from motormem.env import make_env
    env = make_env(EpisodeSpec(suite="libero_pro_goal_task", task_id=7, init_index=3, arm="UR5e"))
    try:
        obs = env.reset()
        assert env.instruction == "turn off the stove"
        assert obs.proprio.joint_pos.shape == (6,) and obs.cameras["agentview"].rgb.shape == (256, 256, 3)
        poses = env.privileged_object_poses()
        assert all(0.8 < T[2, 3] < 1.2 for T in poses.values())        # nothing fell through the table
    finally:
        env.close()
