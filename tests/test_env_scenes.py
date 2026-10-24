"""Custom scenes behind make_env: robosuite bin sort (Panda/UR5e) and the LIBERO desk-sort study scene."""
import numpy as np
import pytest

pytestmark = pytest.mark.sim

from motormem.types import EpisodeSpec  # noqa: E402


@pytest.mark.parametrize("arm", ["Panda", "UR5e"])
def test_bin_sort(arm):
    from motormem.env import make_env
    env = make_env(EpisodeSpec(suite="bin_sort", task_id=0, init_index=2, arm=arm))
    try:
        obs = env.reset()
        assert env.instruction == "sort each item into its matching bin"
        assert obs.cameras["agentview"].depth.shape == (256, 256)
        assert obs.proprio.joint_pos.shape == ((7,) if arm == "Panda" else (6,))
        poses = env.privileged_object_poses()
        assert {"Milk", "Bread", "Cereal", "Can", "bin1", "bin2"} <= set(poses)
        for name in ("Milk", "Bread", "Cereal", "Can"):                    # start in bin1
            assert np.linalg.norm(poses[name][:2, 3] - poses["bin1"][:2, 3]) < 0.25
        assert env._runner_success() is False
        obs = env.step(np.array([0, 0, -0.3, 0, 0, 0, 1.0]))
        assert obs.step == 1
        masks = env.privileged_segmentation("agentview")
        assert all(masks[n].any() for n in ("Milk", "Bread", "Cereal", "Can"))
    finally:
        env.close()


def test_bin_sort_layout_reproducible():
    from motormem.env.bin_sort import make_bin_sort_env
    a, b, c = (make_bin_sort_env("Panda", init_index=i, settle_steps=0) for i in (0, 0, 1))
    try:
        pa, pb, pc = (e.reset() and e.privileged_object_poses() for e in (a, b, c))
        assert all(np.allclose(pa[k], pb[k]) for k in pa)
        assert any(not np.allclose(pa[k], pc[k], atol=1e-3) for k in ("Milk", "Bread", "Cereal", "Can"))
    finally:
        for e in (a, b, c):
            e.close()


def test_desk_sort_bddls_generated():
    from motormem.env import desk_sort
    paths = desk_sort.generate_bddls()
    assert len(paths) == len(desk_sort.TASKS) * desk_sort.N_LAYOUTS
    assert all(p.exists() and p.parent == desk_sort.SCENES_DIR for p in paths)


@pytest.mark.parametrize("layout", range(5))
def test_desk_sort_layout_objects_on_table(layout):
    from motormem.env import desk_sort, make_env
    arm = "UR5e" if layout == 4 else "Panda"
    env = make_env(EpisodeSpec(suite="desk_sort", task_id=0, init_index=layout, arm=arm))
    try:
        env.reset()
        assert env.instruction == "sort the desk: books in the caddy, mugs on the tray"
        poses = env.privileged_object_poses()
        names = [f"{n}_1" for n in desk_sort.OBJECTS]
        assert set(names) <= set(poses) and 6 <= len(names) <= 10
        for n in names:
            assert 0.86 < poses[n][2, 3] < 0.93, (n, poses[n][2, 3])              # resting on the table top
            assert abs(poses[n][2, 2]) > 0.9 or n.startswith("black_book") or n.startswith("yellow_book"), n
        d = [np.linalg.norm(poses[a][:2, 3] - poses[b][:2, 3]) for i, a in enumerate(names) for b in names[i + 1:]]
        assert min(d) > 0.1
        expected = desk_sort.layout_positions(layout)
        for n, (x, y) in expected.items():                                       # placed where the layout says
            assert np.linalg.norm(poses[f"{n}_1"][:2, 3] - np.array([x - 0.2, y])) < 0.05, n
        assert env._runner_success() is False
    finally:
        env.close()


def test_desk_sort_object_status_starts_unsorted():
    from motormem.env import desk_sort, make_env
    env = make_env(EpisodeSpec(suite="desk_sort", task_id=0, init_index=1, arm="Panda"))
    try:
        env.reset()
        st = desk_sort.object_status(env)
        assert set(st) == set(desk_sort.BOOKS + desk_sort.MUGS) and not any(st.values())
    finally:
        env.close()


@pytest.mark.parametrize("dummy", [0])
def test_scene_demo_mark_selection(dummy):
    """Scripted-program helpers: workspace filter, label de-duplication, status parsing (no sim)."""
    import importlib.util
    from pathlib import Path
    from types import SimpleNamespace as NS
    p = Path(__file__).resolve().parents[1] / "scripts" / "scene_demo.py"
    spec = importlib.util.spec_from_file_location("scene_demo", p)
    sd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sd)
    mk = lambda i, label, xyz: NS(mark=i, label=label, xyz=np.array(xyz))  # noqa: E731
    marks = {1: mk(1, "black book", [-1.1, -1.0, -0.26]),      # background false positive
             2: mk(2, "black book", [0.34, -0.11, 0.07]),
             3: mk(3, "book", [0.343, -0.105, 0.07]),          # same object, generic label
             4: mk(4, "book", [0.35, 0.10, 0.07]),
             5: mk(5, "bin", [0.6, -0.25, -0.1]), 6: mk(6, "bin", [0.6, 0.25, -0.1])}
    assert sd.pick_mark(marks, "black book") == 2
    assert sd.pick_mark(marks, "book") == 4
    assert sd.pick_mark(marks, "mug") is None
    assert sd.pick_mark(marks, "bin", far_from=np.array([0.6, -0.2, 0])) == 6
    assert sd._ok("1. pick(3) -> OK: picked book (mark 3)")
    assert not sd._ok("1. pick(3) -> NO_CANDIDATE: pick: no OK segment")
