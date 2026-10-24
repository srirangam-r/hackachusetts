"""Property 4: strip_privileged never lets a privileged key through and keeps every allowed proprio key."""
import string

from hypothesis import given, settings
from hypothesis import strategies as st

from motormem import config
from motormem.env import strip_privileged

OBJECTS = ["akita_black_bowl_1", "plate_1", "wine_bottle_1", "Milk", "Can", "wooden_cabinet_1"]
privileged_keys = st.one_of(
    st.sampled_from(OBJECTS).map(lambda o: f"{o}_pos"),
    st.sampled_from(OBJECTS).map(lambda o: f"{o}_quat"),
    st.sampled_from(OBJECTS).map(lambda o: f"{o}_to_robot0_eef_pos"),
    st.sampled_from(OBJECTS).map(lambda o: f"{o}_to_robot0_eef_quat"),
    st.just("object-state"),
    st.sampled_from(["agentview", "robot0_eye_in_hand"]).map(lambda c: f"{c}_segmentation_instance"),
    st.text(string.ascii_letters + "_", min_size=1, max_size=12).map(lambda s: s + "_pos"),
)
allowed_keys = st.sampled_from(sorted(config.ALLOWED_PROPRIO_KEYS))
other_keys = st.one_of(
    st.sampled_from(["agentview_image", "agentview_depth", "robot0_eye_in_hand_image",
                     "robot0_eye_in_hand_depth", "robot0_joint_vel", "robot0_gripper_qvel"]),
    st.text(min_size=0, max_size=20),
)
keys = st.one_of(privileged_keys, allowed_keys, other_keys)


@settings(max_examples=300, deadline=None)
@given(st.dictionaries(keys, st.integers(), max_size=30))
def test_strip_privileged_removes_privileged_keeps_allowed(raw):
    out = strip_privileged(raw)
    assert not any(config.is_privileged_key(k) for k in out)
    for k in raw:
        if k in config.ALLOWED_PROPRIO_KEYS or not config.is_privileged_key(k):
            assert k in out and out[k] == raw[k]
    assert set(out) <= set(raw)


@settings(max_examples=100, deadline=None)
@given(st.sampled_from(OBJECTS))
def test_known_libero_object_keys_are_stripped(obj):
    raw = {f"{obj}_pos": 1, f"{obj}_quat": 2, f"{obj}_to_robot0_eef_pos": 3, "object-state": 4,
           "robot0_eef_pos": 5, "robot0_eef_quat": 6, "robot0_gripper_qpos": 7, "robot0_joint_pos": 8}
    assert strip_privileged(raw) == {"robot0_eef_pos": 5, "robot0_eef_quat": 6,
                                     "robot0_gripper_qpos": 7, "robot0_joint_pos": 8}
