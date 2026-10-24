import json
from types import SimpleNamespace

import numpy as np

from motormem.brain.recording import Recorder


class _Env:
    def __init__(self):
        self.t = 0

    def step(self, action):
        self.t += 1
        return SimpleNamespace(step=self.t)

    def render(self, camera, size):
        return np.full((size, size, 3), self.t % 255, np.uint8)


def test_recorder_tags_calls_and_patches_segment(tmp_path):
    env = _Env()
    rec = Recorder(env, tmp_path, every=2, size=32)
    rec.begin("pick(1)", 1)
    for _ in range(6):
        env.step(np.zeros(7))
    rec.end("libero_goal/t/demo_3/grasp0", "OK")
    rec.begin("home()", 2)
    env.step(np.zeros(7)); env.step(np.zeros(7))
    rec.end(None, "OK")
    rec.close()
    doc = json.loads((tmp_path / "frames.json").read_text())
    assert len(doc["frames"]) == 1 + 3 + 1
    assert [f["call"] for f in doc["frames"]] == [-1, 0, 0, 0, 1]
    c0, c1 = doc["calls"]
    assert (c0["first"], c0["last"], c0["segment_id"]) == (1, 3, "libero_goal/t/demo_3/grasp0")
    assert c1["segment_id"] == "primitive" and c1["run"] == 2
    assert all((tmp_path / f["file"]).exists() for f in doc["frames"])
    assert env.step.__func__ is _Env.step          # unwrapped on close
