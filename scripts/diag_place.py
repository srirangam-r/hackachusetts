"""DIAGNOSTIC ONLY (reads privileged sim poses to explain a failure; never used by skills).

    diag_place.py <task_id> <held label> <target label> <relation> [init]
Runs pick(held) + place(target, relation) with SAM 3 and prints where the held object ended up relative to
the target's ground-truth pose, the registered target pose used for the place, and the env verdict.
"""
import sys

import numpy as np

from motormem import config
from motormem.brain.backend import RealBackend
from motormem.types import EpisodeSpec

task, held, target, rel = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
init = int(sys.argv[5]) if len(sys.argv) > 5 else 0
b = RealBackend(EpisodeSpec(suite="libero_goal", task_id=task, init_index=init, perception="sam3",
                            index_path=str(config.INDEX_DIR / "libero_goal.pkl")))
b.observe([held, target])
est = {e.label: e for e in b.estimates.values()}
h = next(e for k, e in est.items() if held.split()[-1] in k)
t = next(e for k, e in est.items() if target.split()[-1] in k)
gt = b.env.privileged_object_poses()


def gt_of(word):
    return {k: v[:3, 3] for k, v in gt.items() if word in k}


print("pick:", b.skills.pick(h).message)
r = b.skills.place(t, rel)
print("place:", r.status.value, r.message, r.segment_id)
print("registered target (plan):", r.info.get("T_world_obj_pos") if r.info else None)
print("perceived target centroid:", np.round(t.T_world_obj[:3, 3], 3), "extent", np.round(t.extent, 3))
for k, p in gt_of(held.split()[-1]).items():
    print("GT held", k, np.round(p, 3))
for k, p in gt_of(target.split()[-1].replace(" ", "_")).items():
    print("GT target", k, np.round(p, 3))
print("env success:", b.env._runner_success())
