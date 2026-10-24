"""DIAGNOSTIC ONLY (reads privileged sim poses to explain a failure; never used by skills).

    diag_pick.py <task_id> <label> [perception=sam3|oracle] [init]
Prints, for the pick of `label`: the perceived centroid vs ground truth, each plan tried, and where the
gripper was when it closed relative to the true object.
"""
import sys

import numpy as np

from motormem import config
from motormem.brain.backend import RealBackend
from motormem.types import EpisodeSpec

task, label = int(sys.argv[1]), sys.argv[2]
perc = sys.argv[3] if len(sys.argv) > 3 else "sam3"
init = int(sys.argv[4]) if len(sys.argv) > 4 else 0
import os
b = RealBackend(EpisodeSpec(suite=os.environ.get("SUITE", "libero_goal"), task_id=task, init_index=init, perception=perc,
                            index_path=str(config.INDEX_DIR / os.environ.get("INDEX", "libero_goal.pkl"))))
_, text = b.observe([label])
print(text)
est = next(e for e in b.estimates.values() if label.split()[-1] in e.label)
gt = {k: v for k, v in b.env.privileged_object_poses().items() if label.split()[-1].replace(" ", "_") in k}
for k, T in gt.items():
    print(f"GT {k}: {np.round(T[:3, 3], 3)}  perceived centroid {np.round(est.T_world_obj[:3, 3], 3)} extent {np.round(est.extent, 3)}"
          f"  pts z-range {est.points_world[:, 2].min():.3f}..{est.points_world[:, 2].max():.3f}")
sk = b.skills
if os.environ.get('MS'):
    sk.motion_source = os.environ['MS']
_orig_do = sk._do_grasp
def _dbg(plan, obj):
    r = _orig_do(plan, obj)
    print('  plan', plan.segment_id, '->', r.status.value, r.message[:80], 'steps', r.steps)
    return r
sk._do_grasp = _dbg
if os.environ.get('NOCLEAR'):
    sk._clear_height = lambda *a, **k: -float('inf')
closes = []
orig = sk.ctrl.gripper


def spy(cmd, steps=10, *a, **k):
    if cmd > 0:
        closes.append(sk.ctrl.pose()[:3, 3].copy())
    return orig(cmd, steps, *a, **k)


sk.ctrl.gripper = spy
import time as _t; _t0=_t.time(); n0=b.env.observe().step
res = sk.pick(est)
print('steps used', b.env.observe().step-n0, 'wall', round(_t.time()-_t0,1))
print("RESULT", res.status.value, res.message, res.segment_id, "fallback" if res.used_fallback else "")
for p in closes:
    for k, T in gt.items():
        print(f"close at {np.round(p, 3)}; offset from GT {k}: {np.round(p - T[:3, 3], 3)}")
