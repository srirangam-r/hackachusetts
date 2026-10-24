"""Rung-0 diagnostic (offline, never reported as a result): re-target a stored GRASP segment onto the
ground-truth object pose of a different demo's initial state and execute it with OSC deltas.

  env PYTHONPATH=$PWD MUJOCO_GL=egl python -m motormem.memory.smoke --index data/index/libero_goal.pkl
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from motormem import config, geometry as g
from motormem.memory.index import MotorMemory
from motormem.types import SegmentKind

TASK = "put_the_bowl_on_the_plate"


def _act(T_from, T_to, grip):
    dp, dr = g.pose_error(T_from, T_to)
    a = np.concatenate([dp / config.ACTION_POS_SCALE, dr / config.ACTION_ROT_SCALE])
    return np.concatenate([np.clip(a, -1, 1), [grip]])


def run_once(rp, seg, init_state, mode: str, obj="akita_black_bowl_1"):
    env = rp.env
    env.reset()
    env.set_init_state(init_state)
    rp.sim = sim = env.env.sim   # hard reset rebuilds the sim
    for _ in range(5):
        env.step(np.r_[np.zeros(6), -1.0])
    T_obj = g.make_pose(sim.data.body_xpos[rp.root_bid[obj]].copy(),
                        sim.data.body_xmat[rp.root_bid[obj]].reshape(3, 3).copy())   # rung-0 GT pose
    z0 = T_obj[2, 3]
    targets = g.compose(T_obj[None], seg.T_obj_eef)
    eef = lambda: g.make_pose(sim.data.site_xpos[rp.site_id].copy(), sim.data.site_xmat[rp.site_id].reshape(3, 3).copy())
    # free-space approach to the first segment pose (closed loop), gripper open
    for _ in range(80):
        if np.linalg.norm(targets[0, :3, 3] - eef()[:3, 3]) < 0.005:
            break
        env.step(_act(eef(), targets[0], -1.0))
    for i in range(1, len(targets)):
        base = targets[i - 1] if mode == "open_loop" else eef()
        env.step(_act(base, targets[i], float(seg.gripper[i])))
    # lift ~10 cm with the gripper closed
    for _ in range(25):
        up = eef().copy()
        up[2, 3] += 0.01
        env.step(_act(eef(), up, 1.0))
    z1 = sim.data.body_xpos[rp.root_bid[obj]][2]
    return float(z1 - z0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=str(config.INDEX_DIR / "libero_goal.pkl"))
    ap.add_argument("--n", type=int, default=5)
    a = ap.parse_args()
    import h5py
    from motormem.memory.replay import LiberoReplayer
    mem = MotorMemory.load(a.index)
    segs = [s for s in mem.query(SegmentKind.GRASP, "akita_black_bowl", k=10000) if s.meta.get("task") == TASK]
    segs = sorted(segs, key=lambda s: int(s.demo_id.split("_")[-1]))[: a.n]
    fn = config.LIBERO_HDF5_DIR / "libero_goal" / f"{TASK}_demo.hdf5"
    rp = LiberoReplayer("libero_goal", fn)
    with h5py.File(fn, "r") as f:
        inits = {k: f["data"][k]["states"][0] for k in [f"demo_{i}" for i in range(40, 40 + a.n)]}
    res = []
    for s in segs:
        for k, st in inits.items():
            for mode in ("open_loop", "tracking"):
                lift = run_once(rp, s, st, mode)
                res.append({"segment": s.segment_id, "init": k, "mode": mode, "lift_m": round(lift, 3),
                            "lifted": lift > 0.05})
                print(json.dumps(res[-1]), flush=True)
    for mode in ("open_loop", "tracking"):
        r = [x for x in res if x["mode"] == mode]
        print(f"{mode}: lifted {sum(x['lifted'] for x in r)}/{len(r)}")
    rp.close()


if __name__ == "__main__":
    main()
