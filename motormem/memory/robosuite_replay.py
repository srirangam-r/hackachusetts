"""Replay of robosuite (MimicGen source) demos, e.g. PickPlace with Sawyer. Offline index building only."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from motormem.memory.replay import LiberoReplayer, canonical_instance


def _patch_xml(xml: str) -> str:
    # mujoco 3.x rejects Sawyer's tiny head meshes; shell inertia is irrelevant for state replay
    return re.sub(r'<mesh ([^>]*name="robot0_head[^"]*")', r'<mesh inertia="shell" \1', xml)


class RobosuiteReplayer(LiberoReplayer):
    """Same interface as LiberoReplayer; builds the robosuite env named in the HDF5 env_args."""

    def __init__(self, source: str, hdf5_path):
        import json

        import h5py
        import robosuite as suite
        from robosuite.utils import binding_utils

        self.suite = source
        self.hdf5_path = Path(hdf5_path)
        self.task_name = self.hdf5_path.stem
        with h5py.File(hdf5_path, "r") as f:
            ea = json.loads(f["data"].attrs["env_args"])
        name = re.sub(r"_D\d+$", "", ea["env_name"])
        kw = ea["env_kwargs"]
        orig = binding_utils.MjSim.from_xml_string.__func__
        binding_utils.MjSim.from_xml_string = classmethod(lambda cls, xml: orig(cls, _patch_xml(xml)))
        try:
            env = suite.make(name, robots=kw["robots"], has_renderer=False, has_offscreen_renderer=True,
                             use_camera_obs=False, ignore_done=True, control_freq=20,
                             controller_configs=kw.get("controller_configs"))
            env.reset()
        finally:
            binding_utils.MjSim.from_xml_string = classmethod(orig)
        self.env = _Wrap(env)
        self.arm = kw["robots"][0]
        e = env
        self.sim = e.sim
        m = self.sim.model
        self.site_id = e.robots[0].eef_site_id
        self.instances, self.movable = {}, set()
        for o in getattr(e, "objects", []):
            self.instances[o.name] = o
            self.movable.add(o.name)
        self.root_bid = {k: m.body_name2id(v.root_body) for k, v in self.instances.items()}
        for b in ("bin1", "bin2"):
            try:
                self.root_bid[b] = m.body_name2id(b)
                self.instances[b] = None
            except Exception:
                pass
        bid_to_inst = {b: k for k, b in self.root_bid.items()}
        self.inst_index = {k: i for i, k in enumerate(self.instances)}
        self.geom_inst = np.full(m.ngeom, -1, dtype=np.int64)
        for gi in range(m.ngeom):
            b = int(m.geom_bodyid[gi])
            while b != 0 and b not in bid_to_inst:
                b = int(m.body_parentid[b])
            if b in bid_to_inst:
                self.geom_inst[gi] = self.inst_index[bid_to_inst[b]]
        self.categories = {k: re.sub(r"\d+$", "", canonical_instance(k).lower()) for k in self.instances}
        self.joint_info, self.jnt_ids, self.jnt_qadr, self.jnt_bid = {}, {}, {}, {}

    def language(self, f) -> str:
        return "put each object in the bin"


class _Wrap:
    """Gives a robosuite env the `.env` attribute LiberoReplayer expects."""

    def __init__(self, env):
        self.env = env

    def close(self):
        self.env.close()
