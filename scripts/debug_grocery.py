"""Drive the grocery scene without Kiro: observe nouns, then run a program on the printed marks.

    debug_grocery.py "<noun,noun,...>" ['<call;call>']
"""
import sys

from motormem import config
from motormem.brain.backend import RealBackend
from motormem.brain.program import parse_program
from motormem.types import EpisodeSpec

import os
b = RealBackend(EpisodeSpec(suite=os.environ.get("SUITE", "clutter_desk"), task_id=0, init_index=0, perception="sam3",
                            index_path=str(config.INDEX_DIR / "live_all.pkl"), max_steps=4000))
print(b.observe(sys.argv[1].split(","))[1])
if len(sys.argv) > 2 and sys.argv[2]:
    lines, _, text = b.run(parse_program(sys.argv[2].replace(";", "\n")))
    print("\n".join(lines))
    print(text.split("objects")[-1][:600])
from motormem.env.clutter_desk import object_status  # noqa: E402  (diagnostic only)
print("STATUS", object_status(b.env), {k: round(float(v), 3) for k, v in b.env.privileged_joint_positions().items()})
if len(sys.argv) > 3:
    import numpy as np
    P = b.env.privileged_object_poses()
    base = b.obs.T_world_base
    for k in ("butter_1", "butter_2", "wooden_cabinet_1"):
        print("POSE", k, np.round((np.linalg.inv(base) @ P[k])[:3, 3] * 100, 1))
    sim = b.env._rs.sim
    for site in ("wooden_cabinet_1_top_region",):
        try:
            sid = sim.model.site_name2id(site)
            print("SITE", site, np.round((np.linalg.inv(base) @ np.r_[sim.data.site_xpos[sid], 1])[:3] * 100, 1), sim.model.site_size[sid])
        except Exception as e:
            print("site err", e)
