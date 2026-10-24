"""Render a scene's starting state (agentview + what the brain sees) and print object positions (diagnostic)."""
import sys
import numpy as np
from PIL import Image
from motormem.env import make_env
from motormem.types import EpisodeSpec
suite, out = sys.argv[1], sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
e = make_env(EpisodeSpec(suite=suite, task_id=0, init_index=0, seed=seed))
o = e.reset()
Image.fromarray(o.cameras["agentview"].rgb).save(out)
base = o.T_world_base
P = e.privileged_object_poses()
print("POS", {k: tuple(int(x) for x in np.round((np.linalg.inv(base) @ v)[:3, 3] * 100)) for k, v in P.items() if "table" not in k})
