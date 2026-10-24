"""Debug helper: build a RealBackend for one LIBERO episode and print observe() output (or the traceback)."""
import sys
import traceback

from motormem import config
from motormem.brain.backend import RealBackend
from motormem.types import EpisodeSpec

suite, task, init = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
finds = sys.argv[4].split(",")
b = RealBackend(EpisodeSpec(suite=suite, task_id=task, init_index=init, perception="sam3",
                            index_path=str(config.INDEX_DIR / "libero_goal.pkl")))
try:
    print(b.observe(finds)[1])
except Exception:
    traceback.print_exc()
