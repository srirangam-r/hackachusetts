"""Debug helper: drive the MCP server's RobotSession directly (no Kiro) to reproduce tool errors.

    debug_session.py <task_id> "<find1,find2>" ["<program>"]
Errors with tracebacks go to <out_dir>/errors.log.
"""
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

from motormem import config
from motormem.types import EpisodeSpec

out = Path(tempfile.mkdtemp(prefix="dbg_session_"))
spec = EpisodeSpec(suite="libero_goal", task_id=int(sys.argv[1]), init_index=int(os.environ.get("INIT", "0")), perception="sam3", motion_source=os.environ.get("MS", "hybrid"),
                   index_path=str(config.INDEX_DIR / "libero_goal.pkl"), out_dir=str(out))
(out / "spec.json").write_text(json.dumps(asdict(spec)))
os.environ["ROBOT_EPISODE"] = str(out / "spec.json")
os.environ["MOTORMEM_BACKEND"] = "real"

from motormem.brain.mcp_server import session_from_env  # noqa: E402

s = session_from_env()
for o in s.call("observe", {"find": sys.argv[2].split(",")}):
    if isinstance(o, str):
        print(o)
if len(sys.argv) > 3:
    for o in s.call("run", {"program": sys.argv[3].replace(";", "\n")}):
        if isinstance(o, str):
            print(o)
try:  # DIAGNOSTIC ONLY (runner-side): success flag
    print("RUNNER success:", s.backend.runner_result().get("success"))
except Exception as e:
    print("runner_result failed", e)
s.shutdown()
print("out:", out)
if (out / "errors.log").exists():
    print((out / "errors.log").read_text()[-2500:])
