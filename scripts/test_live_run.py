"""Smoke-test the live demo's run path without the web server: Live.start() -> real episode -> summary."""
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live", ROOT / "scripts" / "live.py")
live_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_mod)

from motormem import config  # noqa: E402

root = config.RUNS_DIR / "live" / "live_ui_test"
root.mkdir(parents=True, exist_ok=True)
live_mod.scene_preview(root)
print("scene preview:", (root / "scene.jpg").exists())
lv = live_mod.Live(root, "Panda", "claude-opus-5.5", 4000, 600)
err = lv.start(sys.argv[1] if len(sys.argv) > 1 else "put the soup cans on the tray")
print("start error:", err, "| busy check:", lv.start("second"))
while lv.running:
    time.sleep(3)
print("state:", lv.state())
print("latest.jpg:", (root / lv.ep / "latest.jpg").exists())
