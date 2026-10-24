"""Live demo: give the robot a natural-language sorting task on a cluttered grocery table and watch it work.

    python scripts/live.py --prompt "put all the ketchup bottles in the basket and the butter on the tray"

Opens a viewer at http://localhost:8765 (robot camera at ~10 fps + the Kiro brain's turns and skill statuses),
runs one episode with the real stack (Kiro brain via headless kiro-cli, SAM 3, motor memory, OSC), then renders
an mp4 composite into runs/videos/. The brain only sees the robot MCP tools; nothing privileged.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path

from motormem import config
from motormem.brain.runner import run_kiro_episode
from motormem.types import EpisodeSpec

VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>Motor Memory: live</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;gap:16px;padding:16px}
#cam{width:640px;height:640px;border-radius:8px;background:#000}#side{flex:1;min-width:0}
h1{font-size:18px;margin:0 0 8px}#prompt{color:#9cf;font-size:20px;margin:0 0 12px}
pre{white-space:pre-wrap;font-size:13px;line-height:1.35;background:#1b1b1b;padding:12px;border-radius:8px;
height:540px;overflow:auto}.ok{color:#7d7}.bad{color:#f88}</style></head><body>
<img id="cam" alt="robot camera"><div id="side"><h1>Motor Memory - Kiro brain, SAM 3, demo-retrieved motion</h1>
<div id="prompt">PROMPT</div><pre id="log">starting...</pre></div><script>
const cam=document.getElementById('cam'),log=document.getElementById('log');
setInterval(()=>{cam.src='ep_000/latest.jpg?'+Date.now()},120);
function keep(l){return l.trim()&&!/^\\[(INFO|ERROR|WARN)\\]|ExperimentalWarning|trace-warnings|status: (InProgress|Completed)|Use `node/.test(l)}
async function poll(){try{const t=await (await fetch('ep_000/transcript.log?'+Date.now())).text();
const html=t.split('\\n').filter(keep).map(l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;')
.replace(/-&gt; (OK[^\\n]*)/,'-&gt; <span class="ok">$1</span>')
.replace(/-&gt; ((GRASP_FAILED|STALL|TIMEOUT|IK_FAILURE|NO_CANDIDATE|PERCEPTION_FAILED|ERROR)[^\\n]*)/,'-&gt; <span class="bad">$1</span>')).join('\\n');
if(html!==log.dataset.h){log.innerHTML=html;log.dataset.h=html;log.scrollTop=log.scrollHeight}}catch(e){}}
setInterval(poll,700);</script></body></html>"""


def serve(root: Path, port: int) -> None:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="put the ketchup bottles in the basket and the soup cans on the tray")
    ap.add_argument("--arm", default="Panda")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--name", default=time.strftime("live_%H%M%S"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    root = config.RUNS_DIR / "live" / a.name
    ep = root / "ep_000"
    ep.mkdir(parents=True, exist_ok=True)
    spec = EpisodeSpec(suite="grocery_sort", task_id=0, init_index=0, arm=a.arm, motion_source="hybrid",
                       perception="sam3", index_path=str(config.INDEX_DIR / "libero90_study_living.pkl"),
                       max_steps=a.max_steps, out_dir=str(ep.resolve()), model=a.model)
    (ep / "spec.json").write_text(json.dumps({**asdict(spec), "instruction": a.prompt, "backend": "real",
                                              "record": True}, indent=2))
    (root / "index.html").write_text(VIEWER.replace("PROMPT", a.prompt.replace("<", "&lt;")))
    serve(root, a.port)
    url = f"http://localhost:{a.port}/"
    print(f"viewer: {url}\nprompt: {a.prompt}\nepisode dir: {ep}", flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    t0 = time.time()
    rec = run_kiro_episode(ep, a.model, "real", a.timeout, keep_sessions=False)
    print(f"done in {time.time() - t0:.0f} s; tool calls {rec.get('tool_calls')}; statuses {rec.get('skill_statuses')}")
    out = config.RUNS_DIR / "videos" / f"{a.name}.mp4"
    subprocess.run([sys.executable, str(config.REPO_ROOT / "scripts" / "make_video.py"), str(ep), "--out", str(out)])
    print(f"video: {out}")
    if not a.no_browser:
        print("viewer stays up for 60 s ...")
        time.sleep(60)


if __name__ == "__main__":
    main()
