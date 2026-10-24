"""Live demo: type a natural-language sorting task in the browser and watch the robot do it.

    python scripts/live.py                 # opens http://localhost:8765 ; type a prompt, press Run
    python scripts/live.py --prompt "put the ketchup bottles in the basket"   # also starts a first run

Each run is one episode on the cluttered grocery table (fresh scene every run) with the real stack: Kiro brain
(headless kiro-cli, robot MCP tools only), SAM 3, motor memory, OSC. The page shows the robot camera (~10 fps)
and the brain's turns and skill statuses live; an mp4 composite is written to runs/videos/ after each run.
"""
from __future__ import annotations

import argparse
import http.server
import json
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

from motormem import config
from motormem.brain.runner import run_kiro_episode
from motormem.types import EpisodeSpec

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Motor Memory: live</title>
<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;margin:0;display:flex;gap:16px;padding:16px}
#cam{width:640px;height:640px;border-radius:8px;background:#000;object-fit:contain}#side{flex:1;min-width:0}
h1{font-size:18px;margin:0 0 10px}form{display:flex;gap:8px;margin-bottom:8px}
input{flex:1;font-size:18px;padding:10px;border-radius:6px;border:1px solid #444;background:#1b1b1b;color:#eee}
button{font-size:18px;padding:10px 18px;border-radius:6px;border:0;background:#2b7;color:#000;cursor:pointer}
button:disabled{background:#555;color:#999;cursor:default}#status{color:#9cf;margin:0 0 8px;min-height:1.2em}
pre{white-space:pre-wrap;font-size:13px;line-height:1.35;background:#1b1b1b;padding:12px;border-radius:8px;
height:520px;overflow:auto}.ok{color:#7d7}.bad{color:#f88}.hint{color:#888;font-size:13px}</style></head><body>
<img id="cam" alt="robot camera"><div id="side">
<h1>Motor Memory - Kiro brain + SAM 3 + demo-retrieved motion</h1>
<form id="f"><input id="p" autocomplete="off" placeholder="e.g. put the ketchup bottles in the basket and the soup cans on the tray">
<button id="go">Run</button></form>
<div class="hint">Objects on the table: ketchup x2, soup can x2, milk x2, butter x2, orange juice, a basket and a wooden tray.</div>
<div id="status">idle</div><pre id="log"></pre></div><script>
const cam=document.getElementById('cam'),log=document.getElementById('log'),st=document.getElementById('status'),
go=document.getElementById('go'),inp=document.getElementById('p');let ep=null;
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const p=inp.value.trim();if(!p)return;
go.disabled=true;const r=await fetch('/run',{method:'POST',body:JSON.stringify({prompt:p})});
const j=await r.json();if(!r.ok){st.textContent=j.error;go.disabled=false}};
function keep(l){return l.trim()&&!/^\\[(INFO|ERROR|WARN)\\]|ExperimentalWarning|trace-warnings|status: (InProgress|Completed)|Use `node/.test(l)}
function fmt(t){return t.split('\\n').filter(keep).map(l=>l.replace(/&/g,'&amp;').replace(/</g,'&lt;')
.replace(/-&gt; (OK[^\\n]*)/,'-&gt; <span class="ok">$1</span>')
.replace(/-&gt; ((GRASP_FAILED|STALL|TIMEOUT|IK_FAILURE|NO_CANDIDATE|PERCEPTION_FAILED|NOT_HOLDING|ERROR)[^\\n]*)/,'-&gt; <span class="bad">$1</span>')).join('\\n')}
async function tick(){try{const s=await (await fetch('/state?'+Date.now())).json();
go.disabled=s.running;ep=s.ep;
st.textContent=s.running?('running: "'+s.prompt+'"  ('+s.elapsed+' s)'):(s.ep?('done: "'+s.prompt+'"  '+(s.summary||'')):'idle - type a prompt and press Run');
if(ep){cam.src=ep+'/latest.jpg?'+Date.now();const t=await (await fetch(ep+'/transcript.log?'+Date.now())).text();
const h=fmt(t);if(h!==log.dataset.h){log.innerHTML=h;log.dataset.h=h;log.scrollTop=log.scrollHeight}}
else{cam.src='scene.jpg'}}catch(e){}}
setInterval(tick,300);</script></body></html>"""


class Live:
    def __init__(self, root: Path, arm: str, model: str, max_steps: int, timeout: float):
        self.root, self.arm, self.model, self.max_steps, self.timeout = root, arm, model, max_steps, timeout
        self.lock = threading.Lock()
        self.running = False
        self.ep: str | None = None
        self.prompt = ""
        self.t0 = 0.0
        self.summary = ""
        self.n = 0

    def state(self) -> dict:
        return {"running": self.running, "ep": self.ep, "prompt": self.prompt, "summary": self.summary,
                "elapsed": int(time.time() - self.t0) if self.running else 0}

    def start(self, prompt: str) -> str | None:
        with self.lock:
            if self.running:
                return "a run is already in progress"
            self.running, self.prompt, self.summary, self.t0 = True, prompt, "", time.time()
            name = f"ep_{self.n:03d}"
            self.n += 1
        ep = self.root / name
        ep.mkdir(parents=True, exist_ok=True)
        spec = EpisodeSpec(suite="grocery_sort", task_id=0, init_index=0, arm=self.arm, motion_source="hybrid",
                           perception="sam3", index_path=str(config.INDEX_DIR / "libero90_study_living.pkl"),
                           max_steps=self.max_steps, out_dir=str(ep.resolve()), model=self.model)
        (ep / "spec.json").write_text(json.dumps({**asdict(spec), "instruction": prompt, "backend": "real",
                                                  "record": True}, indent=2))
        (ep / "transcript.log").write_text("starting the robot and the Kiro brain ...\n")
        self.ep = name
        threading.Thread(target=self._run, args=(ep,), daemon=True).start()
        return None

    def _run(self, ep: Path) -> None:
        try:
            rec = run_kiro_episode(ep, self.model, "real", self.timeout, keep_sessions=False)
            ok = [s for s in (rec.get("skill_statuses") or [])]
            self.summary = (f"{int(time.time() - self.t0)} s, {rec.get('tool_calls', 0)} Kiro tool calls, "
                            f"{sum(s == 'OK' for s in ok)}/{len(ok)} skill calls OK")
            out = config.RUNS_DIR / "videos" / f"{self.root.name}_{ep.name}.mp4"
            subprocess.Popen([sys.executable, str(config.REPO_ROOT / "scripts" / "make_video.py"), str(ep),
                              "--out", str(out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[{ep.name}] {self.summary}; video -> {out}", flush=True)
        except Exception as e:  # noqa: BLE001
            self.summary = f"error: {type(e).__name__}: {e}"
        finally:
            self.running = False


def make_handler(live: Live):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(live.root), **k)

        def log_message(self, *a, **k):
            pass

        def _json(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                body = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/state":
                self._json(200, live.state())
            else:
                super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path != "/run":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                prompt = str(json.loads(self.rfile.read(n) or b"{}").get("prompt", "")).strip()[:300]
            except Exception:
                prompt = ""
            if not prompt:
                return self._json(400, {"error": "empty prompt"})
            err = live.start(prompt)
            return self._json(409 if err else 200, {"error": err} if err else {"ok": True})

    return H


def scene_preview(root: Path) -> None:
    """Render the starting scene once so the page shows it before the first run."""
    try:
        from PIL import Image

        from motormem.env import make_env
        env = make_env(EpisodeSpec(suite="grocery_sort", task_id=0, init_index=0))
        env.reset()
        Image.fromarray(env.render("agentview", 512)).save(root / "scene.jpg", quality=88)
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"(scene preview unavailable: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="", help="optional: start a first run with this prompt")
    ap.add_argument("--arm", default="Panda")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--name", default=time.strftime("live_%H%M%S"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    root = config.RUNS_DIR / "live" / a.name
    root.mkdir(parents=True, exist_ok=True)
    scene_preview(root)
    live = Live(root, a.arm, a.model, a.max_steps, a.timeout)
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(live))
    url = f"http://localhost:{a.port}/"
    print(f"Motor Memory live demo: {url}  (Ctrl+C to stop)\nruns are saved under {root}", flush=True)
    if a.prompt:
        live.start(a.prompt)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
