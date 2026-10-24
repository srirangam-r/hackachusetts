"""Live demo: type a natural-language sorting task in the browser and watch the robot do it.

    python scripts/live.py                 # opens http://localhost:8765 ; type a prompt, press Run task
    python scripts/live.py --prompt "put the milk on the tray"     # also starts a first run
    python scripts/live.py --name live_184743 --no-browser           # reopen a saved session (its runs replay)

Each run is one episode on the messy desk (fresh scene every run) with the real stack: Kiro brain (headless
kiro-cli, robot MCP tools only), SAM 3, motor memory, OSC. The page (`scripts/live_page.html`) shows

  - the robot camera live (~10 fps), then a replay of the recorded frames once the run ends;
  - Kiro's turns: what it said, the program it sent and each skill's status;
  - what Kiro saw: the two marked camera images and the object list from the robot MCP server;
  - for every motion, the human demo it was copied from (frames cut from the raw LIBERO hdf5, read-only),
    or "geometry" when no demo matched;
  - a timeline of the skill calls coloured by motion source.

Everything is read from the files the episode already writes (`trace.jsonl`, `transcript.log`, `frames.json`,
`frames/`, `rec/`, `result.json`); the page never talks to the sim. An mp4 composite is written to
runs/videos/ after each run.
"""
from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import re
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from motormem import config
from motormem.brain.runner import run_kiro_episode
from motormem.types import Arm, EpisodeSpec

PAGE_FILE = Path(__file__).resolve().with_name("live_page.html")

SCENES = {
    "clutter_desk": {
        "about": "A messy desk with twelve groceries (ketchup, butter, cream cheese, chocolate pudding, alphabet "
                 "soup, tomato sauce, milk, orange juice, salad dressing, BBQ sauce, mac and cheese, popcorn), "
                 "a wooden tray and a cabinet with three drawers.",
        "examples": ["put the milk and the orange juice on the tray",
                     "open the top drawer and put the ketchup in it",
                     "open the drawer, put the butter in it and close it"],
        "items": ["ketchup", "butter", "cream cheese", "chocolate pudding", "alphabet soup", "tomato sauce", "milk",
                  "orange juice", "salad dressing", "bbq sauce", "macaroni and cheese", "popcorn", "wooden tray",
                  "wooden cabinet"],
    },
    "grocery_sort": {
        "about": "A grocery table with two ketchup bottles, two soup cans, butter, milk and more, a basket, "
                 "a wooden tray and a cabinet.",
        "examples": ["put the ketchup bottles in the basket and the soup cans on the tray",
                     "put one of the ketchup bottles in the basket",
                     "put the orange juice inside the top drawer"],
        "items": ["ketchup", "alphabet soup", "butter", "milk", "orange juice", "basket", "wooden tray",
                  "wooden cabinet"],
    },
}
# run options the page offers (EpisodeSpec.motion_source / index_exclude / arm)
MOTIONS = {"hybrid": "Demos, geometry if none match", "index": "Demos only", "primitive": "Geometry only"}
ARMS = [a.value for a in Arm]

SUITE_NAMES = {"libero_90": "LIBERO-90", "libero_10": "LIBERO-10", "libero_goal": "LIBERO-Goal",
               "libero_object": "LIBERO-Object", "libero_spatial": "LIBERO-Spatial"}

_LOG = re.compile(r"\[(INFO|WARN|WARNING|ERROR|DEBUG)\][^\n]*")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_TOOL = re.compile(r"^\[tool\] @robot/(\w+)")
_OBJ = re.compile(r"^\s*\[(\d+)\]\s+(.+?):\s+x=(-?[\d.]+)\s+y=(-?[\d.]+)\s+z=(-?[\d.]+)\s+size\s+(\S+)"
                  r"(?:.*?score\s+([\d.]+))?")
_STATUS = re.compile(r"^(\d+)\.\s+(.*?)\s+->\s+([A-Z_]+):?\s*(.*)$")
_CALL = re.compile(r"^(\w+)\((.*)\)$")
_SCENE_PREFIX = re.compile(r"^[A-Z_]*SCENE\d+_")


# ---------------------------------------------------------------------------- reading an episode
def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return default


def brain_text(transcript: str) -> tuple[list[str], list[str], str]:
    """(started robot tools, text Kiro wrote before each of them, text after the last one)."""
    txt = _LOG.sub("", _ANSI.sub("", transcript))
    tools, chunks, cur = [], [], []
    for line in txt.splitlines():
        m = _TOOL.match(line)
        if m:
            tools.append(m.group(1))
            chunks.append("\n".join(cur))
            cur = []
        elif not line.startswith(("[tool]", "(Use `node")):
            cur.append(line)
    return tools, chunks, "\n".join(cur)


def split_turn_text(chunk: str) -> dict:
    """Kiro's free text for one turn -> {"say": [...], "plan": [...], "thought": "..."}."""
    say, plan, thought, in_plan = [], [], [], False
    for raw in chunk.splitlines():
        line = raw.strip()
        if line.startswith("# Plan"):
            in_plan = True
        elif line.startswith("# Thought"):
            in_plan = False
            thought.append(line.split(":", 1)[1].strip() if ":" in line else "")
        elif in_plan and line:
            plan.append(re.sub(r"^(\d+[.)]|[-*])\s*", "", line))
        elif in_plan and plan:
            in_plan = False
        elif line and not line.startswith("starting the robot"):
            say.append(line)
    return {"say": say, "plan": plan, "thought": " ".join(t for t in thought if t)}


def parse_objects(text: str) -> tuple[list[dict], list[str]]:
    """Object lines of an observation report -> marks, plus the other lines (gripper, not found, ...)."""
    objs, notes = [], []
    for line in (text or "").splitlines():
        m = _OBJ.match(line)
        if m:
            mark, label, x, y, z, size, score = m.groups()
            objs.append({"mark": int(mark), "label": label, "x": float(x), "y": float(y), "z": float(z),
                         "size": size, "score": float(score) if score else None})
        elif line.strip() and not line.startswith(("Status:", "objects (")) and not _STATUS.match(line.strip()):
            notes.append(line.strip())
    return objs, notes


def describe_call(call: str, marks: dict[int, str]) -> str:
    """`place(3, "in")` -> `place in drawer`, using the marks Kiro saw when it wrote the call."""
    m = _CALL.match(call.strip())
    if not m:
        return call
    name, args = m.group(1), [a.strip().strip("\"'") for a in m.group(2).split(",") if a.strip()]
    pos = [a for a in args if "=" not in a]
    kw = dict(a.split("=", 1) for a in args if "=" in a)
    obj = marks.get(int(pos[0]), f"mark {pos[0]}") if pos and pos[0].lstrip("-").isdigit() else ""
    if "part" in kw:
        obj = f"{kw['part'].strip(chr(34) + chr(39))} of {obj}"
    if name == "place":
        return f"place {pos[1] if len(pos) > 1 else 'on'} {obj}"
    if name == "actuate":
        return f"{(pos[1] if len(pos) > 1 else 'actuate').replace('_', ' ')} {obj}"
    if name == "look":
        return f"look at {obj}"
    if name == "home":
        return "go home"
    return f"{name} {obj}".strip()


def motion_source(segment_id: str | None) -> dict:
    """A recorder segment id -> where the motion came from, in words."""
    parts = (segment_id or "").split("/")
    if len(parts) < 4 or not parts[2].startswith("demo_"):
        return {"kind": "geometry"}
    scene = re.match(r"^([A-Z_]*SCENE\d+)_", parts[1])
    return {"kind": "demo", "id": segment_id, "suite": SUITE_NAMES.get(parts[0], parts[0]),
            "task": _SCENE_PREFIX.sub("", parts[1]).replace("_", " "),
            "scene": scene.group(1).replace("_", " ").lower().replace("scene", "scene ") if scene else "",
            "demo": parts[2].replace("demo_", ""), "segment": re.sub(r"\d+$", "", parts[3])}


def episode_view(ep: Path, running: bool) -> dict:
    """Everything the page shows about one episode, from the files it has written so far."""
    spec = _read_json(ep / "spec.json", {})
    result = _read_json(ep / "result.json", {}) or {}
    trace = []
    if (ep / "trace.jsonl").exists():
        for line in (ep / "trace.jsonl").read_text(errors="replace").splitlines():
            try:
                trace.append(json.loads(line))
            except ValueError:
                pass                                         # a half-written last line
    transcript = (ep / "transcript.log").read_text(errors="replace") if (ep / "transcript.log").exists() else ""
    started, chunks, tail = brain_text(transcript)
    rec = _read_json(ep / "frames.json", {}) or {}
    rec_calls = rec.get("calls", [])
    n_rec = sum(1 for _ in (ep / "rec").glob("*.jpg")) if (ep / "rec").is_dir() else 0

    def images(i: int) -> dict | None:
        a, w = ep / "frames" / f"call{i:02d}_agentview.png", ep / "frames" / f"call{i:02d}_wrist.png"
        return {"front": f"{ep.name}/frames/{a.name}", "wrist": f"{ep.name}/frames/{w.name}"} if a.exists() else None

    def calls_of(run_no: int, marks: dict[int, str]) -> list[dict]:
        out = []
        for k, c in enumerate(rec_calls):
            if c.get("run") == run_no:
                out.append({"k": k, "call": c["call"], "label": describe_call(c["call"], marks),
                            "status": c.get("status"), "first": c.get("first", 0), "last": c.get("last", 0),
                            "source": motion_source(c.get("segment_id"))})
        return out

    turns, marks, run_no, seen = [], {}, 0, None
    for r in trace:
        i = int(r.get("i", len(turns) + 1))
        t = {"i": i, "tool": r.get("tool"), "dt": r.get("dt"), "t": r.get("t"),
             "brain": split_turn_text(chunks[i - 1] if 0 < i <= len(chunks) else ""),
             "saw": seen, "images": images(i)}
        objs, notes = parse_objects(r.get("text", ""))
        if r.get("tool") == "observe":
            t["find"] = (r.get("args") or {}).get("find") or []
        elif r.get("tool") == "run":
            if r.get("program_error"):
                t["error"] = r["program_error"]
                t["program"] = [(r.get("args") or {}).get("program", "")]
            else:
                run_no += 1
                t["run"] = run_no
                lines = [_STATUS.match(s) for s in r.get("status_lines") or []]
                t["steps"] = [{"call": m.group(2), "label": describe_call(m.group(2), marks), "status": m.group(3),
                               "message": m.group(4).replace(" [fallback]", "")} if m else {"call": "", "status": ""}
                              for m in lines]
                t["calls"] = calls_of(run_no, marks)
        elif r.get("tool") == "done":
            t["summary"] = (r.get("args") or {}).get("summary", "")
        if objs:
            t["objects"], t["notes"] = objs, notes
            marks = {o["mark"]: o["label"] for o in objs}
        if t["images"]:
            seen = i
        turns.append(t)

    finished = bool(result.get("ended_by") or result.get("error")) or not running
    pending = None
    if not finished and len(started) > len(trace):
        k = len(trace)
        pending = {"i": k + 1, "tool": started[k], "brain": split_turn_text(chunks[k]), "saw": seen}
        if started[k] == "run":
            pending["run"] = run_no + 1
            pending["calls"] = calls_of(run_no + 1, marks)
    if finished:
        phase = "done" if result.get("ended_by") else "stopped"
    elif not trace and not started:
        phase = "starting"
    elif pending:
        phase = {"observe": "looking", "run": "moving", "done": "finishing"}.get(pending["tool"], "moving")
    else:
        phase = "thinking"

    calls = [{**c, "k": k, "label": c["call"], "source": motion_source(c.get("segment_id"))}
             for k, c in enumerate(rec_calls)]
    by_k = {c["k"]: c["label"] for t in turns + ([pending] if pending else []) for c in t.get("calls", [])}
    for c in calls:
        c["label"] = by_k.get(c["k"], c["call"])
    statuses = [c.get("status") for c in calls]
    summary = None
    if finished:
        summary = {"seconds": round(result.get("server_uptime_s") or 0),
                   "tool_calls": result.get("tool_calls", len(trace)),
                   "skills_ok": sum(s == "OK" for s in statuses), "skills": len(statuses),
                   "from_demos": sum(c["source"]["kind"] == "demo" for c in calls),
                   "from_geometry": sum(c["source"]["kind"] == "geometry" for c in calls),
                   "sim_steps": result.get("sim_steps"), "ended_by": result.get("ended_by"),
                   "error": result.get("error"), "kiro_summary": result.get("done_summary") or ""}
    return {"name": ep.name, "instruction": spec.get("instruction", ""), "arm": spec.get("arm"),
            "motion": spec.get("motion_source", "hybrid"),
            "forget": (spec.get("index_exclude") or {}).get("objects", []),
            "phase": phase, "turns": turns, "pending": pending, "final": " ".join(split_turn_text(tail)["say"]),
            "calls": calls, "n_frames": max(len(rec.get("frames", [])), n_rec), "summary": summary}


# ---------------------------------------------------------------------------- human-demo clips
class DemoClips:
    """Frames of the human demo a motion was copied from, as one horizontal JPEG strip per segment
    (128 px frames over the segment's recorded `t_range`). Reads the raw LIBERO hdf5 read-only."""

    MAX_FRAMES = 40

    def __init__(self, index_path: Path, cache: Path):
        self.cache, self.lock = cache, threading.Lock()
        self.t_range: dict[str, tuple[int, int]] = {}
        self.stats: dict | None = None
        try:
            from motormem.memory import MotorMemory
            segs = MotorMemory.load(index_path).segments
            self.t_range = {s.segment_id: tuple(s.meta["t_range"]) for s in segs if s.meta and "t_range" in s.meta}
            self.stats = {"motions": len(segs), "demos": len({s.demo_id for s in segs}),
                          "objects": len({s.object_name for s in segs}),
                          "tasks": len({s.segment_id.split("/")[1] for s in segs})}
        except Exception as e:  # noqa: BLE001
            print(f"(motor memory index unavailable for demo clips: {e})")

    def strip(self, segment_id: str) -> Path | None:
        src = motion_source(segment_id)
        if src["kind"] != "demo":
            return None
        out = self.cache / (hashlib.sha1(segment_id.encode()).hexdigest()[:16] + ".jpg")
        with self.lock:
            if out.exists():
                return out
            suite, task, demo = segment_id.split("/")[:3]
            fn = config.LIBERO_HDF5_DIR / suite / f"{task}_demo.hdf5"
            if not fn.exists():
                return None
            import h5py
            import numpy as np
            from PIL import Image
            with h5py.File(fn, "r") as f:                    # read-only
                d = f[f"data/{demo}/obs/agentview_rgb"]
                t0, t1 = self.t_range.get(segment_id, (0, d.shape[0] - 1))
                t0, t1 = max(0, min(t0, d.shape[0] - 1)), max(0, min(t1, d.shape[0] - 1))
                keep = np.unique(np.linspace(t0, t1, min(self.MAX_FRAMES, t1 - t0 + 1)).round().astype(int))
                imgs = d[t0:t1 + 1][keep - t0][:, ::-1]      # rows stored OpenGL-flipped
            self.cache.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.concatenate(list(imgs), axis=1)).save(out, quality=88)
        return out


# ---------------------------------------------------------------------------- runs
class Live:
    def __init__(self, root: Path, arm: str, model: str, max_steps: int, timeout: float, scene: str = "clutter_desk"):
        self.root, self.arm, self.model, self.max_steps, self.timeout = root, arm, model, max_steps, timeout
        self.scene = scene
        self.lock = threading.Lock()
        self.running = False
        self.ep: str | None = None
        self.prompt = ""
        self.t0 = 0.0
        self.summary = ""
        self.n = len(list(root.glob("ep_*")))              # a reopened session keeps its earlier runs
        self.memory: dict | None = None

    def state(self) -> dict:
        return {"running": self.running, "ep": self.ep, "prompt": self.prompt, "summary": self.summary,
                "elapsed": int(time.time() - self.t0) if self.running else 0,
                "scene": self.scene, "arm": self.arm, "model": self.model, "memory": self.memory,
                "arms": ARMS, "motions": MOTIONS, **SCENES.get(self.scene, {"about": "", "examples": [], "items": []}),
                "episodes": self.episodes()}

    def episodes(self) -> list[dict]:
        out = []
        for ep in sorted(self.root.glob("ep_*")):
            spec, res = _read_json(ep / "spec.json", {}) or {}, _read_json(ep / "result.json", {}) or {}
            st = res.get("skill_statuses") or []
            out.append({"name": ep.name, "instruction": spec.get("instruction", ""),
                        "skills_ok": sum(s == "OK" for s in st), "skills": len(st),
                        "running": self.running and ep.name == self.ep,
                        "video": (config.RUNS_DIR / "videos" / f"{self.root.name}_{ep.name}.mp4").exists()})
        return out

    def start(self, prompt: str, arm: str | None = None, motion: str = "hybrid", forget: str = "") -> str | None:
        arm = arm if arm in ARMS else self.arm
        motion = motion if motion in MOTIONS else "hybrid"
        forget_objs = [f.strip()[:40] for f in (forget or "").split(",") if f.strip()][:4]
        with self.lock:
            if self.running:
                return "a run is already in progress"
            self.running, self.prompt, self.summary, self.t0 = True, prompt, "", time.time()
            name = f"ep_{self.n:03d}"
            self.n += 1
        ep = self.root / name
        ep.mkdir(parents=True, exist_ok=True)
        spec = EpisodeSpec(suite=self.scene, task_id=0, init_index=0, arm=arm, motion_source=motion,
                           perception="sam3", index_path=str(config.INDEX_DIR / "live_all.pkl"),
                           index_exclude={"objects": forget_objs} if forget_objs else {},
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


def make_handler(live: Live, clips: DemoClips | None = None):
    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(live.root), **k)

        def log_message(self, *a, **k):
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj: dict) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json", {"Cache-Control": "no-store"})

        def end_headers(self):
            if self.path.split("?")[0].endswith(("latest.jpg", ".json", ".log", ".jsonl")):
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ("/", "/index.html"):
                self._send(200, PAGE_FILE.read_bytes(), "text/html; charset=utf-8", {"Cache-Control": "no-store"})
            elif u.path == "/state":
                self._json(200, live.state())
            elif u.path == "/episode":
                ep = live.root / Path(q.get("ep", "")).name
                if not q.get("ep") or not (ep / "spec.json").exists():
                    return self._json(404, {"error": "no such run"})
                self._json(200, episode_view(ep, live.running and ep.name == live.ep))
            elif u.path == "/demo":
                p = clips.strip(q.get("seg", "")) if clips else None
                if p is None:
                    return self._json(404, {"error": "no demo frames for this motion"})
                self._send(200, p.read_bytes(), "image/jpeg", {"Cache-Control": "max-age=86400"})
            elif u.path.startswith("/video/"):
                f = config.RUNS_DIR / "videos" / f"{live.root.name}_{Path(u.path).stem}.mp4"
                if not f.exists():
                    return self._json(404, {"error": "the video is still being written"})
                self._send(200, f.read_bytes(), "video/mp4",
                           {"Content-Disposition": f'attachment; filename="{f.name}"'})
            else:
                super().do_GET()

        def do_POST(self):
            if urlparse(self.path).path != "/run":
                return self._json(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                body = {}
            prompt = str(body.get("prompt", "")).strip()[:300]
            if not prompt:
                return self._json(400, {"error": "Type a task first."})
            err = live.start(prompt, arm=str(body.get("arm") or ""), motion=str(body.get("motion") or "hybrid"),
                             forget=str(body.get("forget") or ""))
            return self._json(409 if err else 200, {"error": err} if err else {"ok": True})

    return H


def scene_preview(root: Path, scene: str = "clutter_desk") -> None:
    """Render the starting scene once so the page shows it before the first run."""
    if (root / "scene.jpg").exists():
        return
    try:
        from PIL import Image

        from motormem.env import make_env
        env = make_env(EpisodeSpec(suite=scene, task_id=0, init_index=0))
        env.reset()
        Image.fromarray(env.render("agentview", 512)).save(root / "scene.jpg", quality=88)
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"(scene preview unavailable: {e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="", help="optional: start a first run with this prompt")
    ap.add_argument("--scene", default="clutter_desk", choices=sorted(SCENES))
    ap.add_argument("--arm", default="Panda")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--name", default=time.strftime("live_%H%M%S"),
                    help="session folder under runs/live/; an existing one is reopened with its runs")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    root = config.RUNS_DIR / "live" / a.name
    root.mkdir(parents=True, exist_ok=True)
    scene_preview(root, a.scene)
    live = Live(root, a.arm, a.model, a.max_steps, a.timeout, a.scene)
    clips = DemoClips(config.INDEX_DIR / "live_all.pkl", root / "demo_clips")
    live.memory = clips.stats
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(live, clips))
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
