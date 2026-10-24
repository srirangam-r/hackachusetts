"""Composite video of one recorded episode (run the runner with `--record` first).

    python scripts/make_video.py runs/<name>/ep_000 [--out x.mp4] [--index data/index/libero_goal.pkl]

1280x720 @ 20 fps:
  left          the 512 px agentview sim view (`rec/*.jpg`) with the skill call in progress;
  right top     the brain's latest turn: `# Plan:` / `# Thought:` lines from `transcript.log` and the
                program it sent (from `trace.jsonl`);
  right bottom  "what it's copying": the source demo's agentview frames (raw LIBERO hdf5, read-only) over
                the segment's recorded `t_range`, or "hand-written primitive" for fallback motions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motormem import config  # noqa: E402

W, H = 1280, 720
BG, FG, DIM, ACC, WARN = (18, 20, 24), (235, 235, 235), (150, 155, 165), (110, 200, 255), (255, 170, 60)
_LOG = re.compile(r"\[(INFO|WARN|WARNING|ERROR|DEBUG)\][^\n]*")
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def font(size: int, bold: bool = False):
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"),
                 "/usr/share/fonts/truetype/dejavu/" + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def wrap(text: str, f, width: int) -> list[str]:
    out: list[str] = []
    for para in text.split("\n"):
        line = ""
        for w in para.split(" "):
            cand = (line + " " + w) if line else w
            if f.getlength(cand) <= width:
                line = cand
            else:
                if line:
                    out.append(line)
                line = w
        out.append(line)
    return out


# ---------------------------------------------------------------------------- brain turns
def transcript_chunks(path: Path) -> list[str]:
    """Text the brain wrote before each `@robot/<tool>` call; chunk k belongs to tool call k+1."""
    if not path.exists():
        return []
    txt = _LOG.sub("", _ANSI.sub("", path.read_text(errors="replace")))
    chunks, cur = [], []
    for line in txt.splitlines():
        if line.startswith("[tool] @robot/"):
            chunks.append("\n".join(cur))
            cur = []
        elif not line.startswith(("[tool]", "(Use `node")):
            cur.append(line)
    return chunks


def brain_lines(chunk: str) -> tuple[list[str], list[str]]:
    """(plan lines, thought lines) of one turn."""
    plan, thought, in_plan = [], [], False
    for raw in chunk.splitlines():
        line = raw.strip()
        if line.startswith("# Plan"):
            in_plan, plan = True, [line]
        elif line.startswith("# Thought"):
            in_plan = False
            thought.append(line)
        elif in_plan and line:
            plan.append(line)
        elif in_plan and not line and len(plan) > 1:
            in_plan = False
    return plan, thought


def turn_texts(ep: Path, rec_calls: list[dict]) -> dict[int, str]:
    """Run index (recorder `run`) -> panel text."""
    trace = [json.loads(l) for l in (ep / "trace.jsonl").read_text().splitlines() if l.strip()] \
        if (ep / "trace.jsonl").exists() else []
    chunks = transcript_chunks(ep / "transcript.log")
    runs = [t for t in trace if t.get("tool") == "run"]
    first_call = {}
    for c in rec_calls:
        first_call.setdefault(c["run"], c["call"])
    out, ptr, plan = {}, 0, []
    for r in sorted(first_call):
        while ptr < len(runs) and first_call[r] not in runs[ptr].get("program", []):
            ptr += 1
        if ptr >= len(runs):
            break
        t = runs[ptr]
        p, th = brain_lines(chunks[t["i"] - 1]) if 0 < t["i"] <= len(chunks) else ([], [])
        plan = p or plan
        body = plan + [""] + th + ["", "Program:"] + ["> " + c for c in t.get("program", [])]
        out[r] = "\n".join(body).strip()
        ptr += 1
    return out


# ---------------------------------------------------------------------------- demo frames
class DemoFrames:
    def __init__(self, index_path: Path | None):
        self.meta: dict[str, dict] = {}
        if index_path and index_path.exists():
            from motormem.memory import MotorMemory
            self.meta = {s.segment_id: s.meta for s in MotorMemory.load(index_path).segments}
        self._cache: dict[tuple, np.ndarray] = {}

    def frames(self, seg_id: str) -> tuple[np.ndarray, int, int] | None:
        """(T,128,128,3) agentview frames of the source demo and the segment's [t0, t1]."""
        parts = seg_id.split("/")
        if len(parts) < 4 or not parts[2].startswith("demo_"):
            return None
        suite, task, demo = parts[0], parts[1], parts[2]
        key = (suite, task, demo)
        if key not in self._cache:
            import h5py
            fn = config.RAW_DIR / "libero_hdf5" / suite / f"{task}_demo.hdf5"
            if not fn.exists():
                return None
            with h5py.File(fn, "r") as f:                     # read-only
                self._cache[key] = f[f"data/{demo}/obs/agentview_rgb"][()][:, ::-1]   # rows stored OpenGL-flipped
        imgs = self._cache[key]
        tr = self.meta.get(seg_id, {}).get("t_range")
        t0, t1 = (int(tr[0]), int(tr[1])) if tr else (len(imgs) // 2, len(imgs) // 2)
        return imgs, min(t0, len(imgs) - 1), min(t1, len(imgs) - 1)


# ---------------------------------------------------------------------------- compositing
def compose(sim: Image.Image, instruction: str, call: dict | None, brain: str, demo: Image.Image | None,
            F: dict) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    d.text((24, 14), wrap(instruction, F["h"], W - 48)[0], fill=FG, font=F["h"])
    # left: sim
    d.text((24, 58), "Simulation (agentview)", fill=DIM, font=F["s"])
    img.paste(sim.resize((512, 512)), (24, 82))
    if call:
        prim = call.get("segment_id") in (None, "primitive")
        d.text((24, 604), f"Executing: {call['call']}", fill=FG, font=F["b"])
        src = "hand-written primitive" if prim else "motor memory (demo segment)"
        d.text((24, 634), f"Motion source: {src}", fill=WARN if prim else ACC, font=F["s"])
        if call.get("status"):
            d.text((24, 660), f"Result: {call['status']}", fill=DIM, font=F["s"])
    x0 = 560
    d.line([(x0 - 12, 58), (x0 - 12, H - 20)], fill=(50, 55, 62), width=2)
    # right top: brain
    d.text((x0, 58), "Brain (frontier model, latest turn)", fill=DIM, font=F["s"])
    y = 84
    for line in wrap(brain or "(thinking)", F["t"], W - x0 - 24)[:13]:
        d.text((x0, y), line, fill=ACC if line.startswith("> ") else FG, font=F["t"])
        y += 21
    # right bottom: what it's copying
    yb = 372
    d.line([(x0, yb - 10), (W - 24, yb - 10)], fill=(50, 55, 62), width=2)
    d.text((x0, yb), "What it's copying", fill=DIM, font=F["s"])
    if call is None:
        d.text((x0, yb + 40), "(between skill calls)", fill=DIM, font=F["t"])
    elif demo is None:
        d.rectangle([x0, yb + 28, x0 + 300, yb + 328], outline=WARN, width=2)
        d.text((x0 + 150, yb + 178), "hand-written primitive", fill=WARN, font=F["b"], anchor="mm")
        d.text((x0 + 320, yb + 40), "No demo segment:\nscripted top-down motion\n(fallback / ablation).",
               fill=FG, font=F["t"], spacing=6)
    else:
        img.paste(demo.resize((300, 300), Image.BILINEAR), (x0, yb + 28))
        sid = call["segment_id"].split("/")
        lines = ["Source demo (LIBERO training set):", sid[1].replace("_", " "), "",
                 f"demo: {sid[2]}", f"segment: {sid[3]}", "", "Gripper poses replayed", "relative to the live object."]
        y = yb + 34
        for ln in lines:
            for w in wrap(ln, F["t"], W - x0 - 340):
                d.text((x0 + 318, y), w, fill=FG, font=F["t"])
                y += 21
    return img


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode_dir")
    ap.add_argument("--out", default="")
    ap.add_argument("--index", default="", help="index pickle with segment meta (default: from spec.json)")
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args(argv)
    import imageio.v2 as imageio

    ep = Path(a.episode_dir).resolve()
    doc = json.loads((ep / "frames.json").read_text())
    spec = json.loads((ep / "spec.json").read_text()) if (ep / "spec.json").exists() else {}
    idx = Path(a.index or spec.get("index_path") or config.INDEX_DIR / "libero_goal.pkl")
    out = Path(a.out) if a.out else config.RUNS_DIR / "videos" / f"{ep.parent.name}_{ep.name}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    F = {"h": font(22, True), "b": font(19, True), "t": font(16), "s": font(15)}
    calls = doc["calls"]
    brains = turn_texts(ep, calls)
    demos = DemoFrames(idx)
    w = imageio.get_writer(out, fps=a.fps, codec="libx264", quality=7, macro_block_size=16)
    last_call = None
    for j, fr in enumerate(doc["frames"]):
        c = calls[fr["call"]] if 0 <= fr["call"] < len(calls) else None
        last_call = c or last_call
        demo = None
        if c and c.get("segment_id") not in (None, "primitive"):
            got = demos.frames(c["segment_id"])
            if got:                                   # play the segment's demo frames in step with the call
                imgs, t0, t1 = got
                u = min(1.0, max(0.0, (j - c["first"]) / max(1, c["last"] - c["first"])))
                demo = Image.fromarray(imgs[t0 + round((t1 - t0) * u)])
        brain = brains.get((last_call or {}).get("run"), "")
        frame = compose(Image.open(ep / fr["file"]).convert("RGB"), spec.get("instruction", ""), c, brain, demo, F)
        w.append_data(np.asarray(frame))
    w.close()
    print(out)
    return out


if __name__ == "__main__":
    main()
