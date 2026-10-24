"""Episode video recording (media only; never seen by the brain).

`Recorder(env, out_dir)` wraps `env.step` and keeps every `every`-th sim step as a 512 px agentview
JPEG in `out_dir/rec/`, tagged with the skill call in progress. The backend brackets each call with
`begin(call_text, run_idx)` / `end(segment_id, status)`; the segment id is only known after the call,
so it is patched onto that call's frames in `frames.json`:

    {"every": 2, "size": 512, "frames": [{"file": "rec/000000.jpg", "step": 0, "call": 0}, ...],
     "calls": [{"call": "pick(3)", "run": 1, "segment_id": "libero_goal/.../grasp0" | "primitive",
                "status": "OK", "first": 0, "last": 41}, ...]}

`call` is -1 for frames outside any skill call. Frames are written as they arrive and the json after
every call, so a killed episode still leaves a usable recording.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


class Recorder:
    def __init__(self, env, out_dir: str | Path, every: int = 2, size: int = 512):
        self.env, self.every, self.size = env, max(1, int(every)), int(size)
        self.out = Path(out_dir)
        (self.out / "rec").mkdir(parents=True, exist_ok=True)
        self.frames: list[dict] = []
        self.calls: list[dict] = []
        self.current = -1
        self._n = 0
        self._step = env.step
        env.step = self._wrapped
        self.grab(0)

    def _wrapped(self, action, *a, **kw):
        obs = self._step(action, *a, **kw)
        self._n += 1
        if self._n % self.every == 0:
            self.grab(int(getattr(obs, "step", self._n)))
        return obs

    def grab(self, step: int) -> None:
        f = f"rec/{len(self.frames):06d}.jpg"
        img = Image.fromarray(np.asarray(self.env.render("agentview", self.size)))
        img.save(self.out / f, quality=88)
        tmp = self.out / "latest.tmp.jpg"        # for the live viewer (scripts/live.py)
        img.save(tmp, quality=80)
        tmp.replace(self.out / "latest.jpg")
        self.frames.append({"file": f, "step": step, "call": self.current})

    def begin(self, call_text: str, run_idx: int) -> None:
        self.current = len(self.calls)
        self.calls.append({"call": call_text, "run": run_idx, "segment_id": None, "status": None,
                           "first": len(self.frames), "last": len(self.frames) - 1})

    def end(self, segment_id: str | None, status: str) -> None:
        if self.current < 0:
            return
        c = self.calls[self.current]
        c.update(segment_id=segment_id or "primitive", status=status, last=len(self.frames) - 1)
        self.current = -1
        self.save()

    def save(self) -> None:
        doc = {"every": self.every, "size": self.size, "frames": self.frames, "calls": self.calls}
        tmp = self.out / "frames.json.tmp"
        tmp.write_text(json.dumps(doc))
        tmp.replace(self.out / "frames.json")

    def close(self) -> None:
        self.env.step = self._step
        self.save()
