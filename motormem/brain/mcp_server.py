"""Robot MCP server (stdio): the only tools the `robot-brain` Kiro agent has (Req 5.1, 5.3, 5.4).

    observe(find)   -> [agentview overlay PNG, wrist overlay PNG, text report]
    run(program)    -> [status lines, agentview PNG, wrist PNG, text report]
    done(summary)   -> EPISODE_OVER

Environment:
    ROBOT_EPISODE        path to an EpisodeSpec JSON (extra keys such as "instruction" are kept aside)
    MOTORMEM_BACKEND     fake | real (default fake)
    ROBOT_MAX_TOOL_CALLS hard ceiling on tool calls per episode (default 10)

Writes into `spec.out_dir`: `trace.jsonl` (one record per tool call), `frames/*.png` and
`result.json` (runner-only; contains `success` from the backend's runner-only check, which is
never sent to the agent). Every outgoing text passes through `sanitize()`.

Run: python -m motormem.brain.mcp_server
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable

from motormem.brain.program import ProgramError, format_call, parse_program
from motormem.config import is_privileged_key
from motormem.types import EpisodeSpec

EPISODE_OVER = "EPISODE_OVER"
_TOKEN = re.compile(r"[A-Za-z0-9_\-]+")


# ---------------------------------------------------------------------------- privacy guard
def _bad_token(tok: str) -> bool:
    parts = [tok, *tok.split("-")]
    return any("success" in p.lower() or is_privileged_key(p) for p in parts if p)


def sanitize(text: str) -> str:
    """Redact any word token that is a privileged key name or mentions success."""
    return _TOKEN.sub(lambda m: "[redacted]" if _bad_token(m.group(0)) else m.group(0), str(text))


def load_spec(path: str | os.PathLike) -> tuple[EpisodeSpec, dict]:
    raw = json.loads(Path(path).read_text())
    names = {f.name for f in fields(EpisodeSpec)}
    spec = EpisodeSpec(**{k: v for k, v in raw.items() if k in names})
    return spec, {k: v for k, v in raw.items() if k not in names}


def _png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------- session
class RobotSession:
    """Transport-independent tool logic. All backend work runs on ONE worker thread (MuJoCo/EGL
    contexts are thread-bound). Tool methods return lists of `str | bytes(PNG)`."""

    def __init__(self, spec: EpisodeSpec | None, backend_factory: Callable[[EpisodeSpec], Any] | None,
                 extras: dict | None = None, max_tool_calls: int = 10, save_frames: bool = True):
        self.t0 = time.time()
        self.spec, self.extras = spec, extras or {}
        self.max_tool_calls = max_tool_calls
        self.save_frames = save_frames
        self.calls = 0
        self.tool_counts: dict[str, int] = {}
        self.program_errors = 0
        self.first_call_t: float | None = None
        self.last_call_t: float | None = None
        self.limit_hit = False
        self.error: str | None = None
        self.out = Path(spec.out_dir) if spec and spec.out_dir else None
        if self.out:
            self.out.mkdir(parents=True, exist_ok=True)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="robot-backend")
        self.backend = None
        self._ready = self.pool.submit(self._build, backend_factory) if spec and backend_factory else None

    def _build(self, factory):
        try:
            self.backend = factory(self.spec)
        except Exception as e:  # reported as a tool error, never crashes the server
            self.error = f"backend failed to start: {type(e).__name__}: {str(e)[:200]}"
        self._write_result()

    def _on_backend(self, fn, *a):
        """Run fn(*a) on the backend thread and wait."""
        return self.pool.submit(fn, *a).result()

    # ------------------------------------------------------------------ bookkeeping
    def _over(self) -> bool:
        return self.limit_hit or (self.backend is not None and self.backend.episode_over())

    def _trace(self, rec: dict) -> None:
        if self.out:
            with open(self.out / "trace.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def _frames(self, imgs) -> list[bytes]:
        pngs = [_png(im) for im in imgs]
        if self.out and self.save_frames:
            d = self.out / "frames"
            d.mkdir(exist_ok=True)
            for k, p in enumerate(pngs):
                (d / f"call{self.calls:02d}_{'agentview' if k == 0 else 'wrist'}.png").write_bytes(p)
        return pngs

    def result(self) -> dict:
        res = {"spec": asdict(self.spec) if self.spec else None,
               "instruction": self.extras.get("instruction") or getattr(self.backend, "instruction", None),
               "model": self.extras.get("model") or (self.spec.model if self.spec else None),
               "tool_calls": self.calls, "tool_counts": dict(self.tool_counts),
               "program_errors": self.program_errors, "tool_limit_hit": self.limit_hit,
               "brain_wall_s": ((self.last_call_t or time.time()) - self.first_call_t) if self.first_call_t else 0.0,
               "server_uptime_s": time.time() - self.t0, "error": self.error,
               "done_summary": getattr(self.backend, "summary", None)}
        if self.backend is not None:
            res.update(self.backend.runner_result())       # RUNNER-ONLY: success lives here
            if self.limit_hit and not res.get("ended_by"):
                res["ended_by"] = "tool_limit"
        else:
            res.setdefault("success", False)
        return res

    def _write_result(self) -> None:
        if self.out:
            tmp = self.out / "result.json.tmp"
            tmp.write_text(json.dumps(self.result(), indent=2, default=str))
            tmp.replace(self.out / "result.json")

    # ------------------------------------------------------------------ dispatch
    def call(self, tool: str, args: dict) -> list:
        t = time.time()
        self.first_call_t = self.first_call_t or t
        self.calls += 1
        self.tool_counts[tool] = self.tool_counts.get(tool, 0) + 1
        rec: dict = {"i": self.calls, "tool": tool, "args": args, "t": round(t - self.t0, 3)}
        try:
            out = self._dispatch(tool, args, rec)
        except Exception as e:                              # never crash the episode
            import traceback
            tb = traceback.format_exc()
            if self.out:
                with open(self.out / "errors.log", "a") as f:
                    f.write(tb + "\n")
            out = [f"ERROR: {type(e).__name__}: {str(e)[:200]}"]
        out = [sanitize(o) if isinstance(o, str) else o for o in out]
        self.last_call_t = time.time()
        rec["dt"] = round(self.last_call_t - t, 3)
        rec["text"] = "\n".join(o for o in out if isinstance(o, str))
        rec["n_images"] = sum(isinstance(o, bytes) for o in out)
        rec["episode_over"] = self._over()
        self._trace(rec)
        if self.backend is not None:
            self._on_backend(self._write_result)
        else:
            self._write_result()
        return out

    def _dispatch(self, tool: str, args: dict, rec: dict) -> list:
        if self.spec is None:
            return ["ERROR: no episode configured (ROBOT_EPISODE is not set)."]
        if self._ready is not None:
            self._ready.result()
        if self.backend is None:
            return [f"ERROR: {self.error or 'backend unavailable'}"]
        if self._over():
            return [EPISODE_OVER]
        if self.calls > self.max_tool_calls:
            self.limit_hit = True
            self._on_backend(self.backend.done, "tool call limit reached")
            return [EPISODE_OVER + f": tool call limit ({self.max_tool_calls}) reached"]
        b = self.backend
        if tool == "observe":
            find = args.get("find") or []
            if isinstance(find, str):
                find = [f.strip() for f in find.split(",")]
            find = [str(f)[:60] for f in find][:12]
            imgs, text = self._on_backend(b.observe, find)
            return [*self._frames(imgs), text]
        if tool == "run":
            try:
                calls = parse_program(args.get("program", ""))
            except ProgramError as e:
                self.program_errors += 1
                rec["program_error"] = str(e)
                return [f"PROGRAM_ERROR: {e}. Nothing was executed."]
            rec["program"] = [format_call(c) for c in calls]
            lines, imgs, text = self._on_backend(b.run, calls)
            rec["status_lines"] = lines
            out = ["Status:\n" + "\n".join(lines)]
            if self._over():
                return out + [EPISODE_OVER]
            return [*out, *self._frames(imgs), text]
        if tool == "done":
            self._on_backend(b.done, str(args.get("summary", ""))[:500])
            return [EPISODE_OVER]
        return [f"ERROR: unknown tool {tool}"]

    def shutdown(self) -> None:
        try:
            if self.backend is not None:
                self._on_backend(self._write_result)
                self._on_backend(self.backend.close)
            else:
                self._write_result()
        finally:
            self.pool.shutdown(wait=False)


# ---------------------------------------------------------------------------- MCP wiring
def build_server(session: RobotSession):
    from mcp.server.mcpserver import Image, MCPServer

    server = MCPServer(
        "robot",
        instructions="Robot arm tools. observe(find) first, then run(program) with skill calls on "
                     "numbered marks, then done(summary).",
    )

    def wrap(out: list) -> list:
        return [Image(data=o, format="png") if isinstance(o, bytes) else o for o in out]

    async def call(tool: str, args: dict) -> list:
        return wrap(await asyncio.to_thread(session.call, tool, args))

    @server.tool(structured_output=False)
    async def observe(find: list[str] | str) -> list:
        """Look at the scene. `find`: object or part nouns from the instruction, e.g.
        ["bowl", "plate"]. Returns the front camera and wrist camera images with numbered marks,
        plus a text list of marks with positions (robot base frame, cm) and the gripper state.
        Marks are valid until the next observe/run."""
        return await call("observe", {"find": find})

    @server.tool(structured_output=False)
    async def run(program: str) -> list:
        """Execute a skill program, one call per line (max 12):
        pick(M) | place(M, "on"|"in"|"top_of"|"left_of"|"right_of"|"front_of"|"back_of") |
        push(M, dx=<m>, dy=<m>) | actuate(M, "open"|"close"|"turn_on"|"turn_off") | look(M) | home().
        M = mark number from the latest observation. Stops at the first failure. Returns one
        status line per call and a fresh observation with new marks."""
        return await call("run", {"program": program})

    @server.tool(structured_output=False)
    async def done(summary: str) -> list:
        """Declare the task finished (after verifying it in an observation). Ends the episode."""
        return await call("done", {"summary": summary})

    return server


def _env(name: str, default: str | None = None) -> str | None:
    """Environment lookup that treats an unexpanded '${VAR}' placeholder as unset."""
    v = os.environ.get(name)
    return default if not v or v.startswith("${") else v


def session_from_env() -> RobotSession:
    path = _env("ROBOT_EPISODE")
    max_calls = int(_env("ROBOT_MAX_TOOL_CALLS", "10"))
    if not path or not Path(path).is_file():                 # e.g. the power's default, before a spec is written
        return RobotSession(None, None, max_tool_calls=max_calls)
    spec, extras = load_spec(path)
    if not spec.out_dir:
        spec.out_dir = str(Path(path).resolve().parent)
    from motormem.brain.backend import make_backend
    kind = _env("MOTORMEM_BACKEND") or extras.get("backend", "fake")
    return RobotSession(spec, lambda s: make_backend(s, kind, record=bool(extras.get("record"))), extras, max_tool_calls=max_calls)


def main() -> None:
    session = session_from_env()

    def _term(*_):
        session.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    try:
        build_server(session).run("stdio")
    finally:
        session.shutdown()


if __name__ == "__main__":
    main()
