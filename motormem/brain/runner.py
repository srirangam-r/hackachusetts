"""Parallel evaluation runner (Req 6.1-6.4).

    python -m motormem.brain.runner --suite libero_goal --tasks 0-9 --episodes 10 --arm Panda \
        --model auto --parallel 4 --name <run> [--backend real|fake] [--motion-source hybrid] \
        [--perception sam3] [--scripted] [--timeout 600]

Per episode: `runs/<name>/ep_XXX/spec.json` (EpisodeSpec + instruction), then a headless
`kiro-cli chat --agent robot-brain` session run from an ISOLATED workspace `ep_XXX/ws/` that holds
only the agent, the robot protocol and workspace settings (tool search off, no inherited
steering/skills). The repo's own steering and hooks therefore never reach the brain.
The MCP server writes `trace.jsonl` / `result.json`; the runner adds wall time and writes
`runs/<name>/summary.json`, `summary.md` and copies the table to `runs/latest_summary.txt`.

`--scripted` bypasses Kiro and runs `brain/scripted.py` programs (RUNG-0 DIAGNOSTIC, NOT A RESULT).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from motormem import config
from motormem.types import EpisodeSpec

AGENT_TEMPLATE = config.REPO_ROOT / ".kiro" / "agents" / "robot-brain.md"
PROTOCOL = config.REPO_ROOT / ".kiro" / "steering" / "robot-protocol.md"
# toolSearch.enabled=false is not honoured per-workspace by kiro-cli 2.34.0 headless runs (robot tools
# stay deferred: one extra tool_load turn). It only takes effect as a global setting; kept for later versions.
WS_SETTINGS = {"toolSearch.enabled": False, "chat.disableInheritingDefaultResources": True}
KIRO = os.environ.get("KIRO_CLI", "kiro-cli")


def parse_tasks(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def make_workspace(ws: Path) -> None:
    """Isolated Kiro workspace: agent (absolute repo path), protocol, settings. Nothing else."""
    (ws / ".kiro" / "agents").mkdir(parents=True, exist_ok=True)
    (ws / ".kiro" / "steering").mkdir(parents=True, exist_ok=True)
    (ws / ".kiro" / "settings").mkdir(parents=True, exist_ok=True)
    agent = AGENT_TEMPLATE.read_text().replace("${PWD}", str(config.REPO_ROOT))
    (ws / ".kiro" / "agents" / "robot-brain.md").write_text(agent)
    shutil.copy(PROTOCOL, ws / ".kiro" / "steering" / "robot-protocol.md")
    (ws / ".kiro" / "settings" / "cli.json").write_text(json.dumps(WS_SETTINGS, indent=2))


def prompt_for(instruction: str) -> str:
    return f"Instruction: {instruction}"


def delete_sessions(ws: Path) -> int:
    """Delete the kiro-cli sessions created in this episode workspace (and only those)."""
    try:
        p = subprocess.run([KIRO, "chat", "--list-sessions", "-f", "json"], cwd=ws, capture_output=True,
                           text=True, timeout=60)
        groups = json.loads(p.stdout or "[]")
    except Exception:
        return 0
    n = 0
    for g in groups:
        if Path(g.get("cwd", "")).resolve() != ws.resolve():
            continue
        for s in g.get("sessions", []):
            subprocess.run([KIRO, "chat", "--delete-session", s["sessionId"]], cwd=ws, capture_output=True,
                           timeout=60)
            n += 1
    return n


def run_kiro_episode(ep_dir: Path, model: str, backend: str, timeout: float, keep_sessions: bool) -> dict:
    spec_path = ep_dir / "spec.json"
    instruction = json.loads(spec_path.read_text())["instruction"]
    ws = ep_dir / "ws"
    make_workspace(ws)
    env = {**os.environ, "ROBOT_EPISODE": str(spec_path), "MOTORMEM_BACKEND": backend, "MUJOCO_GL": "egl"}
    env.pop("PYTHONPATH", None)
    cmd = [KIRO, "chat", "--no-interactive", "-a", "--trust-workspace", "--require-mcp-startup",
           "--agent", "robot-brain", "--model", model, prompt_for(instruction)]
    t0 = time.time()
    timed_out = False
    with open(ep_dir / "transcript.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=ws, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                code = proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait()
    wall = time.time() - t0
    time.sleep(0.5)                                    # let the MCP server flush result.json
    rec = _read_result(ep_dir)
    rec.update({"wall_s": round(wall, 2), "exit_code": code, "timed_out": timed_out})
    if not keep_sessions:
        rec["sessions_deleted"] = delete_sessions(ws)
    return rec


def _scripted_episode(ep_dir: str, backend: str) -> dict:
    """RUNG-0 DIAGNOSTIC: scripted program on the backend, no brain. Runs in a worker process."""
    from motormem.brain.backend import make_backend
    from motormem.brain.mcp_server import RobotSession, load_spec
    from motormem.brain.scripted import run_scripted

    ep = Path(ep_dir)
    spec, extras = load_spec(ep / "spec.json")
    t0 = time.time()
    s = RobotSession(spec, lambda sp: make_backend(sp, backend, record=bool(extras.get("record"))), extras, max_tool_calls=100)
    try:
        log = run_scripted(s, int(spec.task_id))
    finally:
        s.shutdown()
    (ep / "scripted.log").write_text("\n".join(log) + "\n")
    rec = _read_result(ep)
    rec.update({"wall_s": round(time.time() - t0, 2), "exit_code": 0, "timed_out": False})
    return rec


def _read_result(ep_dir: Path) -> dict:
    p = ep_dir / "result.json"
    if not p.exists():
        return {"success": False, "error": "no result.json (server never started?)", "tool_calls": 0}
    return json.loads(p.read_text())


def summarize(name: str, records: list[dict], scripted: bool, args: dict) -> tuple[dict, str]:
    per_task: dict[str, dict] = {}
    for r in records:
        t = per_task.setdefault(str(r["task_id"]), {"instruction": r.get("instruction"), "n": 0, "successes": 0,
                                                    "wall": [], "calls": []})
        t["n"] += 1
        t["successes"] += int(bool(r.get("success")))
        t["wall"].append(r.get("wall_s", 0.0))
        t["calls"].append(r.get("tool_calls", 0))
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    rows, table = {}, []
    hdr = "RUNG-0 DIAGNOSTIC (scripted, no brain) - NOT A RESULT\n\n" if scripted else ""
    table.append(f"{hdr}Run `{name}` ({args.get('suite')}, {args.get('arm')}, model {args.get('model') if not scripted else '-'}, "
                 f"backend {args.get('backend')}, motion {args.get('motion_source')}, perception {args.get('perception')})\n")
    table.append("| task | instruction | success | mean wall (s) | mean tool calls |")
    table.append("|---|---|---|---|---|")
    for k in sorted(per_task, key=lambda x: int(x) if x.isdigit() else x):
        t = per_task[k]
        rows[k] = {"instruction": t["instruction"], "n": t["n"], "successes": t["successes"],
                   "success_rate": t["successes"] / t["n"], "mean_wall_s": mean(t["wall"]),
                   "mean_tool_calls": mean(t["calls"])}
        table.append(f"| {k} | {t['instruction']} | {t['successes']}/{t['n']} | {mean(t['wall']):.1f} | "
                     f"{mean(t['calls']):.1f} |")
    n = len(records)
    succ = sum(int(bool(r.get("success"))) for r in records)
    overall = {"n": n, "successes": succ, "success_rate": succ / n if n else 0.0,
               "mean_wall_s": mean([r.get("wall_s", 0.0) for r in records]),
               "mean_tool_calls": mean([r.get("tool_calls", 0) for r in records]),
               "timeouts": sum(int(bool(r.get("timed_out"))) for r in records)}
    table.append(f"| all | | {succ}/{n} ({100 * overall['success_rate']:.0f}%) | {overall['mean_wall_s']:.1f} | "
                 f"{overall['mean_tool_calls']:.1f} |")
    summary = {"name": name, "scripted": scripted, "args": args, "overall": overall, "per_task": rows,
               "episodes": records}
    return summary, "\n".join(table) + "\n"


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--episodes", type=int, default=10, help="episodes per task (init_index 0..N-1)")
    ap.add_argument("--arm", default="Panda")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--name", required=True)
    ap.add_argument("--backend", choices=["fake", "real"], default="real")
    ap.add_argument("--motion-source", default="hybrid", choices=["index", "primitive", "hybrid"])
    ap.add_argument("--perception", default="sam3")
    ap.add_argument("--index-path", default="")
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--scripted", action="store_true", help="RUNG-0 diagnostic: scripted programs, no brain")
    ap.add_argument("--runs-dir", default=str(config.RUNS_DIR))
    ap.add_argument("--record", action="store_true", help="record 512 px video frames (brain/recording.py)")
    ap.add_argument("--keep-sessions", action="store_true", help="do not delete the kiro-cli sessions")
    a = ap.parse_args(argv)

    from motormem.brain.backend import instruction_for

    run_dir = Path(a.runs_dir) / a.name
    run_dir.mkdir(parents=True, exist_ok=True)
    eps: list[tuple[Path, EpisodeSpec, str]] = []
    k = 0
    for task in parse_tasks(a.tasks):
        for e in range(a.episodes):
            ep_dir = run_dir / f"ep_{k:03d}"
            ep_dir.mkdir(exist_ok=True)
            for stale in ("result.json", "trace.jsonl"):
                (ep_dir / stale).unlink(missing_ok=True)
            spec = EpisodeSpec(suite=a.suite, task_id=task, init_index=e, arm=a.arm, motion_source=a.motion_source,
                               perception=a.perception, index_path=(str(Path(a.index_path).resolve()) if a.index_path else ""), max_steps=a.max_steps, seed=e,
                               out_dir=str(ep_dir.resolve()), model=a.model)
            instr = instruction_for(spec, a.backend)
            (ep_dir / "spec.json").write_text(json.dumps({**asdict(spec), "instruction": instr,
                                                           "backend": a.backend,
                                                           **({"record": True} if a.record else {})}, indent=2))
            eps.append((ep_dir, spec, instr))
            k += 1

    if a.scripted:
        with ProcessPoolExecutor(max_workers=a.parallel) as pool:
            futs = [pool.submit(_scripted_episode, str(d), a.backend) for d, _, _ in eps]
            results = [f.result() for f in futs]
    else:
        with ThreadPoolExecutor(max_workers=a.parallel) as pool:
            futs = [pool.submit(run_kiro_episode, d, a.model, a.backend, a.timeout, a.keep_sessions)
                    for d, _, _ in eps]
            results = [f.result() for f in futs]

    records = []
    for (d, spec, instr), r in zip(eps, results):
        records.append({"episode": d.name, "task_id": spec.task_id, "init_index": spec.init_index,
                        "instruction": instr, **{k: v for k, v in r.items() if k not in ("spec", "instruction")}})
    summary, table = summarize(a.name, records, a.scripted, vars(a))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (run_dir / "summary.md").write_text(table)
    latest = Path(a.runs_dir) / "latest_summary.txt"
    latest.write_text(table)
    print(table)
    return summary


if __name__ == "__main__":
    main()
