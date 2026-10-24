"""Runner: task parsing, isolated Kiro workspace, scripted rung-0 summary on the fake backend."""
from __future__ import annotations

import json
import sys

import yaml

from motormem import config
from motormem.brain import runner
from motormem.brain.backend import FakeBackend
from motormem.brain.mcp_server import RobotSession
from motormem.brain.scripted import LIBERO_GOAL, run_scripted
from motormem.types import EpisodeSpec


def test_parse_tasks():
    assert runner.parse_tasks("0-3,7") == [0, 1, 2, 3, 7]
    assert runner.parse_tasks("5") == [5]


def test_workspace_is_isolated_and_restricted(tmp_path):
    ws = tmp_path / "ws"
    runner.make_workspace(ws)
    files = sorted(str(p.relative_to(ws)) for p in ws.rglob("*") if p.is_file())
    assert files == [".kiro/agents/robot-brain.md", ".kiro/settings/cli.json", ".kiro/steering/robot-protocol.md"]
    text = (ws / ".kiro/agents/robot-brain.md").read_text()
    front = yaml.safe_load(text.split("---")[1])
    assert front["tools"] == ["@robot"] and front["includeMcpJson"] is False
    assert front["mcpServers"]["robot"]["env"]["PYTHONPATH"] == str(config.REPO_ROOT)
    assert front["mcpServers"]["robot"]["command"] == sys.executable          # the venv running the runner
    assert front["resources"] == ["file://.kiro/steering/robot-protocol.md"]
    assert "inclusion: manual" in (ws / ".kiro/steering/robot-protocol.md").read_text()


def test_scripted_programs_solve_every_fake_task(tmp_path):
    for task in LIBERO_GOAL:
        spec = EpisodeSpec(suite="libero_goal", task_id=task, init_index=1, seed=3, out_dir=str(tmp_path / str(task)))
        s = RobotSession(spec, lambda sp: FakeBackend(sp), max_tool_calls=100, save_frames=False)
        run_scripted(s, task)
        s.shutdown()
        assert json.loads((tmp_path / str(task) / "result.json").read_text())["success"], task


def test_runner_scripted_fake_writes_summary(tmp_path):
    summary = runner.main(["--backend", "fake", "--scripted", "--tasks", "7-8", "--episodes", "2",
                           "--parallel", "2", "--name", "t", "--runs-dir", str(tmp_path)])
    assert summary["overall"]["n"] == 4 and summary["overall"]["successes"] == 4
    assert (tmp_path / "t" / "summary.json").exists()
    table = (tmp_path / "latest_summary.txt").read_text()
    assert "NOT A RESULT" in table and "| 7 |" in table
    assert sorted(p.name for p in (tmp_path / "t").glob("ep_*")) == ["ep_000", "ep_001", "ep_002", "ep_003"]
