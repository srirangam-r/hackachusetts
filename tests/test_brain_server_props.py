"""Property 7: no MCP tool output leaks privileged key names or success information (Req 5.4).
Plus server behaviour: program errors never move the robot, EPISODE_OVER after the end,
trace/result files, tool-call ceiling."""
from __future__ import annotations

import asyncio
import json
import re

from hypothesis import HealthCheck, given, settings, strategies as st

from motormem.brain.backend import FAKE_LIBERO_GOAL, FakeBackend
from motormem.brain.mcp_server import EPISODE_OVER, RobotSession, build_server, sanitize
from motormem.brain.program import ACTUATE_ACTIONS, PLACE_RELATIONS
from motormem.config import is_privileged_key
from motormem.types import EpisodeSpec

FORBIDDEN_WORDS = ("success", "check_success", "_runner_success")
TOKEN_RES = (re.compile(r"[A-Za-z0-9_\-]+"), re.compile(r"[A-Za-z0-9_]+"))


def assert_clean(text: str) -> None:
    for rx in TOKEN_RES:
        for tok in rx.findall(text):
            assert not is_privileged_key(tok), (tok, text)
            assert not any(w in tok.lower() for w in FORBIDDEN_WORDS), (tok, text)


def session(tmp_path, task=0, seed=0, fail=0.0, max_calls=10, out=True):
    spec = EpisodeSpec(suite="libero_goal", task_id=task, init_index=seed % 5, seed=seed,
                       out_dir=str(tmp_path) if out else "")
    return RobotSession(spec, lambda s: FakeBackend(s, fail_prob=fail), max_tool_calls=max_calls,
                        save_frames=False)


def texts(out: list) -> list[str]:
    return [o for o in out if isinstance(o, str)]


# Programs mixing valid calls, invalid calls and privileged-looking junk.
nouns = st.sampled_from(["bowl", "black bowl", "plate", "stove", "cabinet", "top drawer", "middle drawer",
                         "wine bottle", "rack", "cream cheese", "akita_black_bowl_pos", "plate_quat",
                         "object-state", "success", "check_success", "segmentation", "unicorn"])
line = st.one_of(
    st.builds(lambda m: f"pick({m})", st.integers(0, 12)),
    st.builds(lambda m, r: f'place({m}, "{r}")', st.integers(1, 12), st.sampled_from(PLACE_RELATIONS + ("under",))),
    st.builds(lambda m, a: f'actuate({m}, "{a}")', st.integers(1, 12), st.sampled_from(ACTUATE_ACTIONS)),
    st.builds(lambda m, x, y: f"push({m}, dx={x}, dy={y})", st.integers(1, 12),
              st.floats(-0.4, 0.4).map(lambda v: round(v, 3)), st.floats(-0.4, 0.4).map(lambda v: round(v, 3))),
    st.builds(lambda m: f"look({m})", st.integers(1, 12)),
    st.just("home()"), st.just("plate_pos()"), st.just("check_success()"), st.just("# success?"),
    st.text(max_size=20),
)
action = st.one_of(
    st.tuples(st.just("observe"), st.fixed_dictionaries({"find": st.lists(nouns, max_size=4)})),
    st.tuples(st.just("run"), st.fixed_dictionaries({"program": st.lists(line, min_size=1, max_size=5).map("\n".join)})),
    st.tuples(st.just("done"), st.fixed_dictionaries({"summary": st.sampled_from(["success!", "bowl_pos ok", "x"])})),
)


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(task=st.integers(0, 9), seed=st.integers(0, 50), fail=st.sampled_from([0.0, 0.5]),
       actions=st.lists(action, min_size=1, max_size=8))
def test_tool_outputs_never_leak(tmp_path, task, seed, fail, actions):
    s = session(tmp_path, task, seed, fail, out=False)
    try:
        for tool, args in [("observe", {"find": ["bowl", "plate", "stove", "drawer"]}), *actions]:
            for t in texts(s.call(tool, args)):
                assert_clean(t)
    finally:
        s.shutdown()


@given(st.text(max_size=200))
def test_sanitize_is_clean(text):
    assert_clean(sanitize(text))


def test_sanitize_keeps_normal_text():
    t = 'Marks: [1] black bowl: x=40.0 y=-3.2\n1. place(2, "on") -> OK: released'
    assert sanitize(t) == t


def test_program_error_does_not_move_robot(tmp_path):
    s = session(tmp_path, task=8)
    s.call("observe", {"find": ["bowl", "plate"]})
    steps = s.backend.sim_steps
    out = s.call("run", {"program": "pick(1)\n__import__('os')"})
    assert out[0].startswith("PROGRAM_ERROR") and s.backend.sim_steps == steps and s.backend.holding is None
    s.shutdown()


def test_full_fake_episode_writes_trace_and_result(tmp_path):
    s = session(tmp_path, task=8)            # put the bowl on the plate
    out = s.call("observe", {"find": ["bowl", "plate"]})
    assert sum(isinstance(o, bytes) for o in out) == 2
    marks = {mi.label: m for m, mi in s.backend.last_marks().items()}
    out = s.call("run", {"program": f'pick({marks["black bowl"]})\nplace({marks["plate"]}, "on")'})
    assert "OK" in out[0] and out[-1] == EPISODE_OVER          # runner ended it on success
    assert texts(s.call("observe", {"find": ["bowl"]})) == [EPISODE_OVER]
    s.shutdown()
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["success"] is True and res["ended_by"] == "success" and res["tool_calls"] == 3
    trace = [json.loads(l) for l in (tmp_path / "trace.jsonl").read_text().splitlines()]
    assert [r["tool"] for r in trace] == ["observe", "run", "observe"]
    assert len(trace[1]["program"]) == 2 and trace[1]["status_lines"][0].endswith("holding black bowl")


def test_failure_stops_program_and_done_ends(tmp_path):
    s = session(tmp_path, task=1)
    s.call("observe", {"find": ["bowl", "stove"]})
    out = s.call("run", {"program": "place(1, \"on\")\nhome()"})
    assert "NOT_HOLDING" in out[0] and "SKIPPED" in out[0]
    assert s.call("done", {"summary": "gave up"}) == [EPISODE_OVER]
    assert s.call("run", {"program": "home()"}) == [EPISODE_OVER]
    s.shutdown()
    res = json.loads((tmp_path / "result.json").read_text())
    assert res["success"] is False and res["ended_by"] == "done"


def test_tool_call_ceiling(tmp_path):
    s = session(tmp_path, task=7, max_calls=3)
    for _ in range(3):
        assert texts(s.call("observe", {"find": ["stove"]}))[0].startswith("Marks")
    assert texts(s.call("observe", {"find": ["stove"]}))[0].startswith(EPISODE_OVER)
    s.shutdown()


def test_unconfigured_server_answers_error():
    s = RobotSession(None, None)
    assert s.call("observe", {"find": ["bowl"]})[0].startswith("ERROR: no episode")
    s.shutdown()


def test_mcp_layer_returns_images_and_text(tmp_path):
    s = session(tmp_path, task=7)
    srv = build_server(s)

    async def go():
        names = [t.name for t in await srv.list_tools()]
        r = await srv.call_tool("observe", {"find": ["stove"]})
        return names, [c.type for c in r.content]

    names, types = asyncio.run(go())
    s.shutdown()
    assert names == ["observe", "run", "done"] and types == ["image", "image", "text"]


def test_fake_instructions_cover_libero_goal():
    assert len(FAKE_LIBERO_GOAL) == 10
