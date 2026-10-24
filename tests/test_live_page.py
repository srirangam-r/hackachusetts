"""The live-demo web page: serves the prompt box, reports state, starts one run at a time, and turns an episode's
files into the page's view (no sim, no Kiro)."""
import http.server
import importlib.util
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live", ROOT / "scripts" / "live.py")
live_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_mod)

DEMO_SEG = "libero_object/pick_up_the_butter_and_place_it_in_the_basket/demo_26/grasp0"


def _serve(tmp_path, started):
    live = live_mod.Live(tmp_path, "Panda", "auto", 100, 10)

    def fake_start(prompt, **opts):
        if live.running:
            return "a run is already in progress"
        live.running, live.prompt, live.ep = True, prompt, "ep_000"
        started.append((prompt, opts))
        return None

    live.start = fake_start
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), live_mod.make_handler(live))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _get(url):
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_state_and_run(tmp_path):
    started = []
    httpd, base = _serve(tmp_path, started)
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "<input id=\"p\"" in page and "Run task</button>" in page
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["running"] is False and st["ep"] is None and st["episodes"] == []
        assert st["examples"] and "UR5e" in st["arms"] and set(st["motions"]) == {"hybrid", "index", "primitive"}
        assert _post(base + "/run", {"prompt": ""})[0] == 400
        assert _post(base + "/run", {"prompt": "put the soup cans on the tray", "arm": "UR5e", "motion": "index",
                                     "forget": "ketchup"}) == (200, {"ok": True})
        assert started == [("put the soup cans on the tray", {"arm": "UR5e", "motion": "index", "forget": "ketchup"})]
        code, body = _post(base + "/run", {"prompt": "again"})
        assert code == 409 and "already" in body["error"]
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["running"] is True and st["prompt"] == "put the soup cans on the tray"
        assert _get(base + "/episode?ep=ep_404")[0] == 404
        assert _get(base + "/demo?seg=primitive")[0] == 404           # geometry motions have no demo clip
    finally:
        httpd.shutdown()


def test_start_writes_run_options_into_the_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(live_mod, "run_kiro_episode", lambda *a, **k: {"tool_calls": 0, "skill_statuses": []})
    monkeypatch.setattr(live_mod.subprocess, "Popen", lambda *a, **k: None)
    (tmp_path / "ep_000").mkdir()                                     # a reopened session keeps numbering
    live = live_mod.Live(tmp_path, "Panda", "auto", 100, 10)
    assert live.start("put the milk on the tray", arm="UR5e", motion="index", forget="milk, butter") is None
    sp = json.loads((tmp_path / "ep_001" / "spec.json").read_text())
    assert sp["arm"] == "UR5e" and sp["motion_source"] == "index"
    assert sp["index_exclude"] == {"objects": ["milk", "butter"]} and sp["instruction"] == "put the milk on the tray"
    while live.running:
        time.sleep(0.01)
    assert live.start("x", arm="KUKA", motion="cheat") is None         # unknown options fall back to the defaults
    sp = json.loads((tmp_path / "ep_002" / "spec.json").read_text())
    assert sp["arm"] == "Panda" and sp["motion_source"] == "hybrid" and sp["index_exclude"] == {}


def _episode(ep: Path, finished: bool) -> None:
    ep.mkdir()
    (ep / "spec.json").write_text(json.dumps({"instruction": "put the butter in the drawer", "arm": "Panda",
                                              "motion_source": "hybrid", "index_exclude": {"objects": ["milk"]}}))
    obs = ("gripper: opening 1.00 (open)\nobjects (robot base frame: x forward, y left, z up; cm):\n"
           "  [1] drawer: x=72.1 y=-28.4 z=15.5  size 23x0x20  (agentview, score 0.44)\n"
           "  [2] butter: x=42.3 y=-17.4 z=3.5  size 8x4x1  (agentview, score 0.86)")
    trace = [{"i": 1, "tool": "observe", "args": {"find": ["drawer", "butter"]}, "t": 1.0, "dt": 2.0, "text": obs},
             {"i": 2, "tool": "run", "args": {"program": "pick(2)\nplace(1, \"in\")"}, "t": 5.0, "dt": 4.0,
              "program": ["pick(2)", "place(1, \"in\")"],
              "status_lines": ["1. pick(2) -> OK: picked butter (mark 2)",
                               "2. place(1, \"in\") -> GRASP_FAILED: lost it [fallback]"], "text": "Status:\n" + obs}]
    if finished:
        trace.append({"i": 3, "tool": "done", "args": {"summary": "Put the butter away."}, "t": 10.0, "dt": 0.0,
                      "text": "EPISODE_OVER"})
    (ep / "trace.jsonl").write_text("".join(json.dumps(t) + "\n" for t in trace) + '{"i": 4, "tool"')  # torn last line
    (ep / "transcript.log").write_text(
        "[INFO] kas.server.starting\nI'll look first.\n# Plan:\n1. Pick the butter\n2. Place it in the drawer\n\n"
        "# Thought: find the marks.\n[tool] @robot/observe\n[tool] status: Completed\nButter is mark 2.\n"
        "# Thought: pick and place.\n[tool] @robot/run\n[tool] status: InProgress\n"
        + ("[tool] status: Completed\nAll done.\n[tool] @robot/done\nFinished." if finished else
           "[tool] status: Completed\nRetry.\n[tool] @robot/run\n"))
    (ep / "frames").mkdir()
    for i in (1, 2):
        (ep / "frames" / f"call{i:02d}_agentview.png").write_bytes(b"png")
        (ep / "frames" / f"call{i:02d}_wrist.png").write_bytes(b"png")
    (ep / "frames.json").write_text(json.dumps({"frames": [{}] * 30, "calls": [
        {"call": "pick(2)", "run": 1, "segment_id": DEMO_SEG, "status": "OK", "first": 1, "last": 12},
        {"call": "place(1, \"in\")", "run": 1, "segment_id": "primitive", "status": "GRASP_FAILED", "first": 13,
         "last": 29}]}))
    (ep / "result.json").write_text(json.dumps({"success": True, "tool_calls": len(trace), "server_uptime_s": 12.4,
                                                "ended_by": "done" if finished else None,
                                                "done_summary": "Put the butter away." if finished else None}))


def test_episode_view_finished(tmp_path):
    _episode(tmp_path / "ep_000", finished=True)
    v = live_mod.episode_view(tmp_path / "ep_000", running=False)
    assert v["phase"] == "done" and v["pending"] is None and v["final"] == "Finished."
    assert v["forget"] == ["milk"] and v["n_frames"] == 30
    t1, t2, t3 = v["turns"]
    assert t1["brain"]["plan"] == ["Pick the butter", "Place it in the drawer"]
    assert t1["brain"]["thought"] == "find the marks." and t1["brain"]["say"] == ["I'll look first."]
    assert [o["label"] for o in t1["objects"]] == ["drawer", "butter"]
    assert t1["images"]["front"].endswith("call01_agentview.png")
    assert t2["run"] == 1 and t2["saw"] == 1                         # Kiro wrote turn 2 looking at turn 1's images
    assert [(s["label"], s["status"]) for s in t2["steps"]] == [("pick butter", "OK"),
                                                                 ("place in drawer", "GRASP_FAILED")]
    assert t2["steps"][1]["message"] == "lost it"
    assert [c["source"]["kind"] for c in t2["calls"]] == ["demo", "geometry"]
    src = v["calls"][0]["source"]
    assert src["task"] == "pick up the butter and place it in the basket"
    assert src["demo"] == "26" and src["suite"] == "LIBERO-Object"
    assert t3["summary"] == "Put the butter away."
    s = v["summary"]
    assert (s["skills_ok"], s["skills"], s["from_demos"], s["from_geometry"], s["seconds"]) == (1, 2, 1, 1, 12)
    assert "success" not in json.dumps(v)                             # the runner-only flag never reaches the page


def test_episode_view_in_progress(tmp_path):
    _episode(tmp_path / "ep_000", finished=False)
    v = live_mod.episode_view(tmp_path / "ep_000", running=True)
    assert v["phase"] == "moving" and len(v["turns"]) == 2
    assert v["pending"]["tool"] == "run" and v["pending"]["run"] == 2 and v["pending"]["brain"]["say"] == ["Retry."]
    assert v["summary"] is None
    assert live_mod.episode_view(tmp_path / "ep_000", running=False)["phase"] == "stopped"


def test_call_labels_and_sources():
    marks = {1: "drawer", 2: "butter", 3: "tray"}
    assert live_mod.describe_call('actuate(1, "open")', marks) == "open drawer"
    assert live_mod.describe_call('place(3, "on")', marks) == "place on tray"
    assert live_mod.describe_call('pick(1, part="handle")', marks) == "pick handle of drawer"
    assert live_mod.describe_call("push(2, 0.1, 0.0)", marks) == "push butter"
    assert live_mod.describe_call("pick(9)", marks) == "pick mark 9"
    assert live_mod.motion_source("primitive") == {"kind": "geometry"}
    assert live_mod.motion_source(None)["kind"] == "geometry"
    s = live_mod.motion_source("libero_90/LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray"
                               "/demo_47/place1")
    assert s["scene"] == "living room scene 4" and s["segment"] == "place" and s["suite"] == "LIBERO-90"
    assert s["task"] == "pick up the black bowl on the left and put it in the tray"
