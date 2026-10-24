"""The live-demo web page: serves the prompt box, reports state, starts one run at a time (no sim, no Kiro)."""
import http.server
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live", ROOT / "scripts" / "live.py")
live_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live_mod)


def _serve(tmp_path, started):
    live = live_mod.Live(tmp_path, "Panda", "auto", 100, 10)

    def fake_start(prompt):
        if live.running:
            return "a run is already in progress"
        live.running, live.prompt, live.ep = True, prompt, "ep_000"
        started.append(prompt)
        return None

    live.start = fake_start
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), live_mod.make_handler(live))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


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
        assert "<input id=\"p\"" in page and "Run</button>" in page
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["running"] is False and st["ep"] is None
        assert _post(base + "/run", {"prompt": ""})[0] == 400
        assert _post(base + "/run", {"prompt": "put the soup cans on the tray"}) == (200, {"ok": True})
        assert started == ["put the soup cans on the tray"]
        code, body = _post(base + "/run", {"prompt": "again"})
        assert code == 409 and "already" in body["error"]
        st = json.loads(urllib.request.urlopen(base + "/state", timeout=5).read())
        assert st["running"] is True and st["prompt"] == "put the soup cans on the tray"
    finally:
        httpd.shutdown()
