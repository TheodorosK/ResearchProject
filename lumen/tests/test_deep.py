import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import db, deep
from lumen.server import app

STORIES = {
    "r": "Two friends laughing together at a table, a warm and playful moment.",
    "g": "A child alone on a swing looking down, quiet and a little sad.",
    "b": "A couple hugging tenderly on a beach at sunset, calm blue water behind them.",
}


class FakeModelServer:
    """Speaks enough of the OpenAI chat API to 'describe' a picture by its dominant colour."""

    def __init__(self):
        self.calls = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj):
                data = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send({"data": [{"id": "test-vlm"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls += 1
                url = body["messages"][1]["content"][0]["image_url"]["url"]
                im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB").resize((1, 1))
                rgb = im.getpixel((0, 0))
                self._send({"choices": [{"message": {"content": STORIES["rgb"[rgb.index(max(rgb))]]}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def wait(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("deep")
    for name, rgb in [("dp_friends", (200, 40, 40)), ("dp_child", (40, 200, 40)), ("dp_couple", (40, 40, 200)),
                      ("dp_plain", (120, 120, 120))]:
        Image.new("RGB", (640, 480), rgb).save(root / f"{name}.jpg")
    server = FakeModelServer()
    with TestClient(app) as c:
        before = c.get("/api/status").json()["ready"]
        r = c.post("/api/folders", json={"path": str(root)})
        assert r.status_code == 200
        wait(lambda: (lambda s: s["pending"] == 0 and s["state"] == "idle" and s["ready"] >= before + 4)(
            c.get("/api/status").json()))
        c.put("/api/settings", json={"rerank_url": server.url, "rerank_model": "test-vlm"})
        c.folder_id = r.json()["id"]
        yield c, server
    server.close()


def ids_by_name(c):
    return {i["name"]: i["id"] for i in c.get("/api/photos", params={"limit": 500}).json()["items"]
            if i["name"].startswith("dp_")}


def test_nothing_runs_until_asked(env):
    c, server = env
    assert server.calls == 0
    st = c.get("/api/deep").json()
    assert st["queued"] == 0 and st["state"] == "idle"
    assert c.post("/api/deep/queue", json={}).status_code == 400


def test_only_chosen_photos_are_described_and_shown(env):
    c, server = env
    names = ids_by_name(c)
    chosen = [names["dp_friends.jpg"], names["dp_child.jpg"], names["dp_couple.jpg"]]
    before = c.get("/api/deep").json()["described"]
    r = c.post("/api/deep/queue", json={"photo_ids": chosen}).json()
    assert r["added"] == 3
    wait(lambda: c.get("/api/deep").json()["queued"] == 0)
    st = c.get("/api/deep").json()
    assert st["described"] == before + 3 and st["error"] is None and server.calls == 3
    assert c.get(f"/api/photos/{names['dp_child.jpg']}").json()["deep"]["text"] == STORIES["g"]
    assert c.get(f"/api/photos/{names['dp_plain.jpg']}").json()["deep"] is None
    # asking again doesn't describe them twice
    assert c.post("/api/deep/queue", json={"photo_ids": chosen}).json()["added"] == 0
    assert server.calls == 3


def test_search_finds_photos_by_what_is_going_on(env):
    c, server = env
    for q, want in [("friends laughing together", "dp_friends.jpg"), ("a sad child alone", "dp_child.jpg"),
                    ("couple hugging on a beach", "dp_couple.jpg")]:
        items = [i for i in c.get("/api/search", params={"q": q}).json()["items"] if i["name"].startswith("dp_")]
        assert items and items[0]["name"] == want, (q, [i["name"] for i in items])


def test_a_folder_can_be_queued_and_paused(env):
    c, server = env
    assert c.post("/api/deep/pause", json={"paused": True}).json()["paused"] is True
    r = c.post("/api/deep/queue", json={"folder_id": c.folder_id}).json()
    assert r["added"] == 1 and r["state"] == "paused"                       # only dp_plain was left
    time.sleep(0.4)
    assert c.get("/api/deep").json()["queued"] == 1 and server.calls == 3   # paused means paused
    assert c.post("/api/deep/clear").json()["queued"] == 0
    c.post("/api/deep/pause", json={"paused": False})


def test_server_down_is_reported_and_retried_later(env):
    c, server = env
    names = ids_by_name(c)
    c.put("/api/settings", json={"rerank_url": "http://127.0.0.1:9/v1"})
    c.post("/api/deep/queue", json={"photo_ids": [names["dp_plain.jpg"]]})
    wait(lambda: c.get("/api/deep").json()["error"])
    st = c.get("/api/deep").json()
    assert "Is Ollama running" in st["error"] and st["queued"] == 1 and st["state"] == "waiting"
    c.put("/api/settings", json={"rerank_url": server.url})
    deep.worker.nudge()
    wait(lambda: c.get("/api/deep").json()["queued"] == 0, timeout=40)
    assert c.get("/api/deep").json()["error"] is None
    assert db.one("SELECT COUNT(*) n FROM photo_descriptions WHERE photo_id=?", (names["dp_plain.jpg"],))["n"] == 1
