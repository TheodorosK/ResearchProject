import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import rerank
from lumen.server import app


class FakeModelServer:
    """Speaks just enough of the OpenAI chat API. It rates a photo by how green it is, which is
    nothing like the stand-in image model's ranking, so a re-order is visible."""

    def __init__(self, models=("test-vlm",)):
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
                self._send({"data": [{"id": m} for m in models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.calls += 1
                outer.last = body
                url = body["messages"][1]["content"][0]["image_url"]["url"]
                im = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGB").resize((1, 1))
                r, g, b = im.getpixel((0, 0))
                self._send({"choices": [{"message": {"content": f"Score: {round(g / 255 * 10)}/10"}}]})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def settled(c, n):
    s = c.get("/api/status").json()
    return s["pending"] == 0 and s["state"] == "idle" and s["ready"] >= n


def wait(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("rr")
    # Reds with a growing amount of green in them: the stand-in image model ranks pure red first
    # for "red", the fake vision-language model likes the greenest best.
    for i, g in enumerate([0, 40, 80, 120]):
        Image.new("RGB", (640 + i, 480), (230, g, 20)).save(root / f"rr_red_{i}.jpg")
    server = FakeModelServer()
    with TestClient(app) as c:
        assert c.post("/api/folders", json={"path": str(root)}).status_code == 200
        wait(lambda: settled(c, 4))
        c.put("/api/settings", json={"rerank": True, "rerank_url": server.url, "rerank_model": "test-vlm",
                                     "rerank_top": 4})
        yield c, server
        c.put("/api/settings", json={"rerank": False})
    server.close()


def ours(items):
    return [i for i in items if i["name"].startswith("rr_red_")]


def run_job(c, q, ids):
    r = c.post("/api/search/rerank", json={"q": q, "ids": ids})
    assert r.status_code == 200, r.text
    job = r.json()["job_id"]
    wait(lambda: c.get(f"/api/jobs/{job}").json()["state"] != "running")
    return c.get(f"/api/jobs/{job}").json()


def test_parse_score():
    assert rerank.parse_score("7") == 7
    assert rerank.parse_score("Score: 8/10") == 8
    assert rerank.parse_score(" 10.") == 10
    assert rerank.parse_score("I'd say 6.5 out of 10") == 6.5
    assert rerank.parse_score("no idea") is None
    assert rerank.parse_score("42") is None


def test_status_explains_what_is_missing(env):
    c, server = env
    assert c.get("/api/rerank").json()["available"] is True
    c.put("/api/settings", json={"rerank_model": "not-installed"})
    st = c.get("/api/rerank").json()
    assert st["available"] is False and "ollama pull not-installed" in st["detail"]
    assert c.post("/api/search/rerank", json={"q": "red", "ids": [1]}).status_code == 409
    c.put("/api/settings", json={"rerank_url": "http://127.0.0.1:9/v1", "rerank_model": "test-vlm"})
    st = c.get("/api/rerank").json()
    assert st["available"] is False and "Is Ollama running" in st["detail"]
    assert c.put("/api/settings", json={"rerank_url": "file:///etc/passwd"}).status_code == 400
    c.put("/api/settings", json={"rerank_url": server.url})
    assert c.get("/api/rerank").json()["available"] is True


def test_closer_look_reorders_the_top_and_is_remembered(env):
    c, server = env
    q = "red -blue"
    plain = c.get("/api/search", params={"q": q}).json()
    top = plain["items"][:4]
    assert plain["deep"] == 0 and all("deep" not in i for i in top)
    # asking for deep before anything was rated changes nothing
    assert [i["id"] for i in c.get("/api/search", params={"q": q, "deep": "true"}).json()["items"]] == \
           [i["id"] for i in plain["items"]]

    calls = server.calls
    job = run_job(c, q, [i["id"] for i in top])
    assert job["state"] == "done" and job["result"]["rated"] == 4 and job["result"]["reused"] == 0
    assert server.calls == calls + 4
    # the model is asked about the description only, not the exclusion
    assert 'Search: "red"' in server.last["messages"][1]["content"][1]["text"]

    deep = c.get("/api/search", params={"q": q, "deep": "true"}).json()
    assert deep["deep"] == 4
    head = deep["items"][:4]
    assert {i["id"] for i in head} == {i["id"] for i in top}            # same photos...
    greens = [i["deep"] for i in head]
    assert greens == sorted(greens, reverse=True)                        # ...best-rated first
    mine = ours(head)
    assert [i["name"] for i in mine] == sorted((i["name"] for i in mine), reverse=True)
    assert [i["tier"] for i in head] == [i["tier"] for i in top]         # tiers stay with positions
    assert [i["id"] for i in deep["items"][4:]] == [i["id"] for i in plain["items"][4:]]
    # without deep, the usual order is untouched
    assert [i["id"] for i in c.get("/api/search", params={"q": q}).json()["items"]] == [i["id"] for i in plain["items"]]

    again = run_job(c, q, [i["id"] for i in top])
    assert again["result"]["rated"] == 0 and again["result"]["reused"] == 4 and server.calls == calls + 4


def test_only_the_top_few_are_ever_sent(env):
    c, server = env
    c.put("/api/settings", json={"rerank_top": 2})
    items = c.get("/api/search", params={"q": "green"}).json()["items"][:4]
    calls = server.calls
    job = run_job(c, "green", [i["id"] for i in items])
    assert job["total"] == 2 and server.calls == calls + 2
    c.put("/api/settings", json={"rerank_top": 4})


def test_switched_off_means_off(env):
    c, server = env
    c.put("/api/settings", json={"rerank": False})
    assert c.get("/api/rerank").json()["available"] is False
    assert c.post("/api/search/rerank", json={"q": "red", "ids": [1]}).status_code == 409
    assert c.get("/api/search", params={"q": "red -blue", "deep": "true"}).json()["deep"] == 0
    c.put("/api/settings", json={"rerank": True})
