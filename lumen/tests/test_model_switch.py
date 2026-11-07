import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import config, db, embedder, indexer
from lumen.server import app


def settled(c):
    # Other test modules may have left photos in the shared library, so don't count exactly.
    s = c.get("/api/status").json()
    return s["pending"] == 0 and s["state"] == "idle" and s["ready"] >= 4


def wait(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError("timed out")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("sw")
    for i, rgb in enumerate([(220, 30, 30), (30, 30, 220), (30, 200, 30), (220, 220, 30)]):
        Image.new("RGB", (640 + i, 480), rgb).save(root / f"p{i}.jpg")
    with TestClient(app) as c:
        assert c.post("/api/folders", json={"path": str(root)}).status_code == 200
        wait(lambda: settled(c))
        yield c


def test_failed_download_changes_nothing(client, monkeypatch):
    before = client.get("/api/settings").json()["model"]
    ready = client.get("/api/status").json()["ready"]

    def boom(key):
        raise OSError("no network")

    monkeypatch.setattr(embedder, "prepare", boom)
    other = next(k for k in config.MODELS if k != before)
    assert client.put("/api/settings", json={"model": other}).status_code == 200
    wait(lambda: not indexer.switch_status()["to"])
    st = client.get("/api/status").json()
    assert "no network" in st["model_switch"]["error"] and "Still using" in st["model_switch"]["error"]
    assert client.get("/api/settings").json()["model"] == before
    assert st["ready"] == ready                                # nothing was marked for re-reading
    assert db.get_setting("model") in (None, before)
    assert len(client.get("/api/search", params={"q": "red"}).json()["items"]) > 0


def test_successful_switch_rereads_with_new_model(client):
    before = client.get("/api/settings").json()["model"]
    other = next(k for k in config.MODELS if k != before)
    assert client.put("/api/settings", json={"model": other}).status_code == 200
    wait(lambda: client.get("/api/settings").json()["model"] == other)
    wait(lambda: settled(client))
    assert client.get("/api/status").json()["model_switch"] == {"to": None, "error": None}
    assert {r["model"] for r in db.q("SELECT model FROM photos WHERE status='ready'")} == {other}
    assert client.get("/api/search", params={"q": "blue"}).json()["items"]
    # and back, so later tests see the default
    client.put("/api/settings", json={"model": before})
    wait(lambda: client.get("/api/settings").json()["model"] == before)
    wait(lambda: settled(client))


def test_second_switch_while_one_runs_is_refused(client, monkeypatch):
    import threading
    gate = threading.Event()
    real = embedder.prepare
    monkeypatch.setattr(embedder, "prepare", lambda key: (gate.wait(10), real(key))[1])
    before = client.get("/api/settings").json()["model"]
    other = next(k for k in config.MODELS if k != before)
    assert client.put("/api/settings", json={"model": other}).status_code == 200
    assert client.put("/api/settings", json={"model": other}).status_code == 409
    gate.set()
    wait(lambda: client.get("/api/settings").json()["model"] == other)
    client.put("/api/settings", json={"model": before})
    wait(lambda: client.get("/api/settings").json()["model"] == before)
