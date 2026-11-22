import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import db, taste
from lumen.server import app


def test_auc():
    y = np.array([1, 1, -1, -1.0])
    assert taste.auc(np.array([4, 3, 2, 1.0]), y) == 1.0
    assert taste.auc(np.array([1, 2, 3, 4.0]), y) == 0.0
    assert taste.auc(np.array([4, 1, 3, 2.0]), y) == 0.5
    assert taste.auc(np.array([1.0, 2.0]), np.array([1.0, 1.0])) == 0.5      # nothing to compare


def clustered(n, seed=0):
    """Liked photos come from two separate clusters, with a disliked cluster between them: a single
    'average of what you like' direction can't tell them apart, a trained model can."""
    rng = np.random.default_rng(seed)
    centres = np.zeros((3, 24))
    centres[0, 0], centres[1, 1], centres[2, :2] = 1, 1, 0.75
    X = np.concatenate([c + 0.12 * rng.normal(size=(n, 24)) for c in centres])
    y = np.array([1.0] * (2 * n) + [-1.0] * n)
    return X, y, np.ones(len(y))


def test_trained_model_beats_the_average_when_taste_is_not_one_direction():
    X, y, w = clustered(60)
    trained, simple = taste._cross_validate(X, y, w)
    assert trained > 0.95 and trained > simple + 0.01      # the margin taste._train asks for


def wait(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("taste")
    for i in range(6):
        Image.new("RGB", (640 + i, 480), (200, 30 + i, 30)).save(root / f"ts_red_{i}.jpg")
        Image.new("RGB", (640 + i, 480), (30, 30 + i, 200)).save(root / f"ts_blue_{i}.jpg")
    with TestClient(app) as c:
        before = c.get("/api/status").json()["ready"]
        assert c.post("/api/folders", json={"path": str(root)}).status_code == 200
        wait(lambda: (lambda s: s["pending"] == 0 and s["state"] == "idle" and s["ready"] >= before + 12)(
            c.get("/api/status").json()))
        yield c


def named(c, prefix):
    return {i["name"]: i["id"] for i in c.get("/api/photos", params={"limit": 500}).json()["items"]
            if i["name"].startswith(prefix)}


def test_feedback_and_collections_count_as_examples(client, monkeypatch):
    reds, blues = named(client, "ts_red_"), named(client, "ts_blue_")
    liked0, disliked0 = taste.examples()
    a, b, cc = list(reds.values())[:3]
    assert client.post("/api/feedback", json={"q": "Red  Things", "photo_id": a, "verdict": 1}).status_code == 200
    client.post("/api/feedback", json={"q": "red things", "photo_id": list(blues.values())[0], "verdict": -1})
    col = client.post("/api/collections", json={"name": "ts keepers", "photo_ids": [b]}).json()
    client.post("/api/photos/marks", json={"photo_ids": [cc], "rating": 1})
    liked, disliked = taste.examples()
    assert liked[a] == taste.FEEDBACK_WEIGHT and liked[b] == 1.0 and a not in liked0 and b not in liked0
    assert disliked[list(blues.values())[0]] == taste.FEEDBACK_WEIGHT and disliked[cc] == 1.0
    # taking a thumbs-up back removes it; an unknown photo is refused
    client.post("/api/feedback", json={"q": "red things", "photo_id": a, "verdict": 0})
    assert a not in taste.examples()[0]
    assert client.post("/api/feedback", json={"q": "x", "photo_id": 10**9, "verdict": 1}).status_code == 404
    client.delete(f"/api/collections/{col['id']}")
    client.post("/api/photos/marks", json={"photo_ids": [cc], "rating": 0})


def test_status_says_what_is_in_use(client, monkeypatch):
    monkeypatch.setattr(taste, "MIN_EXAMPLES", 3)
    reds = named(client, "ts_red_")
    everyone = [i["id"] for i in client.get("/api/photos", params={"limit": 500}).json()["items"]]
    client.post("/api/photos/marks", json={"photo_ids": everyone, "favorite": False, "rating": 0})
    with db.tx() as c:
        c.execute("DELETE FROM search_feedback")
    st = client.get("/api/settings").json()["taste"]
    assert st["method"] == "none" or st["liked"] >= 3          # collections from other tests may count
    client.post("/api/photos/marks", json={"photo_ids": list(reds.values()), "favorite": True})
    st = client.get("/api/settings").json()["taste"]
    assert st["liked"] >= 6 and st["method"] in ("simple", "trained")
    from lumen.vectors import index
    ids, mat = index.snapshot()
    s = dict(zip(ids.tolist(), taste.scores(ids, mat)))
    blues = named(client, "ts_blue_")
    assert np.mean([s[i] for i in reds.values()]) > np.mean([s[i] for i in blues.values()])
    client.post("/api/photos/marks", json={"photo_ids": list(reds.values()), "favorite": False})
