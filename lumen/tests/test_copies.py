import shutil
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen.server import app


def picture(seed, tint):
    """A textured picture (so its fingerprint is its own) with a colour cast the stand-in model sees."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 80, (48, 64, 3)).repeat(10, 0).repeat(10, 1).astype(np.int16)
    return Image.fromarray(np.clip(base + np.array(tint), 0, 255).astype(np.uint8))


def wait(pred, timeout=30):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError("timed out")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("copies")
    picture(1, (170, 0, 0)).save(root / "cp_red_a.jpg", quality=95)
    for n in (2, 3):                                           # the same file saved twice more
        shutil.copy(root / "cp_red_a.jpg", root / f"cp_red_a copy {n}.jpg")
    picture(2, (170, 0, 0)).save(root / "cp_red_b.jpg", quality=95)      # a different red picture
    picture(3, (0, 0, 170)).save(root / "cp_blue.jpg", quality=95)
    with TestClient(app) as c:
        before = c.get("/api/status").json()["ready"]
        assert c.post("/api/folders", json={"path": str(root)}).status_code == 200
        wait(lambda: (lambda s: s["pending"] == 0 and s["state"] == "idle" and s["ready"] >= before + 5)(
            c.get("/api/status").json()))
        yield c


def mine(items):
    return [i for i in items if i["name"].startswith("cp_")]


def test_copies_are_folded_into_one_result(client):
    every = mine(client.get("/api/search", params={"q": "red", "copies": "true"}).json()["items"])
    assert sum(i["name"].startswith("cp_red_a") for i in every) == 3
    res = client.get("/api/search", params={"q": "red", "copies": "false"}).json()
    folded = mine(res["items"])
    a = [i for i in folded if i["name"].startswith("cp_red_a")]
    assert len(a) == 1 and len(a[0]["copies"]) == 2 and res["hidden_copies"] >= 2
    assert {a[0]["name"], *a[0]["copy_names"]} == {"cp_red_a.jpg", "cp_red_a copy 2.jpg", "cp_red_a copy 3.jpg"}
    # a different picture is never folded away, and order among the rest is unchanged
    b = [i for i in folded if i["name"] == "cp_red_b.jpg"]
    assert len(b) == 1 and "copies" not in b[0]
    kept = {i["id"] for i in folded}
    assert [i["id"] for i in every if i["id"] in kept] == [i["id"] for i in folded]


def test_dismissed_groups_are_not_folded(client):
    group = next(g for g in client.get("/api/duplicates").json()["groups"]
                 if any(p["name"].startswith("cp_red_a") for p in g["photos"]))
    client.post("/api/duplicates/ignore", json={"key": group["key"]})
    folded = mine(client.get("/api/search", params={"q": "red", "copies": "false"}).json()["items"])
    assert sum(i["name"].startswith("cp_red_a") for i in folded) == 3
