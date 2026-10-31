import io
import os
import time
import zipfile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import search
from lumen.server import app

COLORS = {"red": (230, 20, 20), "green": (20, 200, 40), "blue": (30, 40, 230)}


def make_photos(root, n_per_color=6):
    for name, rgb in COLORS.items():
        for i in range(n_per_color):
            sub = root / ("trip" if i % 2 else "")
            sub.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (640 + i, 480), rgb).save(sub / f"{name}_{i}.jpg")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("photos")
    make_photos(root)
    with TestClient(app) as c:
        c.root = root
        r = c.post("/api/folders", json={"path": str(root)})
        assert r.status_code == 200, r.text
        wait_idle(c, 18)
        yield c


def wait_idle(c, expected_ready, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get("/api/status").json()
        if s["ready"] == expected_ready and s["pending"] == 0 and s["state"] == "idle":
            return s
        time.sleep(0.1)
    raise AssertionError(f"indexer never settled: {s}")


def test_indexes_every_photo_with_previews(client):
    s = client.get("/api/status").json()
    assert s["total"] == 18 and s["errors"] == 0
    first = client.get("/api/photos?limit=1").json()["items"][0]
    r = client.get(f"/api/photos/{first['id']}/preview")
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    assert max(Image.open(io.BytesIO(r.content)).size) <= 512


def test_search_ranks_matching_photos_first(client):
    items = client.get("/api/search", params={"q": "red"}).json()["items"]
    top = [i["name"] for i in items[:6]]
    assert all(n.startswith("red_") for n in top), top
    assert items[0]["tier"] == "best"


def test_negative_terms_are_parsed():
    assert search.parse_query('beach -people -"red car" black-and-white') == (
        "beach black-and-white", ["people", "red car"])


def test_similar_photos(client):
    red = client.get("/api/search", params={"q": "red"}).json()["items"][0]
    sim = client.get(f"/api/photos/{red['id']}/similar").json()
    assert sim and sim[0]["id"] != red["id"] and sim[0]["name"].startswith("red_")


def test_nested_folder_is_rejected(client):
    r = client.post("/api/folders", json={"path": str(client.root / "trip")})
    assert r.status_code == 409


def test_manual_collection_lifecycle(client):
    reds = [i["id"] for i in client.get("/api/search", params={"q": "red"}).json()["items"][:6]]
    c = client.post("/api/collections", json={"name": "Reds", "emoji": "❤️", "photo_ids": reds[:3]}).json()
    assert c["count"] == 3
    assert client.post("/api/collections", json={"name": "reds"}).status_code == 409

    added = client.post(f"/api/collections/{c['id']}/photos", json={"photo_ids": reds}).json()
    assert added["count"] == 6 and added["added"] == 3

    client.patch(f"/api/collections/{c['id']}", json={"name": "Red things", "cover_photo_id": reds[4]})
    detail = client.get(f"/api/collections/{c['id']}").json()
    assert detail["name"] == "Red things" and detail["cover_photo_id"] == reds[4]

    client.post(f"/api/collections/{c['id']}/photos/remove", json={"photo_ids": reds[:2]})
    assert client.get(f"/api/collections/{c['id']}").json()["count"] == 4

    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/collections/{c['id']}/export").content))
    assert len(z.namelist()) == 4 and all(n.startswith("red_") for n in z.namelist())

    assert client.delete(f"/api/collections/{c['id']}").status_code == 200
    assert client.get(f"/api/collections/{c['id']}").status_code == 404
    assert client.get("/api/status").json()["total"] == 18  # photos untouched


def test_smart_collection_fills_itself_and_can_be_frozen(client):
    c = client.post("/api/collections", json={"name": "Blues", "kind": "smart", "query": "blue"}).json()
    detail = client.get(f"/api/collections/{c['id']}").json()
    assert detail["count"] == 6 and all(i["name"].startswith("blue_") for i in detail["items"])
    assert client.post(f"/api/collections/{c['id']}/photos", json={"photo_ids": [1]}).status_code == 400
    frozen = client.post(f"/api/collections/{c['id']}/freeze").json()
    assert frozen["kind"] == "manual" and frozen["count"] == 6


def test_rescan_picks_up_new_changed_and_deleted_files(client):
    root = client.root
    Image.new("RGB", (300, 300), COLORS["green"]).save(root / "new_green.jpg")
    os.remove(root / "blue_0.jpg")
    time.sleep(1.1)
    Image.new("RGB", (500, 400), COLORS["red"]).save(root / "green_0.jpg")  # same name, new content
    client.post("/api/scan")
    time.sleep(0.3)
    wait_idle(client, 18)
    names = {i["name"] for i in client.get("/api/photos?limit=100").json()["items"]}
    assert "new_green.jpg" in names and "blue_0.jpg" not in names
    reds = [i["name"] for i in client.get("/api/search", params={"q": "red"}).json()["items"][:7]]
    assert "green_0.jpg" in reds


def test_parent_folder_absorbs_child_and_removal_forgets_photos(client, tmp_path):
    child = tmp_path / "parent" / "child"
    make_photos(child, n_per_color=1)
    client.post("/api/folders", json={"path": str(child)})
    wait_idle(client, 21)
    p = client.post("/api/folders", json={"path": str(tmp_path / "parent")}).json()
    folders = {f["path"] for f in client.get("/api/folders").json()}
    assert str(child) not in folders and p["path"] in folders
    wait_idle(client, 21)
    removed = client.delete(f"/api/folders/{p['id']}").json()
    assert removed["removed_photos"] == 3
    assert client.get("/api/status").json()["total"] == 18


def test_folder_browser_lists_directories(client):
    d = client.get("/api/fs", params={"path": str(client.root)}).json()
    assert [x["name"] for x in d["dirs"]] == ["trip"] and d["images"] > 0
