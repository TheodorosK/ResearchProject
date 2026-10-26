"""Favourites/ratings/tags + XMP, live watching, phone access, sharing, acceleration, autostart."""
import io
import os
import shutil
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import autostart, db, embedder, marks
from lumen.server import app

COLORS = [(230, 20, 20), (20, 200, 40), (30, 40, 230), (230, 220, 30)]


def make(path, k):
    arr = np.clip(np.array(COLORS[k % 4]) + np.random.default_rng(k).integers(-20, 20, (240, 320, 3)), 0, 255)
    Image.fromarray(arr.astype(np.uint8)).save(path)


def wait_for(c, cond, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get("/api/status").json()
        if cond(s):
            return s
        time.sleep(0.2)
    raise AssertionError(f"timed out: {s}")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("roundC")
    for k in range(12):
        make(root / f"p{k:02d}.jpg", k)
    # A Lightroom-style sidecar that should be imported.
    (root / "p00.xmp").write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmp:Rating="4"><dc:subject><rdf:Bag><rdf:li>Portfolio</rdf:li><rdf:li>Favorite</rdf:li></rdf:Bag></dc:subject>'
        '</rdf:Description></rdf:RDF></x:xmpmeta>')
    with TestClient(app) as c:
        c.root = root
        c.post("/api/folders", json={"path": str(root)})
        wait_for(c, lambda s: s["ready"] == 12 and s["pending"] == 0 and s["state"] == "idle"
                 and not db.one("SELECT 1 FROM photos WHERE xmp_v=0"))
        yield c
        for f in c.get("/api/folders").json():
            c.delete(f"/api/folders/{f['id']}")
        db.set_setting("lan", "0")
        db.set_setting("write_xmp", "0")


def by_name(c):
    return {i["name"]: i for i in c.get("/api/photos?limit=500").json()["items"]}


def test_existing_sidecar_is_imported(client):
    p = by_name(client)["p00.jpg"]
    assert p["rating"] == 4 and p["fav"] is True
    assert client.get(f"/api/photos/{p['id']}").json()["tags"] == ["Portfolio"]


def test_favourites_ratings_tags_and_search(client):
    items = by_name(client)
    ids = [items[f"p0{k}.jpg"]["id"] for k in (1, 2, 3)]
    client.post("/api/photos/marks", json={"photo_ids": ids, "favorite": True, "rating": 5, "add_tags": ["#Crete", "summer"]})
    assert {t["tag"] for t in client.get("/api/tags").json()} >= {"Crete", "summer", "Portfolio"}
    fav = {i["name"] for i in client.get("/api/photos", params={"favorite": True}).json()["items"]}
    assert fav == {"p00.jpg", "p01.jpg", "p02.jpg", "p03.jpg"}
    assert {i["name"] for i in client.get("/api/photos", params={"tag": "crete"}).json()["items"]} == {"p01.jpg", "p02.jpg", "p03.jpg"}
    assert {i["name"] for i in client.get("/api/search", params={"q": "#summer"}).json()["items"]} == {"p01.jpg", "p02.jpg", "p03.jpg"}
    assert {i["name"] for i in client.get("/api/search", params={"q": "my favourites"}).json()["items"]} == fav
    client.post("/api/photos/marks", json={"photo_ids": ids[:1], "remove_tags": ["summer"], "rating": 2})
    d = client.get(f"/api/photos/{ids[0]}").json()
    assert d["tags"] == ["Crete"] and d["rating"] == 2


def test_xmp_sidecars_are_written_without_losing_other_data(client):
    client.put("/api/settings", json={"write_xmp": True})
    p = by_name(client)["p00.jpg"]
    client.post("/api/photos/marks", json={"photo_ids": [p["id"]], "rating": 5, "add_tags": ["Best of"]})
    rating, keywords = marks.read_sidecar(str(client.root / "p00.jpg"))
    assert rating == 5 and set(keywords) == {"Portfolio", "Best of", "Favorite"}
    p5 = by_name(client)["p05.jpg"]
    client.post("/api/photos/marks", json={"photo_ids": [p5["id"]], "rating": 3})
    assert marks.read_sidecar(str(client.root / "p05.jpg")) == (3, [])
    assert (client.root / "p05.xmp").exists()


def test_taste_learns_from_favourites(client, monkeypatch):
    from lumen import quality
    monkeypatch.setattr(quality, "TASTE_MIN_EXAMPLES", 3)
    items = by_name(client)
    everyone = [i["id"] for i in items.values()]
    client.post("/api/photos/marks", json={"photo_ids": everyone, "favorite": False, "rating": 0})
    reds = [i["id"] for n, i in items.items() if int(n[1:3]) % 4 == 0]
    before = quality.scores()
    client.post("/api/photos/marks", json={"photo_ids": reds, "favorite": True})
    after = quality.scores()
    gain = lambda ids, s: np.mean([s[i][1] for i in ids])  # noqa: E731
    others = [i for i in everyone if i not in reds]
    # Photos like the favourites move up the ranking; the rest move down.
    assert gain(reds, after) - gain(others, after) > gain(reds, before) - gain(others, before)


def test_new_files_are_picked_up_live(client):
    make(client.root / "new_live.jpg", 2)
    wait_for(client, lambda s: s["total"] == 13 and s["pending"] == 0, timeout=20)
    assert "new_live.jpg" in by_name(client)
    os.remove(client.root / "new_live.jpg")
    wait_for(client, lambda s: s["total"] == 12, timeout=20)


def test_phone_access_needs_the_pin_and_is_limited(client):
    phone = TestClient(app, client=("192.168.1.50", 5555))
    assert phone.get("/api/status").status_code == 403          # off by default
    r = client.post("/api/lan", json={"enabled": True, "pin": "246810"}).json()
    try:
        assert r["enabled"] and r["pin"] == "246810" and r["url"].startswith("http://")
        assert phone.get("/api/status").status_code == 401
        assert phone.get("/", follow_redirects=False).headers["location"] == "/login.html"
        assert phone.get("/login.html").status_code == 200
        assert phone.post("/api/login", json={"pin": "000000"}).status_code == 401
        assert phone.post("/api/login", json={"pin": "246810"}).status_code == 200
        assert phone.get("/api/status").status_code == 200
        assert phone.get("/api/photos").status_code == 200
        # Computer-only things stay computer-only.
        assert phone.get("/api/fs").status_code == 403
        assert phone.post("/api/photos/trash", json={"photo_ids": [1]}).status_code == 403
        assert phone.put("/api/settings", json={"write_xmp": False}).status_code == 403
        assert phone.post("/api/lan", json={"enabled": False}).status_code == 403
        # Brute force is slowed down.
        intruder = TestClient(app, client=("192.168.1.66", 5555))
        codes = [intruder.post("/api/login", json={"pin": f"{k:06d}"}).status_code for k in range(6)]
        assert codes[-1] == 429
        # A new PIN signs everyone out.
        client.post("/api/lan", json={"new_pin": True})
        assert phone.get("/api/status").status_code == 401
    finally:
        client.post("/api/lan", json={"enabled": False})


def test_requests_for_other_host_names_are_refused(client):
    # DNS rebinding: evil.example resolving to 127.0.0.1 must not reach the API.
    assert client.get("/api/status", headers={"host": "evil.example:8765"}).status_code == 400
    assert client.get("/api/status", headers={"host": "127.0.0.1:8765"}).status_code == 200
    assert client.get("/api/status", headers={"host": "localhost"}).status_code == 200


def test_share_as_gallery_and_video(client):
    ids = [i["id"] for i in by_name(client).values()][:5]
    col = client.post("/api/collections", json={"name": "Share me", "photo_ids": ids}).json()

    def run(kind):
        job = client.post(f"/api/collections/{col['id']}/share", json={"kind": kind}).json()["job_id"]
        for _ in range(600):
            j = client.get(f"/api/jobs/{job}").json()
            if j["state"] != "running":
                break
            time.sleep(0.1)
        assert j["state"] == "done", j
        return client.get(f"/api/jobs/{job}/download")

    z = zipfile.ZipFile(io.BytesIO(run("gallery").content))
    names = z.namelist()
    assert "index.html" in names and sum(n.startswith("photos/") for n in names) == 5
    assert "Share me" in z.read("index.html").decode()

    import av
    r = run("video")
    path = os.path.join(os.environ["LUMEN_HOME"], "out.mp4")
    open(path, "wb").write(r.content)
    with av.open(path) as v:
        s = v.streams.video[0]
        assert s.codec_context.name == "h264" and (s.width, s.height) == (1920, 1080)
        assert 15 <= float(v.duration / av.time_base) <= 25   # title + 5 photos x 4 s - crossfades


def test_cpu_setting_and_autostart(client, tmp_path, monkeypatch):
    client.put("/api/settings", json={"device": "cpu"})
    assert embedder.pick_device() == (["CPUExecutionProvider"], "CPU")
    client.put("/api/settings", json={"device": "auto"})
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    if autostart.platform.system() == "Linux":
        autostart.set_enabled(True)
        f = tmp_path / "autostart" / "lumen.desktop"
        assert f.exists() and "-m lumen --no-browser" in f.read_text().replace('"', "")
        assert autostart.is_enabled()
        autostart.set_enabled(False)
        assert not f.exists()


def test_photo_app_libraries_index_only_originals(client):
    lib = client.root / "Photos Library.photoslibrary"
    for sub in ("originals/A", "resources/derivatives/A", "database"):
        (lib / sub).mkdir(parents=True)
    make(lib / "originals/A/IMG_1.jpg", 1)
    make(lib / "resources/derivatives/A/IMG_1_thumb.jpg", 1)     # Apple's cached copy: must be skipped
    (client.root / "Catalog Previews.lrdata").mkdir()
    make(client.root / "Catalog Previews.lrdata/preview.jpg", 2)  # Lightroom cache
    (client.root / "@eaDir").mkdir()
    make(client.root / "@eaDir/SYNOPHOTO_THUMB.jpg", 3)           # Synology thumbnail
    client.post("/api/scan")
    s = wait_for(client, lambda s: s["pending"] == 0 and s["state"] == "idle" and s["total"] >= 13)
    names = set(by_name(client))
    assert "IMG_1.jpg" in names and s["total"] == 13
    assert not names & {"IMG_1_thumb.jpg", "preview.jpg", "SYNOPHOTO_THUMB.jpg"}
    listing = client.get("/api/fs", params={"path": str(client.root)}).json()
    assert any(d["library"] for d in listing["dirs"] if d["name"].endswith(".photoslibrary"))
