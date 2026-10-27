"""People, places, duplicates and copy-to-folder."""
import os
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from PIL import ImageFilter

from lumen import config, quality, search
from lumen.indexer import dhash
from lumen.server import app


def textured(seed, size=(640, 480)):
    """Blotchy colours plus fine grain, so there is both structure (for hashes) and detail (for sharpness)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (12, 16, 3), dtype=np.uint8)
    base = np.asarray(Image.fromarray(small).resize(size, Image.Resampling.BICUBIC), dtype=np.int16)
    grain = rng.integers(-40, 40, (size[1], size[0], 1), dtype=np.int16)
    return Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))


def with_gps(im, path, lat, lon):
    exif = Image.Exif()
    gps = exif.get_ifd(0x8825)
    gps.update({1: "N" if lat >= 0 else "S", 2: (abs(lat), 0.0, 0.0),
                3: "E" if lon >= 0 else "W", 4: (abs(lon), 0.0, 0.0)})
    im.save(path, exif=exif)


def wait_settled(c, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get("/api/status").json()
        if s["pending"] == 0 and s["enrich_pending"] == 0 and s["state"] == "idle":
            return s
        time.sleep(0.1)
    raise AssertionError(f"indexer never settled: {s}")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("library")
    base = textured(1)
    base.save(root / "beach.jpg", quality=95)
    shutil.copy(root / "beach.jpg", root / "beach copy.jpg")
    base.resize((320, 240)).save(root / "beach small.jpg", quality=70)
    textured(2).save(root / "other.jpg")
    with_gps(textured(3), root / "athens.jpg", 37.9715, 23.7257)
    with_gps(textured(4), root / "paris.jpg", 48.8584, -2.2945)
    # A burst: two framings of one scene, one of them out of focus.
    scene = textured(7, size=(1000, 600))
    # Similar shots need real capture times (a camera date, or one in the file name).
    scene.crop((0, 0, 640, 480)).save(root / "IMG_20240601_101500.jpg", quality=95)
    scene.crop((120, 60, 760, 540)).filter(ImageFilter.GaussianBlur(6)).save(root / "IMG_20240601_101530.jpg", quality=95)
    with TestClient(app) as c:
        c.root = root
        assert c.post("/api/folders", json={"path": str(root)}).status_code == 200
        wait_settled(c)
        yield c
        # Leave an empty library for the other test modules, which share this LUMEN_HOME.
        for f in c.get("/api/folders").json():
            c.delete(f"/api/folders/{f['id']}")
        for col in c.get("/api/collections").json():
            c.delete(f"/api/collections/{col['id']}")


def by_name(c):
    return {i["name"]: i["id"] for i in c.get("/api/photos?limit=500").json()["items"]}


def test_dhash_survives_resize_but_not_different_photos():
    a, b = dhash(textured(1)), dhash(textured(1).resize((200, 150)))
    assert bin((a ^ b) & (2**64 - 1)).count("1") <= 6
    assert bin((a ^ dhash(textured(9))) & (2**64 - 1)).count("1") > 12


def test_duplicates_are_grouped_with_best_copy_first(client):
    ids = by_name(client)
    res = client.get("/api/duplicates").json()
    groups = [set(g["photo_ids"]) for g in res["groups"]]
    trio = {ids["beach.jpg"], ids["beach copy.jpg"], ids["beach small.jpg"]}
    assert trio in groups
    g = next(g for g in res["groups"] if set(g["photo_ids"]) == trio)
    assert g["kind"] == "near" and g["keep"] in (ids["beach.jpg"], ids["beach copy.jpg"])
    assert not any(ids["other.jpg"] in s for s in groups)

    assert client.post("/api/duplicates/ignore", json={"key": g["key"]}).json()["ok"]
    assert trio not in [set(x["photo_ids"]) for x in client.get("/api/duplicates").json()["groups"]]


def test_quality_measures_sharpness_and_clipping():
    sharp = textured(1, size=(512, 384))
    blurry = sharp.filter(ImageFilter.GaussianBlur(4))
    assert quality.measure(sharp)["sharpness"] > 3 * quality.measure(blurry)["sharpness"]
    blown = Image.new("RGB", (100, 100), (255, 255, 255))
    assert quality.measure(blown)["hi_clip"] == 1.0


def test_bursts_keep_the_sharpest_shot(client):
    ids = by_name(client)
    groups = client.get("/api/duplicates?kind=bursts").json()["groups"]
    g = next(g for g in groups if ids["IMG_20240601_101500.jpg"] in g["photo_ids"])
    # (The stand-in embedder sees all these grey-ish test images as alike, so the burst may hold more.)
    assert ids["IMG_20240601_101530.jpg"] in g["photo_ids"]
    assert g["photo_ids"].index(ids["IMG_20240601_101500.jpg"]) < g["photo_ids"].index(ids["IMG_20240601_101530.jpg"])


def test_best_shots_and_quality_sort(client):
    ids = by_name(client)
    s = quality.scores()
    assert s[ids["IMG_20240601_101500.jpg"]][0] > s[ids["IMG_20240601_101530.jpg"]][0]
    home = client.get("/api/home").json()
    assert "moods" in home and home["best"]
    best = client.get("/api/photos", params={"order": "best", "limit": 100}).json()["items"]
    names = [i["name"] for i in best]
    assert names.index("IMG_20240601_101500.jpg") < names.index("IMG_20240601_101530.jpg")
    detail = client.get(f"/api/photos/{ids['IMG_20240601_101500.jpg']}").json()
    assert 0 <= detail["quality"] <= 100


def test_expression_words_are_understood():
    assert search.expression_in_query("everyone smiling at dinner") == ("happiness", True)
    assert search.expression_in_query("Maria crying") == ("sadness", False)
    assert search.expression_in_query("χαμογελαστά παιδιά") == ("happiness", False)
    assert search.expression_in_query("όλοι γελάνε") == ("happiness", True)
    assert search.expression_in_query("a mountain lake") == (None, False)


def test_places_come_from_gps(client):
    ids = by_name(client)
    pts = {p[0]: (p[1], p[2]) for p in client.get("/api/map").json()["points"]}
    assert set(pts) == {ids["athens.jpg"], ids["paris.jpg"]}
    lat, lon = pts[ids["paris.jpg"]]
    assert abs(lat - 48.8584) < 1e-3 and abs(lon + 2.2945) < 1e-3
    detail = client.get(f"/api/photos/{ids['athens.jpg']}").json()
    assert abs(detail["lat"] - 37.9715) < 1e-3


def test_copy_collection_to_folder_never_overwrites(client, tmp_path):
    ids = by_name(client)
    col = client.post("/api/collections", json={"name": "Trip", "photo_ids": [ids["athens.jpg"], ids["paris.jpg"]]}).json()

    def copy():
        r = client.post(f"/api/collections/{col['id']}/copy", json={"dest": str(tmp_path), "subfolder": "Trip: best/of"}).json()
        for _ in range(100):
            j = client.get(f"/api/jobs/{r['job_id']}").json()
            if j["state"] != "running":
                return r, j
            time.sleep(0.05)

    r, j = copy()
    assert j["state"] == "done" and j["result"]["copied"] == 2
    assert os.path.basename(r["dest"]) == "Trip bestof"
    copy()
    assert sorted(os.listdir(r["dest"])) == ["athens (1).jpg", "athens.jpg", "paris (1).jpg", "paris.jpg"]


def test_trash_moves_original_and_forgets_it(client):
    ids = by_name(client)
    res = client.post("/api/photos/trash", json={"photo_ids": [ids["beach copy.jpg"]]}).json()
    assert res["trashed"] == 1
    assert not (client.root / "beach copy.jpg").exists()
    assert "beach copy.jpg" not in by_name(client)
    trash = os.path.join(os.environ["XDG_DATA_HOME"], "Trash", "files")
    assert "beach copy.jpg" in os.listdir(trash)


def test_old_library_is_migrated(tmp_path):
    """A database from the first release gains the new columns and gets a catch-up pass."""
    home = tmp_path / "old"
    home.mkdir()
    conn = sqlite3.connect(home / "lumen.db")
    conn.executescript("""
        CREATE TABLE folders (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, added_at REAL NOT NULL, last_scan REAL);
        CREATE TABLE photos (id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL, folder_id INTEGER NOT NULL,
            size INTEGER, mtime REAL, width INTEGER, height INTEGER, taken_at REAL, camera TEXT,
            status TEXT NOT NULL DEFAULT 'pending', error TEXT, embedding BLOB, model TEXT);
        INSERT INTO folders VALUES (1, '/nowhere', 0, NULL);
        INSERT INTO photos(id, path, folder_id, status) VALUES (1, '/nowhere/a.jpg', 1, 'ready');
    """)
    conn.commit()
    conn.close()
    code = ("from lumen import db; db.init(); "
            "print(sorted(r['name'] for r in db.q('PRAGMA table_info(photos)')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(__file__)),
                         env={**os.environ, "LUMEN_HOME": str(home)}).stdout
    for col in ("phash", "lat", "lon", "meta_v", "faces_v"):
        assert col in out


faces_on = pytest.mark.skipif(not config.FACES, reason="set LUMEN_TEST_FACE_MODELS to run face tests")


@faces_on
def test_people_are_found_named_searched_and_corrected(client, tmp_path):
    from skimage import data

    astro = Image.fromarray(data.astronaut())
    folder = tmp_path / "faces"
    folder.mkdir()
    astro.save(folder / "a1.jpg")
    astro.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(folder / "a2.jpg")
    astro.crop((60, 0, 460, 400)).resize((640, 640)).save(folder / "a3.jpg")
    Image.fromarray(data.camera()).convert("RGB").save(folder / "cameraman.jpg")
    textured(5).save(folder / "noface.jpg")
    client.post("/api/folders", json={"path": str(folder)})
    time.sleep(0.3)
    wait_settled(client)

    people = client.get("/api/people").json()
    assert len(people) == 1 and people[0]["count"] == 3 and people[0]["name"] is None
    pid = people[0]["id"]

    # Expressions: the astronaut smiles, the cameraman doesn't.
    ids = by_name(client)
    face = client.get(f"/api/photos/{ids['a1.jpg']}").json()["people"][0]
    assert face["expression"] == "happiness" and face["expression_emoji"] == "😊"
    res = client.get("/api/search", params={"q": "smiling"}).json()
    assert res["parsed"]["expression"]["label"] == "happiness"
    ranked = [i["name"] for i in res["items"]]
    assert ranked.index("a1.jpg") < ranked.index("cameraman.jpg")
    assert client.get(f"/api/faces/{people[0]['face_id']}/thumb").status_code == 200

    client.patch(f"/api/people/{pid}", json={"name": "Eileen"})
    names = {i["name"] for i in client.get("/api/search", params={"q": "photos of Eileen"}).json()["items"]}
    assert names == {"a1.jpg", "a2.jpg", "a3.jpg"}
    res = client.get("/api/search", params={"q": "Eileen red"}).json()
    assert res["parsed"]["people"][0]["name"] == "Eileen"
    assert {i["name"] for i in res["items"]} <= {"a1.jpg", "a2.jpg", "a3.jpg"}

    ids = by_name(client)
    detail = client.get(f"/api/photos/{ids['a1.jpg']}").json()
    assert detail["people"][0]["name"] == "Eileen"

    client.post(f"/api/people/{pid}/remove-photos", json={"photo_ids": [ids["a3.jpg"]]})
    assert client.get(f"/api/people/{pid}").json()["count"] == 2
    # Re-grouping must not put the rejected face back.
    from lumen import faces
    faces.cluster_new_faces()
    assert client.get(f"/api/people/{pid}").json()["count"] == 2


def test_framed_copies_are_duplicates_and_the_original_is_kept(client, tmp_path):
    from lumen.indexer import trim_frame
    photo = textured(11, size=(600, 400))
    framed = Image.new("RGB", (700, 700), "white")
    framed.paste(photo, (50, 150))
    assert trim_frame(framed).size == (600, 400)
    assert trim_frame(photo) is photo                      # no frame: untouched
    folder = tmp_path / "framed"
    folder.mkdir()
    photo.save(folder / "original.jpg", quality=95)
    framed.save(folder / "instagram.jpg", quality=95)
    client.post("/api/folders", json={"path": str(folder)})
    time.sleep(0.3)
    wait_settled(client)
    ids = by_name(client)
    g = next(g for g in client.get("/api/duplicates").json()["groups"] if ids["original.jpg"] in g["photo_ids"])
    assert set(g["photo_ids"]) == {ids["original.jpg"], ids["instagram.jpg"]} and g["keep"] == ids["original.jpg"]
