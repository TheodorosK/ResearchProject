"""Text in photos, descriptions, pets, search by picture and by colour."""
import glob
import io
import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from lumen import colors, config, describe
from lumen.server import app


def wait_settled(c, timeout=90):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get("/api/status").json()
        if s["pending"] == 0 and s["enrich_pending"] == 0 and s["state"] == "idle":
            # OCR and pets run after indexing; wait until no candidate is left.
            from lumen import db
            if not db.one("SELECT 1 FROM photos WHERE status='ready' AND (ocr_v=0 OR pets_v=0) AND kind='photo'") \
                    or not config.FACES and not db.one("SELECT 1 FROM photos WHERE ocr_v=0 AND kind='photo'"):
                return s
        time.sleep(0.2)
    raise AssertionError(f"indexer never settled: {s}")


def sign(path, lines, bg=(240, 235, 220)):
    fonts = glob.glob("/usr/share/fonts/**/DejaVuSans-Bold.ttf", recursive=True) + \
        glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    font = ImageFont.truetype(fonts[0], 52) if fonts else ImageFont.load_default()
    im = Image.new("RGB", (900, 120 + 110 * len(lines)), bg)
    d = ImageDraw.Draw(im)
    for k, line in enumerate(lines):
        d.text((40, 50 + 110 * k), line, fill=(20, 20, 20), font=font)
    im.save(path)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["LUMEN_OCR"] = "all"
    os.environ["LUMEN_PETS"] = "all"
    root = tmp_path_factory.mktemp("roundB")
    sign(root / "menu.jpg", ["TAVERNA NIKOS", "Grilled octopus 14.50"])
    sign(root / "ticket.jpg", ["BOARDING PASS", "ATHENS - PARIS"], bg=(225, 235, 245))
    sign(root / "plaque.jpg", ["MANKIND IS CAPABLE", "OFANAWARENESS"], bg=(250, 250, 250))
    for k, rgb in enumerate([(230, 20, 20), (220, 40, 30), (30, 40, 230), (40, 60, 220), (20, 180, 60)]):
        arr = np.clip(np.array(rgb) + np.random.default_rng(k).integers(-10, 10, (200, 300, 3)), 0, 255)
        Image.fromarray(arr.astype(np.uint8)).save(root / f"plain_{k}.jpg")
    if config.FACES:
        from skimage import data
        cat = Image.fromarray(data.chelsea())
        cat.save(root / "cat1.jpg")
        cat.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(root / "cat2.jpg")
        cat.resize((600, 400)).save(root / "cat3.jpg")
    with TestClient(app) as c:
        c.post("/api/folders", json={"path": str(root)})
        wait_settled(c)
        yield c
        for f in c.get("/api/folders").json():
            c.delete(f"/api/folders/{f['id']}")
    os.environ["LUMEN_OCR"] = "auto"
    os.environ["LUMEN_PETS"] = "auto"


def by_name(c):
    return {i["name"]: i for i in c.get("/api/photos?limit=200").json()["items"]}


def test_text_in_photos_is_read_and_searchable(client):
    items = by_name(client)
    text = client.get(f"/api/photos/{items['menu.jpg']['id']}").json()["text"]
    assert "TAVERNA" in text and "octopus" in text.replace(" ", "").lower()
    res = client.get("/api/search", params={"q": "taverna"}).json()
    assert res["items"][0]["name"] == "menu.jpg"
    exact = client.get("/api/search", params={"q": '"boarding pass"'}).json()
    assert [i["name"] for i in exact["items"]] == ["ticket.jpg"] and exact["parsed"]["phrases"] == ["boarding pass"]
    # Words run together on a sign are still found.
    assert client.get("/api/search", params={"q": "awareness"}).json()["items"][0]["name"] == "plaque.jpg"


def test_read_text_fixes_zero_and_o_lookalikes():
    from lumen.ocr import tidy

    assert tidy("19014 KALAM0S, HOTEL 0LYMPIA") == "19014 KALAMOS, HOTEL OLYMPIA"
    assert tidy("SINCE 198O, OK 2O24") == "SINCE 1980, OK 2024"
    assert tidy("Room 101, Windows10, O2 arena, B2B") == "Room 101, Windows10, O2 arena, B2B"


def test_descriptions_have_a_shape():
    assert describe.caption({"subject": "a dog", "setting": "the beach", "moment": "at sunset"}) == \
        "A dog on the beach at sunset"
    assert describe.caption({"activity": "swimming", "setting": "the sea"}) == "Swimming by the sea"
    assert describe.caption({}) is None


def test_photo_detail_includes_description(client):
    pid = by_name(client)["plain_0.jpg"]["id"]
    seen = client.get(f"/api/photos/{pid}").json()["seen"]
    assert set(seen) == {"tags", "caption"}


def test_palette_and_colour_search(client):
    pal = colors.palette(Image.new("RGB", (50, 50), (255, 0, 0)))
    assert pal.startswith("ff0000:1.00")
    res = client.get("/api/search", params={"q": "", "colors": "e61414"}).json()
    assert [i["name"] for i in res["items"][:2]] == ["plain_0.jpg", "plain_1.jpg"] or \
        {i["name"] for i in res["items"][:2]} == {"plain_0.jpg", "plain_1.jpg"}
    assert res["parsed"]["colors"] == ["e61414"]


def test_search_by_picture(client):
    buf = io.BytesIO()
    Image.new("RGB", (300, 200), (35, 50, 225)).save(buf, "PNG")
    res = client.post("/api/search/image", files={"file": ("q.png", buf.getvalue(), "image/png")}).json()
    assert {i["name"] for i in res["items"][:2]} == {"plain_2.jpg", "plain_3.jpg"}
    bad = client.post("/api/search/image", files={"file": ("q.txt", b"not an image", "text/plain")})
    assert bad.status_code == 400


@pytest.mark.skipif(not config.FACES, reason="needs the face/pet models (LUMEN_TEST_FACE_MODELS)")
def test_pets_are_found_named_and_searchable(client):
    pets = client.get("/api/people", params={"kind": "pet"}).json()
    assert len(pets) == 1 and pets[0]["count"] == 3 and pets[0]["species"] == "cat"
    assert client.get("/api/people").json() == [] or all(p["kind"] == "person" for p in client.get("/api/people").json())
    client.patch(f"/api/people/{pets[0]['id']}", json={"name": "Luna"})
    res = client.get("/api/search", params={"q": "Luna"}).json()
    assert {i["name"] for i in res["items"]} == {"cat1.jpg", "cat2.jpg", "cat3.jpg"}
    detail = client.get(f"/api/photos/{by_name(client)['cat1.jpg']['id']}").json()
    assert any(p["kind"] == "pet" and p["name"] == "Luna" for p in detail["people"])
