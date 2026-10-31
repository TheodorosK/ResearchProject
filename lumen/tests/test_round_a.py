"""Places, dates, memories, refinement and videos."""
import time
from datetime import date, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from lumen import dates, places
from lumen.server import app

COLORS = {"red": (230, 20, 20), "green": (20, 200, 40), "blue": (30, 40, 230)}


def photo(path, rgb, when=None, gps=None):
    exif = Image.Exif()
    if when:
        exif.get_ifd(0x8769)[36867] = when.strftime("%Y:%m:%d %H:%M:%S")
    if gps:
        lat, lon = gps
        exif.get_ifd(0x8825).update({1: "N" if lat >= 0 else "S", 2: (abs(lat), 0.0, 0.0),
                                     3: "E" if lon >= 0 else "W", 4: (abs(lon), 0.0, 0.0)})
    rng = np.random.default_rng(abs(hash(path.name)) % 2**32)
    arr = np.clip(np.array(rgb, dtype=np.int16) + rng.integers(-15, 15, (240, 320, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(arr).save(path, exif=exif)


def video(path, seconds=10, fps=4):
    """First half red, second half blue."""
    import av

    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = 320, 240, "yuv420p"
        for k in range(seconds * fps):
            rgb = COLORS["red"] if k < seconds * fps / 2 else COLORS["blue"]
            frame = av.VideoFrame.from_ndarray(np.full((240, 320, 3), rgb, dtype=np.uint8), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def wait_settled(c, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        s = c.get("/api/status").json()
        if s["pending"] == 0 and s["enrich_pending"] == 0 and s["state"] == "idle":
            return s
        time.sleep(0.1)
    raise AssertionError(f"indexer never settled: {s}")


CRETE, PARIS, HOME = (35.51, 24.02), (48.853, 2.349), (40.640, 22.944)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("roundA")
    # A trip to Crete (3 days, 9 photos), a day in Paris, and home photos in Thessaloniki.
    for k in range(9):
        photo(root / f"crete_{k}.jpg", COLORS["blue"], datetime(2019, 7, 12 + k // 3, 10 + k % 3), CRETE)
    for k in range(6):
        photo(root / f"paris_{k}.jpg", COLORS["green"], datetime(2021, 12, 24, 18, k * 5), PARIS)
    for k in range(8):
        photo(root / f"home_{k}.jpg", COLORS["red"], datetime(2022, 3, 1 + k * 3, 9), HOME)
    video(root / "clip.mp4")
    with TestClient(app) as c:
        c.root = root
        c.post("/api/folders", json={"path": str(root)})
        wait_settled(c)
        yield c
        for f in c.get("/api/folders").json():
            c.delete(f"/api/folders/{f['id']}")


def names(res):
    return [i["name"] for i in res["items"]]


def test_place_names_are_found_offline(client):
    items = {i["name"]: i for i in client.get("/api/photos?limit=100").json()["items"]}
    assert items["crete_0.jpg"]["place"] == "Chania, Crete, Greece"
    assert items["paris_0.jpg"]["place"].startswith("Paris")
    assert {p["city"] for p in client.get("/api/places").json()} >= {"Chania", "Paris"}
    # Home photos are labelled with a suburb, but the city's name still finds them.
    res = client.get("/api/search", params={"q": "Thessaloniki"}).json()
    assert set(names(res)) == {f"home_{k}.jpg" for k in range(8)}


def test_search_by_place(client):
    res = client.get("/api/search", params={"q": "photos in Crete"}).json()
    assert res["parsed"]["places"] == ["Crete"]
    assert sorted(names(res)) == sorted(f"crete_{k}.jpg" for k in range(9))
    assert set(names(client.get("/api/search", params={"q": "blue in Paris"}).json())) <= {f"paris_{k}.jpg" for k in range(6)}
    # Greek alias
    assert set(names(client.get("/api/search", params={"q": "Κρήτη"}).json())) == {f"crete_{k}.jpg" for k in range(9)}


def test_date_parsing():
    today = date(2026, 10, 26)
    rest, w = dates.parse("beach summer 2019", today)
    assert rest == "beach" and w.label == "Summer 2019"
    rest, w = dates.parse("last christmas with family", today)
    assert rest == "with family" and w.label == "Christmas 2025"
    assert dates.parse("photos from the 90s", today)[1].years == set(range(1990, 2000))
    assert dates.parse("kids 2015-2018", today)[1].years == {2015, 2016, 2017, 2018}
    rest, w = dates.parse("city at night", today)
    assert rest == "city at night" and w.hours and not w.is_filter  # soft: night stays in the query
    assert dates.parse("καλοκαίρι 2021 θάλασσα", today)[0] == "θάλασσα"
    assert not dates.parse("a red car", today)[1]


def test_search_by_date_and_place_together(client):
    res = client.get("/api/search", params={"q": "Christmas in Paris"}).json()
    assert res["parsed"]["when"] == "Christmas" and set(names(res)) == {f"paris_{k}.jpg" for k in range(6)}
    assert names(client.get("/api/search", params={"q": "Paris summer 2019"}).json()) == []
    assert set(names(client.get("/api/search", params={"q": "July 2019"}).json())) == {f"crete_{k}.jpg" for k in range(9)}


def test_memories_find_the_trip_and_events(client):
    ms = client.get("/api/memories").json()["items"]
    titles = [m["title"] for m in ms]
    trip = next(m for m in ms if m["kind"] == "trip")
    assert trip["title"] == "Trip to Chania" and trip["count"] == 9 and trip["when"] == "12–14 Jul 2019"
    assert any("Paris" in t for t in titles)
    detail = client.get(f"/api/memories/{trip['key']}").json()
    assert len(detail["items"]) == 9
    col = client.post(f"/api/memories/{trip['key']}/save").json()
    assert col["count"] == 9 and col["name"] == "Trip to Chania"


def test_feedback_refines_results(client):
    base = client.get("/api/search", params={"q": "red"}).json()["items"]
    blue = next(i["id"] for i in client.get("/api/search", params={"q": "blue"}).json()["items"]
                if i["name"].startswith("crete"))
    first_red = base[0]["id"]
    refined = client.get("/api/search", params={"q": "red", "like": str(blue), "dislike": str(first_red)}).json()
    ids = [i["id"] for i in refined["items"]]
    assert first_red not in ids
    assert refined["parsed"]["like"] == [blue]


def test_videos_are_searchable_by_moment(client):
    items = {i["name"]: i for i in client.get("/api/photos?limit=100").json()["items"]}
    v = items["clip.mp4"]
    assert v["kind"] == "video" and 9 <= v["duration"] <= 11
    hit = next(i for i in client.get("/api/search", params={"q": "blue"}).json()["items"] if i["name"] == "clip.mp4")
    assert hit["t"] >= 5  # the blue half
    hit = next(i for i in client.get("/api/search", params={"q": "red"}).json()["items"] if i["name"] == "clip.mp4")
    assert hit["t"] < 5
    r = client.get(f"/api/photos/{v['id']}/original", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206 and len(r.content) == 100


def test_dates_come_from_camera_then_file_name_then_file():
    from lumen.indexer import date_from_filename
    from datetime import datetime as dt
    assert dt.fromtimestamp(date_from_filename("IMG_20230712_142233.jpg")) == dt(2023, 7, 12, 14, 22, 33)
    assert dt.fromtimestamp(date_from_filename("PXL_20230712_142233123.jpg")) == dt(2023, 7, 12, 14, 22, 33)
    assert dt.fromtimestamp(date_from_filename("WhatsApp Image 2023-07-12 at 14.22.33.jpeg")) == dt(2023, 7, 12, 14, 22, 33)
    assert date_from_filename("DSC_0042.JPG") is None and date_from_filename("photo_2019-02-30.jpg") is None


def test_file_dates_stay_out_of_memories(client):
    # Photos with camera dates are in memories; ones dated only by the file system are not.
    items = {i["name"]: i for i in client.get("/api/photos?limit=100").json()["items"]}
    assert items["crete_0.jpg"]["date_source"] == "camera"
    assert items["clip.mp4"]["date_source"] == "file"
    in_memories = {i for m in client.get("/api/memories").json()["items"]
                   for i in [x["id"] for x in client.get(f"/api/memories/{m['key']}").json()["items"]]}
    assert items["clip.mp4"]["id"] not in in_memories
