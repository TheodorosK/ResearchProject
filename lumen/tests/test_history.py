import pytest

from lumen import db, history


@pytest.fixture(autouse=True)
def clean():
    db.init()
    with db.tx() as c:
        c.execute("DELETE FROM search_history")
    yield


def test_records_and_orders_recent_first():
    for q in ["beach", "dog", "snow"]:
        history.record(q)
    assert [i["query"] for i in history.items()] == ["snow", "dog", "beach"]


def test_repeat_moves_to_front_and_whitespace_is_ignored():
    history.record("beach")
    history.record("dog")
    history.record("  beach  ")
    assert [i["query"] for i in history.items()] == ["beach", "dog"]
    assert db.one("SELECT uses FROM search_history WHERE query='beach'")["uses"] == 2


def test_empty_is_ignored():
    history.record("   ")
    assert history.items() == []
    assert history.pin("  ") is False


def test_pinned_come_first_and_survive_clear():
    history.record("beach")
    history.record("dog")
    history.pin("beach")
    history.pin("never run before")
    items = history.items()
    assert [i["pinned"] for i in items] == [True, True, False]
    history.clear()
    assert {i["query"] for i in history.items()} == {"beach", "never run before"}
    history.pin("beach", False)
    history.clear()
    assert [i["query"] for i in history.items()] == ["never run before"]


def test_forget_and_limit():
    for i in range(history.KEEP_RECENT + 20):
        history.record(f"q{i}")
    assert db.one("SELECT COUNT(*) n FROM search_history")["n"] == history.KEEP_RECENT
    assert history.items(5)[0]["query"] == f"q{history.KEEP_RECENT + 19}"
    history.forget("q119")
    assert "q119" not in {i["query"] for i in history.items(200)}


def test_api_remembers_only_when_asked(tmp_path):
    from fastapi.testclient import TestClient
    from lumen.server import app

    with TestClient(app) as c:
        c.post("/api/searches/clear")
        c.get("/api/search", params={"q": "red"})                       # typing as you go: not remembered
        assert c.get("/api/searches").json() == []
        c.get("/api/search", params={"q": "red", "remember": "true"})
        c.get("/api/search", params={"q": "  ", "remember": "true"})   # blank is ignored
        assert [i["query"] for i in c.get("/api/searches").json()] == ["red"]
        assert c.post("/api/searches/pin", json={"query": "red"}).json()[0]["pinned"] is True
        assert c.get("/api/home").json()["history"][0]["query"] == "red"
        assert c.post("/api/searches/pin", json={"query": "  "}).status_code in (400, 422)
        assert c.post("/api/searches/forget", json={"query": "red"}).json() == []
