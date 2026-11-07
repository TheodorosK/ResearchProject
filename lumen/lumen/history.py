"""Remembered searches: the latest ones you ran, plus ones you pinned."""
import time

from . import db

MAX_LEN = 200
KEEP_RECENT = 100      # unpinned searches kept; the oldest beyond this are dropped


def _clean(q: str) -> str:
    return " ".join((q or "").split())[:MAX_LEN]


def record(q: str) -> None:
    q = _clean(q)
    if not q:
        return
    now = time.time()
    with db.tx() as c:
        c.execute(
            "INSERT INTO search_history(query, last_used, uses) VALUES(?,?,1) "
            "ON CONFLICT(query) DO UPDATE SET last_used=excluded.last_used, uses=uses+1", (q, now))
        c.execute(
            "DELETE FROM search_history WHERE pinned=0 AND query NOT IN "
            "(SELECT query FROM search_history WHERE pinned=0 ORDER BY last_used DESC LIMIT ?)", (KEEP_RECENT,))


def pin(q: str, pinned: bool = True) -> bool:
    """Pin (or unpin) a search. Pinning one that was never run saves it too."""
    q = _clean(q)
    if not q:
        return False
    with db.tx() as c:
        c.execute(
            "INSERT INTO search_history(query, last_used, uses, pinned) VALUES(?,?,0,?) "
            "ON CONFLICT(query) DO UPDATE SET pinned=excluded.pinned", (q, time.time(), int(pinned)))
    return True


def forget(q: str) -> None:
    with db.tx() as c:
        c.execute("DELETE FROM search_history WHERE query=?", (_clean(q),))


def clear() -> None:
    """Forget every unpinned search."""
    with db.tx() as c:
        c.execute("DELETE FROM search_history WHERE pinned=0")


def items(limit: int = 12):
    """Pinned searches first (most used), then the most recent."""
    rows = db.q("SELECT query, pinned, uses, last_used FROM search_history "
                "ORDER BY pinned DESC, CASE WHEN pinned=1 THEN uses END DESC, last_used DESC LIMIT ?", (limit,))
    return [{"query": r["query"], "pinned": bool(r["pinned"])} for r in rows]
