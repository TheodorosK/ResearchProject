"""Find copies of the same photo, and bursts of near-identical shots."""
import threading
import time

from . import db
from .vectors import index

NEAR_BITS = 6           # max differing bits of the 64-bit hash for "same picture, re-saved or resized"
NEAR_COSINE = 0.93      # ...and the model must agree they look the same
BURST_SECONDS = 120     # shots this close in time...
BURST_COSINE = 0.90     # ...and this alike count as one moment
MAX_BUCKET = 200        # ignore hash buckets this crowded (e.g. all-black frames) to stay fast

_cache = {"key": None, "data": None, "at": 0.0}
_lock = threading.Lock()


def _u64(v):
    return v & 0xFFFFFFFFFFFFFFFF


def _popcount(x: int) -> int:
    return bin(x).count("1")


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        out = {}
        for a in self.parent:
            out.setdefault(self.find(a), []).append(a)
        return [g for g in out.values() if len(g) > 1]


def _cosine_lookup():
    ids, mat = index.snapshot()
    pos = {int(i): k for k, i in enumerate(ids)}

    def cos(a, b):
        ia, ib = pos.get(a), pos.get(b)
        if ia is None or ib is None:
            return None
        return float(mat[ia] @ mat[ib])

    return cos


def _duplicate_groups(rows, cos):
    """Locality-sensitive hashing: split each 64-bit hash into four 16-bit bands. Two hashes within
    6 bits of each other must share at least one band exactly, so we only compare within buckets."""
    hashes = {r["id"]: _u64(r["phash"]) for r in rows}
    buckets = {}
    for pid, hsh in hashes.items():
        for band in range(4):
            buckets.setdefault((band, (hsh >> (16 * band)) & 0xFFFF), []).append(pid)
    uf = _UnionFind()
    seen = set()
    for members in buckets.values():
        if len(members) < 2 or len(members) > MAX_BUCKET:
            continue
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                if _popcount(hashes[a] ^ hashes[b]) <= NEAR_BITS:
                    c = cos(a, b)
                    if c is None or c >= NEAR_COSINE:
                        uf.union(a, b)
    return uf.groups()


def _burst_groups(rows, cos, exclude):
    uf = _UnionFind()
    timed = sorted((r["taken_at"], r["id"]) for r in rows if r["taken_at"])
    for i, (t, a) in enumerate(timed):
        if a in exclude:
            continue
        for k in range(i + 1, len(timed)):
            t2, b = timed[k]
            if t2 - t > BURST_SECONDS:
                break
            if b in exclude:
                continue
            c = cos(a, b)
            if c is not None and c >= BURST_COSINE:
                uf.union(a, b)
    return uf.groups()


def _describe(group, info, kind):
    items = sorted(group, key=lambda i: (-(info[i]["width"] or 0) * (info[i]["height"] or 0),
                                         -(info[i]["size"] or 0), info[i]["taken_at"] or 0))
    sizes = {info[i]["size"] for i in group}
    hashes = {info[i]["phash"] for i in group}
    if kind == "duplicate":
        kind = "identical" if len(sizes) == 1 and len(hashes) == 1 else "near"
    return {"key": ",".join(map(str, sorted(group))), "kind": kind, "keep": items[0], "photo_ids": items}


def find(kind="duplicates"):
    """Groups of duplicates (`identical`/`near`) or of similar shots (`burst`), best photo first."""
    with _lock:
        # Removing photos or dismissing a group must show at once; other changes (indexing in
        # progress) refresh at most once a minute, since a full pass over 100k photos takes a moment.
        hard = (db.one("SELECT COUNT(*) n FROM photos")["n"], db.one("SELECT COUNT(*) n FROM dup_ignored")["n"])
        soft = (index.version, tuple(db.one("SELECT MAX(id), SUM(meta_v) FROM photos")))
        stale = _cache["key"] is None or _cache["key"][0] != hard or (
            _cache["key"][1] != soft and time.time() - _cache["at"] > 60)
        if stale:
            version = (hard, soft)
            rows = db.q("SELECT id, phash, size, width, height, taken_at FROM photos "
                        "WHERE status='ready' AND phash IS NOT NULL")
            info = {r["id"]: r for r in rows}
            cos = _cosine_lookup()
            ignored = {r["group_key"] for r in db.q("SELECT group_key FROM dup_ignored")}
            dups = [_describe(g, info, "duplicate") for g in _duplicate_groups(rows, cos)]
            in_dups = {i for g in dups for i in g["photo_ids"]}
            bursts = [_describe(g, info, "burst") for g in _burst_groups(rows, cos, in_dups)]
            dups = [g for g in dups if g["key"] not in ignored]
            bursts = [g for g in bursts if g["key"] not in ignored]
            for groups in (dups, bursts):
                groups.sort(key=lambda g: -max(info[i]["taken_at"] or 0 for i in g["photo_ids"]))
            _cache.update(key=version, at=time.time(), data={"duplicates": dups, "bursts": bursts})
        return _cache["data"][kind]


def ignore(group_key: str):
    with db.tx() as c:
        c.execute("INSERT OR IGNORE INTO dup_ignored(group_key) VALUES(?)", (group_key,))


def summary():
    d, b = find("duplicates"), find("bursts")
    return {
        "duplicate_groups": len(d),
        "duplicate_extra": sum(len(g["photo_ids"]) - 1 for g in d),
        "burst_groups": len(b),
        "burst_extra": sum(len(g["photo_ids"]) - 1 for g in b),
    }
