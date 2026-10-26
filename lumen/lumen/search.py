"""Semantic search over the vector index, plus the 'Explore' and 'On this day' discoveries."""
import random
import re
import threading
from datetime import datetime

import numpy as np

from . import db, embedder, faces
from .vectors import index

# Cutoffs on a robust z-score (median/MAD of the whole library), so they hold for any model and
# still work when a large share of the library matches the query.
TIER_BEST, TIER_GOOD, TIER_MAYBE = 3.5, 2.5, 1.5
# Searches always show at least this many closest photos, even if none match strongly.
MIN_RESULTS = 24

CONCEPTS = [
    ("Beaches & sea", "🏖️", "a beach by the sea"),
    ("Mountains", "🏔️", "mountains and hiking"),
    ("Sunsets", "🌅", "a sunset sky"),
    ("City life", "🏙️", "city streets and buildings"),
    ("Night lights", "🌃", "a city at night with lights"),
    ("Food & drinks", "🍝", "a plate of food on a table"),
    ("Coffee moments", "☕", "a cup of coffee"),
    ("Dogs", "🐶", "a dog"),
    ("Cats", "🐱", "a cat"),
    ("Wildlife", "🦌", "wild animals in nature"),
    ("Birds", "🐦", "a bird"),
    ("Flowers & gardens", "🌸", "flowers in a garden"),
    ("Forests", "🌲", "a forest with trees"),
    ("Snow & winter", "❄️", "snow in winter"),
    ("Celebrations", "🎉", "a party celebration"),
    ("Birthdays", "🎂", "a birthday cake with candles"),
    ("Weddings", "💍", "a wedding"),
    ("Kids", "🧒", "children playing"),
    ("Group photos", "👨‍👩‍👧", "a group of people posing together"),
    ("Selfies & portraits", "🤳", "a close-up portrait of a person"),
    ("Sports", "⚽", "people playing sports"),
    ("Concerts", "🎸", "a concert with a stage"),
    ("Road trips", "🚗", "a car on a road trip"),
    ("Boats", "⛵", "a boat on the water"),
    ("Flights", "✈️", "an airplane or a view from an airplane window"),
    ("Architecture", "🏛️", "historic architecture and monuments"),
    ("Churches", "⛪", "a church"),
    ("Museums & art", "🖼️", "art in a museum"),
    ("Home & cozy", "🛋️", "a cozy living room at home"),
    ("Pools", "🏊", "a swimming pool"),
    ("Lakes & rivers", "🏞️", "a lake or a river"),
    ("Fields & countryside", "🌾", "the countryside with fields"),
    ("Desert", "🏜️", "a desert"),
    ("Rain & clouds", "🌧️", "a rainy cloudy day"),
    ("Cars", "🏎️", "a close-up of a car"),
    ("Screenshots", "📱", "a screenshot of a phone screen"),
    ("Documents", "📄", "a document or receipt with text"),
    ("Whiteboards & notes", "📝", "handwritten notes or a whiteboard"),
    ("Babies", "👶", "a baby"),
    ("Christmas", "🎄", "a christmas tree and decorations"),
    ("Fireworks", "🎆", "fireworks"),
    ("Islands", "🏝️", "a greek island village with white houses"),
    ("Market & shopping", "🛍️", "a market or shopping street"),
    ("Restaurants", "🍷", "dinner at a restaurant"),
    ("Desserts", "🍰", "dessert and sweets"),
    ("Underwater", "🤿", "underwater scene"),
    ("Bikes", "🚲", "a bicycle"),
    ("Horses", "🐴", "a horse"),
]

SUGGESTIONS = [
    "golden hour at the beach", "my dog sleeping on the couch", "a birthday cake with candles",
    "snowy mountains", "dinner with friends", "kids laughing", "old stone streets",
    "a cozy rainy day", "fireworks over the city", "the blue sea from a boat",
    "people dancing", "a plate of pasta", "sunrise from an airplane window",
    "a red car", "flowers in spring", "that summer on the island",
]


NEGATIVE = re.compile(r'(?:^|(?<=\s))-("[^"]+"|\S+)')


def parse_query(raw: str):
    """'beach -people "no umbrellas"' -> positive text plus negatives prefixed with '-'."""
    negatives = [m.strip('"') for m in NEGATIVE.findall(raw)]
    positive = NEGATIVE.sub(" ", raw)
    return " ".join(positive.split()), [n for n in negatives if n]


# Words that carry no visual meaning once a person's name is taken out ("photos of Maria").
FILLER = {"photo", "photos", "picture", "pictures", "pics", "pic", "of", "with", "and", "me", "the", "a", "all"}


def people_in_query(text: str):
    """Pull named people out of a query: ('Maria at the beach' -> 'at the beach', [Maria])."""
    found = []
    rows = db.q("SELECT id, name FROM persons WHERE name IS NOT NULL AND name <> '' AND hidden=0")
    for r in sorted(rows, key=lambda r: -len(r["name"])):
        pat = re.compile(r"(?<!\w)" + re.escape(r["name"]) + r"(?:'s|’s)?(?!\w)", re.IGNORECASE)
        if pat.search(text):
            found.append({"id": r["id"], "name": r["name"]})
            text = pat.sub(" ", text)
    text = " ".join(text.split())
    if all(w.lower().strip(",.") in FILLER for w in text.split()):
        text = ""
    return text, found


def _filtered(ids, filters):
    """Boolean mask of ids allowed by folder/year/collection filters."""
    if not filters:
        return None
    where, args = [], []
    if filters.get("folder_id"):
        where.append("folder_id=?")
        args.append(int(filters["folder_id"]))
    if filters.get("year_from"):
        where.append("taken_at >= ?")
        args.append(datetime(int(filters["year_from"]), 1, 1).timestamp())
    if filters.get("year_to"):
        where.append("taken_at < ?")
        args.append(datetime(int(filters["year_to"]) + 1, 1, 1).timestamp())
    if filters.get("collection_id"):
        where.append("id IN (SELECT photo_id FROM collection_items WHERE collection_id=?)")
        args.append(int(filters["collection_id"]))
    if not where:
        return None
    allowed = {r["id"] for r in db.q(f"SELECT id FROM photos WHERE {' AND '.join(where)}", args)}
    return np.fromiter((i in allowed for i in ids), dtype=bool, count=len(ids))


def robust_z(scores):
    med = np.median(scores, axis=0)
    mad = np.median(np.abs(scores - med), axis=0) * 1.4826
    return (scores - med) / np.maximum(mad, 1e-6)


def _rank(scores, ids, limit, min_z, mask=None, exclude=(), min_count=0):
    if not len(scores):
        return []
    z = robust_z(scores)
    if mask is not None:
        z = np.where(mask, z, -np.inf)
    for pid in exclude:
        z = np.where(ids == pid, -np.inf, z)
    k = min(limit, len(z))
    top = np.argpartition(-z, k - 1)[:k]
    top = top[np.argsort(-z[top])]
    out = []
    for i in top:
        if not np.isfinite(z[i]) or (z[i] < min_z and len(out) >= min_count):
            break
        tier = "best" if z[i] >= TIER_BEST else "good" if z[i] >= TIER_GOOD else "maybe"
        out.append({"id": int(ids[i]), "score": round(float(scores[i]), 4), "z": round(float(z[i]), 2), "tier": tier})
    return out


def text_search(raw: str, limit=300, filters=None, min_z=TIER_MAYBE, min_count=MIN_RESULTS):
    ids, mat = index.snapshot()
    if not len(ids):
        return []
    positive, negatives = parse_query(raw)
    positive, people = people_in_query(positive)
    if not positive and not negatives and not people:
        return []
    mask = _filtered(ids, filters)
    if people:
        allowed = set(faces.person_photo_ids(people[0]["id"]))
        for p in people[1:]:
            allowed &= set(faces.person_photo_ids(p["id"]))
        in_people = np.isin(ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        mask = in_people if mask is None else mask & in_people
        min_count = min(limit, int(mask.sum()))
        if not positive and not negatives:
            # Just a name: everything with them, newest first.
            keep = set(ids[mask].tolist())
            ordered = [i for i in faces.person_photo_ids(people[0]["id"]) if i in keep][:limit]
            return [{"id": i, "score": 1.0, "z": 9.0, "tier": "best"} for i in ordered]
    qv = embedder.embed_query(positive) if positive else np.zeros(mat.shape[1], dtype=np.float32)
    scores = mat @ qv
    if negatives:
        # Push down photos that look like what the user excluded.
        neg = np.stack([embedder.embed_query(n) for n in negatives])
        neg_scores = (mat @ neg.T).max(1)
        scores = scores - 0.5 * np.maximum(neg_scores - neg_scores.mean(), 0)
    return _rank(scores, ids, limit, min_z, mask, min_count=min_count)


def similar(photo_id: int, limit=60, filters=None):
    ids, mat = index.snapshot()
    v = index.vector_of(photo_id)
    if v is None:
        return []
    return _rank(mat @ v, ids, limit, TIER_MAYBE, _filtered(ids, filters), exclude=(photo_id,), min_count=12)


def similar_to_many(photo_ids, limit=60):
    """'More like these' for a selection: search by the centroid of several photos."""
    ids, mat = index.snapshot()
    vecs = [v for v in (index.vector_of(p) for p in photo_ids) if v is not None]
    if not vecs:
        return []
    c = np.mean(vecs, axis=0)
    c /= np.linalg.norm(c) or 1
    return _rank(mat @ c, ids, limit, TIER_MAYBE, exclude=photo_ids, min_count=12)


_explore_cache = {"version": None, "data": None}
_explore_lock = threading.Lock()


def explore(per_concept=6):
    """Group the library under everyday themes, zero-shot: each photo joins the theme it matches best."""
    with _explore_lock:
        if _explore_cache["version"] == (index.version, embedder.get().key):
            return _explore_cache["data"]
        ids, mat = index.snapshot()
        if len(ids) < 8:
            return []
        cvecs = embedder.embed_queries([prompt for _, _, prompt in CONCEPTS])
        s = mat @ cvecs.T  # (photos, concepts)
        # Standardize each concept column so "common" concepts don't win everything.
        z = robust_z(s)
        best = z.argmax(1)
        best_z = z[np.arange(len(ids)), best]
        groups = []
        for ci, (name, emoji, prompt) in enumerate(CONCEPTS):
            members = np.nonzero((best == ci) & (best_z >= TIER_GOOD))[0]
            if len(members) < 3:
                continue
            order = members[np.argsort(-z[members, ci])]
            groups.append({
                "name": name, "emoji": emoji, "query": prompt, "count": int(len(members)),
                "photo_ids": [int(ids[i]) for i in order[:per_concept]],
            })
        groups.sort(key=lambda g: -g["count"])
        _explore_cache.update(version=(index.version, embedder.get().key), data=groups)
        return groups


def on_this_day(limit=24):
    today = datetime.now()
    rows = db.q(
        "SELECT id, taken_at FROM photos WHERE status='ready' AND taken_at IS NOT NULL "
        "AND strftime('%m-%d', taken_at, 'unixepoch', 'localtime') = ? "
        "AND strftime('%Y', taken_at, 'unixepoch', 'localtime') < ? ORDER BY taken_at DESC LIMIT ?",
        (today.strftime("%m-%d"), today.strftime("%Y"), limit),
    )
    return [r["id"] for r in rows]


def suggestions(n=6):
    return random.sample(SUGGESTIONS, k=min(n, len(SUGGESTIONS)))
