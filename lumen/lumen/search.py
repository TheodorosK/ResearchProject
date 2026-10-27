"""Semantic search over the vector index, plus the 'Explore' and 'On this day' discoveries."""
import random
import re
import threading
from datetime import datetime

import numpy as np

from . import colors as colour, dates, db, embedder, faces, ocr, places, quality
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

# Moods and styles: overlap with the subject themes above, so they're grouped separately.
MOODS = [
    ("Moody & dark", "🌑", "a moody, dark and atmospheric photo"),
    ("Cinematic", "🎬", "a cinematic film still with dramatic lighting"),
    ("Street photography", "🚶", "candid street photography in the city"),
    ("Black & white", "⚫", "a black and white photograph"),
    ("Golden light", "🌇", "warm golden hour sunlight"),
    ("Dreamy & soft", "☁️", "a dreamy, soft, pastel photo"),
    ("Joyful moments", "😄", "people laughing and having fun together"),
    ("Calm & peaceful", "🕊️", "a calm and peaceful scene"),
    ("Melancholic", "🌧️", "a melancholic, lonely and quiet scene"),
    ("Vibrant colours", "🌈", "a vibrant, colorful photo"),
    ("Minimalist", "◻️", "a minimalist photo with lots of empty space"),
    ("Nostalgic", "📼", "a nostalgic vintage film photo"),
    ("Romantic", "💞", "a romantic moment for a couple"),
    ("Epic landscapes", "🏞️", "an epic, dramatic landscape"),
    ("Cozy", "🕯️", "a cozy warm indoor moment"),
    ("Portraits", "🖼️", "an artistic portrait"),
]

# Words that ask about faces' expressions, mapped to FER+ labels. Greek stems match word starts.
EXPRESSION_WORDS = {
    "smiling": "happiness", "smile": "happiness", "smiles": "happiness", "smiley": "happiness",
    "happy": "happiness", "laughing": "happiness", "laugh": "happiness", "laughs": "happiness",
    "cheerful": "happiness", "grinning": "happiness",
    "sad": "sadness", "crying": "sadness", "cry": "sadness", "tears": "sadness", "unhappy": "sadness",
    "upset": "sadness",
    "surprised": "surprise", "shocked": "surprise", "amazed": "surprise", "astonished": "surprise",
    "angry": "anger", "mad": "anger", "furious": "anger",
    "scared": "fear", "afraid": "fear", "frightened": "fear",
    "disgusted": "disgust", "serious": "neutral", "neutral": "neutral", "expressionless": "neutral",
}
EXPRESSION_STEMS = {
    "χαμογελ": "happiness", "χαρούμεν": "happiness", "χαρουμεν": "happiness", "γελ": "happiness",
    "λυπημέν": "sadness", "λυπημεν": "sadness", "κλαί": "sadness", "κλαι": "sadness",
    "θυμωμέν": "anger", "θυμωμεν": "anger", "έκπληκτ": "surprise", "εκπληκτ": "surprise",
    "φοβισμέν": "fear", "φοβισμεν": "fear", "σοβαρ": "neutral",
}
EVERYONE_WORDS = {"everyone", "everybody", "all", "όλοι", "ολοι"}
EXPRESSION_WEIGHT = 3.0
TEXT_WEIGHT = 2.5        # z-score units for a photo whose recognised text contains the query words   # z-score units added for a confident expression match

SUGGESTIONS = [
    "golden hour at the beach", "my dog sleeping on the couch", "a birthday cake with candles",
    "snowy mountains", "dinner with friends", "kids laughing", "old stone streets",
    "a cozy rainy day", "fireworks over the city", "the blue sea from a boat",
    "people dancing", "a plate of pasta", "sunrise from an airplane window",
    "moody and dark", "cinematic street photography", "everyone smiling", "a melancholic rainy evening",
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
    return _strip_filler(text), found


def _strip_filler(text):
    text = " ".join(text.split())
    return "" if all(w.lower().strip(",.") in FILLER for w in text.split()) else text


def understand(raw: str):
    """Split a query into what to look for and who/where/when it should be."""
    positive, negatives = parse_query(raw)
    phrases = [p for p in re.findall(r'"([^"]+)"', positive) if p.strip()]
    tags = re.findall(r"(?<!\w)#([\w-]+)", positive)
    positive = re.sub(r"(?<!\w)#[\w-]+", " ", positive)
    favourites = bool(re.search(r"(?<!\w)(favou?rites?|faves?|αγαπημένες|αγαπημενες)(?!\w)", positive, re.I))
    positive = re.sub(r"(?<!\w)(?:my\s+)?(favou?rites?|faves?|αγαπημένες|αγαπημενες)(?!\w)", " ", positive, flags=re.I)
    positive = positive.replace('"', " ")
    positive, people = people_in_query(positive)
    positive, where = places.places_in_query(positive)
    positive, when = dates.parse(positive)
    positive = _strip_filler(positive)
    label, everyone = expression_in_query(positive)
    return {"positive": positive, "negatives": negatives, "phrases": phrases, "tags": tags, "favourites": favourites,
            "people": people, "places": where,
            "when": when, "expression": label, "everyone": everyone, "tone": tone_in_query(positive, negatives)}


_taken_cache = {"key": None, "arr": None}


def _taken(ids):
    if _taken_cache["key"] != index.version:
        rows = dict(db.q("SELECT id, taken_at FROM photos WHERE status='ready'"))
        _taken_cache.update(key=index.version,
                            arr=np.array([rows.get(int(i)) or np.nan for i in ids], dtype=float))
    arr = _taken_cache["arr"]
    if len(arr) != len(ids):
        _taken_cache["key"] = None
        return _taken(ids)
    return arr


# Words about light and tone. The image model understands them too; the photos' measured
# brightness backs it up, so "dark" finds dark photos even where the model is unsure.
DARK_WORDS = {
    "dark", "darker", "darkness", "moody", "lowkey", "low-key", "night", "nighttime", "night-time",
    "shadow", "shadows", "shadowy", "gloomy", "gloom", "dim", "dimly", "noir",
    "σκοτεινό", "σκοτεινή", "σκοτεινός", "σκοτεινά", "σκοτεινές", "σκοτεινοί", "σκοτάδι",
    "νύχτα", "νυχτερινό", "νυχτερινή", "νυχτερινές", "σκιά", "σκιές",
}
BRIGHT_WORDS = {
    "bright", "brighter", "sunny", "sunlit", "highkey", "high-key", "airy", "luminous",
    "φωτεινό", "φωτεινή", "φωτεινός", "φωτεινά", "φωτεινές", "ηλιόλουστο", "ηλιόλουστη", "ηλιόλουστες",
}
TONE_WEIGHT = 1.0   # z-score units per unit of (clipped) brightness z


def tone_in_query(text: str, negatives=()):
    """-1 for dark ('moody dark', 'low key'), +1 for bright ('sunny'), 0 if the words cancel or are absent.
    An excluded word counts the other way ('beach -dark' prefers bright beaches)."""
    def votes(t):
        t = re.sub(r"\b(low|high)\s+key\b", r"\1key", t.lower())
        words = re.findall(r"[\w-]+", t)
        return sum(w in BRIGHT_WORDS for w in words) - sum(w in DARK_WORDS for w in words)
    v = votes(text) - sum(votes(n) for n in negatives)
    return int(np.sign(v))


def _tone_bonus(ids, tone):
    rows = dict(db.q("SELECT id, brightness FROM photos WHERE brightness IS NOT NULL"))
    b = np.array([rows.get(int(i), np.nan) for i in ids], dtype=float)
    have = ~np.isnan(b)
    out = np.zeros(len(ids))
    if have.sum() > 2:
        out[have] = TONE_WEIGHT * np.clip(tone * robust_z(b[have]), -1.5, 2.5)
    return out


def expression_in_query(text: str):
    """('everyone smiling at dinner') -> ('happiness', everyone=True); (None, False) if none."""
    label, everyone = None, False
    for w in re.findall(r"\w+", text.lower()):
        if w in EVERYONE_WORDS:
            everyone = True
        if label is None:
            label = EXPRESSION_WORDS.get(w) or next(
                (v for stem, v in EXPRESSION_STEMS.items() if w.startswith(stem)), None)
    return label, everyone


def _expression_scores(ids, label, person_ids=(), everyone=False):
    """Per-photo probability of `label` from its faces: the named people's faces if any were named,
    else any face (or the least-matching face, for 'everyone')."""
    k = faces.EXPRESSIONS.index(label)
    sql = "SELECT photo_id, expr_probs FROM faces WHERE expr_probs IS NOT NULL"
    args = []
    if person_ids:
        sql += f" AND person_id IN ({','.join('?' * len(person_ids))})"
        args = list(person_ids)
    per_photo = {}
    for r in db.q(sql, args):
        p = float(np.frombuffer(r["expr_probs"], dtype=np.float16)[k])
        per_photo.setdefault(r["photo_id"], []).append(p)
    agg = min if everyone else max
    return np.array([agg(per_photo[i]) if i in per_photo else np.nan for i in ids.tolist()])


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


def _rank(scores, ids, limit, min_z, mask=None, exclude=(), min_count=0, bonus=None):
    if not len(scores):
        return []
    z = robust_z(scores)
    if bonus is not None:
        z = z + bonus
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


def text_search(raw: str, limit=300, filters=None, min_z=TIER_MAYBE, min_count=MIN_RESULTS, sort="relevance",
                like=(), dislike=(), colors=()):
    hits = _text_search(raw, limit, filters, min_z, min_count, like=like, dislike=dislike, colors=colors)
    if sort == "quality":
        s = quality.scores()
        order = {"best": 0, "good": 1, "maybe": 2}
        hits.sort(key=lambda h: (order[h["tier"]], -s.get(h["id"], (0.0, 50))[0]))
    return hits


def _text_search(raw, limit, filters, min_z, min_count, like=(), dislike=(), colors=()):
    ids, mat = index.snapshot()
    if not len(ids):
        return []
    u = understand(raw)
    positive, negatives, people = u["positive"], u["negatives"], u["people"]
    if not (positive or negatives or people or u["places"] or u["when"] or like or u["phrases"] or colors
            or u["tags"] or u["favourites"]):
        return []
    mask = _filtered(ids, filters)

    def narrow(allowed):
        nonlocal mask
        m = np.isin(ids, np.fromiter(allowed, dtype=np.int64, count=len(allowed)))
        mask = m if mask is None else mask & m

    for p in people:
        narrow(set(faces.person_photo_ids(p["id"])))
    for pl in u["places"]:
        narrow(places.photo_ids_for(pl))
    if u["phrases"]:
        # "Quoted words" must be written in the photo.
        narrow(ocr.phrase_ids(u["phrases"]))
    for t in u["tags"]:
        narrow({r["photo_id"] for r in db.q("SELECT photo_id FROM photo_tags WHERE tag=?", (t,))})
    if u["favourites"]:
        narrow({r["id"] for r in db.q("SELECT id FROM photos WHERE favorite=1")})
    date_mask, date_bonus = dates.mask_and_bonus(u["when"], ids, _taken(ids))
    if date_mask is not None:
        mask = date_mask if mask is None else mask & date_mask
    if mask is not None and (people or u["places"] or date_mask is not None or u["phrases"] or u["tags"]
                             or u["favourites"]):
        # Every photo that satisfies who/where/when should be shown, ranked by the rest.
        min_count = min(limit, int(mask.sum()))
    if not positive and not negatives and not like and not colors and mask is not None:
        # Only who/where/when ("Maria in Crete last summer"): all of them, newest first.
        taken = _taken(ids)
        sel = np.nonzero(mask)[0]
        sel = sel[np.argsort(-np.nan_to_num(taken[sel], nan=0))][:limit]
        return [{"id": int(ids[i]), "score": 1.0, "z": 9.0, "tier": "best"} for i in sel]

    qv = embedder.embed_query(positive) if positive else np.zeros(mat.shape[1], dtype=np.float32)
    if like or dislike:
        qv = refine(qv, like, dislike, ids, mat)
    if colors and not positive and not like:
        # Colours alone: rank purely by how much of the photo is made of them.
        scores, moments = colour.scores(ids, colors), {}
    else:
        scores, moments = _video_moments(qv, ids, mat @ qv)
    # Among similar matches, prefer the better-looking photo.
    bonus = quality.bonus(ids)
    if date_bonus is not None:
        bonus = bonus + date_bonus
    if colors and (positive or like):
        bonus = bonus + 1.5 * np.clip(robust_z(colour.scores(ids, colors)), -1, 3)
    if positive:
        # Words written in the photo (signs, menus, receipts) count too.
        text_hits = ocr.matches(positive)
        if text_hits:
            bonus = bonus + TEXT_WEIGHT * np.array([text_hits.get(int(i), 0.0) for i in ids])
    if u["tone"]:
        bonus = bonus + _tone_bonus(ids, u["tone"])
    if u["expression"]:
        p = _expression_scores(ids, u["expression"], [x["id"] for x in people], u["everyone"])
        # Faces that show the expression rise; photos without readable faces sink a little.
        bonus = bonus + np.where(np.isnan(p), -0.5, EXPRESSION_WEIGHT * (np.nan_to_num(p) - 0.3))
    if negatives:
        # Push down photos that look like what the user excluded.
        neg = np.stack([embedder.embed_query(n) for n in negatives])
        neg_scores = (mat @ neg.T).max(1)
        scores = scores - 0.5 * np.maximum(neg_scores - neg_scores.mean(), 0)
    hits = _rank(scores, ids, limit, min_z, mask, min_count=min_count, bonus=bonus, exclude=dislike)
    for h in hits:
        if h["id"] in moments:
            h["t"] = moments[h["id"]]
    return hits


_frames_cache = {"key": None}


def _frames():
    """(video ids, times, matrix) of every sampled video frame for the current model."""
    model = embedder.get().key
    key = (model, tuple(db.one("SELECT COUNT(*), MAX(rowid) FROM video_frames WHERE model=?", (model,))))
    if _frames_cache["key"] != key:
        rows = db.q("SELECT photo_id, t, embedding FROM video_frames WHERE model=?", (model,))
        _frames_cache.update(
            key=key,
            ids=np.array([r["photo_id"] for r in rows], dtype=np.int64),
            t=np.array([r["t"] for r in rows], dtype=float),
            mat=np.stack([np.frombuffer(r["embedding"], dtype=np.float16) for r in rows]).astype(np.float32)
            if rows else np.zeros((0, 0), dtype=np.float32),
        )
    return _frames_cache["ids"], _frames_cache["t"], _frames_cache["mat"]


def _video_moments(qv, ids, scores):
    """Let a video score as well as its best-matching frame; remember where that moment is."""
    fids, ft, fmat = _frames()
    if not len(fids):
        return scores, {}
    fs = fmat @ qv
    best = {}
    for k in np.argsort(fs):  # ascending, so the best frame per video is written last
        best[int(fids[k])] = (float(fs[k]), float(ft[k]))
    pos = {int(i): k for k, i in enumerate(ids)}
    scores = scores.copy()
    for vid, (sc, _) in best.items():
        k = pos.get(vid)
        if k is not None and sc > scores[k]:
            scores[k] = sc
    return scores, {vid: t for vid, (_, t) in best.items()}


def refine(qv, like, dislike, ids, mat, alpha=1.0, beta=0.8, gamma=0.5):
    """Rocchio relevance feedback: move the query towards liked photos and away from disliked ones."""
    pos = {int(i): k for k, i in enumerate(ids)}
    lv = [mat[pos[i]] for i in like if i in pos]
    dv = [mat[pos[i]] for i in dislike if i in pos]
    q = alpha * qv
    if lv:
        q = q + beta * np.mean(lv, axis=0)
    if dv:
        q = q - gamma * np.mean(dv, axis=0)
    n = np.linalg.norm(q)
    return q / n if n else qv


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


_explore_cache = {}
_explore_lock = threading.Lock()


def _themes(concepts, per_concept, min_members=3):
    """Zero-shot grouping: each photo joins the theme it matches best, if it matches it well."""
    ids, mat = index.snapshot()
    if len(ids) < 8:
        return []
    s = mat @ embedder.embed_queries([prompt for _, _, prompt in concepts]).T  # (photos, concepts)
    # Standardize each concept column so "common" concepts don't win everything.
    z = robust_z(s)
    best = z.argmax(1)
    best_z = z[np.arange(len(ids)), best]
    q = quality.bonus(ids, weight=0.5)
    groups = []
    for ci, (name, emoji, prompt) in enumerate(concepts):
        members = np.nonzero((best == ci) & (best_z >= TIER_GOOD))[0]
        if len(members) < min_members:
            continue
        # The cover photos: strong matches that also look good.
        order = members[np.argsort(-(z[members, ci] + q[members]))]
        groups.append({
            "name": name, "emoji": emoji, "query": prompt, "count": int(len(members)),
            "photo_ids": [int(ids[i]) for i in order[:per_concept]],
        })
    groups.sort(key=lambda g: -g["count"])
    return groups


def _cached(kind, fn):
    with _explore_lock:
        key = (index.version, embedder.get().key)
        hit = _explore_cache.get(kind)
        if hit and hit[0] == key:
            return hit[1]
        data = fn()
        _explore_cache[kind] = (key, data)
        return data


def explore(per_concept=6):
    """Everyday subjects: beaches, food, pets, celebrations..."""
    return _cached("subjects", lambda: _themes(CONCEPTS, per_concept))


def moods(per_concept=6):
    """Moods and styles: moody & dark, cinematic, black & white..."""
    return _cached("moods", lambda: _themes(MOODS, per_concept))


def on_this_day(limit=24):
    today = datetime.now()
    rows = db.q(
        "SELECT id, taken_at FROM photos WHERE status='ready' AND taken_at IS NOT NULL AND date_source IN ('camera', 'filename') "
        "AND strftime('%m-%d', taken_at, 'unixepoch', 'localtime') = ? "
        "AND strftime('%Y', taken_at, 'unixepoch', 'localtime') < ? ORDER BY taken_at DESC LIMIT ?",
        (today.strftime("%m-%d"), today.strftime("%Y"), limit),
    )
    return [r["id"] for r in rows]


def suggestions(n=6):
    return random.sample(SUGGESTIONS, k=min(n, len(SUGGESTIONS)))
