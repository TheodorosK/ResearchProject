"""Your taste, learned on this computer from what you do in Lumen.

What counts as "I like this": favourites, 4-5 star ratings, photos you put in a collection, and
👍 on a search result. What counts against: 1-2 stars and 👎. From those examples Lumen trains a
small model on top of the image model's vectors (a regularised linear classifier, a few
milliseconds of work) and uses its score in the best-shot ranking and as a nudge in every search.

A trained model can overfit a handful of examples, so it is checked first: the examples are split
five ways, and the trained model is only used if it ranks held-out examples better than the
simple "average of what you like" direction. Otherwise the simple direction stays in use.
"""
import threading

import numpy as np

from . import db

MIN_EXAMPLES = 10          # liked photos needed before any taste is used
MIN_FOR_TRAINING = 20      # liked photos needed before a trained model is considered
BACKGROUND_PER_LIKED = 4   # unlabelled photos sampled per liked one, as weak "ordinary" examples
BACKGROUND_WEIGHT = 0.25
FEEDBACK_WEIGHT = 0.5      # 👍/👎 say "right for this search", a weaker sign of taste than a favourite
L2 = 1.0

_cache = {"key": None, "data": None}
_lock = threading.Lock()


def record_feedback(photo_id: int, query: str, verdict: int) -> None:
    """Remember a 👍 (1) or 👎 (-1) on a search result; 0 takes it back."""
    q = " ".join((query or "").lower().split())[:200]
    with db.tx() as c:
        if verdict == 0:
            c.execute("DELETE FROM search_feedback WHERE photo_id=? AND query=?", (photo_id, q))
        else:
            import time

            c.execute("INSERT OR REPLACE INTO search_feedback(photo_id, query, verdict, at) VALUES(?,?,?,?)",
                      (photo_id, q, 1 if verdict > 0 else -1, time.time()))


def examples():
    """({photo_id: weight} liked, {photo_id: weight} disliked). A photo is never in both."""
    liked, disliked = {}, {}
    for r in db.q("SELECT id, favorite, rating FROM photos WHERE status='ready' AND (favorite=1 OR rating>0)"):
        if r["favorite"] or (r["rating"] or 0) >= 4:
            liked[r["id"]] = 1.0
        elif 0 < (r["rating"] or 0) <= 2:
            disliked[r["id"]] = 1.0
    for r in db.q("SELECT DISTINCT ci.photo_id FROM collection_items ci JOIN collections c ON c.id=ci.collection_id "
                  "WHERE c.kind='manual'"):
        liked.setdefault(r["photo_id"], 1.0)
    for r in db.q("SELECT photo_id, SUM(verdict) v FROM search_feedback GROUP BY photo_id"):
        if r["v"] > 0:
            liked.setdefault(r["photo_id"], FEEDBACK_WEIGHT)
        elif r["v"] < 0 and r["photo_id"] not in liked:
            disliked.setdefault(r["photo_id"], FEEDBACK_WEIGHT)
    for pid in list(disliked):
        if pid in liked:
            del disliked[pid]
    return liked, disliked


def _fit(X, y, w, l2=L2):
    """Weighted ridge regression onto +1/-1 with a bias: a linear classifier in closed form."""
    mu = np.average(X, axis=0, weights=w)
    Xc = X - mu
    A = (Xc * w[:, None]).T @ Xc + l2 * np.eye(X.shape[1], dtype=np.float64)
    coef = np.linalg.solve(A, (Xc * w[:, None]).T @ (y - np.average(y, weights=w)))
    return coef.astype(np.float32)


def _mean_direction(X, y, w):
    pos = y > 0
    return (np.average(X[pos], axis=0, weights=w[pos]) - np.average(X, axis=0, weights=w)).astype(np.float32)


def auc(scores, y) -> float:
    """Chance that a liked example scores above a not-liked one (0.5 = no better than a coin)."""
    pos, neg = scores[y > 0], scores[y <= 0]
    if not len(pos) or not len(neg):
        return 0.5
    order = np.argsort(np.concatenate([pos, neg]), kind="stable")
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _cross_validate(X, y, w, folds=5, seed=0):
    """Held-out ranking quality of the trained model and of the simple direction."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for cls in (y > 0, y <= 0):                      # keep liked/not-liked balanced across folds
        idx = np.nonzero(cls)[0]
        fold[rng.permutation(idx)] = np.arange(len(idx)) % folds
    trained, simple = np.zeros(len(y)), np.zeros(len(y))
    for k in range(folds):
        tr, te = fold != k, fold == k
        trained[te] = X[te] @ _fit(X[tr], y[tr], w[tr])
        simple[te] = X[te] @ _mean_direction(X[tr], y[tr], w[tr])
    return auc(trained, y), auc(simple, y)


def _train(ids, mat):
    liked, disliked = examples()
    pos = {int(i): k for k, i in enumerate(ids)}
    liked = {p: wt for p, wt in liked.items() if p in pos}
    disliked = {p: wt for p, wt in disliked.items() if p in pos}
    info = {"liked": len(liked), "disliked": len(disliked), "method": "none", "trained_auc": None, "simple_auc": None}
    if len(liked) < MIN_EXAMPLES:
        return None, info
    labelled = set(liked) | set(disliked)
    rest = np.array([p for p in pos if p not in labelled], dtype=np.int64)
    rng = np.random.default_rng(0)
    background = rng.choice(rest, size=min(len(rest), BACKGROUND_PER_LIKED * len(liked)), replace=False) \
        if len(rest) else np.array([], dtype=np.int64)
    rows = [pos[p] for p in liked] + [pos[p] for p in disliked] + [pos[int(p)] for p in background]
    X = mat[rows].astype(np.float64)
    y = np.array([1.0] * len(liked) + [-1.0] * (len(disliked) + len(background)))
    w = np.array(list(liked.values()) + list(disliked.values()) + [BACKGROUND_WEIGHT] * len(background))
    direction, info["method"] = _mean_direction(X, y, w), "simple"
    if len(liked) >= MIN_FOR_TRAINING and (y <= 0).sum() >= MIN_FOR_TRAINING:
        info["trained_auc"], info["simple_auc"] = (round(a, 3) for a in _cross_validate(X, y, w))
        if info["trained_auc"] > info["simple_auc"] + 0.01 and info["trained_auc"] >= 0.6:
            direction, info["method"] = _fit(X, y, w), "trained"
    if not np.linalg.norm(direction):
        return None, info
    return direction, info


def _state():
    from . import embedder
    from .vectors import index

    with _lock:
        key = (index.version, embedder.get().key,
               tuple(db.one("SELECT SUM(favorite), SUM(rating), COUNT(*) FROM photos")),
               tuple(db.one("SELECT COUNT(*), MAX(at) FROM search_feedback")),
               tuple(db.one("SELECT COUNT(*), MAX(added_at) FROM collection_items")))
        if _cache["key"] != key:
            ids, mat = index.snapshot()
            direction, info = _train(ids, mat) if len(ids) else (None, {"liked": 0, "disliked": 0, "method": "none",
                                                                       "trained_auc": None, "simple_auc": None})
            _cache.update(key=key, data=(direction, info))
        return _cache["data"]


def scores(ids, mat):
    """A taste score per photo (robust z, clipped), or None while there are too few examples."""
    direction, _ = _state()
    if direction is None or not len(ids) or mat.shape[1] != len(direction):
        return None
    s = mat @ direction
    med = np.median(s)
    mad = np.median(np.abs(s - med)) * 1.4826
    return np.clip((s - med) / max(mad, 1e-6), -3, 3)


def status():
    return dict(_state()[1], needed=MIN_EXAMPLES, needed_for_training=MIN_FOR_TRAINING)
