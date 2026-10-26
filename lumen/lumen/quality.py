"""How good does each photo look? A blend of the image model's taste and measured sharpness/exposure.

The aesthetic half is zero-shot: we compare every photo's embedding with descriptions of great
photos versus poor ones (the CLIP-IQA idea), so it needs no extra model and works with any of
Lumen's models. The technical half is measured on the preview while indexing.
"""
import threading

import numpy as np
from PIL import Image

from . import db, embedder
from .vectors import index

GOOD = [
    "a beautiful, stunning photograph",
    "an award-winning photo with great composition",
    "a sharp, well exposed, professional photo",
]
BAD = [
    "a blurry, out of focus photo",
    "a badly composed, boring snapshot",
    "a dark, noisy, low quality photo",
]
AESTHETIC_WEIGHT, SHARPNESS_WEIGHT = 0.65, 0.35

_cache = {"key": None, "data": None}
_lock = threading.Lock()


def measure(im: Image.Image) -> dict:
    """Technical measurements on a preview-sized image."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    lap = a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:] - 4 * a[1:-1, 1:-1]
    from .colors import palette

    return {
        "palette": palette(im),
        "sharpness": _focus(lap),
        "brightness": float(a.mean()),
        "hi_clip": float((a >= 250).mean()),
        "lo_clip": float((a <= 5).mean()),
    }


def _focus(lap, grid=4, top=3):
    """Sharpness of the in-focus part: Laplacian variance of the sharpest few tiles of a 4x4 grid.
    A portrait with a soft background is still sharp; a busy texture doesn't win by being busy."""
    h, w = lap.shape
    if h < grid or w < grid:
        return float(lap.var()) if lap.size else 0.0
    tiles = [lap[r * h // grid:(r + 1) * h // grid, c * w // grid:(c + 1) * w // grid].var()
             for r in range(grid) for c in range(grid)]
    return float(np.mean(sorted(tiles)[-top:]))


def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) * 1.4826
    return (x - med) / max(mad, 1e-6)


def scores():
    """{photo_id: (quality_z, percentile 0-100)} for every indexed photo; cached until the library changes."""
    with _lock:
        key = (index.version, embedder.get().key, db.one("SELECT SUM(meta_v) s FROM photos")["s"])
        if _cache["key"] == key:
            return _cache["data"]
        ids, mat = index.snapshot()
        if not len(ids):
            return {}
        direction = embedder.embed_queries(GOOD).mean(0) - embedder.embed_queries(BAD).mean(0)
        aes = np.clip(_robust_z(mat @ direction), -3, 3)
        tech = {r["id"]: r for r in db.q("SELECT id, sharpness, hi_clip, lo_clip FROM photos WHERE sharpness IS NOT NULL")}
        sharp = np.array([np.log1p(tech[i]["sharpness"]) if i in tech else np.nan for i in ids.tolist()])
        have = ~np.isnan(sharp)
        sharp_z = np.zeros(len(ids))
        if have.sum() > 2:
            sharp_z[have] = np.clip(_robust_z(sharp[have]), -3, 2)
        # Blown highlights hurt; deep shadows only when most of the frame is black (moody is fine).
        penalty = np.array([
            2 * max(0.0, (tech[i]["hi_clip"] or 0) - 0.15) + max(0.0, (tech[i]["lo_clip"] or 0) - 0.6)
            if i in tech else 0.0 for i in ids.tolist()
        ])
        q = AESTHETIC_WEIGHT * aes + SHARPNESS_WEIGHT * sharp_z - 4 * penalty
        order = q.argsort().argsort()
        pct = np.round(100 * order / max(len(q) - 1, 1)).astype(int)
        data = {int(i): (float(qq), int(pp)) for i, qq, pp in zip(ids, q, pct)}
        _cache.update(key=key, data=data)
        return data


def bonus(ids, weight=0.3):
    """Small ranking nudge (in z-score units) towards better-looking photos."""
    s = scores()
    return weight * np.clip(np.array([s.get(int(i), (0.0, 50))[0] for i in ids]), -2, 2)


def best(limit=24, min_gap=0.92):
    """The library's best-looking photos, skipping near-repeats of ones already picked."""
    s = scores()
    if not s:
        return []
    ids, mat = index.snapshot()
    pos = {int(i): k for k, i in enumerate(ids)}
    picked, vecs = [], []
    for pid, _ in sorted(s.items(), key=lambda kv: -kv[1][0]):
        k = pos.get(pid)
        if k is None:
            continue
        v = mat[k]
        if vecs and max(float(v @ w) for w in vecs) >= min_gap:
            continue
        picked.append(pid)
        vecs.append(v)
        if len(picked) >= limit:
            break
    return picked


def sort_by_quality(ids):
    s = scores()
    return sorted(ids, key=lambda i: -s.get(i, (0.0, 50))[0])
