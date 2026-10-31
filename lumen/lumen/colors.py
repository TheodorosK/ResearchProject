"""Colour palettes of photos, and searching by colour ('mostly teal and orange')."""
import threading

import numpy as np
from PIL import Image

from . import db
from .vectors import index

PALETTE_SIZE = 5
SPREAD = 14.0   # ΔE (CIELAB distance) at which a palette colour counts about a third as much

_cache = {"key": None}
_lock = threading.Lock()


def palette(im: Image.Image) -> str:
    """'rrggbb:0.42,rrggbb:0.30,...': the dominant colours of an image with their shares."""
    small = im.convert("RGB").resize((64, 64))
    q = small.quantize(colors=PALETTE_SIZE, method=Image.Quantize.FASTOCTREE)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts) or 1
    return ",".join(f"{pal[3 * i]:02x}{pal[3 * i + 1]:02x}{pal[3 * i + 2]:02x}:{c / total:.2f}" for c, i in counts[:PALETTE_SIZE])


def to_lab(rgb):
    """sRGB (0-255, shape (..., 3)) to CIELAB."""
    c = np.asarray(rgb, dtype=float) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.2126, 0.0193], [0.3576, 0.7152, 0.1192], [0.1805, 0.0722, 0.9505]])
    xyz /= np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def hex_to_rgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)]


def _palettes(ids):
    """(labs (n, 5, 3), weights (n, 5)) aligned with ids; photos without a palette weigh 0."""
    with _lock:
        key = (index.version, db.one("SELECT COUNT(palette) n FROM photos")["n"])
        if _cache["key"] != key or len(_cache["ids"]) != len(ids):
            rows = dict(db.q("SELECT id, palette FROM photos WHERE status='ready' AND palette IS NOT NULL"))
            rgb = np.zeros((len(ids), PALETTE_SIZE, 3))
            w = np.zeros((len(ids), PALETTE_SIZE))
            for k, i in enumerate(ids.tolist()):
                for j, part in enumerate((rows.get(i) or "").split(",")[:PALETTE_SIZE]):
                    if ":" in part:
                        h, share = part.split(":")
                        rgb[k, j] = hex_to_rgb(h)
                        w[k, j] = float(share)
            _cache.update(key=key, ids=ids, lab=to_lab(rgb), w=w)
        return _cache["lab"], _cache["w"]


def scores(ids, hexes):
    """How much each photo is made of the chosen colours (higher is better), shape (n,)."""
    labs, w = _palettes(ids)
    out = np.zeros(len(ids))
    for h in hexes:
        target = to_lab(hex_to_rgb(h))
        d = np.linalg.norm(labs - target, axis=-1)          # (n, 5)
        out += (w * np.exp(-d / SPREAD)).sum(1)
    return out / max(len(hexes), 1)
