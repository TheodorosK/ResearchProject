"""Reading text inside photos (signs, receipts, menus, screenshots, whiteboards) on-device with RapidOCR.

OCR costs ~0.5 s per photo, too much for a whole 100k library, so only photos the image model
thinks contain text are read. Set LUMEN_OCR=all to read everything, or off to skip it.
"""
import os
import threading

import numpy as np

from . import db, embedder, search
from .vectors import index

TEXT_PROMPTS = [
    "a document with printed text", "a sign with words on it", "a screenshot of a phone screen",
    "a receipt or a menu", "handwritten notes or a whiteboard", "a page of a book", "a poster with text",
    "a ticket or a label with writing",
]
MIN_Z = 2.0          # how text-like a photo must look, relative to the library, to be read
BATCH = 8
MAX_SIDE = 1600
MIN_CONFIDENCE = 0.65  # below this lines are mostly guesses: old handwriting, or scripts the model can't read

_engine = None
_lock = threading.Lock()
state = {"value": "idle"}


def mode():
    return os.environ.get("LUMEN_OCR", "auto").lower()


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


def tidy(text: str) -> str:
    """KALAM0S -> KALAMOS, 198O -> 1980: a zero among capitals, or an O among digits, is a misread."""
    import re

    text = re.sub(r"(?<=[A-Z])0(?=[A-Z]|\b)|\b0(?=[A-Z]{2})", "O", text)
    return re.sub(r"(?<=\d)O(?=\d|\b)|\bO(?=\d{2})", "0", text)


def read(im) -> str:
    """Recognised text of a PIL image, top to bottom."""
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    result, _ = _get_engine()(np.ascontiguousarray(np.asarray(im)[:, :, ::-1]))
    if not result:
        return ""
    lines = sorted(result, key=lambda r: (round(r[0][0][1] / 20), r[0][0][0]))  # rows, then left to right
    return "\n".join(tidy(r[1]) for r in lines if float(r[2]) >= MIN_CONFIDENCE)  # RapidOCR reports scores as strings


def _select():
    """Decide which not-yet-considered photos to read; mark the rest as skipped."""
    rows = db.q("SELECT id FROM photos WHERE status='ready' AND kind='photo' AND ocr_v=0")
    if not rows:
        return []
    todo = [r["id"] for r in rows]
    if mode() == "all":
        return todo
    ids, mat = index.snapshot()
    if len(ids) < 8:
        return []  # too few photos to judge what's "text-like" yet
    z = search.robust_z(mat @ embedder.embed_queries(TEXT_PROMPTS).T).max(1)
    texty = {int(i) for i, zz in zip(ids, z) if zz >= MIN_Z}
    skip = [i for i in todo if i not in texty]
    if skip:
        with db.tx() as c:
            c.executemany("UPDATE photos SET ocr_v=2 WHERE id=?", [(i,) for i in skip])
    return [i for i in todo if i in texty]


def run_pending(should_stop=lambda: False) -> int:
    """Read text from the next batch of selected photos. Returns how many were processed."""
    if mode() == "off":
        return 0
    from PIL import Image, ImageOps

    todo = _select()[:BATCH]
    if not todo:
        return 0
    state["value"] = f"reading text ({len(todo)})"
    done = []
    for pid in todo:
        if should_stop():
            break
        row = db.one("SELECT path FROM photos WHERE id=?", (pid,))
        text = ""
        try:
            with Image.open(row["path"]) as im:
                im.draft("RGB", (MAX_SIDE, MAX_SIDE))
                text = read(ImageOps.exif_transpose(im))
        except Exception:
            pass
        done.append((pid, text))
    with db.tx() as c:
        for pid, text in done:
            c.execute("DELETE FROM photo_text WHERE rowid=?", (pid,))
            if text.strip():
                c.execute("INSERT INTO photo_text(rowid, text) VALUES(?, ?)", (pid, text))
            c.execute("UPDATE photos SET ocr_v=1 WHERE id=?", (pid,))
    state["value"] = "idle"
    return len(done)


def text_of(photo_id: int) -> str | None:
    row = db.one("SELECT text FROM photo_text WHERE rowid=?", (photo_id,))
    return row["text"] if row else None


def _q(s):
    return '"' + s.replace('"', "") + '"'


def _fts_query(words):
    # The recogniser sometimes drops spaces ("BOARDINGPASS"), so neighbouring words also count joined up.
    joined = [a + b for a, b in zip(words, words[1:])]
    return " OR ".join(_q(w) for w in words + joined)


def _phrase_query(ph):
    words = ph.split()
    return f"{_q(ph)} OR {_q(''.join(words))}" if len(words) > 1 else _q(ph)


def matches(text: str, phrases=()) -> dict:
    """{photo id: strength 0..1} for photos whose text contains words of `text` (or exact `phrases`)."""
    import re

    out = {}
    words = [w for w in re.findall(r"\w+", text) if len(w) >= 3]
    if words:
        rows = db.q("SELECT rowid, bm25(photo_text) r FROM photo_text WHERE photo_text MATCH ? "
                    "ORDER BY r LIMIT 2000", (_fts_query(words),))
        if rows:
            best = min(r["r"] for r in rows)  # bm25: lower is better, negative
            for r in rows:
                out[r["rowid"]] = max(0.2, r["r"] / best) if best else 1.0
    # Words run together with their neighbours on signs ("OFANAWARENESS") only match as part of a word.
    for w in {w.lower().replace("_", "") for w in words if len(w) >= 5}:
        for r in db.q("SELECT rowid FROM photo_text WHERE text LIKE ?", (f"%{w}%",)):
            out.setdefault(r["rowid"], 0.5)
    for ph in phrases:
        for r in db.q("SELECT rowid FROM photo_text WHERE photo_text MATCH ?", (_phrase_query(ph),)):
            out[r["rowid"]] = 1.0
    return out


def phrase_ids(phrases) -> set | None:
    """Photos containing every exact phrase (for quoted searches), or None if there are none."""
    if not phrases:
        return None
    ids = None
    for ph in phrases:
        found = {r["rowid"] for r in db.q("SELECT rowid FROM photo_text WHERE photo_text MATCH ?",
                                          (_phrase_query(ph),))}
        ids = found if ids is None else ids & found
    return ids or set()
