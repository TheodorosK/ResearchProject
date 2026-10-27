"""Reading text inside photos (signs, receipts, menus, screenshots, whiteboards) on-device with RapidOCR.

OCR costs ~1 s per photo, too much for a whole 100k library, so only photos the image model
thinks contain text are read. Set LUMEN_OCR=all to read everything, or off to skip it.

Every line of text is read by the multilingual model that ships with RapidOCR (Latin-script
languages, Chinese, Japanese) and by a Greek model fetched once on first use; the Greek reading
wins when it has Greek letters and is at least as confident.
"""
import logging
import os
import re
import threading

import numpy as np

from . import config, db, embedder, search, textnorm
from .vectors import index

log = logging.getLogger("lumen.ocr")

TEXT_PROMPTS = [
    "a document with printed text", "a sign with words on it", "a screenshot of a phone screen",
    "a receipt or a menu", "handwritten notes or a whiteboard", "a page of a book", "a poster with text",
    "a ticket or a label with writing",
]
MIN_Z = 2.0          # how text-like a photo must look, relative to the library, to be read
BATCH = 8
MAX_SIDE = 1600
MIN_CONFIDENCE = 0.65  # below this lines are mostly guesses: old handwriting, or scripts the model can't read
ENGINE = "rapidocr3-ppocrv6"
GREEK = re.compile(r"[Ͱ-Ͽἀ-῿]")

_engine = None
_lock = threading.Lock()
state = {"value": "idle"}


def mode():
    return os.environ.get("LUMEN_OCR", "auto").lower()


def best_reading(main, greek):
    """Per line, the Greek model's (text, score) if it found Greek letters and is at least as sure."""
    return [g if g[0] and GREEK.search(g[0]) and g[1] >= m[1] else m for m, g in zip(main, greek)]


def _make_engine():
    from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

    quiet = {"Global.log_level": "error"}  # a photo without text is normal, not a warning

    class Reader(RapidOCR):
        greek = None

        # The one step we extend (rapidocr is pinned): every detected line is also read in Greek.
        def recognize_txt(self, img):
            res = super().recognize_txt(img)
            if self.greek is None:
                return res
            try:
                g = self.greek.recognize_txt(img)
            except Exception as e:
                log.warning("Reading Greek failed, carrying on without it: %s", e)
                self.greek = None
                return res
            merged = best_reading(list(zip(res.txts, res.scores)), list(zip(g.txts, g.scores)))
            res.txts = tuple(t for t, _ in merged)
            res.scores = [s for _, s in merged]
            res.word_results = tuple((t, s, None) for t, s in merged)
            return res

    reader = Reader(params=quiet)
    if config.OCR_GREEK:
        try:
            config.OCR_DIR.mkdir(parents=True, exist_ok=True)
            greek = RapidOCR(params={**quiet, "Rec.lang_type": LangRec.EL, "Rec.ocr_version": OCRVersion.PPOCRV5,
                                     "Rec.model_type": ModelType.MOBILE,
                                     "Global.model_root_dir": str(config.OCR_DIR)})
            greek._load_rec_model()  # downloads (and checks) the model now, so we know whether Greek is on
            reader.greek = greek
        except Exception as e:  # offline, or the download host is blocked: try again next start
            log.warning("Greek text reading unavailable: %s", e)
    return reader


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            _engine = _make_engine()
        return _engine


def engine_key() -> str:
    """Which readers are in use, stored with each photo's text."""
    return ENGINE + ("+el" if _get_engine().greek is not None else "")


def tidy(text: str) -> str:
    """KALAM0S -> KALAMOS, 198O -> 1980: a zero among capitals, or an O among digits, is a misread."""
    text = re.sub(r"(?<=[A-Z])0(?=[A-Z]|\b)|\b0(?=[A-Z]{2})", "O", text)
    return re.sub(r"(?<=\d)O(?=\d|\b)|\bO(?=\d{2})", "0", text)


def read(im) -> str:
    """Recognised text of a PIL image, top to bottom."""
    im = im.convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    out = _get_engine()(np.ascontiguousarray(np.asarray(im)[:, :, ::-1]))
    if out.boxes is None or not out.txts:
        return ""
    lines = sorted(zip(out.boxes, out.txts, out.scores),
                   key=lambda r: (round(float(r[0][0][1]) / 20), float(r[0][0][0])))  # rows, then left to right
    return "\n".join(tidy(t) for _, t, s in lines if float(s) >= MIN_CONFIDENCE)


_rechecked = {"key": None}


def _reread_with_better_readers():
    """Photos read by an older engine, or before Greek was available, are read again (once per change)."""
    if not db.one("SELECT 1 FROM photos WHERE ocr_v=1 AND (ocr_key IS NULL OR ocr_key NOT LIKE '%+el') LIMIT 1"):
        return
    key = engine_key()
    if _rechecked["key"] == key:
        return
    _rechecked["key"] = key
    with db.tx() as c:
        c.execute("UPDATE photos SET ocr_v=0 WHERE ocr_v=1 AND (ocr_key IS NULL OR (? AND ocr_key NOT LIKE '%+el'))",
                  (key.endswith("+el"),))


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

    _reread_with_better_readers()
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
        done.append((pid, text, engine_key()))
    with db.tx() as c:
        for pid, text, key in done:
            c.execute("DELETE FROM photo_text WHERE rowid=?", (pid,))
            c.execute("DELETE FROM photo_text_plain WHERE rowid=?", (pid,))
            if text.strip():
                c.execute("INSERT INTO photo_text(rowid, text) VALUES(?, ?)", (pid, text))
                c.execute("INSERT INTO photo_text_plain(rowid, plain) VALUES(?, ?)", (pid, textnorm.searchable(text)))
            c.execute("UPDATE photos SET ocr_v=1, ocr_key=? WHERE id=?", (key, pid))
    state["value"] = "idle"
    return len(done)


def text_of(photo_id: int) -> str | None:
    row = db.one("SELECT text FROM photo_text WHERE rowid=?", (photo_id,))
    return row["text"] if row else None


def _q(s):
    return '"' + textnorm.searchable(s).replace('"', "") + '"'


def _fts_query(words):
    # The recogniser sometimes drops spaces ("BOARDINGPASS"), so neighbouring words also count joined up.
    joined = [a + b for a, b in zip(words, words[1:])]
    return " OR ".join(_q(w) for w in words + joined)


def _phrase_query(ph):
    words = ph.split()
    return f"{_q(ph)} OR {_q(''.join(words))}" if len(words) > 1 else _q(ph)


def _plain_match(query):
    return db.q("SELECT rowid, bm25(photo_text_plain) r FROM photo_text_plain WHERE photo_text_plain MATCH ? "
                "ORDER BY r LIMIT 2000", (query,))


def matches(text: str, phrases=()) -> dict:
    """{photo id: strength 0..1} for photos whose text contains words of `text` (or exact `phrases`).
    Case and accents don't matter: "καλημερα" finds "Καλημέρα"."""
    out = {}
    words = [w for w in re.findall(r"\w+", text) if len(w) >= 3]
    if words:
        rows = _plain_match(_fts_query(words))
        if rows:
            best = min(r["r"] for r in rows)  # bm25: lower is better, negative
            for r in rows:
                out[r["rowid"]] = max(0.2, r["r"] / best) if best else 1.0
    # Words run together with their neighbours on signs ("OFANAWARENESS") only match as part of a word.
    for w in {textnorm.searchable(w).replace("_", "") for w in words if len(w) >= 5}:
        for r in db.q("SELECT rowid FROM photo_text_plain WHERE plain LIKE ?", (f"%{w}%",)):
            out.setdefault(r["rowid"], 0.5)
    for ph in phrases:
        for r in _plain_match(_phrase_query(ph)):
            out[r["rowid"]] = 1.0
    return out


def phrase_ids(phrases) -> set | None:
    """Photos containing every exact phrase (for quoted searches), or None if there are none."""
    if not phrases:
        return None
    ids = None
    for ph in phrases:
        found = {r["rowid"] for r in _plain_match(_phrase_query(ph))}
        ids = found if ids is None else ids & found
    return ids or set()
