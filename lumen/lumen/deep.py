"""Deep understanding: a vision-language model describes the photos you choose, once.

The image model gives each photo one overall impression. It is fast, but weak on what people are
doing to each other, what they feel, and what kind of moment it is. Here a vision-language model
(Qwen2.5-VL by default, the same local server as the closer look) writes a short description of
each chosen photo: who is in it, how they interact, expressions and emotions, colours and light,
the setting, the mood. The description is stored, shown in the viewer, and matched against
searches with a multilingual text model, so "two people laughing together" or "a tender moment"
can find photos by what is going on in them.

It is slow (seconds per photo), so it only runs on photos you queue: a folder, a collection or a
selection. It works through the queue in the background and picks up where it left off.
"""
import hashlib
import threading
import time

import numpy as np
import requests
from PIL import Image

from . import config, db, rerank
from .indexer import preview_path

TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"   # ~220 MB, ~50 languages
SYSTEM = ("You describe photos for a personal photo search engine. Describe only what is visible. "
          "Be concrete. Do not guess names.")
PROMPT = ("Describe this photo in 50 to 80 words of plain prose, without lists or headings. Cover: who or what "
          "is in it and how many; what they are doing and how they interact (touching, looking at each other, "
          "playing, talking, alone); facial expressions and emotions; the dominant colours and the light; the "
          "place and setting; and the mood and kind of moment (for example quiet, festive, tender, tense, playful).")


def describe_image(im: Image.Image, cfg) -> str:
    """Ask the model for a description of one picture. cfg needs "url" and "model"."""
    body = {
        "model": cfg["model"], "temperature": 0, "max_tokens": 200,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": rerank._data_url(im)}},
                {"type": "text", "text": PROMPT},
            ]},
        ],
    }
    r = requests.post(cfg["url"] + "/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return " ".join((r.json()["choices"][0]["message"]["content"] or "").split())


# ---------------------------------------------------------------- text embeddings
class _FakeText:
    """Deterministic stand-in for tests: a bag of hashed words."""
    DIM = 64

    def embed(self, texts):
        out = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in "".join(c.lower() if c.isalnum() else " " for c in t).split():
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.DIM] += 1.0
        return out


_text = None
_text_lock = threading.Lock()


def embed_texts(texts) -> np.ndarray:
    """Normalised vectors for descriptions or queries (same space for both)."""
    global _text
    with _text_lock:
        if _text is None:
            if config.EMBEDDER == "fake":
                _text = _FakeText()
            else:
                from fastembed import TextEmbedding

                _text = TextEmbedding(TEXT_MODEL, cache_dir=str(config.MODEL_DIR))
        v = np.asarray(list(_text.embed(list(texts))), dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


# ---------------------------------------------------------------- queue and worker
def enqueue(photo_ids) -> int:
    """Queue photos for a description. Ones already described by the current model are skipped."""
    model = rerank.settings()["model"]
    ids = [int(i) for i in photo_ids]
    added = 0
    with db.tx() as c:
        for k in range(0, len(ids), 500):
            chunk = ids[k : k + 500]
            marks = ",".join("?" * len(chunk))
            done = {r["photo_id"] for r in c.execute(
                f"SELECT photo_id FROM photo_descriptions WHERE model=? AND photo_id IN ({marks})", [model, *chunk])}
            ready = {r["id"] for r in c.execute(
                f"SELECT id FROM photos WHERE status='ready' AND id IN ({marks})", chunk)}
            for pid in chunk:
                if pid in ready and pid not in done:
                    added += c.execute("INSERT OR IGNORE INTO deep_queue(photo_id, added_at) VALUES(?,?)",
                                       (pid, time.time())).rowcount
    worker.nudge()
    return added


def clear_queue() -> None:
    with db.tx() as c:
        c.execute("DELETE FROM deep_queue")


def get(photo_id: int):
    r = db.one("SELECT text, model FROM photo_descriptions WHERE photo_id=?", (photo_id,))
    return {"text": r["text"], "model": r["model"]} if r else None


class Worker:
    def __init__(self):
        self.paused = False
        self.error = None
        self.current = None
        self._rate = None            # seconds per photo, smoothed
        self._wake = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="lumen-deep", daemon=True)
            self._thread.start()

    def nudge(self):
        self._wake.set()

    def status(self):
        queued = db.one("SELECT COUNT(*) n FROM deep_queue")["n"]
        described = db.one("SELECT COUNT(*) n FROM photo_descriptions")["n"]
        return {
            "queued": queued, "described": described, "paused": self.paused, "error": self.error,
            "state": "paused" if self.paused and queued else "describing" if queued and not self.error
            else "waiting" if queued else "idle",
            "seconds_per_photo": round(self._rate, 1) if self._rate else None,
            "eta_seconds": int(queued * self._rate) if self._rate and queued else None,
            "model": rerank.settings()["model"],
        }

    def _next(self):
        r = db.one("SELECT photo_id FROM deep_queue ORDER BY added_at, photo_id LIMIT 1")
        return r["photo_id"] if r else None

    def step(self) -> bool:
        """Describe one queued photo. False when there was nothing to do or the server is away."""
        pid = self._next()
        if pid is None:
            return False
        cfg = rerank.settings()
        t0 = time.time()
        try:
            with Image.open(preview_path(pid)) as im:
                text = describe_image(im, cfg)
        except FileNotFoundError:
            text = ""                                    # the photo went away meanwhile
        except requests.ConnectionError:
            self.error = f"No model server is answering at {cfg['url']}. Is Ollama running?"
            return False
        except requests.HTTPError as e:
            self.error = (f"The model server refused the request ({e.response.status_code}). "
                          f"Is “{cfg['model']}” installed?")
            return False
        self.error = None
        with db.tx() as c:
            if text:
                emb = embed_texts([text])[0].astype(np.float16).tobytes()
                c.execute("INSERT OR REPLACE INTO photo_descriptions(photo_id, model, text, embedding, created_at) "
                          "VALUES(?,?,?,?,?)", (pid, cfg["model"], text, emb, time.time()))
            c.execute("DELETE FROM deep_queue WHERE photo_id=?", (pid,))
        took = time.time() - t0
        self._rate = took if self._rate is None else 0.8 * self._rate + 0.2 * took
        return True

    def _run(self):
        db.init()
        while True:
            self._wake.wait(timeout=30)
            self._wake.clear()
            try:
                while not self.paused and self.step():
                    pass
            except Exception as e:      # keep going whatever happens
                self.error = f"{type(e).__name__}: {e}"
            if self.error:
                time.sleep(5)


worker = Worker()

# ---------------------------------------------------------------- search
_cache = {"key": None, "ids": None, "mat": None}


def _matrix():
    key = tuple(db.one("SELECT COUNT(*), MAX(created_at) FROM photo_descriptions"))
    if _cache["key"] != key:
        rows = db.q("SELECT photo_id, embedding FROM photo_descriptions")
        _cache.update(
            key=key, ids=np.array([r["photo_id"] for r in rows], dtype=np.int64),
            mat=np.stack([np.frombuffer(r["embedding"], dtype=np.float16) for r in rows]).astype(np.float32)
            if rows else np.zeros((0, 0), dtype=np.float32))
    return _cache["ids"], _cache["mat"]


def scores(ids, query: str):
    """How well each photo's description matches the query: an array aligned with ids, NaN where
    the photo has no description. None when nothing is described at all."""
    d_ids, mat = _matrix()
    if not len(d_ids) or not query.strip():
        return None
    sim = mat @ embed_texts([query])[0]
    pos = {int(p): k for k, p in enumerate(d_ids)}
    return np.array([sim[pos[int(i)]] if int(i) in pos else np.nan for i in ids], dtype=np.float32)
