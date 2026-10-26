"""Turns images and text into vectors in one shared space, entirely on this machine."""
import functools
import hashlib
import threading

import numpy as np

from . import config

# Short queries work better phrased like a caption; we average a few phrasings.
TEMPLATES = ["a photo of {}.", "{}", "a picture showing {}."]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


class FastEmbedder:
    def __init__(self, model_key: str):
        spec = config.MODELS[model_key]
        self.key = model_key
        self.spec = spec
        self._image = None
        self._text = None
        self._lock = threading.Lock()
        self.state = "not loaded"
        self.loading = False

    def _load(self):
        with self._lock:
            if self._image is not None:
                return
            from fastembed import ImageEmbedding, TextEmbedding

            self.loading = True
            self.state = "downloading / loading model (first run only)"
            try:
                cache = str(config.MODEL_DIR)
                self._image = ImageEmbedding(self.spec["image"], cache_dir=cache)
                self._text = TextEmbedding(self.spec["text"], cache_dir=cache)
                self.state = "ready"
            except Exception as e:
                self.state = f"failed to load: {e}"
                raise
            finally:
                self.loading = False

    def embed_images(self, images) -> np.ndarray:
        self._load()
        vecs = list(self._image.embed(images, batch_size=config.BATCH_SIZE))
        return _normalize(np.stack(vecs))

    def embed_texts(self, texts) -> np.ndarray:
        self._load()
        return _normalize(np.stack(list(self._text.embed(list(texts)))))


class FakeEmbedder:
    """Deterministic stand-in for tests: images map to their average colour, text to colour words."""

    COLORS = {
        "red": (1, 0, 0), "green": (0, 1, 0), "blue": (0, 0, 1),
        "yellow": (1, 1, 0), "white": (1, 1, 1), "black": (0, 0, 0),
    }
    DIM = 16

    def __init__(self, model_key: str):
        self.key = model_key
        self.state = "ready"
        self.loading = False

    def _vec(self, rgb, seed: str):
        h = np.frombuffer(hashlib.sha256(seed.encode()).digest()[: self.DIM - 3], dtype=np.uint8)
        noise = (h.astype(np.float32) / 255.0 - 0.5) * 0.05
        return np.concatenate([np.asarray(rgb, dtype=np.float32) + 0.05, noise])

    def embed_images(self, images) -> np.ndarray:
        out = []
        for im in images:
            arr = np.asarray(im.convert("RGB").resize((8, 8)), dtype=np.float32) / 255.0
            out.append(self._vec(arr.reshape(-1, 3).mean(0), str(arr.sum())))
        return _normalize(np.stack(out))

    def embed_texts(self, texts) -> np.ndarray:
        out = []
        for t in texts:
            rgb = np.zeros(3, dtype=np.float32)
            for word, c in self.COLORS.items():
                if word in t.lower():
                    rgb += c
            out.append(self._vec(rgb, t))
        return _normalize(np.stack(out))


_embedder = None
_embedder_lock = threading.Lock()


def get(model_key: str | None = None):
    global _embedder
    from . import db

    key = model_key or db.get_setting("model", config.DEFAULT_MODEL)
    with _embedder_lock:
        if _embedder is None or _embedder.key != key:
            cls = FakeEmbedder if config.EMBEDDER == "fake" else FastEmbedder
            _embedder = cls(key)
        return _embedder


def embed_query(text: str) -> np.ndarray:
    """One normalized vector for a free-text query, averaged over caption-style phrasings."""
    return _cached_query(get().key, " ".join(text.split()))


def embed_queries(texts) -> np.ndarray:
    return np.stack([embed_query(t) for t in texts])


@functools.lru_cache(maxsize=512)
def _cached_query(model_key: str, text: str) -> np.ndarray:
    phrasings = [t.format(text) for t in TEMPLATES] if len(text.split()) <= 6 else [text]
    v = _normalize(get(model_key).embed_texts(phrasings).mean(0))
    v.setflags(write=False)
    return v
