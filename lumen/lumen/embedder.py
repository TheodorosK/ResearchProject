"""Turns images and text into vectors in one shared space, entirely on this machine."""
import functools
import hashlib
import os
import threading

import numpy as np

from . import config

# Short queries work better phrased like a caption; we average a few phrasings.
TEMPLATES = ["a photo of {}.", "{}", "a picture showing {}."]


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


ACCELERATORS = [
    ("CUDAExecutionProvider", "NVIDIA GPU (CUDA)"),
    ("DmlExecutionProvider", "GPU (DirectML)"),
    ("CoreMLExecutionProvider", "Apple Neural Engine / GPU (Core ML)"),
]


def pick_device():
    """(onnxruntime providers, human label) for the fastest hardware available, honouring the
    'device' setting (auto | cpu)."""
    from . import db

    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
    except Exception:
        available = []
    if (db.get_setting("device") or os.environ.get("LUMEN_DEVICE", "auto")) != "cpu":
        for provider, label in ACCELERATORS:
            if provider in available:
                return [provider, "CPUExecutionProvider"], label
    return ["CPUExecutionProvider"], "CPU"


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
        self.device = "CPU"
        self.batch_size = config.BATCH_SIZE

    def _load(self, force_cpu=False):
        with self._lock:
            if self._image is not None:
                return
            from fastembed import ImageEmbedding, TextEmbedding

            self.loading = True
            self.state = "downloading / loading model (first run only)"
            providers, label = (["CPUExecutionProvider"], "CPU") if force_cpu else pick_device()
            try:
                cache = str(config.MODEL_DIR)
                try:
                    self._image = ImageEmbedding(self.spec["image"], cache_dir=cache, providers=providers)
                    self._text = TextEmbedding(self.spec["text"], cache_dir=cache, providers=providers)
                except Exception:
                    if providers == ["CPUExecutionProvider"]:
                        raise
                    # The accelerator wouldn't take the model; the CPU always will.
                    providers, label = ["CPUExecutionProvider"], "CPU (accelerator unavailable)"
                    self._image = ImageEmbedding(self.spec["image"], cache_dir=cache, providers=providers)
                    self._text = TextEmbedding(self.spec["text"], cache_dir=cache, providers=providers)
                self.device = label
                self.batch_size = config.BATCH_SIZE * (4 if providers[0] != "CPUExecutionProvider" else 1)
                self.state = "ready"
            except Exception as e:
                self.state = f"failed to load: {e}"
                raise
            finally:
                self.loading = False

    def _retry_on_cpu(self, fn):
        try:
            return fn()
        except Exception:
            if self.device.startswith("CPU"):
                raise
            # Some accelerators fail on particular inputs at run time: fall back for good.
            with self._lock:
                self._image = self._text = None
            self._load(force_cpu=True)
            return fn()

    def embed_images(self, images) -> np.ndarray:
        self._load()
        images = list(images)
        return self._retry_on_cpu(
            lambda: _normalize(np.stack(list(self._image.embed(images, batch_size=self.batch_size)))))

    def embed_texts(self, texts) -> np.ndarray:
        self._load()
        texts = list(texts)
        return self._retry_on_cpu(lambda: _normalize(np.stack(list(self._text.embed(texts)))))


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
        self.device = "CPU (test stand-in)"

    def _vec(self, rgb, seed: str):
        # Last dimension stays 0 for real content; text without colour words points only there,
        # so it scores every image the same (a neutral query, handy for testing re-ranking).
        h = np.frombuffer(hashlib.sha256(seed.encode()).digest()[: self.DIM - 4], dtype=np.uint8)
        noise = (h.astype(np.float32) / 255.0 - 0.5) * 0.05
        return np.concatenate([np.asarray(rgb, dtype=np.float32) + 0.05, noise, [0.0]])

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
            hits = [c for word, c in self.COLORS.items() if word in t.lower()]
            if not hits:
                out.append(np.eye(self.DIM, dtype=np.float32)[-1])
                continue
            for c in hits:
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
