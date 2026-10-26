"""Paths and tunables. Everything Lumen writes lives under LUMEN_HOME (default ~/.lumen)."""
import os
from pathlib import Path

HOME = Path(os.environ.get("LUMEN_HOME", Path.home() / ".lumen")).expanduser()
DB_PATH = HOME / "lumen.db"
THUMB_DIR = HOME / "previews"
MODEL_DIR = HOME / "models"

# Long edge of the smart previews shown in grids and collections.
PREVIEW_SIZE = 512
PREVIEW_QUALITY = 82

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff",
    ".heic", ".heif", ".avif",
}

# Models the user can choose between in Settings. Image and text halves must share a space.
MODELS = {
    "siglip2": {
        "label": "SigLIP 2 · multilingual",
        "description": "Best understanding, understands Greek and ~100 other languages. ~1.5 GB download.",
        "image": "google/siglip2-base-patch16-224",
        "text": "google/siglip2-base-patch16-224",
    },
    "clip": {
        "label": "CLIP ViT-B/32 · fast",
        "description": "Lighter and roughly 2x faster to index. English queries only. ~0.6 GB download.",
        "image": "Qdrant/clip-ViT-B-32-vision",
        "text": "Qdrant/clip-ViT-B-32-text",
    },
}
DEFAULT_MODEL = os.environ.get("LUMEN_MODEL", "siglip2")

# "fastembed" for real use; "fake" is a deterministic stand-in used by the tests.
EMBEDDER = os.environ.get("LUMEN_EMBEDDER", "fastembed")

BATCH_SIZE = 16

# Bumped when indexing starts extracting something new (faces, places, fingerprints), so
# photos indexed by an older version get a quick catch-up pass instead of a full re-index.
META_VERSION = 2

# Face grouping (people). Small OpenCV models (~40 MB) fetched once on first use.
FACES = os.environ.get("LUMEN_FACES", "1") != "0"
FACE_DIR = MODEL_DIR / "faces"
FACE_MODELS = {
    "detector": {
        "file": "face_detection_yunet_2023mar.onnx",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
               "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    },
    "recognizer": {
        "file": "face_recognition_sface_2021dec.onnx",
        "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
               "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    },
}
FACE_THUMB_DIR = THUMB_DIR / "faces"


def ensure_dirs():
    for d in (HOME, THUMB_DIR, MODEL_DIR, FACE_THUMB_DIR):
        d.mkdir(parents=True, exist_ok=True)
