"""Recognising your own pets: find cats and dogs (YOLOX, on-device), fingerprint each one with the
image model, and group them like people, so 'Luna in the snow' works once Luna has a name.

Detection runs only on photos the image model thinks show a cat or a dog. Set LUMEN_PETS=all to
check every photo, or off to skip it.
"""
import os
import threading

import numpy as np
from PIL import Image, ImageOps

from . import config, db, embedder, faces, search
from .vectors import index

MODEL = {
    "file": "object_detection_yolox_2022nov.onnx",
    "url": "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
           "object_detection_yolox/object_detection_yolox_2022nov.onnx",
    "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
}
SPECIES = {15: "cat", 16: "dog"}   # COCO classes
PET_PROMPTS = ["a photo of a dog", "a photo of a cat"]
MIN_Z = 2.0
MIN_SCORE = 0.5
MIN_SIDE = 64
SIZE = 640
# Image-model similarity between two crops of the same animal. Same-breed animals look alike,
# so this is stricter than for faces; merging groups in the UI is one click.
ASSIGN, LINK = 0.86, 0.88
BATCH = 8

_session = None
_lock = threading.Lock()
_grid = None


def mode():
    return os.environ.get("LUMEN_PETS", "auto").lower() if config.FACES else "off"


def _model():
    global _session, _grid
    with _lock:
        if _session is None:
            import onnxruntime as ort

            path = config.FACE_DIR / MODEL["file"]
            if not (path.exists() and faces._sha256(path) == MODEL["sha256"]):
                config.FACE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".part")
                faces.download(MODEL["url"], tmp)
                if faces._sha256(tmp) != MODEL["sha256"]:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("checksum mismatch for the pet detector")
                tmp.replace(path)
            _session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            g, st = [], []
            for stride in (8, 16, 32):
                n = SIZE // stride
                xv, yv = np.meshgrid(np.arange(n), np.arange(n))
                g.append(np.stack((xv, yv), 2).reshape(-1, 2))
                st.append(np.full((n * n, 1), stride))
            _grid = (np.concatenate(g), np.concatenate(st))
        return _session


def detect(im: Image.Image):
    """[(species, score, (x0, y0, x1, y1) in pixels)] for cats and dogs in a PIL image."""
    import cv2

    sess = _model()
    bgr = np.asarray(im.convert("RGB"))[:, :, ::-1]
    h, w = bgr.shape[:2]
    r = min(SIZE / h, SIZE / w)
    pad = np.full((SIZE, SIZE, 3), 114, np.float32)
    pad[: int(h * r), : int(w * r)] = cv2.resize(bgr, (int(w * r), int(h * r)))
    out = sess.run(None, {sess.get_inputs()[0].name: pad.transpose(2, 0, 1)[None]})[0][0]
    g, st = _grid
    out[:, :2] = (out[:, :2] + g) * st
    out[:, 2:4] = np.exp(out[:, 2:4]) * st
    scores = out[:, 4:5] * out[:, 5:]
    cls, best = scores.argmax(1), scores.max(1)
    keep = np.nonzero(np.isin(cls, list(SPECIES)) & (best >= MIN_SCORE))[0]
    if not len(keep):
        return []
    cx, cy, bw, bh = (out[keep, k] / r for k in range(4))
    boxes = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1)
    picked = cv2.dnn.NMSBoxes(boxes.tolist(), best[keep].tolist(), MIN_SCORE, 0.45)
    found = []
    for k in np.array(picked).reshape(-1):
        x, y, bw_, bh_ = boxes[k]
        # Plain floats: numpy scalars would be stored by SQLite as blobs.
        x0, y0, x1, y1 = float(max(0, x)), float(max(0, y)), float(min(w, x + bw_)), float(min(h, y + bh_))
        if min(x1 - x0, y1 - y0) >= MIN_SIDE:
            found.append((SPECIES[int(cls[keep[k]])], float(best[keep[k]]), (x0, y0, x1, y1)))
    return found


def _select():
    rows = db.q("SELECT id FROM photos WHERE status='ready' AND kind='photo' AND pets_v=0")
    if not rows:
        return []
    todo = [r["id"] for r in rows]
    if mode() == "all":
        return todo
    ids, mat = index.snapshot()
    if len(ids) < 8:
        return []
    z = search.robust_z(mat @ embedder.embed_queries(PET_PROMPTS).T).max(1)
    animal = {int(i) for i, zz in zip(ids, z) if zz >= MIN_Z}
    skip = [i for i in todo if i not in animal]
    if skip:
        with db.tx() as c:
            c.executemany("UPDATE photos SET pets_v=2 WHERE id=?", [(i,) for i in skip])
    return [i for i in todo if i in animal]


def refingerprint(limit=256) -> int:
    """After switching image models, re-fingerprint known pets from their thumbnails (names are kept)."""
    model = embedder.get().key
    rows = db.q("SELECT id FROM faces WHERE kind='pet' AND (model IS NULL OR model<>?) LIMIT ?", (model, limit))
    if not rows:
        return 0
    ims, ids = [], []
    for r in rows:
        try:
            with Image.open(faces.face_thumb_path(r["id"])) as im:
                ims.append(im.convert("RGB"))
            ids.append(r["id"])
        except OSError:
            pass
    vecs = embedder.get().embed_images(ims) if ims else []
    with db.tx() as c:
        c.executemany("UPDATE faces SET embedding=?, model=? WHERE id=?",
                      [(np.asarray(v, dtype=np.float16).tobytes(), model, i) for i, v in zip(ids, vecs)])
        missing = {r["id"] for r in rows} - set(ids)
        c.executemany("DELETE FROM faces WHERE id=?", [(i,) for i in missing])
    return len(rows)


def run_pending(should_stop=lambda: False) -> int:
    """Find pets in the next batch of candidate photos. Returns how many photos were checked."""
    if mode() == "off":
        return 0
    if refingerprint():
        return 1
    todo = _select()[:BATCH]
    if not todo:
        return 0
    found = []  # (photo id, species, score, box fractions, crop)
    checked = []
    for pid in todo:
        if should_stop():
            break
        row = db.one("SELECT path FROM photos WHERE id=?", (pid,))
        try:
            with Image.open(row["path"]) as im:
                im.draft("RGB", (1280, 1280))
                im = ImageOps.exif_transpose(im).convert("RGB")
            im.thumbnail((1280, 1280))
            for species, score, (x0, y0, x1, y1) in detect(im):
                side = max(x1 - x0, y1 - y0) * 1.15
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                crop = im.crop((int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                                int(min(im.width, cx + side / 2)), int(min(im.height, cy + side / 2))))
                found.append((pid, species, score,
                              (x0 / im.width, y0 / im.height, (x1 - x0) / im.width, (y1 - y0) / im.height), crop))
        except Exception:
            pass
        checked.append(pid)
    vecs = embedder.get().embed_images([f[4] for f in found]) if found else []
    with db.tx() as c:
        for (pid, species, score, (x, y, w, h), crop), v in zip(found, vecs):
            fid = c.execute(
                "INSERT INTO faces(photo_id, x, y, w, h, score, embedding, kind, species, model) "
                "VALUES(?,?,?,?,?,?,?,'pet',?,?)",
                (pid, x, y, w, h, score, np.asarray(v, dtype=np.float16).tobytes(), species, embedder.get().key),
            ).lastrowid
            p = faces.face_thumb_path(fid)
            p.parent.mkdir(parents=True, exist_ok=True)
            crop.thumbnail((faces.THUMB_SIZE, faces.THUMB_SIZE))
            crop.save(p, "WEBP", quality=85)
        c.executemany("UPDATE photos SET pets_v=1 WHERE id=?", [(i,) for i in checked])
    if found:
        faces.cluster_new_faces(kind="pet", assign=ASSIGN, link=LINK)
    return len(checked)
