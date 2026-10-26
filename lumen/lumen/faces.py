"""Face detection, face fingerprints and grouping faces into people. All on-device (OpenCV YuNet + SFace)."""
import hashlib
import threading
import urllib.request

import numpy as np

from . import config, db

# SFace cosine similarity: OpenCV's suggested same-person threshold is 0.363. We stay a little
# stricter, because merging two people is one click but untangling a wrong group is tedious.
ASSIGN_THRESHOLD = 0.42   # new face joins an existing person
LINK_THRESHOLD = 0.45     # an unassigned face joins a forming group of new faces
MIN_SCORE = 0.85          # detector confidence
MIN_FACE_PX = 36          # smaller faces are too blurry to recognise reliably
DETECT_SIZE = 1280        # long edge the detector sees
THUMB_SIZE = 160


class FaceEngine:
    def __init__(self):
        self._local = threading.local()
        self._ready_lock = threading.Lock()
        self.state = "not loaded" if config.FACES else "off"

    @property
    def enabled(self):
        return config.FACES and not self.state.startswith("unavailable")

    def _ensure_models(self):
        with self._ready_lock:
            if self.state == "ready":
                return
            config.FACE_DIR.mkdir(parents=True, exist_ok=True)
            for spec in config.FACE_MODELS.values():
                path = config.FACE_DIR / spec["file"]
                if path.exists() and _sha256(path) == spec["sha256"]:
                    continue
                self.state = "downloading face models (first run only)"
                tmp = path.with_suffix(".part")
                urllib.request.urlretrieve(spec["url"], tmp)
                if _sha256(tmp) != spec["sha256"]:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f"checksum mismatch for {spec['file']}")
                tmp.replace(path)
            self.state = "ready"

    def _models(self):
        """OpenCV face models are not thread-safe, so each indexing thread gets its own pair."""
        m = getattr(self._local, "models", None)
        if m is None:
            import cv2

            try:
                cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
            except AttributeError:
                pass
            self._ensure_models()
            det = cv2.FaceDetectorYN.create(
                str(config.FACE_DIR / config.FACE_MODELS["detector"]["file"]), "", (320, 320), MIN_SCORE
            )
            rec = cv2.FaceRecognizerSF.create(str(config.FACE_DIR / config.FACE_MODELS["recognizer"]["file"]), "")
            m = self._local.models = (det, rec)
        return m

    def detect(self, im):
        """Faces in a PIL RGB image: [{box (fractions), score, embedding, thumb (PIL)}]."""
        if not self.enabled:
            return []
        try:
            det, rec = self._models()
        except Exception as e:
            self.state = f"unavailable: {e}"
            return []
        im = im.copy()
        im.thumbnail((DETECT_SIZE, DETECT_SIZE))
        bgr = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])
        h, w = bgr.shape[:2]
        det.setInputSize((w, h))
        _, found = det.detect(bgr)
        out = []
        for f in found if found is not None else []:
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            if score < MIN_SCORE or min(fw, fh) < MIN_FACE_PX:
                continue
            emb = rec.feature(rec.alignCrop(bgr, f)).reshape(-1).astype(np.float32)
            emb /= np.linalg.norm(emb) or 1
            # A square crop with some margin, for the round avatars in the UI.
            side = max(fw, fh) * 1.5
            cx, cy = x + fw / 2, y + fh / 2
            box = (max(0, cx - side / 2), max(0, cy - side / 2), min(w, cx + side / 2), min(h, cy + side / 2))
            thumb = im.crop(tuple(int(v) for v in box))
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))
            out.append({
                "box": (max(0.0, x / w), max(0.0, y / h), fw / w, fh / h),
                "score": score,
                "embedding": emb,
                "thumb": thumb,
            })
        return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def face_thumb_path(face_id: int):
    return config.FACE_THUMB_DIR / f"{face_id % 256:02x}" / f"{face_id}.webp"


def save_faces(conn, photo_id: int, faces):
    """Insert detected faces for one photo (inside the caller's transaction) and write their thumbnails."""
    for f in faces:
        x, y, w, h = f["box"]
        fid = conn.execute(
            "INSERT INTO faces(photo_id, x, y, w, h, score, embedding) VALUES(?,?,?,?,?,?,?)",
            (photo_id, x, y, w, h, f["score"], f["embedding"].astype(np.float16).tobytes()),
        ).lastrowid
        p = face_thumb_path(fid)
        p.parent.mkdir(parents=True, exist_ok=True)
        f["thumb"].save(p, "WEBP", quality=85)


def delete_face_thumbs(face_ids):
    for fid in face_ids:
        face_thumb_path(fid).unlink(missing_ok=True)


def _emb(blob):
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


def _unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def cluster_new_faces():
    """Assign unassigned faces to people. Existing people stay put; only new faces move.

    1. A face close enough to an existing person's centre joins that person.
    2. The rest are grouped with each other; every group of two or more becomes a new,
       unnamed person. Lone faces wait for company.
    """
    loose = db.q("SELECT id, embedding FROM faces WHERE person_id IS NULL AND locked=0 ORDER BY score DESC")
    if not loose:
        return 0
    ids = np.array([r["id"] for r in loose])
    vecs = _unit(np.stack([_emb(r["embedding"]) for r in loose]))
    assignments = {}

    people = db.q("SELECT person_id, embedding FROM faces WHERE person_id IS NOT NULL")
    if people:
        pids = np.array([r["person_id"] for r in people])
        pvecs = np.stack([_emb(r["embedding"]) for r in people])
        uniq = np.unique(pids)
        centres = _unit(np.stack([pvecs[pids == p].mean(0) for p in uniq]))
        sims = vecs @ centres.T
        best = sims.argmax(1)
        for i in np.nonzero(sims[np.arange(len(ids)), best] >= ASSIGN_THRESHOLD)[0]:
            assignments[int(ids[i])] = int(uniq[best[i]])

    # Greedy centroid ("leader") clustering: unlike single-link it doesn't chain different
    # people together through a series of look-alike faces.
    rest = [i for i, fid in enumerate(ids) if int(fid) not in assignments]
    sums = np.zeros((max(len(rest), 1), vecs.shape[1]), dtype=np.float32)
    centres = np.zeros_like(sums)
    members = []
    for i in rest:
        v = vecs[i]
        k = len(members)
        if k:
            sims = centres[:k] @ v
            j = int(sims.argmax())
            if sims[j] >= LINK_THRESHOLD:
                members[j].append(int(ids[i]))
                sums[j] += v
                centres[j] = sums[j] / (np.linalg.norm(sums[j]) + 1e-8)
                continue
        members.append([int(ids[i])])
        sums[k] = v
        centres[k] = v
    new_groups = [g for g in members if len(g) >= 2]

    with db.tx() as c:
        c.executemany("UPDATE faces SET person_id=? WHERE id=?", [(p, f) for f, p in assignments.items()])
        for g in new_groups:
            pid = c.execute("INSERT INTO persons(created_at) VALUES(?)", (db.now(),)).lastrowid
            c.executemany("UPDATE faces SET person_id=? WHERE id=?", [(pid, f) for f in g])
    return len(assignments) + sum(len(g) for g in new_groups)


def person_photo_ids(person_id: int):
    return [r["photo_id"] for r in db.q(
        "SELECT DISTINCT f.photo_id FROM faces f JOIN photos p ON p.id=f.photo_id "
        "WHERE f.person_id=? AND p.status='ready' ORDER BY p.taken_at DESC", (person_id,))]


def cover_face(person_id: int):
    row = db.one(
        "SELECT COALESCE((SELECT cover_face_id FROM persons WHERE id=? AND cover_face_id IN "
        "(SELECT id FROM faces WHERE person_id=?)), "
        "(SELECT id FROM faces WHERE person_id=? ORDER BY score * w DESC LIMIT 1)) fid",
        (person_id, person_id, person_id),
    )
    return row["fid"] if row else None


engine = FaceEngine()
