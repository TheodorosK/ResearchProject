"""Background worker: walks folders, builds smart previews, reads EXIF and embeds photos."""
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import config, db, embedder
from .vectors import index

try:  # iPhone photos
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover - optional
    pass

Image.MAX_IMAGE_PIXELS = 200_000_000

EXIF_IFD = 0x8769
TAG_DATETIME_ORIGINAL = 36867
TAG_DATETIME = 306
TAG_MAKE, TAG_MODEL = 271, 272


def preview_path(photo_id: int) -> Path:
    return config.THUMB_DIR / f"{photo_id % 256:02x}" / f"{photo_id}.webp"


def _parse_exif_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def _read_meta(im: Image.Image):
    try:
        exif = im.getexif()
    except Exception:
        return None, None
    taken = _parse_exif_date(exif.get_ifd(EXIF_IFD).get(TAG_DATETIME_ORIGINAL)) or _parse_exif_date(
        exif.get(TAG_DATETIME)
    )
    make, model = (exif.get(TAG_MAKE) or "").strip(), (exif.get(TAG_MODEL) or "").strip()
    camera = model if make and model.lower().startswith(make.split()[0].lower()) else f"{make} {model}".strip()
    return taken, camera or None


def _prepare(row):
    """Open one photo; return (id, preview image, metadata) or (id, None, error)."""
    pid, path = row["id"], row["path"]
    prev = preview_path(pid)
    try:
        if prev.exists() and row["width"]:
            # Already have a preview (e.g. switching models): embed from it, skip the original.
            with Image.open(prev) as p:
                return pid, p.convert("RGB"), None
        with Image.open(path) as im:
            taken, camera = _read_meta(im)
            im.draft("RGB", (config.PREVIEW_SIZE * 2, config.PREVIEW_SIZE * 2))  # fast JPEG decode
            im = ImageOps.exif_transpose(im)
            if getattr(im, "n_frames", 1) > 1:
                im.seek(0)
            im = im.convert("RGB")
            im.thumbnail((config.PREVIEW_SIZE, config.PREVIEW_SIZE), Image.Resampling.LANCZOS)
            # Aspect ratio of the preview equals the oriented original; that's all the grid needs.
            prev.parent.mkdir(parents=True, exist_ok=True)
            im.save(prev, "WEBP", quality=config.PREVIEW_QUALITY, method=4)
            return pid, im, {"width": im.width, "height": im.height, "taken_at": taken, "camera": camera}
    except Exception as e:  # corrupt, unsupported, vanished...
        return pid, None, {"error": f"{type(e).__name__}: {e}"[:300]}


class Indexer:
    def __init__(self):
        self._wake = threading.Event()
        self._scan_requested = threading.Event()
        self._thread = None
        self.paused = False
        self.state = "idle"  # idle | scanning | indexing | paused
        self.current = ""
        self.scanned_files = 0
        self.batch_started = None
        self._recent = deque(maxlen=40)  # (timestamp, photos done) for rate/ETA
        self.last_error = None
        self._pool = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4))))

    # ---- control -------------------------------------------------------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="lumen-indexer", daemon=True)
            self._thread.start()
            self.request_scan()

    def request_scan(self):
        self._scan_requested.set()
        self._wake.set()

    def nudge(self):
        self._wake.set()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self._wake.set()

    # ---- status --------------------------------------------------------
    def status(self):
        counts = {r["status"]: r["n"] for r in db.q("SELECT status, COUNT(*) n FROM photos GROUP BY status")}
        total = sum(counts.values())
        ready, pending, errors = counts.get("ready", 0), counts.get("pending", 0), counts.get("error", 0)
        rate = None
        if len(self._recent) >= 2:
            (t0, n0), (t1, n1) = self._recent[0], self._recent[-1]
            if t1 > t0 and n1 > n0:
                rate = (n1 - n0) / (t1 - t0)
        emb = embedder.get()
        return {
            "state": "paused" if self.paused and pending else self.state,
            "total": total,
            "ready": ready,
            "pending": pending,
            "errors": errors,
            "rate": round(rate, 2) if rate else None,
            "eta_seconds": int(pending / rate) if rate and pending else None,
            "current": self.current,
            "scanned_files": self.scanned_files,
            "model": emb.key,
            "model_state": emb.state,
            "last_error": self.last_error,
        }

    # ---- worker --------------------------------------------------------
    def _run(self):
        db.init()
        while True:
            self._wake.wait(timeout=30)
            self._wake.clear()
            try:
                if self._scan_requested.is_set():
                    self._scan_requested.clear()
                    self._scan_all()
                if not self.paused:
                    self._index_pending()
            except Exception as e:  # keep the worker alive no matter what
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(2)
            self.state = "idle"
            self.current = ""

    def _scan_all(self):
        self.state = "scanning"
        self.scanned_files = 0
        for folder in db.q("SELECT id, path FROM folders ORDER BY id"):
            self._scan_folder(folder["id"], folder["path"])

    def _scan_folder(self, folder_id: int, root: str):
        self.current = root
        known = {
            r["path"]: (r["id"], r["size"], r["mtime"])
            for r in db.q("SELECT id, path, size, mtime FROM photos WHERE folder_id=?", (folder_id,))
        }
        seen, new_rows, changed = set(), [], []
        home = str(config.HOME)
        if not os.path.isdir(root):
            # Unplugged drive or renamed folder: keep the index rather than wiping it.
            return
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and os.path.join(dirpath, d) != home]
            self.current = dirpath
            for name in filenames:
                if name.startswith(".") or os.path.splitext(name)[1].lower() not in config.IMAGE_EXTENSIONS:
                    continue
                full = os.path.join(dirpath, name)
                try:
                    st = os.stat(full)
                except OSError:
                    continue
                self.scanned_files += 1
                seen.add(full)
                prev = known.get(full)
                if prev is None:
                    new_rows.append((full, folder_id, st.st_size, st.st_mtime))
                elif prev[1] != st.st_size or abs((prev[2] or 0) - st.st_mtime) > 1:
                    changed.append((st.st_size, st.st_mtime, prev[0]))
                if len(new_rows) >= 500:
                    self._insert(new_rows)
                    new_rows = []
        self._insert(new_rows)
        if changed:
            with db.tx() as c:
                c.executemany(
                    "UPDATE photos SET size=?, mtime=?, status='pending', width=NULL, error=NULL WHERE id=?",
                    changed,
                )
            for _, _, pid in changed:
                preview_path(pid).unlink(missing_ok=True)
        gone = [v[0] for p, v in known.items() if p not in seen]
        delete_photos(gone)
        with db.tx() as c:
            c.execute("UPDATE folders SET last_scan=? WHERE id=?", (db.now(), folder_id))

    @staticmethod
    def _insert(rows):
        if rows:
            with db.tx() as c:
                c.executemany(
                    "INSERT OR IGNORE INTO photos(path, folder_id, size, mtime, status) "
                    "VALUES(?, ?, ?, ?, 'pending')",
                    rows,
                )

    def _index_pending(self):
        emb = embedder.get()
        self._recent.clear()
        done = 0
        while not self.paused:
            if self._scan_requested.is_set():
                return  # a new scan was asked for; the loop comes straight back
            rows = db.q(
                "SELECT id, path, width FROM photos WHERE status='pending' ORDER BY id LIMIT ?",
                (config.BATCH_SIZE * 4,),
            )
            if not rows:
                return
            self.state = "indexing"
            if not self._recent:
                self._recent.append((time.time(), 0))
            prepared = list(self._pool.map(_prepare, rows))
            ok = [(pid, im, meta) for pid, im, meta in prepared if im is not None]
            bad = [(meta["error"], pid) for pid, im, meta in prepared if im is None]
            self.current = os.path.dirname(rows[-1]["path"])
            vecs = emb.embed_images([im for _, im, _ in ok]) if ok else np.zeros((0, 0))
            with db.tx() as c:
                for (pid, _, meta), vec in zip(ok, vecs):
                    blob = vec.astype(np.float16).tobytes()
                    if meta:
                        c.execute(
                            "UPDATE photos SET status='ready', error=NULL, embedding=?, model=?, "
                            "width=?, height=?, taken_at=COALESCE(?, mtime), camera=? WHERE id=?",
                            (blob, emb.key, meta["width"], meta["height"], meta["taken_at"], meta["camera"], pid),
                        )
                    else:
                        c.execute(
                            "UPDATE photos SET status='ready', error=NULL, embedding=?, model=? WHERE id=?",
                            (blob, emb.key, pid),
                        )
                c.executemany("UPDATE photos SET status='error', error=? WHERE id=?", bad)
            if ok:
                index.add([pid for pid, _, _ in ok], vecs)
            done += len(rows)
            self._recent.append((time.time(), done))


def delete_photos(ids):
    ids = list(ids)
    if not ids:
        return
    with db.tx() as c:
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            c.execute(f"DELETE FROM photos WHERE id IN ({','.join('?' * len(chunk))})", chunk)
    for pid in ids:
        preview_path(pid).unlink(missing_ok=True)
    index.remove(ids)


def switch_model(model_key: str):
    """Re-embed everything with another model. Previews are kept, so this is much faster than a first index."""
    db.set_setting("model", model_key)
    with db.tx() as c:
        c.execute("UPDATE photos SET status='pending' WHERE status='ready'")
    embedder.get(model_key)
    index.load(model_key)


worker = Indexer()
