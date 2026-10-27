"""Background worker: walks folders, builds smart previews, reads EXIF and embeds photos."""
import os
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from . import config, db, embedder, faces, marks, ocr, pets, places, quality
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
GPS_IFD = 0x8825


def preview_path(photo_id: int) -> Path:
    return config.THUMB_DIR / f"{photo_id % 256:02x}" / f"{photo_id}.webp"


def _parse_exif_date(value):
    if not value:
        return None
    try:
        return datetime.strptime(str(value).strip()[:19], "%Y:%m:%d %H:%M:%S").timestamp()
    except ValueError:
        return None


def _dms(value, ref):
    try:
        d, m, sec = (float(v) for v in value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    deg = d + m / 60 + sec / 3600
    return -deg if str(ref).upper() in ("S", "W") else deg


def _read_meta(im: Image.Image):
    """(taken_at, camera, lat, lon) from EXIF, each None when missing."""
    try:
        exif = im.getexif()
    except Exception:
        return None, None, None, None
    taken = _parse_exif_date(exif.get_ifd(EXIF_IFD).get(TAG_DATETIME_ORIGINAL)) or _parse_exif_date(
        exif.get(TAG_DATETIME)
    )
    make, model = (exif.get(TAG_MAKE) or "").strip(), (exif.get(TAG_MODEL) or "").strip()
    camera = model if make and model.lower().startswith(make.split()[0].lower()) else f"{make} {model}".strip()
    lat = lon = None
    try:
        gps = exif.get_ifd(GPS_IFD)
    except Exception:
        gps = {}
    if gps:
        lat, lon = _dms(gps.get(2), gps.get(1)), _dms(gps.get(4), gps.get(3))
        # Missing, out of range, or the 0,0 that some cameras write when they have no fix.
        if lat is None or lon is None or abs(lat) > 90 or abs(lon) > 180 or (abs(lat) < 1e-6 and abs(lon) < 1e-6):
            lat = lon = None
    return taken, camera or None, lat, lon


def trim_frame(im: Image.Image) -> Image.Image:
    """The photo inside a plain border (the white or black frames added for Instagram and prints),
    so a framed copy is analysed like the original. Returns the image unchanged if there's no frame."""
    from PIL import ImageChops

    rgb = im.convert("RGB")
    w, h = rgb.size
    corners = [rgb.getpixel(p) for p in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1))]
    if max(max(abs(a - b) for a, b in zip(c, corners[0])) for c in corners) > 24:
        return im  # corners differ: no uniform frame
    diff = ImageChops.difference(rgb, Image.new("RGB", rgb.size, corners[0])).convert("L")
    box = diff.point(lambda v: 255 if v > 28 else 0).getbbox()
    if not box:
        return im
    bw, bh = box[2] - box[0], box[3] - box[1]
    # Only a real frame: a few percent at least, and the photo still most of the picture.
    if bw * bh > 0.95 * w * h or bw * bh < 0.3 * w * h:
        return im
    return im.crop(box)


def dhash(im: Image.Image) -> int:
    """64-bit difference hash: survives resizing and re-compression, so it spots copies of a photo."""
    a = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    return int.from_bytes(np.packbits(a[:, 1:] > a[:, :-1]).tobytes(), "big", signed=True)


def _open_original(path):
    """Decode an original, oriented and RGB, at roughly the size the face detector wants."""
    with Image.open(path) as im:
        taken, camera, lat, lon = _read_meta(im)
        im.draft("RGB", (faces.DETECT_SIZE, faces.DETECT_SIZE))  # fast JPEG decode
        im = ImageOps.exif_transpose(im)
        if getattr(im, "n_frames", 1) > 1:
            im.seek(0)
        im = im.convert("RGB")
    return im, {"taken_at": taken, "camera": camera, "lat": lat, "lon": lon}


def subfolders_to_scan(dirpath, dirnames, home=None):
    """Which subfolders of dirpath hold photos worth indexing (skips caches, thumbnails, hidden folders)."""
    for suffix, originals in config.LIBRARY_ORIGINALS.items():
        if dirpath.lower().endswith(suffix):
            return [d for d in dirnames if d in originals]
    return [d for d in dirnames
            if not d.startswith(".") and d not in config.SKIP_DIRS
            and not d.lower().endswith(config.SKIP_DIR_SUFFIXES)
            and (home is None or os.path.join(dirpath, d) != home)]


def _inside_library_cache(dirpath):
    """True for files directly in a photo-app library's top level (its database, not photos)."""
    return any(dirpath.lower().endswith(s) for s in config.LIBRARY_ORIGINALS)


FILENAME_DATE = re.compile(
    r"(?<!\d)((?:19|20)\d\d)[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])"
    r"(?:\D{1,4}?([01]\d|2[0-3])[-_.:h]?([0-5]\d)[-_.:m]?(?:([0-5]\d)\d{0,3})?)?(?!\d)")  # PXL_…_142233123


def date_from_filename(path):
    """IMG_20230712_142233.jpg, PXL_20230712…, 2023-07-12 14.22.33.jpg, 'WhatsApp Image 2023-07-12 at
    14.22.33.jpeg', 'Screenshot 2023-07-12 at 14.22.33.png' -> timestamp, or None."""
    m = FILENAME_DATE.search(os.path.basename(path))
    if not m:
        return None
    y, mo, d, hh, mm, ss = (int(g) if g else 0 for g in m.groups())
    try:
        dt = datetime(y, mo, d, hh, mm, ss)
    except ValueError:
        return None
    return dt.timestamp() if datetime(1990, 1, 1) <= dt <= datetime.now() else None


def _set_date(c, pid, camera_taken):
    """Best available capture date. Camera dates beat dates in file names, which beat the file's
    modification time (that one is often just when the file was copied or exported)."""
    row = c.execute("SELECT path, mtime FROM photos WHERE id=?", (pid,)).fetchone()
    if camera_taken:
        taken, source = camera_taken, "camera"
    elif (from_name := date_from_filename(row["path"])):
        taken, source = from_name, "filename"
    else:
        taken, source = row["mtime"], "file"
    c.execute("UPDATE photos SET taken_at=?, date_source=? WHERE id=?", (taken, source, pid))


def _parse_iso6709(s):
    """'+37.9715+023.7257+050.000/' -> (37.9715, 23.7257)."""
    import re

    m = re.match(r"([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)", s or "")
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def _prepare_video(row):
    """Sample frames from a video: the middle one becomes the preview, all of them get embedded."""
    import av

    pid, path = row["id"], row["path"]
    with av.open(path) as c:
        stream = c.streams.video[0]
        duration = float(c.duration / av.time_base) if c.duration else float((stream.duration or 0) * stream.time_base)
        n = int(min(config.VIDEO_MAX_FRAMES, max(1, duration // config.VIDEO_SAMPLE_SECONDS)))
        times = [duration * (k + 0.5) / n for k in range(n)] if duration else [0.0]
        frames = []
        for t in times:
            try:
                c.seek(int(t / stream.time_base), stream=stream, any_frame=False, backward=True)
                frame = next(c.decode(stream))
            except (StopIteration, av.error.FFmpegError):
                continue
            im = frame.to_image()
            rot = getattr(frame, "rotation", 0) or 0
            if rot:
                im = im.rotate(rot, expand=True)
            im.thumbnail((config.PREVIEW_SIZE, config.PREVIEW_SIZE), Image.Resampling.LANCZOS)
            frames.append((round(t, 2), im.convert("RGB")))
        if not frames:
            raise ValueError("no decodable frames")
        meta = c.metadata or {}
        created = meta.get("creation_time")
        taken = None
        if created:
            try:
                taken = datetime.fromisoformat(created.replace("Z", "+00:00")).timestamp()
            except ValueError:
                pass
        lat, lon = _parse_iso6709(meta.get("location") or meta.get("com.apple.quicktime.location.ISO6709"))
    preview = frames[len(frames) // 2][1]
    prev = preview_path(pid)
    prev.parent.mkdir(parents=True, exist_ok=True)
    preview.save(prev, "WEBP", quality=config.PREVIEW_QUALITY, method=4)
    out = {"taken_at": taken, "camera": None, "lat": lat, "lon": lon, "width": preview.width,
           "height": preview.height, "phash": dhash(trim_frame(preview)), "duration": duration, "frames": frames,
           "faces": faces.engine.detect(preview) if faces.engine.enabled else None,
           **quality.measure(trim_frame(preview))}
    return pid, preview, out


def _prepare(row):
    """Open one photo; return (id, preview image, metadata) or (id, None, error)."""
    pid, path = row["id"], row["path"]
    prev = preview_path(pid)
    if row["kind"] == "video":
        try:
            return _prepare_video(row)
        except Exception as e:
            return pid, None, {"error": f"{type(e).__name__}: {e}"[:300]}
    try:
        if prev.exists() and row["width"]:
            # Already have a preview (e.g. switching models): embed from it, skip the original.
            with Image.open(prev) as p:
                return pid, p.convert("RGB"), None
        im, meta = _open_original(path)
        meta["faces"] = faces.engine.detect(im) if faces.engine.enabled else None
        im.thumbnail((config.PREVIEW_SIZE, config.PREVIEW_SIZE), Image.Resampling.LANCZOS)
        # Aspect ratio of the preview equals the oriented original; that's all the grid needs.
        prev.parent.mkdir(parents=True, exist_ok=True)
        im.save(prev, "WEBP", quality=config.PREVIEW_QUALITY, method=4)
        core = trim_frame(im)
        meta.update(width=im.width, height=im.height, phash=dhash(core), framed=int(core is not im),
                    **quality.measure(core))
        # The image model sees the photo without its frame, so framed copies search alike.
        return pid, core, meta
    except Exception as e:  # corrupt, unsupported, vanished...
        return pid, None, {"error": f"{type(e).__name__}: {e}"[:300]}


def _catch_up(row):
    """Extract what an older Lumen version didn't (places, fingerprint, faces) for an indexed photo."""
    pid = row["id"]
    if row["kind"] == "video":
        return pid, None  # videos get everything when indexed; nothing to catch up from the original
    want_faces = faces.engine.enabled and not row["faces_v"]
    try:
        if row["meta_v"] >= config.META_VERSION and not want_faces:
            return pid, {}
        # Places and faces need the original; the rest is measured on the preview.
        im, meta = (None, {})
        if row["meta_v"] < 2 or want_faces:
            im, meta = _open_original(row["path"])
            meta["faces"] = faces.engine.detect(im) if want_faces else None
        if row["meta_v"] < 5 and "taken_at" not in meta:
            with Image.open(row["path"]) as orig:  # header only: no decoding
                meta["taken_at"], meta["camera"], _, _ = _read_meta(orig)
        prev = preview_path(pid)
        if prev.exists():
            with Image.open(prev) as p:
                p = p.convert("RGB")
        else:
            if im is None:
                im, _ = _open_original(row["path"])
            p = im
            p.thumbnail((config.PREVIEW_SIZE, config.PREVIEW_SIZE))
        core = trim_frame(p)
        meta["framed"] = int(core is not p)
        p = core
        meta["phash"] = dhash(p)
        meta.update(quality.measure(p))
        return pid, meta
    except Exception:
        # The original is unreadable now (moved, unplugged); try again on a later run.
        return pid, None


def _write_meta(c, pid, meta):
    """Store places/fingerprint/faces for one photo inside an open transaction."""
    c.execute(
        "UPDATE photos SET phash=?, sharpness=?, brightness=?, hi_clip=?, lo_clip=?, palette=?, meta_v=? WHERE id=?",
        (meta["phash"], meta["sharpness"], meta["brightness"], meta["hi_clip"], meta["lo_clip"], meta["palette"],
         config.META_VERSION, pid),
    )
    if "lat" in meta:
        c.execute("UPDATE photos SET lat=?, lon=? WHERE id=?", (meta["lat"], meta["lon"], pid))
    if "framed" in meta:
        c.execute("UPDATE photos SET framed=? WHERE id=?", (meta["framed"], pid))
    if "taken_at" in meta and meta.get("width") is None:  # catch-up pass (new photos are dated above)
        _set_date(c, pid, meta["taken_at"])
        if meta.get("camera"):
            c.execute("UPDATE photos SET camera=? WHERE id=?", (meta["camera"], pid))
    if meta.get("faces") is not None and faces.engine.enabled:
        old = [r["id"] for r in c.execute("SELECT id FROM faces WHERE photo_id=? AND kind='person'", (pid,))]
        c.execute("DELETE FROM faces WHERE photo_id=? AND kind='person'", (pid,))
        faces.delete_face_thumbs(old)
        faces.save_faces(c, pid, meta["faces"])
        c.execute("UPDATE photos SET faces_v=1 WHERE id=?", (pid,))
        return len(meta["faces"])
    return 0


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
        self._new_faces = 0
        self._scan_lock = threading.Lock()
        self._scan_only = None   # folder ids to scan, or None for all
        self._pool = ThreadPoolExecutor(max_workers=max(2, min(6, (os.cpu_count() or 4))))

    # ---- control -------------------------------------------------------
    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="lumen-indexer", daemon=True)
            self._thread.start()
            self.request_scan()

    def request_scan(self, folder_ids=None):
        """Re-scan some folders (e.g. ones the watcher saw change), or all of them."""
        with self._scan_lock:
            if folder_ids is None:
                self._scan_only = None
            elif self._scan_only is not None or not self._scan_requested.is_set():
                self._scan_only = (self._scan_only or set()) | set(folder_ids)
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
        enrich = db.one(
            "SELECT COUNT(*) n FROM photos WHERE status='ready' AND (meta_v < ? OR (? AND faces_v=0))",
            (config.META_VERSION, int(faces.engine.enabled)),
        )["n"]
        return {
            "state": "paused" if self.paused and (pending or enrich) else self.state,
            "enrich_pending": enrich,
            "faces_state": faces.engine.state,
            "people": db.one("SELECT COUNT(*) n FROM persons WHERE hidden=0")["n"],
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
                    with self._scan_lock:
                        self._scan_requested.clear()
                        only, self._scan_only = self._scan_only, None
                    self._scan_all(only)
                if not self.paused:
                    self._index_pending()
                if not self.paused and not self._scan_requested.is_set():
                    self._catch_up_all()
                while places.geocode_pending():
                    pass
                while marks.import_sidecars():
                    pass
                while not self.paused and not self._scan_requested.is_set() \
                        and not db.one("SELECT 1 FROM photos WHERE status='pending' LIMIT 1"):
                    self.state = "reading"
                    stop = lambda: self.paused or self._scan_requested.is_set()  # noqa: E731
                    if not (ocr.run_pending(should_stop=stop) + pets.run_pending(should_stop=stop)):
                        break
                self._group_faces()
            except Exception as e:  # keep the worker alive no matter what
                self.last_error = f"{type(e).__name__}: {e}"
                time.sleep(2)
            self.state = "idle"
            self.current = ""

    def _group_faces(self, force=True):
        if self._new_faces and (force or self._new_faces >= 500):
            self._new_faces = 0
            self.current = "Grouping faces into people…"
            faces.cluster_new_faces()
        if force:
            self.current = "Reading expressions…"
            while not self.paused and faces.score_expressions():
                pass

    def _catch_up_all(self):
        while not self.paused and not self._scan_requested.is_set():
            if db.one("SELECT 1 FROM photos WHERE status='pending' LIMIT 1"):
                return
            rows = db.q(
                "SELECT id, path, meta_v, faces_v, kind FROM photos WHERE status='ready' AND "
                "(meta_v < ? OR (? AND faces_v=0)) ORDER BY id LIMIT ?",
                (config.META_VERSION, int(faces.engine.enabled), config.BATCH_SIZE * 4),
            )
            if not rows:
                return
            self.state = "enriching"
            self.current = os.path.dirname(rows[-1]["path"])
            results = list(self._pool.map(_catch_up, rows))
            with db.tx() as c:
                for pid, meta in results:
                    if meta is None:
                        # Can't open the original right now: mark done so we don't spin on it.
                        c.execute("UPDATE photos SET meta_v=?, faces_v=1 WHERE id=?", (config.META_VERSION, pid))
                    elif meta:
                        self._new_faces += _write_meta(c, pid, meta)
            self._group_faces(force=False)

    def _scan_all(self, only=None):
        self.state = "scanning"
        self.scanned_files = 0
        for folder in db.q("SELECT id, path FROM folders ORDER BY id"):
            if only is not None and folder["id"] not in only:
                continue
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
            dirnames[:] = subfolders_to_scan(dirpath, dirnames, home)
            if _inside_library_cache(dirpath):
                continue
            self.current = dirpath
            for name in filenames:
                if name.startswith(".") or os.path.splitext(name)[1].lower() not in config.MEDIA_EXTENSIONS:
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
                    kind = "video" if os.path.splitext(name)[1].lower() in config.VIDEO_EXTENSIONS else "photo"
                    new_rows.append((full, folder_id, st.st_size, st.st_mtime, kind))
                elif prev[1] != st.st_size or abs((prev[2] or 0) - st.st_mtime) > 1:
                    changed.append((st.st_size, st.st_mtime, prev[0]))
                if len(new_rows) >= 500:
                    self._insert(new_rows)
                    new_rows = []
        self._insert(new_rows)
        if changed:
            with db.tx() as c:
                c.executemany(
                    "UPDATE photos SET size=?, mtime=?, status='pending', width=NULL, error=NULL, "
                    "meta_v=0, faces_v=0, ocr_v=0, pets_v=0 WHERE id=?",
                    changed,
                )
                face_ids = [r["id"] for r in c.execute(
                    f"SELECT id FROM faces WHERE photo_id IN ({','.join('?' * len(changed))})",
                    [pid for _, _, pid in changed])]
                c.execute(f"DELETE FROM faces WHERE photo_id IN ({','.join('?' * len(changed))})",
                          [pid for _, _, pid in changed])
            faces.delete_face_thumbs(face_ids)
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
                    "INSERT OR IGNORE INTO photos(path, folder_id, size, mtime, kind, status) "
                    "VALUES(?, ?, ?, ?, ?, 'pending')",
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
                "SELECT id, path, width, kind FROM photos WHERE status='pending' ORDER BY id LIMIT ?",
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
            # Videos contribute several frames; the video's own vector is their average.
            flat, spans = [], []
            for _, im, meta in ok:
                frames = (meta or {}).get("frames") or [(None, im)]
                spans.append((len(flat), len(frames)))
                flat += [f for _, f in frames]
            flat_vecs = emb.embed_images(flat) if flat else np.zeros((0, 0))
            vecs = np.stack([embedder._normalize(flat_vecs[a : a + n].mean(0)) for a, n in spans]) if ok else flat_vecs
            with db.tx() as c:
                for (pid, _, meta), vec, (a, n) in zip(ok, vecs, spans):
                    if meta and meta.get("frames"):
                        c.execute("DELETE FROM video_frames WHERE photo_id=?", (pid,))
                        c.executemany(
                            "INSERT INTO video_frames(photo_id, t, embedding, model) VALUES(?,?,?,?)",
                            [(pid, t, flat_vecs[a + k].astype(np.float16).tobytes(), emb.key)
                             for k, (t, _) in enumerate(meta["frames"])],
                        )
                        c.execute("UPDATE photos SET duration=? WHERE id=?", (meta.get("duration"), pid))
                    blob = vec.astype(np.float16).tobytes()
                    if meta:
                        c.execute(
                            "UPDATE photos SET status='ready', error=NULL, embedding=?, model=?, "
                            "width=?, height=?, camera=? WHERE id=?",
                            (blob, emb.key, meta["width"], meta["height"], meta["camera"], pid),
                        )
                        _set_date(c, pid, meta["taken_at"])
                        self._new_faces += _write_meta(c, pid, meta)
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
            self._group_faces(force=False)


def delete_photos(ids):
    ids = list(ids)
    if not ids:
        return
    face_ids = []
    with db.tx() as c:
        for i in range(0, len(ids), 500):
            chunk = ids[i : i + 500]
            marks = ",".join("?" * len(chunk))
            face_ids += [r["id"] for r in c.execute(f"SELECT id FROM faces WHERE photo_id IN ({marks})", chunk)]
            c.execute(f"DELETE FROM photos WHERE id IN ({marks})", chunk)
            c.execute(f"DELETE FROM photo_text WHERE rowid IN ({marks})", chunk)
            c.execute(f"DELETE FROM photo_text_plain WHERE rowid IN ({marks})", chunk)
        # People left without any face disappear.
        c.execute("DELETE FROM persons WHERE id NOT IN (SELECT DISTINCT person_id FROM faces "
                  "WHERE person_id IS NOT NULL)")
    faces.delete_face_thumbs(face_ids)
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

from .watcher import FolderWatcher  # noqa: E402

watcher = FolderWatcher(on_change=lambda ids: worker.request_scan(ids))
