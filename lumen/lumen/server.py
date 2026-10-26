"""HTTP API and static UI. Binds to localhost only: your photos stay on your machine."""
import io
import os
import platform
import string
import subprocess
import tempfile
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import config, db, describe, duplicates, embedder, events, faces, jobs, ocr, places, quality, search
from .indexer import delete_photos, preview_path, switch_model, worker
from .vectors import index

STATIC = Path(__file__).parent / "static"
BROWSER_FRIENDLY = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".bmp"}


@asynccontextmanager
async def lifespan(app):
    db.init()
    index.load(embedder.get().key)
    worker.start()
    yield


app = FastAPI(title="Lumen", lifespan=lifespan)


# ---------------------------------------------------------------- helpers
def hydrate(items):
    """Attach the fields the grid needs to a list of ids or ranked hits, preserving order."""
    if not items:
        return []
    hits = [i if isinstance(i, dict) else {"id": i} for i in items]
    ids = [h["id"] for h in hits]
    rows = {}
    for i in range(0, len(ids), 900):
        chunk = ids[i : i + 900]
        for r in db.q(
            f"SELECT id, path, width, height, taken_at, camera, folder_id, lat, lon, size, kind, duration, "
            f"place_city, place_region, place_country FROM photos "
            f"WHERE id IN ({','.join('?' * len(chunk))})",
            chunk,
        ):
            rows[r["id"]] = r
    out = []
    for h in hits:
        r = rows.get(h["id"])
        if r is None:
            continue
        out.append({
            **h,
            "name": os.path.basename(r["path"]),
            "w": r["width"] or 4,
            "h": r["height"] or 3,
            "taken_at": r["taken_at"],
            "camera": r["camera"],
            "folder_id": r["folder_id"],
            "lat": r["lat"],
            "lon": r["lon"],
            "size": r["size"],
            "dir": os.path.basename(os.path.dirname(r["path"])),
            "place": places.label(r),
            "kind": r["kind"],
            "duration": r["duration"],
        })
    return out


def photo_or_404(photo_id: int):
    row = db.one("SELECT * FROM photos WHERE id=?", (photo_id,))
    if not row:
        raise HTTPException(404, "Photo not found")
    return row


def collection_or_404(cid: int):
    row = db.one("SELECT * FROM collections WHERE id=?", (cid,))
    if not row:
        raise HTTPException(404, "Collection not found")
    return row


def collection_photo_ids(c) -> list[int]:
    if c["kind"] == "smart":
        return [h["id"] for h in search.text_search(c["query"] or "", limit=500, min_z=search.TIER_GOOD, min_count=0)]
    return [
        r["photo_id"]
        for r in db.q(
            "SELECT photo_id FROM collection_items WHERE collection_id=? ORDER BY added_at DESC", (c["id"],)
        )
    ]


def collection_dict(c, with_preview=True):
    ids = collection_photo_ids(c)
    d = {k: c[k] for k in c.keys()}
    d["count"] = len(ids)
    if with_preview:
        cover = [c["cover_photo_id"]] if c["cover_photo_id"] in ids else []
        best = quality.sort_by_quality(ids[:500])
        d["preview_ids"] = (cover + [i for i in best if i not in cover])[:4]
    return d


def require_model():
    if embedder.get().loading:
        raise HTTPException(503, "Lumen is still downloading its understanding model (first run only). Try again in a minute.")


def filters_from(folder_id=None, year_from=None, year_to=None, collection_id=None):
    f = {"folder_id": folder_id, "year_from": year_from, "year_to": year_to, "collection_id": collection_id}
    return {k: v for k, v in f.items() if v}


# ---------------------------------------------------------------- status & settings
@app.get("/api/status")
def status():
    s = worker.status()
    s["folders"] = db.one("SELECT COUNT(*) n FROM folders")["n"]
    s["collections"] = db.one("SELECT COUNT(*) n FROM collections")["n"]
    return s


@app.post("/api/index/pause")
def pause():
    worker.pause()
    return worker.status()


@app.post("/api/index/resume")
def resume():
    worker.resume()
    return worker.status()


@app.post("/api/index/retry-errors")
def retry_errors():
    with db.tx() as c:
        c.execute("UPDATE photos SET status='pending', width=NULL, error=NULL WHERE status='error'")
    worker.nudge()
    return worker.status()


@app.get("/api/settings")
def get_settings():
    return {
        "model": embedder.get().key,
        "models": [{"key": k, **{f: v[f] for f in ("label", "description")}} for k, v in config.MODELS.items()],
        "data_dir": str(config.HOME),
    }


class SettingsIn(BaseModel):
    model: str


@app.put("/api/settings")
def put_settings(body: SettingsIn):
    if body.model not in config.MODELS:
        raise HTTPException(400, "Unknown model")
    if body.model != embedder.get().key:
        switch_model(body.model)
        worker.nudge()
    return get_settings()


# ---------------------------------------------------------------- folders
@app.get("/api/folders")
def list_folders():
    out = []
    for f in db.q("SELECT * FROM folders ORDER BY path"):
        counts = {
            r["status"]: r["n"]
            for r in db.q("SELECT status, COUNT(*) n FROM photos WHERE folder_id=? GROUP BY status", (f["id"],))
        }
        out.append({
            **{k: f[k] for k in f.keys()},
            "name": os.path.basename(f["path"].rstrip("/\\")) or f["path"],
            "exists": os.path.isdir(f["path"]),
            "photos": sum(counts.values()),
            "ready": counts.get("ready", 0),
            "pending": counts.get("pending", 0),
            "errors": counts.get("error", 0),
        })
    return out


class FolderIn(BaseModel):
    path: str


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(p.strip().strip('"'))))


def _inside(child: str, parent: str) -> bool:
    try:
        return os.path.commonpath([child, parent]) == parent
    except ValueError:  # different drives on Windows
        return False


@app.post("/api/folders")
def add_folder(body: FolderIn):
    path = _norm(body.path)
    if not os.path.isdir(path):
        raise HTTPException(400, "That folder doesn't exist or isn't reachable.")
    if _inside(path, _norm(str(config.HOME))):
        raise HTTPException(400, "That's Lumen's own data folder.")
    existing = db.q("SELECT id, path FROM folders")
    for f in existing:
        if _inside(path, _norm(f["path"])):
            raise HTTPException(409, f"Already included through “{f['path']}”.")
    with db.tx() as c:
        cur = c.execute("INSERT INTO folders(path, added_at) VALUES(?, ?)", (path, db.now()))
        fid = cur.lastrowid
        # Adding a parent of existing folders absorbs them, keeping their indexed photos.
        for f in existing:
            if _inside(_norm(f["path"]), path):
                c.execute("UPDATE photos SET folder_id=? WHERE folder_id=?", (fid, f["id"]))
                c.execute("DELETE FROM folders WHERE id=?", (f["id"],))
    worker.request_scan()
    return {"id": fid, "path": path}


@app.delete("/api/folders/{folder_id}")
def remove_folder(folder_id: int):
    ids = [r["id"] for r in db.q("SELECT id FROM photos WHERE folder_id=?", (folder_id,))]
    delete_photos(ids)
    with db.tx() as c:
        c.execute("DELETE FROM folders WHERE id=?", (folder_id,))
    return {"removed_photos": len(ids)}


@app.post("/api/scan")
def rescan():
    worker.request_scan()
    return {"ok": True}


def _roots():
    home = Path.home()
    roots = [{"name": "Home", "path": str(home)}]
    for name in ("Pictures", "Photos", "Desktop", "Downloads", "OneDrive/Pictures"):
        p = home / name
        if p.is_dir():
            roots.append({"name": name.split("/")[-1], "path": str(p)})
    if platform.system() == "Windows":
        roots += [{"name": f"{d}:", "path": f"{d}:\\"} for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    else:
        for mnt in ("/Volumes", "/media", "/mnt", f"/media/{home.name}", f"/run/media/{home.name}"):
            if os.path.isdir(mnt):
                try:
                    roots += [
                        {"name": e.name, "path": e.path}
                        for e in os.scandir(mnt)
                        if e.is_dir() and not e.name.startswith(".")
                    ]
                except OSError:
                    pass
    return roots


@app.get("/api/fs")
def browse(path: str | None = None):
    """Directory listing for the folder picker. Only names of folders and counts of images, never contents."""
    p = _norm(path) if path else str(Path.home())
    if not os.path.isdir(p):
        raise HTTPException(404, "Folder not found")
    dirs, images = [], 0
    try:
        with os.scandir(p) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    if e.is_dir():
                        dirs.append({"name": e.name, "path": e.path})
                    elif os.path.splitext(e.name)[1].lower() in config.IMAGE_EXTENSIONS:
                        images += 1
                except OSError:
                    continue
    except PermissionError:
        raise HTTPException(403, "No permission to open that folder")
    dirs.sort(key=lambda d: d["name"].lower())
    parent = os.path.dirname(p.rstrip("/\\")) if p.rstrip("/\\") else None
    return {"path": p, "parent": parent if parent and parent != p else None, "dirs": dirs,
            "images": images, "roots": _roots()}


# ---------------------------------------------------------------- photos
@app.get("/api/photos")
def list_photos(
    offset: int = 0, limit: int = Query(200, le=1000), folder_id: int | None = None,
    year: int | None = None, order: str = "newest",
):
    where, args = ["status='ready'"], []
    if folder_id:
        where.append("folder_id=?")
        args.append(folder_id)
    if year:
        where.append("taken_at >= ? AND taken_at < ?")
        args += [datetime(year, 1, 1).timestamp(), datetime(year + 1, 1, 1).timestamp()]
    direction = "ASC" if order == "oldest" else "DESC"
    w = " AND ".join(where)
    total = db.one(f"SELECT COUNT(*) n FROM photos WHERE {w}", args)["n"]
    if order == "best":
        ids = quality.sort_by_quality([r["id"] for r in db.q(f"SELECT id FROM photos WHERE {w}", args)])
        return {"total": total, "items": hydrate(ids[offset : offset + limit])}
    rows = db.q(f"SELECT id FROM photos WHERE {w} ORDER BY taken_at {direction}, id {direction} LIMIT ? OFFSET ?",
                args + [limit, offset])
    return {"total": total, "items": hydrate([r["id"] for r in rows])}


@app.get("/api/years")
def years():
    rows = db.q(
        "SELECT CAST(strftime('%Y', taken_at, 'unixepoch', 'localtime') AS INTEGER) y, COUNT(*) n "
        "FROM photos WHERE status='ready' AND taken_at IS NOT NULL GROUP BY y ORDER BY y DESC"
    )
    return [{"year": r["y"], "count": r["n"]} for r in rows if r["y"]]


@app.get("/api/photos/{photo_id}")
def photo_detail(photo_id: int):
    r = photo_or_404(photo_id)
    cols = db.q(
        "SELECT c.id, c.name, c.emoji, c.color FROM collections c JOIN collection_items i ON i.collection_id=c.id "
        "WHERE i.photo_id=? ORDER BY c.name",
        (photo_id,),
    )
    try:
        size = os.path.getsize(r["path"])
    except OSError:
        size = r["size"]
    return {
        **hydrate([photo_id])[0],
        "path": r["path"],
        "folder": os.path.dirname(r["path"]),
        "size": size,
        "status": r["status"],
        "error": r["error"],
        "collections": [dict(c) for c in cols],
        "people": [_face_entry(p) for p in db.q(
            "SELECT f.id face_id, f.x, f.y, f.w, f.h, f.expr_probs, f.kind, f.species, p.id, p.name FROM faces f "
            "LEFT JOIN persons p ON p.id=f.person_id WHERE f.photo_id=? AND (p.hidden IS NULL OR p.hidden=0) "
            "ORDER BY f.x", (photo_id,))],
        "quality": quality.scores().get(photo_id, (None, None))[1],
        "text": ocr.text_of(photo_id),
        "seen": describe.describe(photo_id),
        "sharpness": r["sharpness"],
    }


def _face_entry(row):
    d = {k: row[k] for k in row.keys() if k != "expr_probs"}
    label, prob = faces.expression_of(row["expr_probs"])
    d["expression"] = label if label and prob >= 0.5 else None
    d["expression_emoji"] = faces.EXPRESSION_EMOJI.get(d["expression"])
    return d


@app.get("/api/photos/{photo_id}/preview")
def photo_preview(photo_id: int):
    p = preview_path(photo_id)
    if not p.exists():
        raise HTTPException(404, "Preview not ready yet")
    return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "public, max-age=604800"})


@app.get("/api/photos/{photo_id}/original")
def photo_original(photo_id: int):
    r = photo_or_404(photo_id)
    path = r["path"]
    if not os.path.exists(path):
        raise HTTPException(410, "The original file has moved or been deleted")
    if r["kind"] == "video" or os.path.splitext(path)[1].lower() in BROWSER_FRIENDLY:
        return FileResponse(path, headers={"Cache-Control": "private, max-age=3600"})
    # HEIC/TIFF: browsers can't show these, so convert on the fly.
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((2560, 2560))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    return Response(buf.getvalue(), media_type="image/jpeg")


@app.get("/api/photos/{photo_id}/download")
def photo_download(photo_id: int):
    r = photo_or_404(photo_id)
    if not os.path.exists(r["path"]):
        raise HTTPException(410, "The original file has moved or been deleted")
    return FileResponse(r["path"], filename=os.path.basename(r["path"]))


@app.post("/api/photos/{photo_id}/open")
def photo_open(photo_id: int):
    """Open the file in its default app (for videos the browser can't play)."""
    path = photo_or_404(photo_id)["path"]
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", path])
        elif system == "Windows":
            os.startfile(path)  # noqa: S606 - local app, user's own file
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as e:
        raise HTTPException(500, f"Couldn't open it: {e}")
    return {"ok": True}


@app.post("/api/photos/{photo_id}/reveal")
def photo_reveal(photo_id: int):
    """Open the photo's folder in Finder / Explorer / the file manager."""
    path = photo_or_404(photo_id)["path"]
    system = platform.system()
    try:
        if system == "Darwin":
            subprocess.Popen(["open", "-R", path])
        elif system == "Windows":
            subprocess.Popen(["explorer", "/select,", path])
        else:
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
    except OSError as e:
        raise HTTPException(500, f"Couldn't open the file manager: {e}")
    return {"ok": True}


@app.get("/api/photos/{photo_id}/similar")
def photo_similar(photo_id: int, limit: int = 60):
    photo_or_404(photo_id)
    return hydrate(search.similar(photo_id, limit))


class IdsIn(BaseModel):
    photo_ids: list[int] = Field(default_factory=list)


@app.post("/api/similar")
def similar_many(body: IdsIn):
    return hydrate(search.similar_to_many(body.photo_ids))


# ---------------------------------------------------------------- search & discovery
def _hex_param(v: str | None):
    import re

    return [h.lower() for h in (v or "").split(",") if re.fullmatch(r"#?[0-9a-fA-F]{6}", h.strip())][:5]


@app.post("/api/search/image")
async def search_by_image(file: UploadFile, limit: int = Query(200, le=1000)):
    """Find photos that look like an image you drop in (from anywhere, even the web)."""
    require_model()
    try:
        im = Image.open(io.BytesIO(await file.read()))
        im = ImageOps.exif_transpose(im).convert("RGB")
    except Exception:
        raise HTTPException(400, "That file isn't an image Lumen can read")
    im.thumbnail((config.PREVIEW_SIZE, config.PREVIEW_SIZE))
    v = embedder.get().embed_images([im])[0]
    ids, mat = index.snapshot()
    hits = search._rank(mat @ v, ids, limit, search.TIER_MAYBE, min_count=24) if len(ids) else []
    return {"items": hydrate(hits), "indexed": len(ids)}


def _ids_param(v: str | None):
    return [int(x) for x in (v or "").split(",") if x.strip().isdigit()][:200]


@app.get("/api/search")
def do_search(
    q: str = "", limit: int = Query(300, le=1000), folder_id: int | None = None,
    year_from: int | None = None, year_to: int | None = None, collection_id: int | None = None,
    sort: str = "relevance", like: str | None = None, dislike: str | None = None, colors: str | None = None,
):
    require_model()
    like_ids, dislike_ids = _ids_param(like), _ids_param(dislike)
    hits = search.text_search(q, limit, sort=sort, filters=filters_from(folder_id, year_from, year_to, collection_id),
                              like=like_ids, dislike=dislike_ids, colors=_hex_param(colors))
    u = search.understand(q)
    label = u["expression"]
    expression = {"label": label, "emoji": faces.EXPRESSION_EMOJI[label], "everyone": u["everyone"]} if label else None
    return {"query": q, "parsed": {"positive": u["positive"], "negatives": u["negatives"], "people": u["people"],
                                   "phrases": u["phrases"], "colors": _hex_param(colors),
                                   "places": [p["label"] for p in u["places"]], "when": u["when"].label or None,
                                   "expression": expression, "like": like_ids, "dislike": dislike_ids},
            "items": hydrate(hits), "indexed": len(index)}


@app.get("/api/home")
def home():
    loading = embedder.get().loading
    groups = [] if loading else search.explore()
    mood_groups = [] if loading else search.moods()
    return {
        "suggestions": search.suggestions(),
        "on_this_day": hydrate(search.on_this_day()),
        "explore": [{**g, "photos": hydrate(g.pop("photo_ids"))} for g in (dict(g) for g in groups)],
        "moods": [{**g, "photos": hydrate(g.pop("photo_ids"))} for g in (dict(g) for g in mood_groups)],
        "best": [] if loading else hydrate(quality.best(18)),
        "recent": hydrate([r["id"] for r in db.q(
            "SELECT id FROM photos WHERE status='ready' ORDER BY id DESC LIMIT 12")]),
        "people": list_people()[:12],
        "pets": list_people(kind="pet")[:8],
        "memories": [] if loading else [_memory_card(m) for m in events.compute()[:10]],
        "tidy": duplicates.summary(),
        "places": db.one("SELECT COUNT(*) n FROM photos WHERE status='ready' AND lat IS NOT NULL")["n"],
    }


# ---------------------------------------------------------------- collections
class CollectionIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    emoji: str | None = None
    color: str | None = None
    description: str | None = None
    kind: str = "manual"
    query: str | None = None
    photo_ids: list[int] = Field(default_factory=list)


class CollectionPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    emoji: str | None = None
    color: str | None = None
    description: str | None = None
    query: str | None = None
    cover_photo_id: int | None = None


@app.get("/api/collections")
def list_collections():
    return [collection_dict(c) for c in db.q("SELECT * FROM collections ORDER BY updated_at DESC")]


@app.post("/api/collections")
def create_collection(body: CollectionIn):
    if body.kind not in ("manual", "smart"):
        raise HTTPException(400, "kind must be manual or smart")
    if body.kind == "smart" and not (body.query or "").strip():
        raise HTTPException(400, "A smart collection needs a description to search for")
    name = body.name.strip()
    if db.one("SELECT id FROM collections WHERE lower(name)=lower(?)", (name,)):
        raise HTTPException(409, f"You already have a collection called “{name}”.")
    t = db.now()
    with db.tx() as c:
        cid = c.execute(
            "INSERT INTO collections(name, emoji, color, description, kind, query, created_at, updated_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (name, body.emoji, body.color, body.description, body.kind, body.query, t, t),
        ).lastrowid
        if body.kind == "manual":
            c.executemany(
                "INSERT OR IGNORE INTO collection_items(collection_id, photo_id, added_at) VALUES(?,?,?)",
                [(cid, p, t) for p in body.photo_ids],
            )
    return collection_dict(collection_or_404(cid))


@app.get("/api/collections/{cid}")
def get_collection(cid: int):
    c = collection_or_404(cid)
    d = collection_dict(c, with_preview=False)
    d["items"] = hydrate(collection_photo_ids(c))
    return d


@app.patch("/api/collections/{cid}")
def update_collection(cid: int, body: CollectionPatch):
    collection_or_404(cid)
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields:
        fields["name"] = fields["name"].strip()
        if db.one("SELECT id FROM collections WHERE lower(name)=lower(?) AND id<>?", (fields["name"], cid)):
            raise HTTPException(409, f"You already have a collection called “{fields['name']}”.")
    if fields:
        fields["updated_at"] = db.now()
        with db.tx() as c:
            c.execute(f"UPDATE collections SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                      [*fields.values(), cid])
    return collection_dict(collection_or_404(cid))


@app.delete("/api/collections/{cid}")
def delete_collection(cid: int):
    """Deletes the collection only. The photos themselves are never touched."""
    with db.tx() as c:
        c.execute("DELETE FROM collections WHERE id=?", (cid,))
    return {"ok": True}


@app.post("/api/collections/{cid}/photos")
def add_to_collection(cid: int, body: IdsIn):
    c = collection_or_404(cid)
    if c["kind"] == "smart":
        raise HTTPException(400, "Smart collections fill themselves from their description")
    t = db.now()
    with db.tx() as conn:
        before = conn.total_changes
        conn.executemany(
            "INSERT OR IGNORE INTO collection_items(collection_id, photo_id, added_at) "
            "SELECT ?, id, ? FROM photos WHERE id=?",
            [(cid, t, p) for p in body.photo_ids],
        )
        added = conn.total_changes - before
        conn.execute("UPDATE collections SET updated_at=? WHERE id=?", (t, cid))
    return {**collection_dict(collection_or_404(cid)), "added": added}


@app.post("/api/collections/{cid}/photos/remove")
def remove_from_collection(cid: int, body: IdsIn):
    collection_or_404(cid)
    with db.tx() as c:
        c.executemany("DELETE FROM collection_items WHERE collection_id=? AND photo_id=?",
                      [(cid, p) for p in body.photo_ids])
        c.execute("UPDATE collections SET updated_at=? WHERE id=?", (db.now(), cid))
    return collection_dict(collection_or_404(cid))


@app.post("/api/collections/{cid}/freeze")
def freeze_collection(cid: int):
    """Turn a smart collection into a regular one holding its current photos."""
    c = collection_or_404(cid)
    ids = collection_photo_ids(c)
    t = db.now()
    with db.tx() as conn:
        conn.execute("UPDATE collections SET kind='manual', updated_at=? WHERE id=?", (t, cid))
        conn.executemany("INSERT OR IGNORE INTO collection_items VALUES(?,?,?)", [(cid, p, t) for p in ids])
    return collection_dict(collection_or_404(cid))


@app.get("/api/collections/{cid}/export")
def export_collection(cid: int, background: BackgroundTasks):
    c = collection_or_404(cid)
    paths = [r["path"] for r in db.q(
        f"SELECT path FROM photos WHERE id IN ({','.join('?' * len(ids))})", ids
    )] if (ids := collection_photo_ids(c)) else []
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    used = set()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:  # photos are already compressed
        for p in paths:
            if not os.path.exists(p):
                continue
            name, n = os.path.basename(p), 1
            while name.lower() in used:
                stem, ext = os.path.splitext(os.path.basename(p))
                name, n = f"{stem} ({n}){ext}", n + 1
            used.add(name.lower())
            z.write(p, name)
    tmp.close()
    background.add_task(os.unlink, tmp.name)
    safe = "".join(ch for ch in c["name"] if ch.isalnum() or ch in " -_").strip() or "collection"
    return FileResponse(tmp.name, media_type="application/zip", filename=f"{safe}.zip")


# ---------------------------------------------------------------- people
class PersonPatch(BaseModel):
    name: str | None = Field(default=None, max_length=80)
    hidden: bool | None = None
    cover_face_id: int | None = None


class MergeIn(BaseModel):
    into: int
    person_ids: list[int]


def person_or_404(pid: int):
    row = db.one("SELECT * FROM persons WHERE id=?", (pid,))
    if not row:
        raise HTTPException(404, "Person not found")
    return row


def person_dict(p, count=None):
    if count is None:
        count = db.one("SELECT COUNT(DISTINCT photo_id) n FROM faces WHERE person_id=?", (p["id"],))["n"]
    species = None
    if p["kind"] == "pet":
        row = db.one("SELECT species, COUNT(*) n FROM faces WHERE person_id=? GROUP BY species ORDER BY n DESC",
                     (p["id"],))
        species = row["species"] if row else None
    return {"id": p["id"], "name": p["name"], "hidden": bool(p["hidden"]), "count": count,
            "face_id": faces.cover_face(p["id"]), "kind": p["kind"], "species": species}


@app.get("/api/people")
def list_people(hidden: bool = False, kind: str = "person"):
    rows = db.q(
        "SELECT p.*, COUNT(DISTINCT f.photo_id) n FROM persons p JOIN faces f ON f.person_id=p.id "
        "WHERE p.hidden=? AND p.kind=? GROUP BY p.id ORDER BY (p.name IS NULL OR p.name=''), n DESC",
        (int(hidden), kind),
    )
    return [person_dict(r, r["n"]) for r in rows]


@app.get("/api/people/{pid}")
def get_person(pid: int):
    d = person_dict(person_or_404(pid))
    d["items"] = hydrate(faces.person_photo_ids(pid))
    return d


@app.patch("/api/people/{pid}")
def update_person(pid: int, body: PersonPatch):
    person_or_404(pid)
    fields = body.model_dump(exclude_unset=True)
    if "name" in fields:
        fields["name"] = (fields["name"] or "").strip() or None
        if fields["name"] and db.one("SELECT id FROM persons WHERE lower(name)=lower(?) AND id<>?",
                                     (fields["name"], pid)):
            raise HTTPException(409, f"Someone is already called “{fields['name']}”. Merge them instead?")
    if "hidden" in fields:
        fields["hidden"] = int(fields["hidden"])
    if fields:
        with db.tx() as c:
            c.execute(f"UPDATE persons SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                      [*fields.values(), pid])
    return person_dict(person_or_404(pid))


@app.post("/api/people/merge")
def merge_people(body: MergeIn):
    """Fold several people into one: 'these are all Maria'."""
    target = person_or_404(body.into)
    # People merge with people, pets with pets.
    others = [p["id"] for p in db.q(
        f"SELECT id FROM persons WHERE kind=? AND id IN ({','.join('?' * len(body.person_ids))})",
        [target["kind"], *body.person_ids]) if p["id"] != body.into] if body.person_ids else []
    with db.tx() as c:
        name = target["name"]
        for p in others:
            row = c.execute("SELECT name FROM persons WHERE id=?", (p,)).fetchone()
            name = name or (row["name"] if row else None)
            c.execute("UPDATE faces SET person_id=? WHERE person_id=?", (body.into, p))
            c.execute("DELETE FROM persons WHERE id=?", (p,))
        c.execute("UPDATE persons SET name=? WHERE id=?", (name, body.into))
    return person_dict(person_or_404(body.into))


@app.post("/api/people/{pid}/remove-photos")
def remove_person_photos(pid: int, body: IdsIn):
    """'That's not them': detach this person from the given photos, and remember it."""
    person_or_404(pid)
    with db.tx() as c:
        c.executemany("UPDATE faces SET person_id=NULL, locked=1 WHERE person_id=? AND photo_id=?",
                      [(pid, ph) for ph in body.photo_ids])
        left = c.execute("SELECT COUNT(*) n FROM faces WHERE person_id=?", (pid,)).fetchone()["n"]
        if not left:
            c.execute("DELETE FROM persons WHERE id=?", (pid,))
    return {"remaining": left}


@app.get("/api/faces/{face_id}/thumb")
def face_thumb(face_id: int):
    p = faces.face_thumb_path(face_id)
    if not p.exists():
        raise HTTPException(404, "No thumbnail")
    return FileResponse(p, media_type="image/webp", headers={"Cache-Control": "public, max-age=604800"})


# ---------------------------------------------------------------- places
@app.get("/api/map")
def map_points():
    rows = db.q("SELECT id, lat, lon FROM photos WHERE status='ready' AND lat IS NOT NULL ORDER BY taken_at DESC")
    return {"points": [[r["id"], round(r["lat"], 6), round(r["lon"], 6)] for r in rows]}


@app.post("/api/photos/batch")
def photos_batch(body: IdsIn):
    """Grid data for a list of ids (e.g. the photos inside the map's current view)."""
    return hydrate(body.photo_ids[:1000])


@app.get("/api/places")
def list_places():
    return places.top_places(40)


# ---------------------------------------------------------------- memories (trips & events)
def _memory_card(m):
    return {k: v for k, v in m.items() if k != "photo_ids"} | {"covers": hydrate(m["cover_ids"])}


@app.get("/api/memories")
def list_memories(offset: int = 0, limit: int = Query(40, le=200), kind: str | None = None):
    ms = [m for m in events.compute() if not kind or m["kind"] == kind]
    return {"total": len(ms), "items": [_memory_card(m) for m in ms[offset : offset + limit]]}


@app.get("/api/memories/{key}")
def get_memory(key: str):
    m = events.get(key)
    if not m:
        raise HTTPException(404, "That memory no longer exists (the library changed)")
    return _memory_card(m) | {"items": hydrate(m["photo_ids"])}


@app.post("/api/memories/{key}/save")
def save_memory(key: str):
    """Keep a memory as a regular collection."""
    m = events.get(key)
    if not m:
        raise HTTPException(404, "That memory no longer exists")
    name, n = m["title"], 2
    while db.one("SELECT id FROM collections WHERE lower(name)=lower(?)", (name,)):
        name, n = f"{m['title']} ({n})", n + 1
    emoji = "🧳" if m["kind"] == "trip" else "✨"
    return create_collection(CollectionIn(name=name, emoji=emoji, description=m["when"], photo_ids=m["photo_ids"]))


# ---------------------------------------------------------------- tidy up
@app.get("/api/duplicates")
def get_duplicates(kind: str = "duplicates"):
    if kind not in ("duplicates", "bursts"):
        raise HTTPException(400, "kind must be duplicates or bursts")
    groups = duplicates.find(kind)
    return {"groups": [{**g, "photos": hydrate(g["photo_ids"])} for g in groups[:300]],
            "total_groups": len(groups), "summary": duplicates.summary()}


class KeyIn(BaseModel):
    key: str


@app.post("/api/duplicates/ignore")
def ignore_duplicates(body: KeyIn):
    duplicates.ignore(body.key)
    return {"ok": True}


@app.post("/api/photos/trash")
def trash_photos(body: IdsIn):
    """Move originals to the system Trash / Recycle Bin (recoverable) and forget them."""
    try:
        from send2trash import send2trash
    except ImportError:
        raise HTTPException(501, "Moving to the Trash needs the send2trash package: pip install send2trash")
    trashed, failed = [], []
    for pid in body.photo_ids:
        row = db.one("SELECT path FROM photos WHERE id=?", (pid,))
        if not row:
            continue
        try:
            if os.path.exists(row["path"]):
                send2trash(row["path"])
            trashed.append(pid)
        except OSError as e:
            failed.append({"id": pid, "error": str(e)})
    delete_photos(trashed)
    return {"trashed": len(trashed), "failed": failed}


# ---------------------------------------------------------------- copy to folder & jobs
class CopyIn(BaseModel):
    dest: str
    subfolder: str | None = None


@app.post("/api/collections/{cid}/copy")
def copy_collection(cid: int, body: CopyIn):
    c = collection_or_404(cid)
    dest = _norm(body.dest)
    if not os.path.isdir(dest):
        raise HTTPException(400, "That destination folder doesn't exist.")
    if body.subfolder and body.subfolder.strip():
        safe = "".join(ch for ch in body.subfolder.strip() if ch not in '<>:"/\\|?*').strip(" .")
        dest = os.path.join(dest, safe or "Lumen collection")
    ids = collection_photo_ids(c)
    rows = {r["id"]: r["path"] for r in db.q(
        f"SELECT id, path FROM photos WHERE id IN ({','.join('?' * len(ids))})", ids)} if ids else {}
    paths = [rows[i] for i in ids if i in rows]
    return {"job_id": jobs.copy_files(paths, dest), "dest": dest, "total": len(paths)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: int):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Unknown job")
    return j


# ---------------------------------------------------------------- UI
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
