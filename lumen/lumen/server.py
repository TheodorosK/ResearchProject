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

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel, Field

from . import config, db, embedder, search
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
            f"SELECT id, path, width, height, taken_at, camera, folder_id FROM photos "
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
        d["preview_ids"] = (cover + [i for i in ids if i not in cover])[:4]
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
    }


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
    if os.path.splitext(path)[1].lower() in BROWSER_FRIENDLY:
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
@app.get("/api/search")
def do_search(
    q: str, limit: int = Query(300, le=1000), folder_id: int | None = None,
    year_from: int | None = None, year_to: int | None = None, collection_id: int | None = None,
):
    require_model()
    hits = search.text_search(q, limit, filters_from(folder_id, year_from, year_to, collection_id))
    return {"query": q, "parsed": dict(zip(("positive", "negatives"), search.parse_query(q))),
            "items": hydrate(hits), "indexed": len(index)}


@app.get("/api/home")
def home():
    groups = [] if embedder.get().loading else search.explore()
    return {
        "suggestions": search.suggestions(),
        "on_this_day": hydrate(search.on_this_day()),
        "explore": [{**g, "photos": hydrate(g.pop("photo_ids"))} for g in (dict(g) for g in groups)],
        "recent": hydrate([r["id"] for r in db.q(
            "SELECT id FROM photos WHERE status='ready' ORDER BY id DESC LIMIT 12")]),
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


# ---------------------------------------------------------------- UI
app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
