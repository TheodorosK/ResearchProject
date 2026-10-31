"""Sharing a collection: a self-contained web gallery (a zip to send or upload anywhere), or a
slideshow video (MP4, H.264) with gentle pan-and-zoom and crossfades."""
import html
import io
import json
import os
import tempfile
import zipfile
from datetime import datetime

import numpy as np
from PIL import Image, ImageOps

from . import db, jobs

GALLERY_FULL = 2048
GALLERY_THUMB = 640
VIDEO_SIZE = (1920, 1080)
VIDEO_FPS = 30
SECONDS_PER_PHOTO = 4.0
FADE_SECONDS = 1.0
MAX_VIDEO_PHOTOS = 120


def _open(path):
    with Image.open(path) as im:
        im.draft("RGB", (GALLERY_FULL, GALLERY_FULL))
        return ImageOps.exif_transpose(im).convert("RGB")


def _rows(photo_ids):
    rows = {r["id"]: r for r in db.q(
        f"SELECT id, path, taken_at, kind FROM photos WHERE id IN ({','.join('?' * len(photo_ids))})", photo_ids)}
    return [rows[i] for i in photo_ids if i in rows and rows[i]["kind"] == "photo" and os.path.exists(rows[i]["path"])]


GALLERY_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
:root{{--bg:#f6f4ef;--fg:#1c1b22;--muted:#6b6878}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0d0e13;--fg:#eef0f6;--muted:#9a9fb3}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{padding:48px 24px 24px;max-width:1500px;margin:0 auto}}h1{{margin:0;font-size:clamp(30px,5vw,54px);letter-spacing:-.03em}}
header p{{margin:6px 0 0;color:var(--muted)}}
.grid{{columns:4 280px;column-gap:8px;padding:0 24px 48px;max-width:1500px;margin:0 auto}}
.grid img{{width:100%;display:block;margin:0 0 8px;border-radius:10px;cursor:zoom-in;break-inside:avoid;transition:opacity .2s}}
.grid img:hover{{opacity:.85}}
#lb{{position:fixed;inset:0;background:rgba(0,0,0,.94);display:none;place-items:center;z-index:9}}
#lb.on{{display:grid}}#lb img{{max-width:96vw;max-height:92vh;border-radius:6px}}
#lb button{{position:fixed;top:50%;transform:translateY(-50%);background:rgba(255,255,255,.1);color:#fff;border:0;width:52px;height:52px;border-radius:50%;font-size:26px;cursor:pointer}}
#prev{{left:12px}}#next{{right:12px}}#close{{top:28px!important;right:12px}}
footer{{text-align:center;color:var(--muted);font-size:13px;padding:0 0 32px}}
</style></head><body>
<header><h1>{title}</h1><p>{subtitle}</p></header>
<main class="grid">{thumbs}</main>
<div id="lb"><img alt=""><button id="prev" aria-label="Previous">‹</button><button id="next" aria-label="Next">›</button><button id="close" aria-label="Close">×</button></div>
<footer>Made with Lumen</footer>
<script>
const photos={photos};let i=0;const lb=document.getElementById("lb"),img=lb.querySelector("img");
const show=k=>{{i=(k+photos.length)%photos.length;img.src=photos[i];lb.classList.add("on")}};
document.querySelectorAll(".grid img").forEach((t,k)=>t.onclick=()=>show(k));
document.getElementById("prev").onclick=()=>show(i-1);document.getElementById("next").onclick=()=>show(i+1);
document.getElementById("close").onclick=()=>lb.classList.remove("on");
lb.onclick=e=>{{if(e.target===lb)lb.classList.remove("on")}};
document.onkeydown=e=>{{if(!lb.classList.contains("on"))return;if(e.key==="Escape")lb.classList.remove("on");if(e.key==="ArrowLeft")show(i-1);if(e.key==="ArrowRight")show(i+1)}};
</script></body></html>"""


def _subtitle(rows, description):
    times = [r["taken_at"] for r in rows if r["taken_at"]]
    parts = []
    if description:
        parts.append(description)
    if times:
        a, b = datetime.fromtimestamp(min(times)), datetime.fromtimestamp(max(times))
        parts.append(a.strftime("%B %Y") if (a.year, a.month) == (b.year, b.month)
                     else f"{a.strftime('%b %Y')} – {b.strftime('%b %Y')}")
    parts.append(f"{len(rows)} photos")
    return " · ".join(parts)


def gallery(photo_ids, title, description=None):
    """Start building a gallery zip. The job's result holds the zip's path."""
    rows = _rows(photo_ids)
    out = tempfile.NamedTemporaryFile(suffix=".zip", delete=False, prefix="lumen-gallery-")
    out.close()

    def work(progress):
        thumbs, fulls = [], []
        with zipfile.ZipFile(out.name, "w", zipfile.ZIP_STORED) as z:
            for k, r in enumerate(rows, 1):
                im = _open(r["path"])
                for size, folder, quality in ((GALLERY_FULL, "photos", 88), (GALLERY_THUMB, "thumbs", 82)):
                    copy = im.copy()
                    copy.thumbnail((size, size), Image.Resampling.LANCZOS)
                    buf = io.BytesIO()
                    copy.save(buf, "JPEG", quality=quality, optimize=True, progressive=True)
                    z.writestr(f"{folder}/{k:04d}.jpg", buf.getvalue())
                thumbs.append(f'<img loading="lazy" src="thumbs/{k:04d}.jpg" alt="">')
                fulls.append(f"photos/{k:04d}.jpg")
                progress(k)
            z.writestr("index.html", GALLERY_HTML.format(
                title=html.escape(title), subtitle=html.escape(_subtitle(rows, description)),
                thumbs="\n".join(thumbs), photos=json.dumps(fulls)))
        return {"path": out.name, "count": len(rows), "filename": _safe(title) + " (gallery).zip"}

    return jobs.start("gallery", len(rows), work)


def _safe(name):
    return "".join(ch for ch in name if ch.isalnum() or ch in " -_").strip() or "Lumen"


def _fit(im, size, zoom, ox, oy):
    """Cover-crop `im` to `size`, zoomed in by `zoom` towards (ox, oy) in 0..1."""
    W, H = size
    scale = max(W / im.width, H / im.height) * zoom
    cw, ch = W / scale, H / scale
    x = (im.width - cw) * ox
    y = (im.height - ch) * oy
    return im.resize(size, Image.Resampling.BILINEAR, box=(x, y, x + cw, y + ch))


def slideshow_video(photo_ids, title):
    """Start rendering an MP4 slideshow. The job's result holds the video's path."""
    import av

    rows = _rows(photo_ids)[:MAX_VIDEO_PHOTOS]
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, prefix="lumen-slideshow-")
    out.close()
    per = int(SECONDS_PER_PHOTO * VIDEO_FPS)
    fade = int(FADE_SECONDS * VIDEO_FPS)
    rng = np.random.default_rng(7)

    def work(progress):
        with av.open(out.name, "w") as c:
            stream = c.add_stream("libx264", rate=VIDEO_FPS)
            stream.width, stream.height = VIDEO_SIZE
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "20", "preset": "veryfast"}

            def emit(arr):
                for p in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                    c.mux(p)

            # Title card
            card = Image.new("RGB", VIDEO_SIZE, (12, 12, 16))
            _draw_title(card, title, _subtitle(rows, None))
            for _ in range(int(2.5 * VIDEO_FPS)):
                emit(np.asarray(card))
            # Frames are rendered on the fly (a buffered 4 s clip would be gigabytes).
            def frame(shot, f):
                im, ox, oy = shot
                return np.asarray(_fit(im, VIDEO_SIZE, 1.0 + 0.10 * f / per, ox, oy))  # slow push in

            prev = None
            for k, r in enumerate(rows, 1):
                cur = (_open(r["path"]), *rng.uniform(0.2, 0.8, 2))
                start_f = 0
                if prev is not None:
                    for f in range(fade):  # crossfade from the previous photo's last moments
                        a = (f + 1) / (fade + 1)
                        mixed = frame(prev, per - fade + f) * (1 - a) + frame(cur, f) * a
                        emit(mixed.astype(np.uint8))
                    start_f = fade
                for f in range(start_f, per - fade):
                    emit(frame(cur, f))
                prev = cur
                progress(k)
            if prev is not None:
                for f in range(per - fade, per):
                    emit(frame(prev, f))
            for p in stream.encode():
                c.mux(p)
        return {"path": out.name, "count": len(rows), "filename": _safe(title) + ".mp4"}

    return jobs.start("video", len(rows), work)


def _draw_title(card, title, subtitle):
    from PIL import ImageDraw, ImageFont

    d = ImageDraw.Draw(card)

    def font(size):
        for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf", "Helvetica.ttc"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default(size)

    W, H = card.size
    for text, f, y, fill in ((title, font(96), H / 2 - 40, (240, 240, 245)), (subtitle, font(40), H / 2 + 60, (160, 160, 175))):
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y - f.size / 2), text, font=f, fill=fill)
