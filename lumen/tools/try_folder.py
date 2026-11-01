"""Try Lumen's models on your own photos and see the results side by side.

    python tools/try_folder.py --folder "~/Pictures/for test" \
        --query "a dog on the beach" --query "ένας σκύλος" --query "moody dark street"

Writes report.html (open it in a browser) with the top results of each model for each query,
and prints how fast each model embedded your photos. Embeddings are cached in
~/.lumen-compare, so re-running with new queries is instant. Your photos are only read.
"""
import argparse
import base64
import hashlib
import io
import os
import sys
import time
from html import escape
from pathlib import Path

HOME = Path(os.environ.get("LUMEN_HOME") or "~/.lumen-compare").expanduser()
os.environ["LUMEN_HOME"] = str(HOME)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from lumen import config, db, embedder  # noqa: E402

EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".avif"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--query", action="append", default=[], help="repeat for several queries")
    ap.add_argument("--queries-file", help="one query per line")
    ap.add_argument("--models", default="siglip2,siglip2-large", help="comma-separated keys from config.MODELS")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", default="report.html")
    a = ap.parse_args()

    queries = list(a.query)
    if a.queries_file:
        queries += [q.strip() for q in Path(a.queries_file).read_text().splitlines() if q.strip()]
    if not queries:
        sys.exit("Give at least one --query")
    folder = Path(a.folder).expanduser()
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXT and not p.name.startswith("."))
    if not files:
        sys.exit(f"No photos found in {folder}")
    print(f"{len(files)} photos in {folder}")

    HOME.mkdir(parents=True, exist_ok=True)
    # Reuse models Lumen already downloaded, so the base model isn't fetched twice.
    real, link = Path("~/.lumen/models").expanduser(), HOME / "models"
    if real.exists() and not link.exists():
        link.symlink_to(real, target_is_directory=True)
    db.init()

    def thumb(p, size=170):
        with Image.open(p) as im:
            im = im.convert("RGB")
            im.thumbnail((size, size))
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=70)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    key = hashlib.sha1("\n".join(f"{p}:{p.stat().st_mtime_ns}" for p in files).encode()).hexdigest()[:12]
    results, speeds = {}, {}
    for model in a.models.split(","):
        emb = embedder.FastEmbedder(model)
        emb._load()
        cache = HOME / f"folder-{key}-{model}.npy"
        if cache.exists():
            vecs = np.load(cache)
        else:
            t0, out = time.time(), []
            for i in range(0, len(files), 16):
                batch = []
                for p in files[i:i + 16]:
                    with Image.open(p) as im:
                        im.draft("RGB", (768, 768))     # fast JPEG decode for big originals
                        batch.append(im.convert("RGB"))
                out.append(emb.embed_images(batch))
                print(f"  {model}: {i + len(batch)}/{len(files)}", end="\r")
            vecs = np.concatenate(out)
            speeds[model] = len(files) / (time.time() - t0)
            np.save(cache, vecs)
            print()
        results[model] = (emb, vecs)
        print(f"{model}: {emb.device}" + (f", {speeds[model]:.1f} photos/s" if model in speeds else ", embeddings cached"))

    thumbs = {p: thumb(p) for p in files}
    parts = ["<meta charset=utf-8><title>Lumen model comparison</title><style>body{font:14px system-ui;margin:20px}"
             "h2{margin:28px 0 4px}h3{margin:10px 0 4px;font-weight:600}.row{display:flex;gap:6px;flex-wrap:wrap}"
             "figure{margin:0;width:170px}img{display:block;border-radius:4px}figcaption{font-size:11px;color:#666;"
             "overflow:hidden;white-space:nowrap;text-overflow:ellipsis}</style><h1>Lumen model comparison</h1>"]
    for q in queries:
        parts.append(f"<h2>{escape(q)}</h2>")
        for model, (emb, vecs) in results.items():
            qv = embedder._normalize(emb.embed_texts([t.format(q) for t in embedder.TEMPLATES]
                                                      if len(q.split()) <= 6 else [q]).mean(0))
            s = vecs @ qv
            parts.append(f"<h3>{escape(config.MODELS[model]['label'])}</h3><div class=row>")
            for i in np.argsort(-s)[:a.top]:
                parts.append(f"<figure><img src='{thumbs[files[i]]}'><figcaption>{s[i]:.3f} · "
                             f"{escape(files[i].name)}</figcaption></figure>")
            parts.append("</div>")
    Path(a.out).write_text("".join(parts))
    print(f"Wrote {a.out}. Open it in a browser.")


if __name__ == "__main__":
    main()
