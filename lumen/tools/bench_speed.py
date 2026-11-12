"""How fast do Lumen's models run on THIS computer?

    python tools/bench_speed.py                                  # both SigLIP 2 models, GPU and CPU
    python tools/bench_speed.py --folder "~/Pictures/for test"   # on your own photos
    python tools/bench_speed.py --closer-look                    # also time the vision-language model

Prints, for each model and for the accelerator (Core ML on Apple Silicon) and the plain CPU:
how long it takes to load, how many photos per second it indexes, and how long a search takes.
From that it estimates the time to index 10,000 photos. Nothing in your Lumen library is touched:
it works in ~/.lumen-compare and reuses models Lumen already downloaded.
"""
import argparse
import os
import sys
import time
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


def load_images(folder, n):
    if folder:
        files = sorted(p for p in Path(folder).expanduser().rglob("*")
                       if p.suffix.lower() in EXT and not p.name.startswith("."))[:n]
        if not files:
            sys.exit(f"No photos found in {folder}")
        out = []
        for p in files:
            with Image.open(p) as im:
                im.draft("RGB", (768, 768))
                im = im.convert("RGB")
                im.thumbnail((768, 768))
                out.append(im)
        return out, f"{len(out)} of your photos"
    rng = np.random.default_rng(0)
    return ([Image.fromarray(rng.integers(0, 255, (512, 640, 3), dtype=np.uint8)) for _ in range(n)],
            f"{n} generated test pictures (use --folder for your own photos)")


def duration(seconds):
    return f"{seconds / 3600:.1f} h" if seconds >= 5400 else f"{seconds / 60:.0f} min"


def bench(model, device, images):
    db.set_setting("device", device)
    emb = embedder.FastEmbedder(model)
    t0 = time.time()
    emb._load()
    load = time.time() - t0
    bs = emb.batch_size
    t0 = time.time()
    emb.embed_images(images[:bs])                      # first batch: Core ML compiles the model here
    warm = time.time() - t0
    t0 = time.time()
    for i in range(0, len(images), bs):
        emb.embed_images(images[i:i + bs])
    rate = len(images) / (time.time() - t0)
    emb.embed_texts(["warm up"])
    t0 = time.time()
    for q in ["a dog on the beach", "golden hour", "ένας σκύλος στην παραλία", "moody dark street at night"]:
        emb.embed_texts([t.format(q) for t in embedder.TEMPLATES])
    query_ms = (time.time() - t0) / 4 * 1000
    return {"device": emb.device, "load": load, "warm": warm, "rate": rate, "query_ms": query_ms}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folder", help="time it on photos from this folder instead of generated pictures")
    ap.add_argument("--photos", type=int, default=64, help="how many photos to time (default 64)")
    ap.add_argument("--models", default="siglip2,siglip2-large")
    ap.add_argument("--devices", default="auto,cpu", help="auto = GPU/Neural Engine if there is one")
    ap.add_argument("--closer-look", action="store_true", help="also time the vision-language model")
    ap.add_argument("--url", default=None, help="model server for --closer-look (default: Ollama on this computer)")
    ap.add_argument("--vlm", default=None, help="model for --closer-look (default qwen2.5vl:7b)")
    a = ap.parse_args()

    HOME.mkdir(parents=True, exist_ok=True)
    real, link = Path("~/.lumen/models").expanduser(), HOME / "models"
    if real.exists() and not link.exists():
        link.symlink_to(real, target_is_directory=True)
    db.init()
    images, what = load_images(a.folder, a.photos)
    import platform
    print(f"{platform.platform()} · {platform.machine()} · {os.cpu_count()} cores")
    print(f"Timing on {what}\n")

    rows = []
    for model in a.models.split(","):
        seen = set()
        for device in a.devices.split(","):
            try:
                r = bench(model, device, images)
            except Exception as e:
                print(f"{model} ({device}): failed: {type(e).__name__}: {e}")
                continue
            if r["device"] in seen:        # no accelerator here: "auto" and "cpu" are the same thing
                continue
            seen.add(r["device"])
            rows.append((model, r))
            print(f"{config.MODELS[model]['label']}\n  on {r['device']}: {r['rate']:.1f} photos/s"
                  f" · 10,000 photos in about {duration(10000 / r['rate'])}"
                  f" · a search takes {r['query_ms']:.0f} ms"
                  f" · loads in {r['load']:.0f} s, first batch {r['warm']:.0f} s\n")
    db.set_setting("device", "auto")
    if len(rows) > 1:
        best = {}
        for model, r in rows:
            best[model] = max(best.get(model, 0), r["rate"])
        names = list(best)
        if len(names) == 2 and best[names[1]]:
            print(f"At their fastest here, {config.MODELS[names[1]]['label'].split(' ·')[0]} is "
                  f"{best[names[0]] / best[names[1]]:.1f}x slower to index than "
                  f"{config.MODELS[names[0]]['label'].split(' ·')[0]}.\n")

    if a.closer_look:
        from lumen import rerank
        cfg = {"url": (a.url or rerank.DEFAULT_URL).rstrip("/"), "model": a.vlm or rerank.DEFAULT_MODEL}
        print(f"Closer look: {cfg['model']} at {cfg['url']}")
        try:
            t0 = time.time()
            rerank.score_image(images[0], "a dog on the beach", cfg)        # loads the model
            first = time.time() - t0
            t0 = time.time()
            answers = [rerank.score_image(im, "a dog on the beach", cfg) for im in images[1:6]]
            each = (time.time() - t0) / len(answers)
            print(f"  first photo {first:.0f} s (loading the model), then {each:.1f} s per photo"
                  f" · looking at the top {rerank.DEFAULT_TOP} of a search takes about {each * rerank.DEFAULT_TOP:.0f} s")
            if any(x is None for x in answers):
                print("  Warning: some answers weren't a 0-10 number; this model may not suit the closer look.")
        except Exception as e:
            print(f"  failed: {type(e).__name__}: {e}\n  Is the model server running, and the model installed "
                  f"(with Ollama: ollama pull {cfg['model']})?")


if __name__ == "__main__":
    main()
