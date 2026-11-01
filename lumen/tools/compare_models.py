"""Compare Lumen's understanding models on captioned photos (text -> photo retrieval).

Needs a folder of images and a CSV with `filename` and `raw` (JSON list of captions) columns,
e.g. the Flickr 1k test set (nlphuji/flickr_1k_test_image_text_retrieval on Hugging Face).

    python tools/compare_models.py --images DIR --csv FILE --model siglip2-large [--limit 500]
        [--vision-file onnx/vision_model.onnx] [--text-file onnx/text_model.onnx]

Embeddings are cached under --cache so a model is only run over the photos once.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("LUMEN_HOME", "/tmp/lumen-compare")

from PIL import Image  # noqa: E402

from lumen import config, db, embedder  # noqa: E402


def load_pairs(images: Path, csv_path: Path, limit: int):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            path = next(images.rglob(r["filename"]), None)
            if path and not path.name.startswith("._"):
                rows.append((path, json.loads(r["raw"])))
            if len(rows) >= limit:
                break
    return rows


def recall(sim: np.ndarray, owner: np.ndarray, ks=(1, 5, 10)):
    """sim: (captions, images). owner[i] is the right image for caption i."""
    ranks = (sim > sim[np.arange(len(owner)), owner][:, None]).sum(1)
    return {f"R@{k}": float((ranks < k).mean() * 100) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--vision-file")
    ap.add_argument("--text-file")
    ap.add_argument("--vision-extra", nargs="*", default=[])
    ap.add_argument("--text-extra", nargs="*", default=[])
    ap.add_argument("--cache", type=Path, default=Path("/tmp/lumen-compare/cache"))
    a = ap.parse_args()

    db.init()
    spec = dict(config.MODELS[a.model])
    tag = a.model
    if a.vision_file:
        spec["vision_file"] = a.vision_file
        tag += "-v" + Path(a.vision_file).stem.replace("vision_model", "") or "-v"
    if a.text_file:
        spec["text_file"] = a.text_file
        tag += "-t" + Path(a.text_file).stem.replace("text_model", "") or "-t"
    if a.vision_file or a.text_file:
        # a distinct fastembed name so variants of one export don't collide
        spec["image"] = spec["text"] = f"{spec['image']}#{tag}"
    spec["vision_extra"], spec["text_extra"] = a.vision_extra, a.text_extra
    config.MODELS[tag] = spec
    pairs = load_pairs(a.images, a.csv, a.limit)
    print(f"{tag}: {len(pairs)} photos, {sum(len(c) for _, c in pairs)} captions")

    emb = embedder.FastEmbedder(tag)
    emb._load()
    print("device:", emb.device)
    a.cache.mkdir(parents=True, exist_ok=True)
    f_img = a.cache / f"{tag}-{len(pairs)}-img.npy"
    if f_img.exists():
        img = np.load(f_img)
        speed = None
    else:
        t0, out = time.time(), []
        for i in range(0, len(pairs), 16):
            ims = [Image.open(p).convert("RGB") for p, _ in pairs[i:i + 16]]
            out.append(emb.embed_images(ims))
            print(f"  {i + len(ims)}/{len(pairs)}  {(i + len(ims)) / (time.time() - t0):.2f} img/s", end="\r")
        img = np.concatenate(out)
        speed = len(pairs) / (time.time() - t0)
        np.save(f_img, img)
        print()
    caps = [c for _, cs in pairs for c in cs]
    owner = np.array([i for i, (_, cs) in enumerate(pairs) for _ in cs])
    t0 = time.time()
    txt = np.concatenate([emb.embed_texts(caps[i:i + 64]) for i in range(0, len(caps), 64)])
    t_text = (time.time() - t0) / len(caps) * 1000
    sim = txt @ img.T
    res = recall(sim, owner)
    # image -> best of its 5 captions
    i2t = sim.T
    first = np.array([np.where(owner == i)[0][0] for i in range(len(pairs))])
    best_rank = []
    for i in range(len(pairs)):
        order = np.argsort(-i2t[i])
        best_rank.append(int(np.where(owner[order] == i)[0][0]))
    res["img->txt R@1"] = float((np.array(best_rank) < 1).mean() * 100)
    res.update(tag=tag, photos=len(pairs), img_per_s=speed, ms_per_query=t_text)
    print(json.dumps(res, indent=1))
    (a.cache / f"{tag}-{len(pairs)}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
