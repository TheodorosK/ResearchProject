"""A closer look at the top results.

The image model finds candidates in milliseconds but reads a photo as one overall impression, so
"a dog on the beach" can rank any dog first. A vision-language model (Qwen2.5-VL by default) looks
at each of the top few photos and rates how well it fits the words. It is slow (seconds per photo),
so it only ever sees the top results of a search you ask it to refine, and every rating is kept so
the same search is instant next time.

The model runs in a separate local server that speaks the OpenAI chat API: Ollama, LM Studio or
llama.cpp's llama-server. Nothing leaves this computer unless you point the address elsewhere.
"""
import base64
import io
import re

import requests
from PIL import Image

from . import db
from .indexer import preview_path

DEFAULT_URL = "http://127.0.0.1:11434/v1"      # Ollama's default
DEFAULT_MODEL = "qwen2.5vl:7b"
DEFAULT_TOP = 12
MAX_TOP = 40
IMAGE_SIDE = 448

SYSTEM = "You judge how well a photo matches a search. Reply with only a whole number from 0 to 10."
PROMPT = ('Search: "{q}"\nHow well does this photo match the search? Every part of it counts: the subject, '
          "the place, the action, the light and mood. 0 means not at all, 10 means perfectly. "
          "Reply with only the number.")


def settings():
    top = db.get_setting("rerank_top")
    return {
        "enabled": db.get_setting("rerank") == "1",
        "url": (db.get_setting("rerank_url") or DEFAULT_URL).rstrip("/"),
        "model": db.get_setting("rerank_model") or DEFAULT_MODEL,
        "top": max(1, min(MAX_TOP, int(top))) if top and top.isdigit() else DEFAULT_TOP,
    }


def status():
    """Settings plus whether the model server answers and has the model."""
    s = settings()
    s.update(available=False, detail="Switched off")
    if not s["enabled"]:
        return s
    try:
        r = requests.get(s["url"] + "/models", timeout=3)
        r.raise_for_status()
        names = {m.get("id") for m in r.json().get("data", [])}
    except Exception as e:
        s["detail"] = f"No model server is answering at {s['url']} ({type(e).__name__}). Is Ollama running?"
        return s
    if s["model"] not in names and s["model"] + ":latest" not in names:
        s["detail"] = f"The server is running but doesn't have “{s['model']}”. With Ollama: ollama pull {s['model']}"
        return s
    s.update(available=True, detail="Ready")
    return s


def _data_url(im: Image.Image) -> str:
    im = im.convert("RGB")
    im.thumbnail((IMAGE_SIDE, IMAGE_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def parse_score(text: str):
    """The model's 0-10 rating, or None if it didn't give one ('7', '7/10', 'Score: 7.')."""
    m = re.search(r"(?<![\d.])(\d{1,2})(?:\.\d+)?", text or "")
    if not m:
        return None
    v = float(m.group(0))
    return v if 0 <= v <= 10 else None


def score_photo(photo_id: int, query: str, cfg) -> float | None:
    with Image.open(preview_path(photo_id)) as im:
        return score_image(im, query, cfg)


def score_image(im: Image.Image, query: str, cfg) -> float | None:
    """Ask the model to rate one picture against the search. cfg needs "url" and "model"."""
    body = {
        "model": cfg["model"], "temperature": 0, "max_tokens": 8,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _data_url(im)}},
                {"type": "text", "text": PROMPT.format(q=query.replace('"', "'"))},
            ]},
        ],
    }
    r = requests.post(cfg["url"] + "/chat/completions", json=body, timeout=180)
    r.raise_for_status()
    return parse_score(r.json()["choices"][0]["message"]["content"])


def query_text(raw: str) -> str:
    """What the model is asked about: the descriptive part of the search, without -exclusions."""
    from . import search

    return search.understand(raw)["positive"] or " ".join(raw.split())


def _key(query: str) -> str:
    return " ".join(query.lower().split())


def cached(ids, query: str, model: str) -> dict:
    ids = [int(i) for i in ids]
    if not ids:
        return {}
    rows = db.q(f"SELECT photo_id, score FROM rerank_scores WHERE query=? AND model=? "
                f"AND photo_id IN ({','.join('?' * len(ids))})", [_key(query), model, *ids])
    return {r["photo_id"]: r["score"] for r in rows}


def run(ids, raw_query: str, progress=None):
    """Rate each photo against the search (skipping ones already rated). Returns how many were
    rated now, reused, or got no usable answer."""
    cfg = settings()
    query = query_text(raw_query)
    ids = [int(i) for i in ids][: cfg["top"]]
    have = cached(ids, query, cfg["model"])
    rated = unclear = 0
    for n, pid in enumerate(ids, 1):
        if pid not in have:
            try:
                score = score_photo(pid, query, cfg)
            except requests.ConnectionError:
                raise RuntimeError(f"No model server is answering at {cfg['url']}. Is Ollama running?")
            except requests.HTTPError as e:
                raise RuntimeError(f"The model server refused the request ({e.response.status_code}). "
                                   f"Is “{cfg['model']}” installed?")
            except FileNotFoundError:
                score = None        # the preview is gone (photo removed meanwhile)
            if score is None:
                unclear += 1
            else:
                rated += 1
                with db.tx() as c:
                    c.execute("INSERT OR REPLACE INTO rerank_scores(photo_id, query, model, score) VALUES(?,?,?,?)",
                              (pid, _key(query), cfg["model"], score))
        if progress:
            progress(n)
    return {"rated": rated, "reused": len(have), "unclear": unclear, "model": cfg["model"]}


def apply(hits, raw_query: str):
    """Re-order the top results by their ratings. Photos that were never rated keep their place,
    and the tier labels stay with the positions. Returns (hits, how many were re-ordered)."""
    cfg = settings()
    head = hits[: cfg["top"]]
    scores = cached([h["id"] for h in head], query_text(raw_query), cfg["model"])
    slots = [i for i, h in enumerate(head) if h["id"] in scores]
    if len(slots) < 2:
        return hits, 0
    ranked = sorted(slots, key=lambda i: (-scores[head[i]["id"]], i))
    out = list(hits)
    for slot, src in zip(slots, ranked):
        out[slot] = {**head[src], "tier": head[slot]["tier"], "deep": scores[head[src]["id"]]}
    return out, len(slots)
