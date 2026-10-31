"""'What Lumen sees': a few tags and a one-line description per photo, zero-shot.

Each tag is scored against every photo with the image model; a tag applies when the photo matches
it much better than the library as a whole does (robust z-score per tag), so tags stay meaningful
whatever the model and whatever the library.
"""
import threading

import numpy as np

from . import embedder
from .vectors import index

# (tag, prompt) by role in the sentence.
SUBJECTS = [
    ("people", "a group of people"), ("a child", "a young child"), ("a baby", "a baby"),
    ("a couple", "a couple together"), ("a portrait", "a close-up portrait of a person"),
    ("a selfie", "a selfie"), ("a dog", "a dog"), ("a cat", "a cat"), ("a bird", "a bird"),
    ("a horse", "a horse"), ("a car", "a car"), ("a bicycle", "a bicycle"), ("a boat", "a boat"),
    ("an airplane", "an airplane"), ("a train", "a train"), ("food", "a plate of food"),
    ("a cake", "a cake"), ("coffee", "a cup of coffee"), ("wine", "a glass of wine"),
    ("cocktails", "cocktails"), ("pizza", "pizza"), ("seafood", "seafood"), ("fruit", "fresh fruit"),
    ("flowers", "flowers"), ("a tree", "a big tree"), ("a church", "a church"),
    ("ruins", "ancient ruins"), ("a castle", "a castle"), ("a bridge", "a bridge"),
    ("a lighthouse", "a lighthouse"), ("a statue", "a statue"), ("a painting", "a painting"),
    ("a building", "a modern building"), ("a house", "a house"), ("a window", "a window"),
    ("a door", "a colorful door"), ("a table", "a table set for a meal"), ("a bed", "a bed"),
    ("a book", "a book"), ("a phone", "a mobile phone"), ("a laptop", "a laptop"),
    ("a guitar", "a guitar"), ("a ball", "a ball"), ("a tent", "a tent"), ("a swing", "a swing"),
    ("a christmas tree", "a christmas tree"), ("balloons", "balloons"), ("candles", "candles"),
    ("fireworks", "fireworks"), ("a sign", "a street sign"), ("a document", "a document with text"),
    ("a screenshot", "a screenshot"), ("a map", "a map"), ("toys", "toys"), ("shoes", "shoes"),
    ("a dress", "a dress"), ("sunglasses", "sunglasses"), ("a hat", "a hat"), ("an umbrella", "an umbrella"),
    ("a waterfall", "a waterfall"), ("a pool", "a swimming pool"), ("a fountain", "a fountain"),
    ("cows", "cows"), ("sheep", "sheep"), ("goats", "goats"), ("fish", "fish"), ("a butterfly", "a butterfly"),
]
SETTINGS = [
    ("the beach", "a beach"), ("the sea", "the sea"), ("a harbour", "a harbour with boats"),
    ("the mountains", "mountains"), ("a lake", "a lake"), ("a river", "a river"), ("a forest", "a forest"),
    ("a park", "a park"), ("a garden", "a garden"), ("a field", "a field in the countryside"),
    ("the desert", "a desert"), ("the snow", "snow"), ("a city street", "a city street"),
    ("an old town", "narrow streets of an old town"), ("a village", "a village"), ("a market", "a market"),
    ("a restaurant", "inside a restaurant"), ("a café", "a café"), ("a bar", "a bar at night"),
    ("a kitchen", "a kitchen"), ("a living room", "a living room"), ("a bedroom", "a bedroom"),
    ("an office", "an office"), ("a classroom", "a classroom"), ("a museum", "a museum"),
    ("a church interior", "the inside of a church"), ("a stadium", "a stadium"), ("a stage", "a stage"),
    ("an airport", "an airport"), ("a car interior", "inside a car"), ("an airplane window", "the view from an airplane window"),
    ("a road", "a road"), ("a rooftop", "a rooftop view"), ("a balcony", "a balcony"), ("a playground", "a playground"),
    ("a wedding venue", "a wedding ceremony"), ("a party", "a party"), ("a concert", "a concert"),
]
MOMENTS = [
    ("at sunset", "at sunset"), ("at sunrise", "at sunrise"), ("at night", "at night"),
    ("on a rainy day", "on a rainy day"), ("in the fog", "in the fog"), ("on a sunny day", "on a sunny day"),
    ("in golden light", "in warm golden light"), ("under a starry sky", "under a starry sky"),
    ("in winter", "in winter"), ("in autumn", "with autumn leaves"), ("in spring", "with spring blossoms"),
]
ACTIVITIES = [
    ("swimming", "people swimming"), ("eating", "people eating together"), ("dancing", "people dancing"),
    ("hiking", "people hiking"), ("skiing", "people skiing"), ("cycling", "people cycling"),
    ("playing", "children playing"), ("celebrating", "people celebrating"), ("cooking", "someone cooking"),
    ("reading", "someone reading"), ("sleeping", "someone sleeping"), ("running", "someone running"),
    ("posing", "people posing for a photo"), ("laughing", "people laughing"), ("working", "someone working"),
]
STYLES = [
    ("black & white", "a black and white photo"), ("moody", "a moody, dark photo"),
    ("cinematic", "a cinematic photo"), ("minimalist", "a minimalist photo"), ("vintage", "a vintage film photo"),
    ("vibrant", "a vibrant, colorful photo"), ("aerial", "an aerial drone photo"), ("macro", "a macro close-up photo"),
    ("panorama", "a wide panorama"), ("long exposure", "a long exposure photo"),
]
GROUPS = {"subject": SUBJECTS, "setting": SETTINGS, "moment": MOMENTS, "activity": ACTIVITIES, "style": STYLES}
MIN_Z = 3.0
ON = {"the beach", "a road", "a balcony", "a rooftop", "a stage", "a playground"}
BY = {"the sea", "a lake", "a river"}

_cache = {"key": None}
_lock = threading.Lock()


def _stats():
    """Per-tag median and spread over the library, plus the tag vectors (cached)."""
    with _lock:
        key = (index.version, embedder.get().key)
        if _cache["key"] != key:
            tags, prompts, roles = [], [], []
            for role, items in GROUPS.items():
                for tag, prompt in items:
                    tags.append(tag)
                    prompts.append(prompt)
                    roles.append(role)
            vecs = embedder.embed_queries(prompts)
            ids, mat = index.snapshot()
            if len(ids) >= 8:
                s = mat @ vecs.T
                med = np.median(s, 0)
                mad = np.median(np.abs(s - med), 0) * 1.4826
            else:
                med, mad = np.zeros(len(tags)), np.ones(len(tags))
            _cache.update(key=key, tags=tags, roles=roles, vecs=vecs, med=med, mad=np.maximum(mad, 1e-6))
        return _cache


def describe(photo_id: int, max_tags=8):
    """{'tags': [...], 'caption': 'a dog on the beach at sunset'} or None if the photo isn't indexed."""
    v = index.vector_of(photo_id)
    if v is None:
        return None
    st = _stats()
    z = (v @ st["vecs"].T - st["med"]) / st["mad"]
    best_by_role = {}
    tags = []
    for k in np.argsort(-z):
        if z[k] < MIN_Z or len(tags) >= max_tags:
            break
        tags.append(st["tags"][k])
        best_by_role.setdefault(st["roles"][k], st["tags"][k])
    return {"tags": tags, "caption": caption(best_by_role)}


def caption(parts):
    subject, setting = parts.get("subject"), parts.get("setting")
    activity, moment, style = parts.get("activity"), parts.get("moment"), parts.get("style")
    if not (subject or setting or activity):
        return None
    if activity and not subject:
        words = [activity.capitalize()]
    else:
        words = [(subject or "A scene").capitalize()]
        if activity:
            words.append(activity)
    if setting:
        prep = "on" if setting in ON else "by" if setting in BY else "in"
        words.append(f"{prep} {setting}")
    if moment:
        words.append(moment)
    text = " ".join(words)
    return f"{text} · {style}" if style else text
