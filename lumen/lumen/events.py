"""Memories: the library cut into trips and events by time and place, each with a fitting title."""
import threading
from collections import Counter
from datetime import datetime

import numpy as np

from . import db, embedder, quality, search
from .vectors import index

EVENT_GAP = 5 * 3600        # a pause this long between photos starts a new event
TRIP_GAP = 48 * 3600        # away-from-home events this close together form one trip
MIN_EVENT_PHOTOS = 6
MIN_TRIP_PHOTOS = 8

# Explore themes that make a good event title, with the wording to use.
EVENT_TITLES = {
    "Birthdays": "Birthday", "Weddings": "Wedding", "Celebrations": "Party", "Christmas": "Christmas",
    "Beaches & sea": "Day at the beach", "Mountains": "Day in the mountains", "Concerts": "Concert",
    "Sports": "Game day", "Restaurants": "Dinner out", "Snow & winter": "Snow day", "Fireworks": "Fireworks",
    "Pools": "Pool day", "Museums & art": "Museum visit", "Road trips": "Road trip", "Boats": "Boat trip",
    "Forests": "Walk in the woods", "Kids": "Kids' day", "Group photos": "Get-together",
}

_cache = {"key": None, "data": None}
_lock = threading.Lock()


def _span(t0, t1):
    """'Sat 12 Jul 2023', '12–19 Jul 2023', '28 Jun – 3 Jul 2023' (portable: no %-d)."""
    a, b = datetime.fromtimestamp(t0), datetime.fromtimestamp(t1)
    if a.date() == b.date():
        return f"{a.strftime('%a')} {a.day} {a.strftime('%b %Y')}"
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}–{b.day} {b.strftime('%b %Y')}"
    if a.year == b.year:
        return f"{a.day} {a.strftime('%b')} – {b.day} {b.strftime('%b %Y')}"
    return f"{a.day} {a.strftime('%b %Y')} – {b.day} {b.strftime('%b %Y')}"


def _holiday(ts):
    d = datetime.fromtimestamp(ts)
    md = (d.month, d.day)
    if md in ((12, 24), (12, 25), (12, 26)):
        return "Christmas"
    if md in ((12, 31), (1, 1)):
        return "New Year's Eve" if md == (12, 31) else "New Year's Day"
    if md == (10, 31):
        return "Halloween"
    return None


def _concepts_by_photo():
    """Photo id -> best-matching Explore theme name (only for strong matches)."""
    ids, mat = index.snapshot()
    if len(ids) < 8:
        return {}
    z = search.robust_z(mat @ embedder.embed_queries([p for _, _, p in search.CONCEPTS]).T)
    best = z.argmax(1)
    strong = z[np.arange(len(ids)), best] >= search.TIER_GOOD
    return {int(ids[i]): search.CONCEPTS[best[i]][0] for i in np.nonzero(strong)[0]}


def _people_by_photo(ids):
    out = {}
    if not ids:
        return out
    for i in range(0, len(ids), 900):
        chunk = ids[i : i + 900]
        for r in db.q(
            f"SELECT f.photo_id, p.name FROM faces f JOIN persons p ON p.id=f.person_id "
            f"WHERE p.name IS NOT NULL AND p.name<>'' AND p.hidden=0 AND f.photo_id IN ({','.join('?' * len(chunk))})",
            chunk,
        ):
            out.setdefault(r["photo_id"], set()).add(r["name"])
    return out


def _home_region(rows):
    """Where most photo *days* happen is home; trips are away from it."""
    days = Counter()
    for r in rows:
        if r["place_region"]:
            days[(r["place_region"], datetime.fromtimestamp(r["taken_at"]).date())] += 1
    regions = Counter(region for region, _ in days)
    return regions.most_common(1)[0][0] if regions else None


def _where(group):
    cities = Counter(r["place_city"] for r in group if r["place_city"])
    regions = Counter(r["place_region"] for r in group if r["place_region"])
    countries = Counter(r["place_country"] for r in group if r["place_country"])
    located = sum(cities.values()) or 1
    if cities and cities.most_common(1)[0][1] >= 0.6 * located:
        return cities.most_common(1)[0][0]
    if len(countries) > 1:
        return countries.most_common(1)[0][0]
    if regions:
        return regions.most_common(1)[0][0]
    return countries.most_common(1)[0][0] if countries else None


def compute():
    with _lock:
        key = (index.version, db.one("SELECT COUNT(place_cc) n FROM photos")["n"],
               db.one("SELECT COUNT(*) n FROM persons WHERE name IS NOT NULL")["n"])
        if _cache["key"] == key:
            return _cache["data"]
        rows = db.q(
            "SELECT id, taken_at, place_city, place_region, place_country FROM photos "
            "WHERE status='ready' AND taken_at IS NOT NULL AND date_source IN ('camera', 'filename') ORDER BY taken_at"
        )
        # 1. Events: runs of photos without long pauses.
        events, cur = [], []
        for r in rows:
            if cur and r["taken_at"] - cur[-1]["taken_at"] > EVENT_GAP:
                events.append(cur)
                cur = []
            cur.append(r)
        if cur:
            events.append(cur)
        # 2. Trips: consecutive events away from home.
        home = _home_region(rows)

        def away(e):
            regions = [r["place_region"] for r in e if r["place_region"]]
            return bool(regions) and home is not None and Counter(regions).most_common(1)[0][0] != home

        groups = []
        for e in events:
            if groups and groups[-1]["away"] and away(e) and e[0]["taken_at"] - groups[-1]["rows"][-1]["taken_at"] <= TRIP_GAP:
                groups[-1]["rows"] += e
            else:
                groups.append({"rows": list(e), "away": away(e)})
        # Filter after joining, so a trip of several small days still counts.
        groups = [g for g in groups if len(g["rows"]) >= MIN_EVENT_PHOTOS]

        concepts = _concepts_by_photo()
        all_ids = [r["id"] for g in groups for r in g["rows"]]
        people = _people_by_photo(all_ids)
        q = quality.scores()
        out = []
        for g in groups:
            g_rows = g["rows"]
            ids = [r["id"] for r in g_rows]
            t0, t1 = g_rows[0]["taken_at"], g_rows[-1]["taken_at"]
            place = _where(g_rows)
            multi_day = datetime.fromtimestamp(t0).date() != datetime.fromtimestamp(t1).date()
            is_trip = g["away"] and len(ids) >= MIN_TRIP_PHOTOS and (multi_day or place)
            # Who: named people in at least a third of the photos.
            seen = Counter(n for i in ids for n in people.get(i, ()))
            who = [n for n, c in seen.most_common(3) if c >= max(2, len(ids) / 3)]
            if is_trip:
                title = f"Trip to {place}" if place else "Trip"
            else:
                theme = Counter(concepts[i] for i in ids if i in concepts).most_common(1)
                if theme and theme[0][1] >= 0.3 * len(ids) and theme[0][0] in EVENT_TITLES:
                    title = EVENT_TITLES[theme[0][0]]
                else:
                    title = _holiday(t0) or datetime.fromtimestamp(t0).strftime("%A")
                if place:
                    title += f" in {place}"
            if who:
                title += " with " + (" & ".join(who) if len(who) <= 2 else f"{who[0]}, {who[1]} & {who[2]}")
            best = sorted(ids, key=lambda i: -q.get(i, (0.0, 50))[0])
            out.append({
                "key": f"{ids[0]}-{ids[-1]}",
                "kind": "trip" if is_trip else "event",
                "title": title,
                "when": _span(t0, t1),
                "start": t0,
                "end": t1,
                "place": place,
                "people": who,
                "count": len(ids),
                "cover_ids": best[:4],
                "photo_ids": ids,
            })
        out.reverse()  # newest first
        _cache.update(key=key, data=out)
        return out


def get(key: str):
    return next((m for m in compute() if m["key"] == key), None)
