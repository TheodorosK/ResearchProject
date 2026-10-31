"""Place names for photos with GPS, fully offline (GeoNames cities via reverse_geocoder), and
finding place names inside search queries ("beach in Crete")."""
import math
import re
import threading
import unicodedata

from . import db

COUNTRIES = dict(line.split(" ", 1) for line in """\
AD Andorra|AE United Arab Emirates|AF Afghanistan|AG Antigua and Barbuda|AL Albania|AM Armenia|AO Angola
AR Argentina|AT Austria|AU Australia|AW Aruba|AZ Azerbaijan|BA Bosnia and Herzegovina|BB Barbados|BD Bangladesh
BE Belgium|BF Burkina Faso|BG Bulgaria|BH Bahrain|BI Burundi|BJ Benin|BM Bermuda|BN Brunei|BO Bolivia|BR Brazil
BS Bahamas|BT Bhutan|BW Botswana|BY Belarus|BZ Belize|CA Canada|CD DR Congo|CF Central African Republic
CG Congo|CH Switzerland|CI Ivory Coast|CL Chile|CM Cameroon|CN China|CO Colombia|CR Costa Rica|CU Cuba
CV Cape Verde|CY Cyprus|CZ Czechia|DE Germany|DJ Djibouti|DK Denmark|DM Dominica|DO Dominican Republic
DZ Algeria|EC Ecuador|EE Estonia|EG Egypt|ER Eritrea|ES Spain|ET Ethiopia|FI Finland|FJ Fiji|FO Faroe Islands
FR France|GA Gabon|GB United Kingdom|GD Grenada|GE Georgia|GH Ghana|GI Gibraltar|GL Greenland|GM Gambia
GN Guinea|GP Guadeloupe|GR Greece|GT Guatemala|GU Guam|GY Guyana|HK Hong Kong|HN Honduras|HR Croatia|HT Haiti
HU Hungary|ID Indonesia|IE Ireland|IL Israel|IN India|IQ Iraq|IR Iran|IS Iceland|IT Italy|JM Jamaica|JO Jordan
JP Japan|KE Kenya|KG Kyrgyzstan|KH Cambodia|KR South Korea|KP North Korea|KW Kuwait|KZ Kazakhstan|LA Laos
LB Lebanon|LI Liechtenstein|LK Sri Lanka|LR Liberia|LS Lesotho|LT Lithuania|LU Luxembourg|LV Latvia|LY Libya
MA Morocco|MC Monaco|MD Moldova|ME Montenegro|MG Madagascar|MK North Macedonia|ML Mali|MM Myanmar|MN Mongolia
MO Macao|MQ Martinique|MR Mauritania|MT Malta|MU Mauritius|MV Maldives|MW Malawi|MX Mexico|MY Malaysia
MZ Mozambique|NA Namibia|NC New Caledonia|NE Niger|NG Nigeria|NI Nicaragua|NL Netherlands|NO Norway|NP Nepal
NZ New Zealand|OM Oman|PA Panama|PE Peru|PF French Polynesia|PG Papua New Guinea|PH Philippines|PK Pakistan
PL Poland|PR Puerto Rico|PS Palestine|PT Portugal|PY Paraguay|QA Qatar|RE Réunion|RO Romania|RS Serbia|RU Russia
RW Rwanda|SA Saudi Arabia|SC Seychelles|SD Sudan|SE Sweden|SG Singapore|SI Slovenia|SK Slovakia|SL Sierra Leone
SM San Marino|SN Senegal|SO Somalia|SR Suriname|SV El Salvador|SY Syria|TG Togo|TH Thailand|TJ Tajikistan
TN Tunisia|TR Turkey|TT Trinidad and Tobago|TW Taiwan|TZ Tanzania|UA Ukraine|UG Uganda|US United States
UY Uruguay|UZ Uzbekistan|VA Vatican City|VE Venezuela|VN Vietnam|XK Kosovo|YE Yemen|ZA South Africa|ZM Zambia
ZW Zimbabwe""".replace("\n", "|").split("|"))

# Names people actually say that aren't GeoNames city/region names, and Greek spellings.
# name -> (lat, lon, radius km)
ALIASES = {
    "santorini": (36.40, 25.43, 15), "σαντορίνη": (36.40, 25.43, 15), "thira": (36.40, 25.43, 15),
    "mykonos": (37.45, 25.36, 12), "μύκονος": (37.45, 25.36, 12),
    "naxos": (37.05, 25.48, 25), "νάξος": (37.05, 25.48, 25),
    "paros": (37.08, 25.15, 15), "πάρος": (37.08, 25.15, 15),
    "corfu": (39.60, 19.87, 40), "κέρκυρα": (39.60, 19.87, 40),
    "rhodes": (36.20, 27.95, 45), "ρόδος": (36.20, 27.95, 45),
    "zakynthos": (37.78, 20.85, 25), "zante": (37.78, 20.85, 25), "ζάκυνθος": (37.78, 20.85, 25),
    "kefalonia": (38.20, 20.55, 35), "κεφαλονιά": (38.20, 20.55, 35),
    "lefkada": (38.71, 20.65, 20), "λευκάδα": (38.71, 20.65, 20),
    "halkidiki": (40.25, 23.55, 55), "chalkidiki": (40.25, 23.55, 55), "χαλκιδική": (40.25, 23.55, 55),
    "meteora": (39.72, 21.63, 10), "μετέωρα": (39.72, 21.63, 10),
    "delphi": (38.48, 22.50, 8), "δελφοί": (38.48, 22.50, 8),
    "acropolis": (37.9715, 23.7257, 1), "ακρόπολη": (37.9715, 23.7257, 1),
    "κρήτη": (35.24, 24.90, 140), "crete": (35.24, 24.90, 140),
    "αθήνα": (37.98, 23.73, 25), "θεσσαλονίκη": (40.64, 22.94, 20), "ελλάδα": (38.5, 23.5, 600),
    "πελοπόννησος": (37.5, 22.3, 120), "peloponnese": (37.5, 22.3, 120),
    "cyclades": (37.0, 25.2, 110), "κυκλάδες": (37.0, 25.2, 110),
}

MAX_CITY_KM = 30  # further than this from the nearest town, we don't claim the town's name
PREPOSITIONS = r"(?:in|at|from|near|around|on|στην|στη|στο|στον|στα|από)\s+"

_lock = threading.Lock()


def _km(lat1, lon1, lat2, lon2):
    p = math.pi / 180
    a = (math.sin((lat2 - lat1) * p / 2) ** 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * math.sin((lon2 - lon1) * p / 2) ** 2)
    return 12742 * math.asin(math.sqrt(a))


def fold(s: str) -> str:
    """Lowercase without accents, so 'Κρήτη' matches 'κρητη' and 'Réunion' matches 'reunion'."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")


def geocode_pending(limit=5000) -> int:
    """Name the places of photos that have GPS but no place yet."""
    rows = db.q("SELECT id, lat, lon FROM photos WHERE lat IS NOT NULL AND place_cc IS NULL LIMIT ?", (limit,))
    if not rows:
        return 0
    import reverse_geocoder as rg

    with _lock:
        found = rg.search([(r["lat"], r["lon"]) for r in rows], mode=1, verbose=False)
    updates = []
    for r, f in zip(rows, found):
        near = _km(r["lat"], r["lon"], float(f["lat"]), float(f["lon"])) <= MAX_CITY_KM
        updates.append((f["name"] if near else None, f["admin1"] or None, f["cc"],
                        COUNTRIES.get(f["cc"], f["cc"]), r["id"]))
    with db.tx() as c:
        c.executemany("UPDATE photos SET place_city=?, place_region=?, place_cc=?, place_country=? WHERE id=?",
                      updates)
    _known.update(key=None)
    return len(rows)


def label(row) -> str | None:
    """'Chania, Crete, Greece' without repeats."""
    parts = []
    for k in ("place_city", "place_region", "place_country"):
        v = row[k] if k in row.keys() else None
        if v and v not in parts:
            parts.append(v)
    return ", ".join(parts) or None


_known = {"key": None, "names": {}}


def _known_places():
    """Folded place name -> (column, value) for places that occur in the library."""
    key = tuple(db.one("SELECT COUNT(place_cc) n, MAX(id) m FROM photos"))
    if _known["key"] != key:
        names = {}
        for col in ("place_country", "place_region", "place_city"):  # cities win on clashes
            for r in db.q(f"SELECT DISTINCT {col} v FROM photos WHERE {col} IS NOT NULL"):
                if len(r["v"]) >= 3:
                    names[fold(r["v"])] = (col, r["v"])
        _known.update(key=key, names=names)
    return _known["names"]


def places_in_query(text: str):
    """('beach in Crete') -> ('beach', [{'label': 'Crete', 'where': sql, 'args': [...]}])."""
    found = []
    folded = fold(text)
    candidates = {**{fold(k): ("alias", k) for k in ALIASES}, **_known_places()}
    for name in sorted(candidates, key=len, reverse=True):
        pat = re.compile(rf"(?:{PREPOSITIONS})?(?<!\w){re.escape(name)}(?!\w)", re.IGNORECASE)
        m = pat.search(folded)
        if not m:
            continue
        col, value = candidates[name]
        if col == "alias":
            lat, lon, radius = ALIASES[value]
            where, args = "lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?", _box(lat, lon, radius)
            shown = value.title() if value.isascii() else value.capitalize()
        else:
            where, args, shown = f"{col}=?", [value], value
        found.append({"label": shown, "where": where, "args": args})
        # Remove the same span from the original text (folding keeps positions for these scripts).
        text = text[: m.start()] + " " + text[m.end():]
        folded = folded[: m.start()] + " " + folded[m.end():]
    if db.one("SELECT 1 FROM photos WHERE lat IS NOT NULL LIMIT 1"):
        for name, place in _world_city_matches(text):
            m = re.search(rf"(?:{PREPOSITIONS})?(?<!\w){re.escape(name)}(?!\w)", folded)
            if m:
                found.append(place)
                text = text[: m.start()] + " " + text[m.end():]
                folded = folded[: m.start()] + " " + folded[m.end():]
    return " ".join(text.split()), found


_world = {"names": None}
CITY_RADIUS_KM = 15


def _world_cities():
    """Folded city name -> [(lat, lon)] for every town in GeoNames' cities1000 list (loaded once)."""
    if _world["names"] is None:
        import csv
        import os

        import reverse_geocoder

        names = {}
        path = os.path.join(os.path.dirname(reverse_geocoder.__file__), "rg_cities1000.csv")
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.setdefault(fold(row["name"]), []).append((float(row["lat"]), float(row["lon"])))
        _world["names"] = names
    return _world["names"]


def _box(lat, lon, radius):
    dlat = radius / 111.0
    dlon = radius / (111.0 * max(math.cos(lat * math.pi / 180), 0.1))
    return [lat - dlat, lat + dlat, lon - dlon, lon + dlon]


def _world_city_matches(text):
    """Any town name in the query that has some of the user's photos nearby ('Thessaloniki' even when
    the photos are labelled with a suburb). Ambiguous names keep only the places the user has been."""
    found = []
    words = re.findall(r"\w+", fold(text))
    world = _world_cities()
    for n in (3, 2, 1):
        for k in range(len(words) - n + 1):
            name = " ".join(words[k : k + n])
            if len(name) < 4 or name not in world:
                continue
            where, args = [], []
            for lat, lon in world[name][:50]:
                b = _box(lat, lon, CITY_RADIUS_KM)
                if db.one("SELECT 1 FROM photos WHERE lat BETWEEN ? AND ? AND lon BETWEEN ? AND ? LIMIT 1", b):
                    where.append("(lat BETWEEN ? AND ? AND lon BETWEEN ? AND ?)")
                    args += b
            if where:
                found.append((name, {"label": name.title(), "where": " OR ".join(where), "args": args}))
    return found


def photo_ids_for(place) -> set:
    return {r["id"] for r in db.q(f"SELECT id FROM photos WHERE status='ready' AND {place['where']}", place["args"])}


def top_places(limit=24):
    return [dict(r) for r in db.q(
        "SELECT place_city city, place_region region, place_country country, COUNT(*) n, "
        "AVG(lat) lat, AVG(lon) lon FROM photos WHERE status='ready' AND place_city IS NOT NULL "
        "GROUP BY place_city, place_region, place_country ORDER BY n DESC LIMIT ?", (limit,))]
