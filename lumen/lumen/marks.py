"""Favourites, star ratings and your own tags, optionally mirrored to XMP sidecar files so Lightroom,
Bridge, Capture One, digiKam and friends see them too. The photo files themselves are never written."""
import os
import xml.etree.ElementTree as ET

from . import db

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)

FAVOURITE_KEYWORD = "Favorite"


def _q(prefix, tag):
    return f"{{{NS[prefix]}}}{tag}"


def sidecar_path(photo_path: str) -> str:
    """IMG_0001.jpg -> IMG_0001.xmp (the Adobe convention; darktable's IMG_0001.jpg.xmp is read too)."""
    return os.path.splitext(photo_path)[0] + ".xmp"


def tags_of(photo_id: int):
    return [r["tag"] for r in db.q("SELECT tag FROM photo_tags WHERE photo_id=? ORDER BY tag", (photo_id,))]


def all_tags():
    return [dict(r) for r in db.q("SELECT tag, COUNT(*) n FROM photo_tags t JOIN photos p ON p.id=t.photo_id "
                                  "WHERE p.status='ready' GROUP BY tag ORDER BY n DESC, tag")]


def update(photo_ids, favorite=None, rating=None, add_tags=(), remove_tags=()):
    """Apply changes to several photos at once; returns how many changed."""
    add_tags = [t.strip().lstrip("#")[:60] for t in add_tags if t.strip().lstrip("#")]
    remove_tags = [t.strip().lstrip("#") for t in remove_tags]
    with db.tx() as c:
        for pid in photo_ids:
            if favorite is not None:
                c.execute("UPDATE photos SET favorite=? WHERE id=?", (int(bool(favorite)), pid))
            if rating is not None:
                c.execute("UPDATE photos SET rating=? WHERE id=?", (max(0, min(5, int(rating))), pid))
            c.executemany("INSERT OR IGNORE INTO photo_tags(photo_id, tag) VALUES(?, ?)", [(pid, t) for t in add_tags])
            c.executemany("DELETE FROM photo_tags WHERE photo_id=? AND lower(tag)=lower(?)",
                          [(pid, t) for t in remove_tags])
    if db.get_setting("write_xmp") == "1":
        for pid in photo_ids:
            write_sidecar(pid)
    return len(photo_ids)


EMPTY_XMP = """<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:dc="http://purl.org/dc/elements/1.1/"/>
</rdf:RDF></x:xmpmeta>"""


def write_sidecar(photo_id: int):
    """Write rating and keywords into the photo's XMP sidecar, keeping anything else already in it."""
    row = db.one("SELECT path, rating, favorite FROM photos WHERE id=?", (photo_id,))
    if not row:
        return
    path = sidecar_path(row["path"])
    try:
        root = ET.parse(path).getroot() if os.path.exists(path) else ET.fromstring(EMPTY_XMP)
    except ET.ParseError:
        return  # someone else's malformed sidecar: leave it alone
    desc = root.find(f".//{_q('rdf', 'Description')}")
    if desc is None:
        return
    desc.set(_q("xmp", "Rating"), str(row["rating"] or 0))
    keywords = tags_of(photo_id) + ([FAVOURITE_KEYWORD] if row["favorite"] else [])
    subject = desc.find(_q("dc", "subject"))
    if subject is not None:
        desc.remove(subject)
    if keywords:
        subject = ET.SubElement(desc, _q("dc", "subject"))
        bag = ET.SubElement(subject, _q("rdf", "Bag"))
        for k in keywords:
            ET.SubElement(bag, _q("rdf", "li")).text = k
    tmp = path + ".lumen-tmp"
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=False)
    os.replace(tmp, path)


def read_sidecar(photo_path: str):
    """(rating, keywords) from an existing XMP sidecar, or (None, [])."""
    for path in (sidecar_path(photo_path), photo_path + ".xmp"):
        if not os.path.exists(path):
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        rating = None
        for desc in root.iter(_q("rdf", "Description")):
            r = desc.get(_q("xmp", "Rating"))
            if r is None:
                el = desc.find(_q("xmp", "Rating"))
                r = el.text if el is not None else None
            if r is not None:
                try:
                    rating = max(0, min(5, int(float(r))))
                except ValueError:
                    pass
        keywords = [li.text.strip() for subject in root.iter(_q("dc", "subject"))
                    for li in subject.iter(_q("rdf", "li")) if li.text and li.text.strip()]
        return rating, keywords
    return None, []


def import_sidecars(limit=500) -> int:
    """Bring in ratings and keywords other apps already wrote, once per photo."""
    rows = db.q("SELECT id, path FROM photos WHERE status='ready' AND xmp_v=0 LIMIT ?", (limit,))
    if not rows:
        return 0
    with db.tx() as c:
        for r in rows:
            rating, keywords = read_sidecar(r["path"])
            if rating:
                c.execute("UPDATE photos SET rating=? WHERE id=? AND (rating IS NULL OR rating=0)", (rating, r["id"]))
            for k in keywords:
                if k == FAVOURITE_KEYWORD:
                    c.execute("UPDATE photos SET favorite=1 WHERE id=?", (r["id"],))
                else:
                    c.execute("INSERT OR IGNORE INTO photo_tags(photo_id, tag) VALUES(?, ?)", (r["id"], k[:60]))
            c.execute("UPDATE photos SET xmp_v=1 WHERE id=?", (r["id"],))
    return len(rows)
