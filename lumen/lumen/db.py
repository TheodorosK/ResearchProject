"""SQLite storage for folders, photos (with their embeddings) and collections."""
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS folders (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    added_at REAL NOT NULL,
    last_scan REAL
);
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    folder_id INTEGER NOT NULL REFERENCES folders(id) ON DELETE CASCADE,
    size INTEGER,
    mtime REAL,
    width INTEGER,
    height INTEGER,
    taken_at REAL,
    camera TEXT,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending | ready | error
    error TEXT,
    embedding BLOB,
    model TEXT
);
CREATE INDEX IF NOT EXISTS photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS photos_folder ON photos(folder_id);
CREATE INDEX IF NOT EXISTS photos_taken ON photos(taken_at);
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY,
    name TEXT,
    hidden INTEGER NOT NULL DEFAULT 0,
    cover_face_id INTEGER,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    x REAL, y REAL, w REAL, h REAL,          -- box as fractions of the photo
    score REAL,
    embedding BLOB NOT NULL,
    person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    locked INTEGER NOT NULL DEFAULT 0         -- 1 = user said "not this person"; never auto-assign
);
CREATE INDEX IF NOT EXISTS faces_photo ON faces(photo_id);
CREATE INDEX IF NOT EXISTS faces_person ON faces(person_id);
CREATE TABLE IF NOT EXISTS video_frames (
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    t REAL NOT NULL,                          -- seconds into the video
    embedding BLOB NOT NULL,
    model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS video_frames_photo ON video_frames(photo_id);
CREATE VIRTUAL TABLE IF NOT EXISTS photo_text USING fts5(text, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS dup_ignored (
    group_key TEXT PRIMARY KEY
);
CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    emoji TEXT,
    color TEXT,
    description TEXT,
    kind TEXT NOT NULL DEFAULT 'manual',        -- manual | smart
    query TEXT,
    cover_photo_id INTEGER REFERENCES photos(id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS collection_items (
    collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
    photo_id INTEGER NOT NULL REFERENCES photos(id) ON DELETE CASCADE,
    added_at REAL NOT NULL,
    PRIMARY KEY (collection_id, photo_id)
);
"""

_local = threading.local()
_write_lock = threading.RLock()


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        config.ensure_dirs()
        conn = sqlite3.connect(config.DB_PATH, timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        _local.conn = conn
    return conn


# Columns added after the first release; added in place to existing libraries.
MIGRATIONS = {
    "photos": {
        "phash": "INTEGER",           # 64-bit difference hash, for duplicate detection
        "lat": "REAL",
        "lon": "REAL",
        "meta_v": "INTEGER NOT NULL DEFAULT 0",
        "faces_v": "INTEGER NOT NULL DEFAULT 0",
        "sharpness": "REAL",          # variance of the Laplacian on the preview
        "brightness": "REAL",         # mean luminance 0-255
        "hi_clip": "REAL",            # share of blown-out highlight pixels
        "lo_clip": "REAL",            # share of crushed shadow pixels
        "place_city": "TEXT",
        "place_region": "TEXT",
        "place_country": "TEXT",
        "place_cc": "TEXT",
        "kind": "TEXT NOT NULL DEFAULT 'photo'",   # photo | video
        "duration": "REAL",           # seconds, videos only
        "ocr_v": "INTEGER NOT NULL DEFAULT 0",    # 0 = not considered, 1 = read, 2 = skipped (no text)
        "pets_v": "INTEGER NOT NULL DEFAULT 0",   # same, for looking for cats and dogs
        "palette": "TEXT",            # dominant colours, see colors.palette
    },
    "persons": {
        "kind": "TEXT NOT NULL DEFAULT 'person'",  # person | pet
    },
    "faces": {
        "expr_probs": "BLOB",         # FER+ probabilities, float16 x 8, see faces.EXPRESSIONS
        "kind": "TEXT NOT NULL DEFAULT 'person'",  # person (SFace fingerprint) | pet (image-model fingerprint)
        "species": "TEXT",            # cat | dog
        "model": "TEXT",              # image model that fingerprinted a pet
    },
}


def init():
    conn = connect()
    conn.executescript(SCHEMA)
    for table, cols in MIGRATIONS.items():
        have = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        for col, decl in cols.items():
            if col not in have:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
    conn.execute("CREATE INDEX IF NOT EXISTS photos_geo ON photos(lat) WHERE lat IS NOT NULL")
    conn.commit()


@contextmanager
def tx():
    """A serialized write transaction (SQLite allows one writer at a time anyway)."""
    with _write_lock:
        conn = connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise


def q(sql, args=()):
    return connect().execute(sql, args).fetchall()


def one(sql, args=()):
    return connect().execute(sql, args).fetchone()


def get_setting(key, default=None):
    row = one("SELECT value FROM settings WHERE key=?", (key,))
    return row["value"] if row else default


def set_setting(key, value):
    with tx() as c:
        c.execute(
            "INSERT INTO settings(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )


def now():
    return time.time()
