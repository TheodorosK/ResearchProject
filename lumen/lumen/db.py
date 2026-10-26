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


def init():
    conn = connect()
    conn.executescript(SCHEMA)
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
