"""SQLite storage: events, users, camera status, search and retention."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL    NOT NULL,           -- unix epoch seconds
    ts_text       TEXT    NOT NULL,           -- local time, YYYY-MM-DD HH:MM:SS
    camera_id     TEXT    NOT NULL,
    camera_name   TEXT    NOT NULL,
    gate          TEXT,
    direction     TEXT    NOT NULL,           -- IN / OUT
    kind          TEXT    NOT NULL,           -- vehicle / person
    plate_text    TEXT,                       -- as displayed (e.g. ASY-3549)
    plate_norm    TEXT,                       -- A-Z0-9 only, for searching
    plate_conf    REAL,
    vehicle_type  TEXT,
    color         TEXT,                       -- vehicle colour
    upper_color   TEXT,                       -- person: shirt/jacket colour
    lower_color   TEXT,                       -- person: trousers colour
    full_path     TEXT,
    crop_path     TEXT,
    plate_path    TEXT,
    face_path     TEXT,
    track_id      INTEGER,
    extra         TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts    ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_plate ON events(plate_norm);
CREATE INDEX IF NOT EXISTS idx_events_kind  ON events(kind, ts);

CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL               -- admin / manager / guard
);

CREATE TABLE IF NOT EXISTS camera_status (
    camera_id   TEXT PRIMARY KEY,
    name        TEXT,
    role        TEXT,
    direction   TEXT,
    last_frame  REAL,
    last_event  REAL,
    fps         REAL,
    updated     REAL,
    skips       TEXT
);
"""

_CONFUSABLE = str.maketrans({"0": "O", "1": "I", "8": "B", "5": "S", "2": "Z", "6": "G"})


def norm_plate(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def canon_plate(text: str | None) -> str:
    """Normalise and fold look-alike characters (0/O, 1/I, 8/B ...)."""
    return norm_plate(text).translate(_CONFUSABLE)


def edit_distance_le1(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) > len(b):
        a, b = b, a
    i = j = diff = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
            continue
        diff += 1
        if diff > 1:
            return False
        if len(a) == len(b):
            i += 1
        j += 1
    return diff + (len(b) - j) + (len(a) - i) <= 1


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self._conn() as c:
            c.executescript(SCHEMA)
            try:                                   # older databases: add the column
                c.execute("ALTER TABLE camera_status ADD COLUMN skips TEXT")
            except sqlite3.OperationalError:
                pass

    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(str(self.path), timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = c
        return c

    # ------------------------------------------------------------ events
    def insert_event(self, ev: dict) -> int:
        ts = float(ev["ts"])
        row = {
            "ts": ts,
            "ts_text": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
            "camera_id": ev["camera_id"],
            "camera_name": ev.get("camera_name", ev["camera_id"]),
            "gate": ev.get("gate"),
            "direction": ev["direction"],
            "kind": ev["kind"],
            "plate_text": ev.get("plate_text"),
            "plate_norm": norm_plate(ev.get("plate_text")) or None,
            "plate_conf": ev.get("plate_conf"),
            "vehicle_type": ev.get("vehicle_type"),
            "color": ev.get("color"),
            "upper_color": ev.get("upper_color"),
            "lower_color": ev.get("lower_color"),
            "full_path": ev.get("full_path"),
            "crop_path": ev.get("crop_path"),
            "plate_path": ev.get("plate_path"),
            "face_path": ev.get("face_path"),
            "track_id": ev.get("track_id"),
            "extra": json.dumps(ev.get("extra") or {}),
        }
        cols = ", ".join(row)
        marks = ", ".join(":" + k for k in row)
        with self._conn() as c:
            cur = c.execute(f"INSERT INTO events ({cols}) VALUES ({marks})", row)
            return cur.lastrowid

    def get_event(self, event_id: int):
        return self._conn().execute(
            "SELECT * FROM events WHERE id=?", (event_id,)
        ).fetchone()

    def latest_events(self, limit: int = 30, after_id: int = 0):
        return self._conn().execute(
            "SELECT * FROM events WHERE id>? ORDER BY id DESC LIMIT ?",
            (after_id, limit),
        ).fetchall()

    def search(self, f: dict, limit: int = 48, offset: int = 0):
        """Filter events. Returns (rows, total).

        Supported keys: plate, fuzzy, kind, direction, camera_id, gate,
        vehicle_type, color, upper_color, lower_color, ts_from, ts_to.
        """
        where, args = [], []

        def eq(col, key):
            if f.get(key):
                where.append(f"{col} = ?")
                args.append(f[key])

        eq("kind", "kind")
        eq("direction", "direction")
        eq("camera_id", "camera_id")
        eq("gate", "gate")
        eq("vehicle_type", "vehicle_type")
        eq("color", "color")
        eq("upper_color", "upper_color")
        eq("lower_color", "lower_color")
        if f.get("ts_from") is not None:
            where.append("ts >= ?")
            args.append(float(f["ts_from"]))
        if f.get("ts_to") is not None:
            where.append("ts <= ?")
            args.append(float(f["ts_to"]))

        plate = norm_plate(f.get("plate"))
        fuzzy = bool(f.get("fuzzy")) and len(plate) >= 3
        if plate and not fuzzy:
            where.append("plate_norm LIKE ?")
            args.append(f"%{plate}%")
        sql_where = ("WHERE " + " AND ".join(where)) if where else ""
        c = self._conn()

        if fuzzy:
            target = canon_plate(plate)
            rows = c.execute(
                f"SELECT * FROM events {sql_where} AND plate_norm IS NOT NULL ORDER BY ts DESC"
                if where
                else "SELECT * FROM events WHERE plate_norm IS NOT NULL ORDER BY ts DESC",
                args,
            ).fetchall()
            hits = [r for r in rows if _fuzzy_match(target, canon_plate(r["plate_norm"]))]
            return hits[offset: offset + limit], len(hits)

        total = c.execute(f"SELECT COUNT(*) FROM events {sql_where}", args).fetchone()[0]
        rows = c.execute(
            f"SELECT * FROM events {sql_where} ORDER BY ts DESC LIMIT ? OFFSET ?",
            args + [limit, offset],
        ).fetchall()
        return rows, total

    def distinct(self, column: str) -> list[str]:
        if column not in {"color", "upper_color", "lower_color", "vehicle_type"}:
            raise ValueError(column)
        rows = self._conn().execute(
            f"SELECT DISTINCT {column} FROM events WHERE {column} IS NOT NULL ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]

    # --------------------------------------------------------- retention
    def delete_older_than(self, days: float, data_dir: str | Path, now: float | None = None) -> int:
        cutoff = (now or time.time()) - days * 86400
        c = self._conn()
        rows = c.execute(
            "SELECT id, full_path, crop_path, plate_path, face_path FROM events WHERE ts < ?",
            (cutoff,),
        ).fetchall()
        data_dir = Path(data_dir)
        for r in rows:
            for key in ("full_path", "crop_path", "plate_path", "face_path"):
                if r[key]:
                    try:
                        (data_dir / r[key]).unlink()
                    except FileNotFoundError:
                        pass
        with c:
            c.execute("DELETE FROM events WHERE ts < ?", (cutoff,))
        images = data_dir / "images"
        if images.exists():  # prune empty day/camera folders
            for d in sorted(images.rglob("*"), key=lambda p: -len(p.parts)):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        return len(rows)

    # ------------------------------------------------------------- users
    def add_user(self, username: str, password_hash: str, role: str):
        with self._conn() as c:
            c.execute(
                "INSERT INTO users(username,password_hash,role) VALUES(?,?,?) "
                "ON CONFLICT(username) DO UPDATE SET password_hash=excluded.password_hash, role=excluded.role",
                (username, password_hash, role),
            )

    def get_user(self, username: str):
        return self._conn().execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()

    def list_users(self):
        return self._conn().execute("SELECT username, role FROM users ORDER BY username").fetchall()

    def delete_user(self, username: str) -> bool:
        with self._conn() as c:
            return c.execute("DELETE FROM users WHERE username=?", (username,)).rowcount > 0

    # ----------------------------------------------------- camera status
    def update_camera_status(self, camera_id, name, role, direction, last_frame, last_event, fps, skips=None):
        with self._conn() as c:
            c.execute(
                "INSERT INTO camera_status(camera_id,name,role,direction,last_frame,last_event,fps,updated,skips) "
                "VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(camera_id) DO UPDATE SET "
                "name=excluded.name, role=excluded.role, direction=excluded.direction, "
                "last_frame=excluded.last_frame, last_event=COALESCE(excluded.last_event, camera_status.last_event), "
                "fps=excluded.fps, updated=excluded.updated, skips=excluded.skips",
                (camera_id, name, role, direction, last_frame, last_event, fps, time.time(), skips),
            )

    def camera_statuses(self):
        return self._conn().execute("SELECT * FROM camera_status ORDER BY camera_id").fetchall()


def _fuzzy_match(target: str, candidate: str) -> bool:
    if edit_distance_le1(target, candidate):
        return True
    # also allow the query to be a fragment of the plate (e.g. digits only)
    return len(target) >= 3 and target in candidate
