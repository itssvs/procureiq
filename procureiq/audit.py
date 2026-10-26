"""SQLite audit log: every agent step, vendor message and human decision.

Slide 9: "Every decision is logged. A human must approve before any award."
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import DATA_DIR

DB_PATH = DATA_DIR / "runs" / "procureiq.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    request_id  TEXT,
    status      TEXT NOT NULL,          -- RUNNING | AWAITING_APPROVAL | APPROVED | REJECTED
    state_json  TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id    TEXT NOT NULL,
    ts        TEXT NOT NULL,
    step      TEXT NOT NULL,            -- e.g. 4_verify, 8_clarify, approval
    actor     TEXT NOT NULL,            -- agent | vendor:C | human:<name>
    message   TEXT NOT NULL,
    data_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _db(path: Optional[Path] = None):
    p = Path(path or DB_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def new_run(request_id: str = "") -> str:
    run_id = datetime.now().strftime("RUN-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    with _db() as con:
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (run_id, _now(), request_id, "RUNNING", None))
    return run_id


def log(run_id: str, step: str, message: str, actor: str = "agent", data: Any = None) -> None:
    with _db() as con:
        con.execute("INSERT INTO events (run_id, ts, step, actor, message, data_json) VALUES (?,?,?,?,?,?)",
                    (run_id, _now(), step, actor, message,
                     None if data is None else json.dumps(data, ensure_ascii=False, default=str)))


def save_state(run_id: str, state: dict, status: Optional[str] = None, request_id: Optional[str] = None) -> None:
    with _db() as con:
        con.execute("UPDATE runs SET state_json = ?, status = COALESCE(?, status), "
                    "request_id = COALESCE(?, request_id) WHERE run_id = ?",
                    (json.dumps(state, ensure_ascii=False, default=str), status, request_id, run_id))


def set_status(run_id: str, status: str) -> None:
    with _db() as con:
        con.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))


def get_run(run_id: str) -> Optional[dict]:
    with _db() as con:
        row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["state"] = json.loads(d.pop("state_json") or "{}")
    return d


def list_runs(limit: int = 20) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT run_id, created_at, request_id, status FROM runs "
                           "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def events(run_id: str) -> list[dict]:
    with _db() as con:
        rows = con.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id", (run_id,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["data"] = json.loads(d.pop("data_json")) if d.get("data_json") else None
        out.append(d)
    return out
