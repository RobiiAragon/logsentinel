"""SQLite alert store (stdlib only, so it also works in Termux)."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT, source TEXT, model TEXT,
  threshold REAL, n_lines INTEGER, n_sessions INTEGER, n_alerts INTEGER, seconds REAL,
  metrics TEXT
);
CREATE TABLE IF NOT EXISTS alerts (
  run_id INTEGER, block TEXT, score REAL, severity TEXT, first_ts TEXT, n_lines INTEGER,
  label INTEGER, top TEXT, PRIMARY KEY (run_id, block)
);
CREATE TABLE IF NOT EXISTS lines (run_id INTEGER, block TEXT, idx INTEGER, text TEXT);
CREATE INDEX IF NOT EXISTS lines_block ON lines (run_id, block);
"""


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def add_run(
    con: sqlite3.Connection,
    run: dict[str, Any],
    alerts: list[dict[str, Any]],
    lines: dict[str, list[str]],
) -> int:
    cur = con.execute(
        "INSERT INTO runs (created, source, model, threshold, n_lines, n_sessions, n_alerts,"
        " seconds, metrics) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            run["created"],
            run["source"],
            run["model"],
            run["threshold"],
            run["n_lines"],
            run["n_sessions"],
            len(alerts),
            run["seconds"],
            json.dumps(run.get("metrics")),
        ),
    )
    run_id = int(cur.lastrowid)
    con.executemany(
        "INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?)",
        [
            (
                run_id,
                a["block"],
                a["score"],
                a["severity"],
                a["first_ts"],
                a["n_lines"],
                a.get("label"),
                json.dumps(a["top"]),
            )
            for a in alerts
        ],
    )
    con.executemany(
        "INSERT INTO lines VALUES (?,?,?,?)",
        [(run_id, blk, i, text) for blk, ls in lines.items() for i, text in enumerate(ls)],
    )
    con.commit()
    return run_id


def runs(con: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = con.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    return [{**dict(r), "metrics": json.loads(r["metrics"] or "null")} for r in rows]


def alerts(
    con: sqlite3.Connection, run_id: int, severity: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM alerts WHERE run_id = ?"
    args: list[Any] = [run_id]
    if severity:
        sql += " AND severity = ?"
        args.append(severity)
    sql += " ORDER BY score DESC LIMIT ?"
    args.append(limit)
    return [{**dict(r), "top": json.loads(r["top"])} for r in con.execute(sql, args)]


def alert(con: sqlite3.Connection, run_id: int, block: str) -> dict[str, Any] | None:
    row = con.execute(
        "SELECT * FROM alerts WHERE run_id = ? AND block = ?", (run_id, block)
    ).fetchone()
    return {**dict(row), "top": json.loads(row["top"])} if row else None


def run(con: sqlite3.Connection, run_id: int) -> dict[str, Any] | None:
    row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return {**dict(row), "metrics": json.loads(row["metrics"] or "null")} if row else None


def alert_lines(con: sqlite3.Connection, run_id: int, block: str) -> list[str]:
    rows = con.execute(
        "SELECT text FROM lines WHERE run_id = ? AND block = ? ORDER BY idx", (run_id, block)
    )
    return [r["text"] for r in rows]
