"""FastAPI alert dashboard over the SQLite store."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise SystemExit('the dashboard needs FastAPI: pip install -e ".[server]"') from exc

from . import __version__, store


def create_app(db_path: Path) -> FastAPI:
    app = FastAPI(
        title="LogSentinel",
        version=__version__,
        description="Log anomaly alerts produced by `logsentinel detect`.",
    )
    con = store.connect(db_path)
    page = files("logsentinel").joinpath("web/index.html").read_text(encoding="utf-8")

    def _run_id(run: int | None) -> int:
        all_runs = store.runs(con)
        if not all_runs:
            raise HTTPException(404, "no detection runs yet; run `logsentinel detect` first")
        if run is None:
            return int(all_runs[0]["id"])
        if not any(r["id"] == run for r in all_runs):
            raise HTTPException(404, f"run {run} not found")
        return run

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def home() -> str:
        return page

    @app.get("/api/runs")
    def runs() -> list[dict[str, Any]]:
        return store.runs(con)

    @app.get("/api/alerts")
    def alerts(
        run: int | None = None,
        severity: str | None = Query(None, pattern="^(critical|high|medium)$"),
        limit: int = Query(500, ge=1, le=5000),
    ) -> list[dict[str, Any]]:
        return store.alerts(con, _run_id(run), severity, limit)

    @app.get("/api/alerts/{block}")
    def alert(block: str, run: int | None = None) -> dict[str, Any]:
        rid = _run_id(run)
        match = [a for a in store.alerts(con, rid, limit=100000) if a["block"] == block]
        if not match:
            raise HTTPException(404, "alert not found")
        return {**match[0], "lines": store.alert_lines(con, rid, block)}

    return app
