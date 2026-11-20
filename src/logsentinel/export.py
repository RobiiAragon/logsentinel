"""Export alerts for a SIEM: ECS-style NDJSON (Elastic Common Schema field names) or CSV.

One line per alert. Standard ECS fields are used where they exist (``@timestamp``,
``event.*``, ``rule.*``, ``message``, ``tags``); everything LogSentinel-specific lives under
the custom ``logsentinel.*`` namespace, as ECS recommends for non-standard fields.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Iterable, Iterator
from typing import Any

# ECS event.severity is numeric; these follow the common low/medium/high/critical bands
# used for risk scores (21 / 47 / 73 / 99).
SEVERITY_NUM = {"medium": 47, "high": 73, "critical": 99}
CSV_FIELDS = ("run_id", "block", "severity", "score", "threshold", "first_ts", "n_lines",
              "label", "top_template", "top_kind")  # fmt: skip


def _iso(ts: str) -> str | None:
    """'2008-11-09 20:35:18' -> '2008-11-09T20:35:18'. HDFS logs carry no time zone, so
    the session's first timestamp is passed through as the log's local time."""
    return ts.replace(" ", "T") if ts else None


def _created(text: str) -> str:
    """Run timestamp '2026-11-20 09:39 UTC' -> ISO 8601 '2026-11-20T09:39:00Z'."""
    if text.endswith(" UTC"):
        return text[:-4].replace(" ", "T") + ":00Z"
    return text


def risk_score(score: float, threshold: float) -> float:
    """0-100 scale aligned with the severity bands: 25 at the threshold, 50 at 2x
    (high), 100 at 4x and above (critical)."""
    if threshold <= 0:
        return 100.0
    return round(min(100.0, 25.0 * score / threshold), 1)


def ecs(run: dict[str, Any], alert: dict[str, Any]) -> dict[str, Any]:
    top = alert.get("top") or []
    why = "; ".join(
        f"{t['kind']}: {t['template']} (seen {t['count']}x, typical {t['typical']}x)" for t in top
    )
    doc: dict[str, Any] = {
        "@timestamp": _iso(alert.get("first_ts", "")) or _created(run["created"]),
        "message": f"Anomalous HDFS session {alert['block']} "
        f"(score {alert['score']:.3f} >= {run['threshold']:.3f}). {why}".strip(),
        "event": {
            "kind": "alert",
            "module": "logsentinel",
            "dataset": "logsentinel.hdfs",
            "category": ["host"],
            "type": ["info"],
            "severity": SEVERITY_NUM.get(alert["severity"], 21),
            "risk_score": risk_score(float(alert["score"]), float(run["threshold"])),
            "created": _created(run["created"]),
        },
        "rule": {"name": f"logsentinel {run['model']}", "ruleset": "logsentinel"},
        "tags": ["logsentinel", alert["severity"]],
        "logsentinel": {
            "run_id": run["id"],
            "source": run["source"],
            "model": run["model"],
            "threshold": run["threshold"],
            "block": alert["block"],
            "score": round(float(alert["score"]), 6),
            "severity": alert["severity"],
            "n_lines": alert["n_lines"],
            "label": alert.get("label"),
            "top_templates": top,
        },
    }
    return doc


def ndjson(run: dict[str, Any], alerts: Iterable[dict[str, Any]]) -> Iterator[str]:
    for a in alerts:
        yield json.dumps(ecs(run, a), ensure_ascii=False, separators=(",", ":")) + "\n"


def csv_rows(run: dict[str, Any], alerts: Iterable[dict[str, Any]]) -> Iterator[str]:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_FIELDS)
    for a in alerts:
        top = (a.get("top") or [{}])[0]
        writer.writerow(
            [run["id"], a["block"], a["severity"], f"{a['score']:.6f}", run["threshold"],
             a["first_ts"], a["n_lines"], "" if a.get("label") is None else a["label"],
             top.get("template", ""), top.get("kind", "")]
        )  # fmt: skip
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate()
    if buf.getvalue():  # header only (no alerts)
        yield buf.getvalue()
