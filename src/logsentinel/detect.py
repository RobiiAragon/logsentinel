"""Score any HDFS-format log file and store alerts (with their log lines) in SQLite."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets, metrics, sessions, store
from .models import Scorer, load_cards, load_extra
from .parser import BLOCK_RE, TemplateMiner


def severity(score: float, threshold: float) -> str:
    ratio = score / threshold if threshold > 0 else float("inf")
    return "critical" if ratio >= 4 else "high" if ratio >= 2 else "medium"


def _explain(
    counts: np.ndarray,
    feat_err: np.ndarray | None,
    names: list[str],
    rarity: np.ndarray,
    typical: np.ndarray,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Top templates behind an alert: reconstruction error (autoencoder) or rarity x count
    (Isolation Forest), each with how often it normally appears in a session."""
    weight = feat_err if feat_err is not None else counts * rarity
    out = []
    for i in [int(i) for i in np.argsort(-weight)[:k] if weight[i] > 0]:
        seen, usual = float(counts[i]), float(typical[i])
        if seen == 0 and usual > 0:
            kind = "missing"
        elif usual < 0.01:
            kind = "unusual"
        else:
            kind = "more than usual" if seen > usual else "fewer than usual"
        out.append(
            {
                "template": names[i],
                "count": int(seen),
                "typical": round(usual, 2),
                "kind": kind,
                "weight": round(float(weight[i]), 4),
            }
        )
    return out


def load_labels(path: Path) -> dict[str, int]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        blk, _, lab = line.partition(",")
        out[blk] = 1 if lab.strip() == "Anomaly" else 0
    return out


def detect(
    log_path: Path,
    models_dir: Path,
    model: str,
    db_path: Path,
    provider: str = "cpu",
    labels_path: Path | None = None,
    max_lines: int = 200,
) -> dict[str, Any]:
    cards = load_cards(models_dir)
    if model not in cards:
        raise SystemExit(f"unknown model {model!r}; available: {', '.join(cards)}")
    card = cards[model]
    miner = TemplateMiner.load(models_dir / "templates.json")
    names = miner.feature_names()
    extra = load_extra(models_dir)
    typical = np.asarray(extra.get("normal_mean_counts", [0.0] * miner.n_features))
    freq = np.array([1] + [t.count for t in miner.templates], dtype=np.float64)
    rarity = -np.log(freq / freq.sum())
    rarity[0] = rarity.max() + 1.0  # never-seen template = rarest of all

    t0 = time.perf_counter()
    raw = sessions.scan(datasets.iter_text_lines(log_path))
    counts = sessions.count_matrix(raw, sessions.key_to_template(raw, miner), miner.n_features)
    scores, feat_err = Scorer(models_dir / card.file, provider)(counts)
    hit = np.flatnonzero(scores >= card.threshold)

    wanted = {raw.block_ids[i] for i in hit}
    lines: dict[str, list[str]] = {b: [] for b in wanted}
    n_per_block = np.bincount(raw.pair_block, minlength=len(raw.block_ids))
    for line in datasets.iter_text_lines(log_path):  # second pass: evidence for alerts only
        for blk in dict.fromkeys(BLOCK_RE.findall(line)):
            if blk in lines and len(lines[blk]) < max_lines:
                lines[blk].append(line.rstrip("\n"))
    seconds = time.perf_counter() - t0

    labels = load_labels(labels_path) if labels_path else {}
    alerts = []
    for i in hit:
        blk = raw.block_ids[i]
        ts = datetime.fromtimestamp(int(raw.first_ts[i])) if raw.first_ts[i] else None
        alerts.append(
            {
                "block": blk,
                "score": float(scores[i]),
                "severity": severity(float(scores[i]), card.threshold),
                "first_ts": ts.isoformat(sep=" ") if ts else "",
                "n_lines": int(n_per_block[i]),
                "label": labels.get(blk),
                "top": _explain(
                    counts[i],
                    feat_err[i] if feat_err is not None else None,
                    names,
                    rarity,
                    typical,
                ),
            }
        )
    run_metrics = None
    if labels:
        y = np.array([labels.get(b, 0) for b in raw.block_ids])
        run_metrics = metrics.report(y, scores, card.threshold)
    run = {
        "created": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "source": log_path.name,
        "model": model,
        "threshold": card.threshold,
        "n_lines": raw.n_lines,
        "n_sessions": len(raw.block_ids),
        "seconds": seconds,
        "metrics": run_metrics,
    }
    con = store.connect(db_path)
    run["id"] = store.add_run(con, run, alerts, lines)
    run["n_alerts"] = len(alerts)
    run["lines_per_s"] = raw.n_lines / seconds if seconds else None
    return run
