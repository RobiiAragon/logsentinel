"""prepare -> train -> evaluate on Loghub HDFS_v1 with a chronological split."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import datasets, metrics, sessions
from .models import MODEL_FILES, ModelCard, Scorer, save_cards
from .parser import BLOCK_RE, TemplateMiner

SPLITS = {"train": 0, "val": 1, "test": 2}


def chronological_split(first_line: np.ndarray, train: float = 0.6, val: float = 0.2):
    order = np.argsort(first_line, kind="stable")
    n = len(order)
    split = np.empty(n, dtype=np.int8)
    a, b = int(train * n), int((train + val) * n)
    split[order[:a]], split[order[a:b]], split[order[b:]] = 0, 1, 2
    return split, int(first_line[order[a]])  # cut = first line of the validation period


def prepare(
    processed: Path,
    models_dir: Path,
    sample_dir: Path,
    sample_blocks: int = 1500,
    limit: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    zip_path = datasets.download_hdfs()
    raw = sessions.scan(datasets.iter_hdfs_lines(zip_path, limit))
    print(
        f"parsed {raw.n_lines:,} lines -> {len(raw.block_ids):,} blocks, {len(raw.keys)} unique "
        f"masked keys in {raw.seconds:.1f}s ({raw.n_lines / raw.seconds:,.0f} lines/s)"
    )
    split, cut = chronological_split(raw.first_line)
    miner = sessions.mine_templates(raw, max_line=cut)  # templates from the train period only
    key_tid = sessions.key_to_template(raw, miner)
    counts = sessions.count_matrix(raw, key_tid, miner.n_features)
    labels = datasets.hdfs_labels(zip_path)
    y = np.array([labels.get(b, 0) for b in raw.block_ids], dtype=np.int8)

    processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    miner.save(models_dir / "templates.json")
    np.savez_compressed(
        processed / "hdfs_features.npz",
        counts=counts.astype(np.uint16),
        y=y,
        split=split,
        block_ids=np.array(raw.block_ids),
    )
    stats = {
        "lines": raw.n_lines,
        "blocks": len(raw.block_ids),
        "unique_keys": len(raw.keys),
        "templates": len(miner.templates),
        "parse_seconds": raw.seconds,
        "lines_per_second": raw.n_lines / raw.seconds,
        "unknown_template_occurrences": int(raw.key_count[key_tid == 0].sum()),
        "splits": {
            name: {"blocks": int((split == i).sum()), "anomaly_rate": float(y[split == i].mean())}
            for name, i in SPLITS.items()
        },
    }
    (processed / "prepare_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats["splits"]))
    _write_sample(zip_path, raw.block_ids, y, split, sample_dir, sample_blocks, seed, limit)
    return stats


def _write_sample(
    zip_path: Path,
    block_ids: list[str],
    y: np.ndarray,
    split: np.ndarray,
    out: Path,
    n: int,
    seed: int,
    limit: int | None,
) -> None:
    """Raw log lines of a random subset of *test* blocks: a small, redistributable demo file
    (CC BY 4.0) for the dashboard and for devices that should not download 187 MB."""
    rng = np.random.default_rng(seed)
    test = np.flatnonzero(split == SPLITS["test"])
    chosen = {
        block_ids[i]: int(y[i]) for i in rng.choice(test, size=min(n, len(test)), replace=False)
    }
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    with gzip.open(out / "hdfs_test_sample.log.gz", "wt", encoding="utf-8") as fh:
        for line in datasets.iter_hdfs_lines(zip_path, limit):
            if any(b in chosen for b in BLOCK_RE.findall(line)):
                fh.write(line)
                kept += 1
    with (out / "hdfs_test_sample_labels.csv").open("w", encoding="utf-8") as fh:
        fh.write("BlockId,Label\n")
        for blk, lab in chosen.items():
            fh.write(f"{blk},{'Anomaly' if lab else 'Normal'}\n")
    print(f"sample: {len(chosen)} test blocks ({sum(chosen.values())} anomalies), {kept} lines")


def load_processed(processed: Path) -> dict[str, np.ndarray]:
    path = processed / "hdfs_features.npz"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `logsentinel prepare` first")
    data = np.load(path)
    return {
        "counts": data["counts"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "split": data["split"],
        "block_ids": data["block_ids"],
    }


def train(processed: Path, models_dir: Path, epochs: int = 8, seed: int = 0) -> dict[str, Any]:
    from .autoencoder import AutoEncoder
    from .models import export_iforest, quantize_int8, train_iforest

    d = load_processed(processed)
    x, y, split = d["counts"], d["y"], d["split"]
    tr, va = split == 0, split == 1
    n_feat = x.shape[1]
    report: dict[str, Any] = {}
    cards: list[ModelCard] = []

    def pick(name: str, s_val: np.ndarray) -> float:
        thr, f1 = metrics.best_f1_threshold(y[va], s_val)
        report[name] = {
            "val_f1": f1,
            "threshold": thr,
            "val_pr_auc": metrics.average_precision(y[va], s_val),
        }
        print(f"{name:<24} val F1={f1:.3f} PR-AUC={report[name]['val_pr_auc']:.3f}")
        return thr

    t0 = time.perf_counter()
    iso = train_iforest(x[tr], seed)
    report["train_seconds_iforest"] = time.perf_counter() - t0
    path = export_iforest(iso, n_feat, models_dir / MODEL_FILES["iforest"])
    s_val = Scorer(path)(x[va])[0]
    thr = pick("iforest", s_val)
    cards.append(
        ModelCard("iforest", path.name, thr, "train period, no labels", report["iforest"]["val_f1"])
    )

    dims = [n_feat, 32, 8, 32, n_feat]
    # Unsupervised variant (all train blocks, ~3% anomalous) - reported for comparison only.
    ae_all = AutoEncoder(dims, seed)
    ae_all.fit(x[tr], epochs=epochs)
    pick("autoencoder-unsupervised", ae_all.score(x[va]))

    # Deployed variant: trained on blocks from the train period that were labelled normal.
    t0 = time.perf_counter()
    ae = AutoEncoder(dims, seed)
    ae.fit(x[tr & (y == 0)], epochs=epochs)
    report["train_seconds_autoencoder"] = time.perf_counter() - t0
    fp32 = models_dir / MODEL_FILES["autoencoder"]
    import onnx

    onnx.save(ae.to_onnx(), str(fp32))
    onnx_scores = Scorer(fp32)(x[va])[0]
    report["onnx_max_abs_diff"] = float(np.abs(onnx_scores - ae.score(x[va])).max())
    thr = pick("autoencoder", onnx_scores)
    cards.append(
        ModelCard(
            "autoencoder",
            fp32.name,
            thr,
            "train period, normal blocks only",
            report["autoencoder"]["val_f1"],
        )
    )

    # Calibrate on the same distribution the model was trained on (normal sessions).
    rng = np.random.default_rng(seed)
    normal_train = np.flatnonzero(tr & (y == 0))
    calib = x[rng.choice(normal_train, 2048, replace=False)]
    for name, bits in (("autoencoder-int8", 8), ("autoencoder-w8a16", 16)):
        qpath = quantize_int8(fp32, models_dir / MODEL_FILES[name], calib, bits)
        thr = pick(name, Scorer(qpath)(x[va])[0])
        cards.append(
            ModelCard(
                name,
                qpath.name,
                thr,
                f"QDQ {'W8A8' if bits == 8 else 'W8A16'} "
                "of autoencoder, calibrated on normal train sessions",
                report[name]["val_f1"],
            )
        )

    # Supervised reference (upper bound; needs labels, not exported).
    from sklearn.linear_model import LogisticRegression

    z = np.log1p(x)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(z[tr], y[tr])
    thr = pick("logreg-supervised", lr.decision_function(z[va]))
    te = split == 2
    report["test_reference"] = {
        "logreg-supervised": metrics.report(y[te], lr.decision_function(z[te]), thr),
        "autoencoder-unsupervised": metrics.report(
            y[te], ae_all.score(x[te]), report["autoencoder-unsupervised"]["threshold"]
        ),
    }
    names = TemplateMiner.load(models_dir / "templates.json").feature_names()
    typical = x[tr & (y == 0)].mean(0)  # expected count of each template in a normal session
    save_cards(
        models_dir,
        cards,
        {
            "feature_names": names,
            "normal_mean_counts": [round(float(v), 3) for v in typical],
            "dataset": "Loghub HDFS_v1",
            "split": "chronological 60/20/20 by first block appearance",
        },
    )
    (models_dir / "train_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _latency(scorer: Scorer, x: np.ndarray, iters: int = 300) -> dict[str, float]:
    one = x[:1]
    for _ in range(20):
        scorer(one)
    t = []
    for i in range(iters):
        row = x[i % len(x) : i % len(x) + 1]
        a = time.perf_counter()
        scorer(row)
        t.append(time.perf_counter() - a)
    ms = np.array(t) * 1000
    return {"p50_ms": float(np.percentile(ms, 50)), "p99_ms": float(np.percentile(ms, 99))}


def evaluate(processed: Path, models_dir: Path, provider_aliases: list[str]) -> list[dict]:
    from .models import load_cards

    d = load_processed(processed)
    te = d["split"] == 2
    x, y = d["counts"][te], d["y"][te]
    out = []
    for card in load_cards(models_dir).values():
        path = models_dir / card.file
        for alias in provider_aliases:
            scorer = Scorer(path, alias)
            t0 = time.perf_counter()
            scores, _ = scorer(x)
            secs = time.perf_counter() - t0
            rec = {
                "model": card.name,
                "provider": alias,
                "size_kb": path.stat().st_size / 1024,
                **metrics.report(y, scores, card.threshold),
                "sessions_per_s": len(x) / secs,
                **_latency(scorer, x),
            }
            out.append(rec)
            print(
                f"{card.name:<17} {alias:<7} P={rec['precision']:.3f} R={rec['recall']:.3f} "
                f"F1={rec['f1']:.3f} PR-AUC={rec['pr_auc']:.3f} "
                f"{rec['sessions_per_s']:,.0f} sessions/s p50={rec['p50_ms']:.3f}ms"
            )
    return out
