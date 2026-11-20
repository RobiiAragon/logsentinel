from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from conftest import REPO

from logsentinel import sessions, store
from logsentinel.autoencoder import AutoEncoder
from logsentinel.detect import detect, severity
from logsentinel.models import MODEL_FILES, ModelCard, save_cards
from logsentinel.pipeline import chronological_split


def test_chronological_split() -> None:
    split, cut = chronological_split(np.arange(10)[::-1])
    assert split.tolist() == [2, 2, 1, 1, 0, 0, 0, 0, 0, 0] and cut == 6


def test_severity_bands() -> None:
    assert severity(5.0, 1.0) == "critical"
    assert severity(2.5, 1.0) == "high"
    assert severity(1.1, 1.0) == "medium"


def _train_tiny(models_dir: Path, log: Path) -> None:
    import onnx

    raw = sessions.scan(log.read_text().splitlines(keepends=True))
    miner = sessions.mine_templates(raw)
    x = sessions.count_matrix(raw, sessions.key_to_template(raw, miner), miner.n_features)
    normal = x[[i for i, b in enumerate(raw.block_ids) if int(b[4:]) < 1300]]
    ae = AutoEncoder([miner.n_features, 8, 3, 8, miner.n_features])
    ae.fit(normal, epochs=30)
    models_dir.mkdir()
    miner.save(models_dir / "templates.json")
    onnx.save(ae.to_onnx(), str(models_dir / MODEL_FILES["autoencoder"]))
    thr = float(ae.score(normal).max()) * 1.5
    save_cards(
        models_dir,
        [ModelCard("autoencoder", MODEL_FILES["autoencoder"], thr, "t", 1.0)],
        {"feature_names": miner.feature_names(), "normal_mean_counts": normal.mean(0).tolist()},
    )


def test_detect_end_to_end_and_dashboard(tmp_path, synthetic) -> None:
    log, labels_csv, labels = synthetic
    _train_tiny(tmp_path / "m", log)
    db = tmp_path / "alerts.db"
    run = detect(log, tmp_path / "m", "autoencoder", db, labels_path=labels_csv)
    assert run["n_sessions"] == len(labels) and run["metrics"]["recall"] == 1.0
    assert run["metrics"]["precision"] == 1.0
    con = store.connect(db)
    alerts = store.alerts(con, run["id"])
    assert len(alerts) == sum(labels.values()) and all(a["label"] == 1 for a in alerts)
    top = alerts[0]["top"]
    assert {t["kind"] for t in top} & {"missing", "unusual"}
    assert any("exception" in ln for ln in store.alert_lines(con, run["id"], alerts[0]["block"]))

    from fastapi.testclient import TestClient

    from logsentinel.api import create_app

    client = TestClient(create_app(db))
    assert "LogSentinel" in client.get("/").text
    assert client.get("/api/runs").json()[0]["n_alerts"] == len(alerts)
    blk = alerts[0]["block"]
    detail = client.get(f"/api/alerts/{blk}").json()
    assert detail["lines"] and detail["block"] == blk
    assert client.get("/api/alerts", params={"severity": "bogus"}).status_code == 422
    assert client.get("/api/alerts/blk_nope").status_code == 404


def test_committed_models_on_committed_sample(tmp_path) -> None:
    """The shipped ONNX models + sample must reproduce a sane detection run."""
    models = REPO / "models"
    sample = REPO / "data/sample/hdfs_test_sample.log.gz"
    cards = json.loads((models / "cards.json").read_text())
    names = {m["name"] for m in cards["models"]}
    assert {"autoencoder", "autoencoder-int8", "autoencoder-w8a16", "iforest"} <= names
    run = detect(
        sample,
        models,
        "autoencoder",
        tmp_path / "a.db",
        labels_path=REPO / "data/sample/hdfs_test_sample_labels.csv",
    )
    assert run["n_sessions"] == 1500
    assert run["metrics"]["f1"] > 0.8


def test_threshold_override_export_and_card(tmp_path, synthetic, capsys) -> None:
    from logsentinel import export
    from logsentinel.cli import main

    log, labels_csv, _ = synthetic
    _train_tiny(tmp_path / "m", log)
    db = tmp_path / "alerts.db"
    base = detect(log, tmp_path / "m", "autoencoder", db, labels_path=labels_csv)
    huge = detect(log, tmp_path / "m", "autoencoder", db, threshold=1e9)
    assert huge["n_alerts"] == 0 and huge["threshold"] == 1e9 and base["n_alerts"] > 0

    out = tmp_path / "alerts.ndjson"
    run_arg = str(base["id"])
    assert main(["export", "--db", str(db), "--run", run_arg, "-o", str(out)]) == 0
    docs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(docs) == base["n_alerts"]
    first = docs[0]
    assert first["event"]["kind"] == "alert" and first["event"]["created"].endswith("Z")
    assert 25 <= first["event"]["risk_score"] <= 100
    assert (
        first["logsentinel"]["block"].startswith("blk_") and first["logsentinel"]["top_templates"]
    )

    assert main(["export", "--db", str(db), "--format", "csv"]) == 0  # latest run: no alerts
    assert capsys.readouterr().out.strip() == ",".join(export.CSV_FIELDS)
    assert main(["export", "--db", str(tmp_path / "missing.db")]) == 2
    assert export.risk_score(2.0, 1.0) == 50.0 and export.risk_score(9.0, 1.0) == 100.0


def test_model_card_matches_committed_results() -> None:
    from logsentinel.card import render

    text = render(REPO / "models", REPO / "results/mac-m5max.json")
    results = json.loads((REPO / "results/mac-m5max.json").read_text())["results"]
    ae = next(r for r in results if r["model"] == "autoencoder" and r["provider"] == "cpu")
    assert text.startswith("---\nlicense: mit") and "model-index:" in text
    assert f"{{type: f1, value: {ae['f1']:.4f}}}" in text
    assert f"{ae['tp']} / {ae['fp']} / {ae['fn']}" in text
