from __future__ import annotations

import numpy as np
import onnx
import pytest

from logsentinel import metrics
from logsentinel.autoencoder import AutoEncoder
from logsentinel.models import Scorer, quantize_int8


def test_metrics_match_sklearn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y = (rng.random(500) < 0.1).astype(int)
    s = rng.random(500) + y * 0.5
    s[::7] = 0.3  # ties
    assert metrics.average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert metrics.roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
    r = metrics.prf(y, s >= 0.8)
    assert r["f1"] == pytest.approx(sk.f1_score(y, s >= 0.8))
    thr, f1 = metrics.best_f1_threshold(y, s)
    assert metrics.prf(y, s >= thr)["f1"] == pytest.approx(f1)


def _data(n: int = 2000, d: int = 6, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    normal = rng.poisson([3, 3, 3, 1, 0.01, 0.01], size=(n, d)).astype(np.float32)
    anomal = normal[:100].copy()
    anomal[:, :3] = 0  # missing events
    anomal[:, 4] = 5  # unusual event
    return normal, anomal


def test_autoencoder_learns_and_onnx_matches(tmp_path) -> None:
    normal, anomal = _data()
    ae = AutoEncoder([6, 8, 3, 8, 6], seed=0)
    hist = ae.fit(normal, epochs=5)
    assert hist[-1] < hist[0]
    assert np.median(ae.score(anomal)) > 10 * np.median(ae.score(normal))
    path = tmp_path / "ae.onnx"
    model = ae.to_onnx()
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    scores, feat = Scorer(path)(anomal)
    assert np.allclose(scores, ae.score(anomal), rtol=1e-4, atol=1e-5)
    assert feat.shape == anomal.shape


@pytest.mark.parametrize("bits", [8, 16])
def test_quantized_autoencoder_keeps_ranking(tmp_path, bits) -> None:
    normal, anomal = _data()
    ae = AutoEncoder([6, 8, 3, 8, 6], seed=0)
    ae.fit(normal, epochs=5)
    fp32 = tmp_path / "ae.onnx"
    onnx.save(ae.to_onnx(), str(fp32))
    q = quantize_int8(fp32, tmp_path / f"ae{bits}.onnx", normal[:256], bits)
    ops = {n.op_type for n in onnx.load(str(q)).graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops
    s = Scorer(q)
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomal))]
    assert metrics.roc_auc(y, s(np.vstack([normal, anomal]))[0]) > 0.95


def test_iforest_export_scores_like_sklearn(tmp_path) -> None:
    pytest.importorskip("skl2onnx")
    from logsentinel.models import export_iforest, train_iforest

    normal, anomal = _data()
    iso = train_iforest(normal)
    path = export_iforest(iso, 6, tmp_path / "if.onnx")
    scores, feat = Scorer(path)(anomal)
    assert feat is None
    assert np.allclose(scores, -iso.decision_function(anomal), atol=1e-4)
