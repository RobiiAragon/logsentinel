"""Train, export and load the detectors. Inference only needs NumPy + ONNX Runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import providers

MODEL_FILES = {
    "autoencoder": "autoencoder.fp32.onnx",
    "autoencoder-int8": "autoencoder.int8.onnx",
    "autoencoder-w8a16": "autoencoder.w8a16.onnx",
    "iforest": "iforest.onnx",
}


@dataclass
class ModelCard:
    name: str
    file: str
    threshold: float  # score >= threshold -> alert (chosen on the validation split)
    trained_on: str
    val_f1: float


def train_iforest(x_train: np.ndarray, seed: int = 0):
    from sklearn.ensemble import IsolationForest

    return IsolationForest(n_estimators=100, random_state=seed, n_jobs=-1).fit(x_train)


def export_iforest(model, n_features: int, path: Path) -> Path:
    from skl2onnx import to_onnx

    onx = to_onnx(
        model, np.zeros((1, n_features), np.float32), target_opset={"": 17, "ai.onnx.ml": 3}
    )
    path.write_bytes(onx.SerializeToString())
    return path


def quantize_int8(fp32: Path, out: Path, calib: np.ndarray, activation_bits: int = 8) -> Path:
    """Static QDQ quantization of the dense layers: int8 per-channel weights and uint8
    (W8A8) or uint16 (W8A16) activations.

    ``calib`` must look like the training data. For an autoencoder trained on normal
    sessions, calibrating with anomalies stretches the activation ranges so much that
    8-bit quantization noise exceeds the anomaly signal (see README).
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    class Reader(CalibrationDataReader):
        def __init__(self) -> None:
            self.batches = iter([{"counts": calib[i : i + 256]} for i in range(0, len(calib), 256)])

        def get_next(self):
            return next(self.batches, None)

    quantize_static(
        str(fp32),
        str(out),
        Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt16 if activation_bits == 16 else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=["Gemm", "MatMul"],
        extra_options={"UseQDQContribOps": activation_bits == 16},
    )
    return out


class Scorer:
    """Anomaly scores from an exported model: higher = more anomalous for every model."""

    def __init__(self, path: Path, provider: str = "cpu") -> None:
        self.path = path
        self.session = providers.session(str(path), provider)
        self.input = self.session.get_inputs()[0].name
        self.outputs = [o.name for o in self.session.get_outputs()]
        self.is_iforest = "scores" in self.outputs and "label" in self.outputs

    def __call__(
        self, counts: np.ndarray, batch: int = 4096
    ) -> tuple[np.ndarray, np.ndarray | None]:
        scores, feats = [], []
        for i in range(0, len(counts), batch):
            chunk = np.ascontiguousarray(counts[i : i + batch], dtype=np.float32)
            out = dict(zip(self.outputs, self.session.run(None, {self.input: chunk}), strict=True))
            if self.is_iforest:
                scores.append(-out["scores"].ravel())  # decision_function: lower = anomalous
            else:
                scores.append(out["score"].ravel())
                feats.append(out["feature_error"])
        return np.concatenate(scores), (np.concatenate(feats) if feats else None)


def save_cards(models_dir: Path, cards: list[ModelCard], extra: dict) -> None:
    payload = {"models": [asdict(c) for c in cards], **extra}
    (models_dir / "cards.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_extra(models_dir: Path) -> dict:
    data = json.loads((models_dir / "cards.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if k != "models"}


def load_cards(models_dir: Path) -> dict[str, ModelCard]:
    path = models_dir / "cards.json"
    if not path.exists():
        raise SystemExit(f"{path} not found; run `logsentinel train` (or use committed models/)")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {m["name"]: ModelCard(**m) for m in data["models"]}
