"""A small dense autoencoder written in NumPy and emitted as an ONNX graph by hand.

The exported graph contains the whole scoring function, so any ONNX Runtime (Mac, Termux,
Snapdragon QNN) can score raw event-count vectors without Python-side preprocessing:

    counts -> log1p -> standardise -> encoder -> decoder -> squared error
           -> outputs: "score" (mean error per session), "feature_error" (per template)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AutoEncoder:
    dims: list[int]  # e.g. [44, 32, 8, 32, 44]
    seed: int = 0
    weights: list[np.ndarray] = field(default_factory=list)
    biases: list[np.ndarray] = field(default_factory=list)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        if not self.weights:
            for a, b in zip(self.dims[:-1], self.dims[1:], strict=True):
                self.weights.append(
                    (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                )
                self.biases.append(np.zeros(b, dtype=np.float32))

    # ----------------------------------------------------------------- numerics
    def preprocess(self, counts: np.ndarray) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        return ((np.log1p(counts) - self.mean) / self.std).astype(np.float32)

    def _forward(self, z: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [z]
        h = z
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            h = h @ w + b
            if i != last:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h, acts

    def feature_error(self, counts: np.ndarray) -> np.ndarray:
        z = self.preprocess(counts)
        recon, _ = self._forward(z)
        return (recon - z) ** 2

    def score(self, counts: np.ndarray) -> np.ndarray:
        return self.feature_error(counts).mean(axis=1)

    def fit(
        self,
        counts: np.ndarray,
        epochs: int = 8,
        batch: int = 512,
        lr: float = 1e-3,
        std_floor: float = 0.1,
        verbose: bool = False,
    ) -> list[float]:
        """``std_floor`` stops templates that are (almost) never seen in normal sessions from
        getting a near-zero std, which would turn a single occurrence into an enormous
        z-score and make both scores and explanations unstable."""
        logc = np.log1p(counts)
        self.mean = logc.mean(0).astype(np.float32)
        self.std = np.maximum(logc.std(0), std_floor).astype(np.float32)
        x = self.preprocess(counts)
        rng = np.random.default_rng(self.seed)
        params = self.weights + self.biases
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        b1, b2, eps, step = 0.9, 0.999, 1e-8, 0
        history = []
        for epoch in range(epochs):
            perm = rng.permutation(len(x))
            total = 0.0
            for i in range(0, len(x), batch):
                xb = x[perm[i : i + batch]]
                out, acts = self._forward(xb)
                diff = out - xb
                total += float((diff**2).sum())
                grad = 2.0 * diff / diff.size
                gw, gb = [], []
                for li in range(len(self.weights) - 1, -1, -1):
                    gw.append(acts[li].T @ grad)
                    gb.append(grad.sum(0))
                    if li:
                        grad = (grad @ self.weights[li].T) * (acts[li] > 0)
                grads = gw[::-1] + gb[::-1]
                step += 1
                for p, g, mi, vi in zip(params, grads, m, v, strict=True):
                    mi *= b1
                    mi += (1 - b1) * g
                    vi *= b2
                    vi += (1 - b2) * g * g
                    p -= lr * (mi / (1 - b1**step)) / (np.sqrt(vi / (1 - b2**step)) + eps)
            history.append(total / x.size)
            if verbose:
                print(f"  epoch {epoch + 1}/{epochs} mse={history[-1]:.5f}")
        return history

    # --------------------------------------------------------------------- ONNX
    def to_onnx(self, opset: int = 17):
        from onnx import TensorProto, helper, numpy_helper

        assert self.mean is not None and self.std is not None
        d = self.dims[0]
        inits = [
            numpy_helper.from_array(np.ones(1, np.float32), "one"),
            numpy_helper.from_array(self.mean, "mean"),
            numpy_helper.from_array(self.std, "std"),
        ]
        nodes = [
            helper.make_node("Add", ["counts", "one"], ["c1"], name="plus_one"),
            helper.make_node("Log", ["c1"], ["logc"], name="log1p"),
            helper.make_node("Sub", ["logc", "mean"], ["centered"], name="center"),
            helper.make_node("Div", ["centered", "std"], ["z"], name="standardize"),
        ]
        h = "z"
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            inits += [numpy_helper.from_array(w, f"W{i}"), numpy_helper.from_array(b, f"B{i}")]
            out = f"h{i}"
            nodes.append(helper.make_node("Gemm", [h, f"W{i}", f"B{i}"], [out], name=f"dense{i}"))
            if i != last:
                nodes.append(helper.make_node("Relu", [out], [f"{out}_relu"], name=f"relu{i}"))
                out = f"{out}_relu"
            h = out
        nodes += [
            helper.make_node("Sub", [h, "z"], ["diff"], name="residual"),
            helper.make_node("Mul", ["diff", "diff"], ["feature_error"], name="square"),
            helper.make_node(
                "ReduceMean", ["feature_error"], ["score"], axes=[1], keepdims=0, name="mean_error"
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "logsentinel_autoencoder",
            [helper.make_tensor_value_info("counts", TensorProto.FLOAT, ["batch", d])],
            [
                helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch"]),
                helper.make_tensor_value_info("feature_error", TensorProto.FLOAT, ["batch", d]),
            ],
            inits,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", opset)], producer_name="logsentinel"
        )
        model.ir_version = 8
        return model
