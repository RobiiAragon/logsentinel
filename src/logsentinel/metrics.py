"""Binary detection metrics in pure NumPy (no scikit-learn needed on edge devices)."""

from __future__ import annotations

import numpy as np


def confusion(y: np.ndarray, pred: np.ndarray) -> dict[str, int]:
    y, pred = y.astype(bool), pred.astype(bool)
    return {
        "tp": int((y & pred).sum()),
        "fp": int((~y & pred).sum()),
        "fn": int((y & ~pred).sum()),
        "tn": int((~y & ~pred).sum()),
    }


def prf(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    c = confusion(y, pred)
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def _sorted_counts(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order].astype(np.int64)
    # keep the last index of each tied score run so thresholds are well defined
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(yy)[distinct]
    fp = (distinct + 1) - tp
    return s[distinct], tp, fp


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, like sklearn)."""
    _, tp, fp = _sorted_counts(y, scores)
    pos = int(y.sum())
    if pos == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney U formulation with average ranks for ties."""
    y = y.astype(bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Threshold (score >= t means anomaly) maximising F1 on a labelled validation set."""
    thr, tp, fp = _sorted_counts(y, scores)
    pos = int(y.sum())
    fn = pos - tp
    f1 = np.where(tp > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def report(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = scores >= threshold
    return {
        **prf(y, pred),
        **confusion(y, pred),
        "pr_auc": average_precision(y, scores),
        "roc_auc": roc_auc(y, scores),
        "threshold": float(threshold),
        "n": len(y),
        "positives": int(y.sum()),
    }
