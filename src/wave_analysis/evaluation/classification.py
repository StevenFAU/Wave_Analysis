"""Classification metrics, including *ordinal* metrics for binned sea states.

Wave-height classes (e.g. Kim et al. 2023's 0.5 m bins; Choi et al. 2020's
low/high) are **ordered**. Confusing 1.25-1.75 m with 1.75-2.25 m is a smaller
error than confusing it with 2.75-3.25 m, and Kim et al. observe that most
errors fall in neighbouring classes. Plain accuracy ignores this, so ordinal
metrics (class-index MAE, within-one accuracy, quadratic-weighted kappa) are
reported alongside the standard set. Under class imbalance (Kim et al.'s
1.25-1.75 m class dominates) balanced accuracy and macro-F1 are preferred to
accuracy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: int | None = None
) -> NDArray[np.int64]:
    """Confusion matrix ``C[i, j]`` = count of true class ``i`` predicted as ``j`` (integer labels)."""
    a = np.asarray(y_true, dtype=np.int64)
    b = np.asarray(y_pred, dtype=np.int64)
    k = n_classes if n_classes is not None else int(max(a.max(initial=0), b.max(initial=0))) + 1
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (a, b), 1)
    return cm


def per_class_precision_recall_f1(cm: NDArray[np.int64]) -> dict[str, NDArray[np.float64]]:
    """Per-class precision, recall (sensitivity) and F1 from a confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = tp / cm.sum(axis=0)
        recall = tp / cm.sum(axis=1)
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": cm.sum(axis=1).astype(np.float64),
    }


def quadratic_weighted_kappa(cm: NDArray[np.int64]) -> float:
    """Cohen's kappa with quadratic weights (agreement beyond chance for ordinal labels)."""
    k = cm.shape[0]
    n = cm.sum()
    if n == 0:
        return float("nan")
    i, j = np.indices((k, k))
    w = (i - j) ** 2 / max((k - 1) ** 2, 1)
    expected = np.outer(cm.sum(axis=1), cm.sum(axis=0)) / n
    denom = (w * expected).sum()
    return float(1 - (w * cm).sum() / denom) if denom > 0 else float("nan")


def classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    n_classes: int | None = None,
    class_names: Sequence[str] | None = None,
    ordinal: bool = True,
) -> dict[str, object]:
    """Accuracy, balanced accuracy, macro/weighted P/R/F1, per-class table, and ordinal metrics."""
    a = np.asarray(y_true, dtype=np.int64)
    b = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(a, b, n_classes)
    pc = per_class_precision_recall_f1(cm)
    support = pc["support"]
    w = support / support.sum() if support.sum() else support
    out: dict[str, object] = {
        "n": int(a.size),
        "accuracy": float(np.mean(a == b)),
        "balanced_accuracy": float(np.nanmean(pc["recall"])),
        "precision_macro": float(np.nanmean(pc["precision"])),
        "recall_macro": float(np.nanmean(pc["recall"])),
        "f1_macro": float(np.nanmean(pc["f1"])),
        "precision_weighted": float(np.nansum(pc["precision"] * w)),
        "recall_weighted": float(np.nansum(pc["recall"] * w)),
        "f1_weighted": float(np.nansum(pc["f1"] * w)),
        "confusion_matrix": cm.tolist(),
        "per_class": {
            (class_names[i] if class_names else str(i)): {k: float(v[i]) for k, v in pc.items()}
            for i in range(cm.shape[0])
        },
    }
    if ordinal:
        out["ordinal_mae"] = float(np.mean(np.abs(a - b)))
        out["within_one_accuracy"] = float(np.mean(np.abs(a - b) <= 1))
        out["quadratic_weighted_kappa"] = quadratic_weighted_kappa(cm)
    return out


def bin_values(values: ArrayLike, edges: Sequence[float]) -> NDArray[np.int64]:
    """Map continuous values to class indices with left-closed bins ``[e_i, e_{i+1})``.

    Values outside ``[edges[0], edges[-1])`` map to -1 (excluded, not clipped),
    mirroring Choi et al. (2020), who excluded the intermediate 1.52-3.05 m band
    from their binary task rather than forcing it into a class.
    """
    v = np.asarray(values, dtype=np.float64)
    idx = np.digitize(v, edges) - 1
    idx[(v < edges[0]) | (v >= edges[-1]) | ~np.isfinite(v)] = -1
    return idx.astype(np.int64)
