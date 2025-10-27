"""
Metrics required by the CUTLASS estimators.

The implementations mirror scikit-learn's behaviour closely while keeping the
package lightweight and without a hard dependency on scikit-learn itself.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

__all__ = [
    "calculate_youden_j",
    "roc_auc_score",
    "precision_recall_curve",
]


def calculate_youden_j(
    y_true: Iterable[int] | np.ndarray,
    y_pred: Iterable[int] | np.ndarray,
) -> float:
    """
    Compute Youden's J statistic (sensitivity + specificity - 1).

    Parameters
    ----------
    y_true :
        Ground truth labels in {0, 1}.
    y_pred :
        Predicted labels in {0, 1}.
    """
    yt = np.asarray(y_true, dtype=bool)
    yp = np.asarray(y_pred, dtype=bool)

    tp = np.sum(yt & yp)
    tn = np.sum(~yt & ~yp)
    fp = np.sum(~yt & yp)
    fn = np.sum(yt & ~yp)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return float(sens + spec - 1.0)


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Tie-aware ranking with averaging; helper for ROC AUC."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    n = a.size
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def roc_auc_score(
    y_true: Iterable[int] | np.ndarray,
    y_score: Iterable[float] | np.ndarray,
) -> float:
    """
    Area under the ROC curve for binary classification.

    This is a lightweight re-implementation that matches scikit-learn's binary
    ROC AUC in functionality.
    """
    y = np.asarray(y_true, dtype=np.int64)
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError("roc_auc_score currently supports binary labels {0,1}.")
    scores = np.asarray(y_score, dtype=np.float64)

    order = np.argsort(scores, kind="mergesort")
    y_sorted = y[order]
    ranks = _rankdata_average(scores)

    pos = np.sum(y_sorted == 1)
    neg = np.sum(y_sorted == 0)
    if pos == 0 or neg == 0:
        raise ValueError("roc_auc_score is undefined when only one class is present.")

    sum_ranks_pos = np.sum(ranks[y == 1])
    u = sum_ranks_pos - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))


def precision_recall_curve(
    y_true: Iterable[int] | np.ndarray,
    probas_pred: Iterable[float] | np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Precision-recall curve for binary classification.

    Returns
    -------
    precision : ndarray
    recall : ndarray
    thresholds : ndarray
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    probas_pred = np.asarray(probas_pred, dtype=np.float64)

    order = np.argsort(probas_pred)[::-1]
    y = y_true[order]
    prob = probas_pred[order]

    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    fn = tp[-1] - tp

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    thresholds = prob

    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    thresholds = np.r_[1.0, thresholds]
    return precision, recall, thresholds

