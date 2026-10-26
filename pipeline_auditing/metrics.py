"""
Evaluation metrics: low-level metric utilities and per-group metric computation.

All functions operate on plain numpy arrays / Python scalars, without any
dependency on sklearn.metrics. The metric functions are used by the pipeline
runner (overall evaluation, Step 2); ``compute_group_metrics`` is used by the
fairness auditor (per-group evaluation, Step 4).

Every metric returns NaN when it is undefined (e.g. precision when there are
no positive predictions).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Confusion-matrix primitives                                                  #
# --------------------------------------------------------------------------- #


def confusion_matrix_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> Tuple[int, int, int, int]:
    """
    Return (TP, FP, TN, FN) for a binary classification problem.

    Parameters
    ----------
    y_true : array-like of shape (n,)
    y_pred : array-like of shape (n,)
    positive_label :
        The value treated as the positive class.

    Returns
    -------
    tp, fp, tn, fn : int
    """
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Derived scalar metrics                                                        #
# --------------------------------------------------------------------------- #


def safe_divide(
    numerator: float, denominator: float, default: float = float("nan")
) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    raise NotImplementedError("TODO")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correct predictions."""
    raise NotImplementedError("TODO")


def positive_rate(y_pred: np.ndarray, positive_label: Any = 1) -> float:
    """Fraction of predictions equal to *positive_label*."""
    raise NotImplementedError("TODO")


def true_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """TPR = TP / (TP + FN).  Also called recall or sensitivity."""
    raise NotImplementedError("TODO")


def false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """FPR = FP / (FP + TN).  Also called fall-out."""
    raise NotImplementedError("TODO")


def precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """Precision = TP / (TP + FP).  Also called positive predictive value."""
    raise NotImplementedError("TODO")


def f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_label: Any = 1,
) -> float:
    """F1 = 2 * precision * recall / (precision + recall)."""
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Per-group metrics                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class GroupMetrics:
    """
    Performance metrics computed for a single group within a protected attribute.

    Attributes
    ----------
    group_value : Any
        The value of the protected attribute that defines this group.
    n_samples : int
        Number of test-set rows belonging to this group.

    The remaining attributes are scalar metric values (``float``).
    ``NaN`` is used when a metric is undefined (e.g. TPR when there are no
    true positives in the group).
    """

    group_value: Any
    n_samples: int

    positive_rate: float

    tpr: float  # true positive rate (recall / sensitivity)
    fpr: float  # false positive rate
    prec: float  # precision (positive predictive value)
    acc: float  # accuracy


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_value: Any,
    positive_label: Any,
) -> GroupMetrics:
    """
    Compute all metrics of :class:`GroupMetrics` for a single group.

    A metric is NaN when it is undefined for the group (e.g. TPR when the
    group contains no actual positives).
    """
    raise NotImplementedError("TODO")
