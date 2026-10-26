"""
Unit tests for the metric functions in pipeline_auditing/metrics.py (Step 2a).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from pipeline_auditing.metrics import (
    accuracy,
    confusion_matrix_counts,
    f1,
    false_positive_rate,
    positive_rate,
    precision,
    safe_divide,
    true_positive_rate,
)


# =========================================================================== #
# Metric utility tests                                                          #
# =========================================================================== #


class TestConfusionMatrixCounts:
    def test_perfect_predictions(self):
        y = np.array([1, 1, 0, 0])
        tp, fp, tn, fn = confusion_matrix_counts(y, y)
        assert (tp, fp, tn, fn) == (2, 0, 2, 0)

    def test_all_wrong(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([0, 0, 1, 1])
        tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
        assert (tp, fp, tn, fn) == (0, 2, 0, 2)

    def test_all_predicted_positive(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1])
        tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred)
        assert (tp, fp, tn, fn) == (2, 2, 0, 0)

    def test_custom_positive_label(self):
        y_true = np.array(["A", "B", "A", "B"])
        y_pred = np.array(["A", "A", "B", "B"])
        tp, fp, tn, fn = confusion_matrix_counts(y_true, y_pred, positive_label="A")
        assert (tp, fp, tn, fn) == (1, 1, 1, 1)


class TestMetricFunctions:
    def test_accuracy_perfect(self):
        y = np.array([0, 1, 1, 0])
        assert accuracy(y, y) == 1.0

    def test_accuracy_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        assert accuracy(y_true, y_pred) == 0.0

    def test_positive_rate(self):
        y = np.array([1, 1, 0, 0, 0])
        assert positive_rate(y) == pytest.approx(0.4)

    def test_tpr_perfect(self):
        y = np.array([1, 1, 0, 0])
        assert true_positive_rate(y, y) == 1.0

    def test_tpr_no_true_positives(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        result = true_positive_rate(y_true, y_pred)
        assert math.isnan(result)

    def test_fpr_no_true_negatives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        result = false_positive_rate(y_true, y_pred)
        assert math.isnan(result)

    def test_precision_zero_denominator(self):
        y_true = np.array([0, 0])
        y_pred = np.array([0, 0])
        result = precision(y_true, y_pred)
        assert math.isnan(result)

    def test_tpr_partial(self):
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([1, 1, 0, 0, 0])
        assert true_positive_rate(y_true, y_pred) == pytest.approx(0.5)

    def test_fpr_value(self):
        y_true = np.array([0, 0, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 0, 1])
        assert false_positive_rate(y_true, y_pred) == pytest.approx(0.25)

    def test_precision_value(self):
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0])
        assert precision(y_true, y_pred) == pytest.approx(2 / 3)

    def test_f1_score(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        # TP=1, FP=1, FN=1 → precision=0.5, recall=0.5 → F1=0.5
        assert f1(y_true, y_pred) == pytest.approx(0.5)

    def test_f1_undefined(self):
        # No actual and no predicted positives: precision and recall are undefined
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        assert math.isnan(f1(y_true, y_pred))

    def test_safe_divide_zero_denominator(self):
        assert math.isnan(safe_divide(1.0, 0.0))

    def test_safe_divide_custom_default(self):
        assert safe_divide(1.0, 0.0, default=-1.0) == -1.0

    def test_safe_divide_normal(self):
        assert safe_divide(3.0, 4.0) == pytest.approx(0.75)


# =========================================================================== #
# Cross-check against scikit-learn                                              #
# =========================================================================== #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_metrics_agree_with_sklearn(seed):
    """On random labels (all metrics defined) your functions match scikit-learn."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    assert confusion_matrix_counts(y_true, y_pred) == (tp, fp, tn, fn)
    assert accuracy(y_true, y_pred) == pytest.approx(accuracy_score(y_true, y_pred))
    assert precision(y_true, y_pred) == pytest.approx(precision_score(y_true, y_pred))
    assert true_positive_rate(y_true, y_pred) == pytest.approx(recall_score(y_true, y_pred))
    assert false_positive_rate(y_true, y_pred) == pytest.approx(fp / (fp + tn))
    assert f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred))
    assert positive_rate(y_pred) == pytest.approx(np.mean(y_pred == 1))
