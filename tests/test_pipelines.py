"""
Unit tests for the pipelines implemented in Step 1.

Each test verifies:
- The pipeline can be instantiated.
- prepare_train returns (X, y) with matching indices and correct dtypes.
- prepare_test uses fitted encoders without refitting, producing the same column count as prepare_train.
- get_model returns an unfitted estimator.
- The full load → filter_train → prepare_train → prepare_test → train → predict sequence works end-to-end.

prepare_train/prepare_test are the runner-facing interface; encode_train/encode_test are pure
feature engineering methods called by those concrete helpers in the base class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipelines import (
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
)
from pipeline_auditing.base import MLPipeline


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

# The pipelines you implement in Step 1 (the provided ones are tested elsewhere)
PIPELINE_CLASSES = [
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
]


# --------------------------------------------------------------------------- #
# Instantiation                                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_instantiation(cls):
    """Each pipeline can be instantiated and is a subclass of MLPipeline."""
    p = cls()
    assert isinstance(p, MLPipeline)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_predict_before_train_raises(cls, synthetic_csv_path):
    """Calling predict before train should raise RuntimeError."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, _ = p.prepare_train(filtered)
    with pytest.raises(RuntimeError, match="not been trained"):
        p.predict(X)


# --------------------------------------------------------------------------- #
# encode_train                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_preserves_index(cls, synthetic_csv_path):
    """X and y must share the same index as the filtered DataFrame."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, y = p.prepare_train(filtered)

    pd.testing.assert_index_equal(X.index, filtered.index)
    pd.testing.assert_index_equal(y.index, filtered.index)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_train_X_is_numeric(cls, synthetic_csv_path):
    """All columns of X must be numeric after encoding."""
    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)
    X, _ = p.prepare_train(filtered)

    non_numeric = X.select_dtypes(exclude="number").columns.tolist()
    assert non_numeric == [], f"Non-numeric columns in X after encoding: {non_numeric}"


# --------------------------------------------------------------------------- #
# get_model                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_get_model_returns_unfitted_estimator(cls):
    from sklearn.base import BaseEstimator

    p = cls()
    model = p.get_model()
    assert isinstance(model, BaseEstimator)
    # Fitted sklearn estimators have instance attributes ending with '_'.
    # Use vars() (not dir()) to check only instance attributes and avoid
    # false positives from class-level properties like _repr_html_.
    fitted_attrs = [
        a for a in vars(model) if a.endswith("_") and not a.startswith("__")
    ]
    assert fitted_attrs == [], (
        f"get_model() returned an already-fitted estimator "
        f"(fitted attributes: {fitted_attrs})"
    )


# --------------------------------------------------------------------------- #
# End-to-end pipeline run                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_pipeline_end_to_end(cls, synthetic_csv_path):
    """Full load → filter_train → prepare_train → prepare_test → train → predict should succeed."""
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    X_train, y_train = p.prepare_train(train_data)
    X_test, _ = p.prepare_test(test_data)

    p.train(X_train, y_train)
    preds = p.predict(X_test)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X_test)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_test_column_count_matches_train(cls, synthetic_csv_path):
    """prepare_test must produce the same number of columns as prepare_train."""
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    X_train, _ = p.prepare_train(train_data)
    X_test, _ = p.prepare_test(test_data)

    assert X_test.shape[1] == X_train.shape[1], (
        f"prepare_test produced {X_test.shape[1]} columns but prepare_train produced {X_train.shape[1]}"
    )


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_encode_test_does_not_refit(cls, synthetic_csv_path):
    """encode_test must only apply what encode_train fitted.

    Encoding a few test rows on their own must give the same result as
    encoding them together with the rest of the test set.
    """
    from sklearn.model_selection import train_test_split

    p = cls()
    raw = p.load_data(str(synthetic_csv_path))
    filtered = p.filter_train(raw)

    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)
    p.prepare_train(train_data)

    X_all, _ = p.prepare_test(test_data)
    X_few, _ = p.prepare_test(test_data.iloc[:5])

    pd.testing.assert_frame_equal(
        X_few, X_all.iloc[:5], check_dtype=False,
        obj="encode_test output (it depends on which test rows are encoded together)",
    )
