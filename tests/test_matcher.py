"""
Unit tests for PredictionMatcher.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_auditing.fairness.matcher import (
    PredictionMatcher,
    COL_IN_MODEL,
    COL_IN_TEST,
    COL_IN_TRAIN,
    COL_Y_PRED,
    COL_Y_TRUE,
)
from pipeline_auditing.runner import PipelineRunner
from pipelines import ML816Pipeline


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def run_pipeline(cls, synthetic_csv_path):
    runner = PipelineRunner(random_state=42)
    return runner.run(cls(), str(synthetic_csv_path))


# --------------------------------------------------------------------------- #
# Column presence                                                               #
# --------------------------------------------------------------------------- #


def test_match_adds_required_columns(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    for col in [COL_Y_TRUE, COL_Y_PRED, COL_IN_MODEL, COL_IN_TEST, COL_IN_TRAIN]:
        assert col in matched.columns, f"Column '{col}' missing from matched DataFrame"


def test_match_preserves_raw_columns(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    raw_cols = set(result.raw_data.columns)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    assert raw_cols.issubset(set(matched.columns))


# --------------------------------------------------------------------------- #
# Row count                                                                     #
# --------------------------------------------------------------------------- #


def test_match_has_same_length_as_raw(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    assert len(matched) == len(result.raw_data)


# --------------------------------------------------------------------------- #
# Membership flags                                                              #
# --------------------------------------------------------------------------- #


def test_in_model_flag_correct(synthetic_csv_path):
    """in_model should be True exactly for rows in X_train or X_test."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    expected_in_model = (
        matched.index.isin(result.X_train.index) | matched.index.isin(result.X_test.index)
    )
    pd.testing.assert_series_equal(
        matched[COL_IN_MODEL].reset_index(drop=True),
        pd.Series(expected_in_model, name=COL_IN_MODEL),
    )


def test_in_test_flag_correct(synthetic_csv_path):
    """in_test should be True exactly for rows in the test split."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    expected_in_test = matched.index.isin(result.X_test.index)
    pd.testing.assert_series_equal(
        matched[COL_IN_TEST].reset_index(drop=True),
        pd.Series(expected_in_test, name=COL_IN_TEST),
    )


def test_train_and_test_flags_disjoint(synthetic_csv_path):
    """No row should be simultaneously in train and test."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    assert not (matched[COL_IN_TRAIN] & matched[COL_IN_TEST]).any()


# --------------------------------------------------------------------------- #
# y_true column                                                                 #
# --------------------------------------------------------------------------- #


def test_y_true_not_nan_for_model_rows(synthetic_csv_path):
    """Rows that survived filtering should have a valid y_true."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    model_rows = matched[matched[COL_IN_MODEL]]
    assert model_rows[COL_Y_TRUE].notna().all(), (
        "All rows that survived filtering should have a valid y_true."
    )


# --------------------------------------------------------------------------- #
# y_pred column                                                                 #
# --------------------------------------------------------------------------- #


def test_y_pred_nan_for_non_test_rows(synthetic_csv_path):
    """Rows not in the test split must have NaN in y_pred (default behaviour)."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result, include_train=False)

    non_test_mask = ~matched[COL_IN_TEST]
    assert matched.loc[non_test_mask, COL_Y_PRED].isna().all()


def test_y_pred_not_nan_for_test_rows(synthetic_csv_path):
    """Test-set rows must have a non-NaN prediction."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    test_rows = matched[matched[COL_IN_TEST]]
    assert test_rows[COL_Y_PRED].notna().all()


def test_y_pred_values_match_runner_output(synthetic_csv_path):
    """Predictions in matched DF must match result.y_pred exactly."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)

    test_preds = matched.loc[result.X_test.index, COL_Y_PRED].values
    np.testing.assert_array_equal(test_preds, result.y_pred)


# --------------------------------------------------------------------------- #
# include_train=True                                                            #
# --------------------------------------------------------------------------- #


def test_include_train_fills_train_predictions(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result, include_train=True)

    train_rows = matched[matched[COL_IN_TRAIN]]
    assert train_rows[COL_Y_PRED].notna().all(), (
        "When include_train=True, training rows should have predictions."
    )


# --------------------------------------------------------------------------- #
# get_auditable_subset                                                          #
# --------------------------------------------------------------------------- #


def test_auditable_subset_all_have_predictions(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    auditable = PredictionMatcher.get_auditable_subset(matched)

    assert auditable[COL_Y_TRUE].notna().all()
    assert auditable[COL_Y_PRED].notna().all()


def test_auditable_subset_equals_test_split(synthetic_csv_path):
    """Auditable subset should contain exactly the test-split rows."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    auditable = PredictionMatcher.get_auditable_subset(matched)

    assert set(auditable.index) == set(result.X_test.index)


def test_auditable_subset_size_equals_n_test(synthetic_csv_path):
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matcher = PredictionMatcher()
    matched = matcher.match(result)
    auditable = PredictionMatcher.get_auditable_subset(matched)

    assert len(auditable) == result.metrics["n_test"]


# --------------------------------------------------------------------------- #
# Rows removed by filter_train, labels                                          #
# --------------------------------------------------------------------------- #


def test_filtered_rows_are_marked(synthetic_csv_path):
    """Rows removed by filter_train are in no split and have neither label nor prediction."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matched = PredictionMatcher().match(result)

    in_a_split = matched.index.isin(result.X_train.index) | matched.index.isin(result.X_test.index)
    filtered = matched[~in_a_split]
    assert len(filtered) > 0, "fixture should contain rows removed by filter_train"
    assert not filtered[COL_IN_MODEL].any()
    assert not filtered[COL_IN_TRAIN].any()
    assert not filtered[COL_IN_TEST].any()
    assert filtered[COL_Y_TRUE].isna().all()
    assert filtered[COL_Y_PRED].isna().all()


def test_in_train_flag_correct(synthetic_csv_path):
    """in_train should be True exactly for rows in the (filtered) training split."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matched = PredictionMatcher().match(result)
    np.testing.assert_array_equal(
        matched[COL_IN_TRAIN].to_numpy(), matched.index.isin(result.X_train.index)
    )


def test_y_true_values_match_labels(synthetic_csv_path):
    """y_true holds the label of every row that was used in the model."""
    result = run_pipeline(ML816Pipeline, synthetic_csv_path)
    matched = PredictionMatcher().match(result)
    np.testing.assert_array_equal(
        matched.loc[result.y.index, COL_Y_TRUE].to_numpy(), result.y.to_numpy()
    )
