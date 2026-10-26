"""
Unit tests for PipelineRunner and PipelineResult.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipeline_auditing.runner import PipelineResult, PipelineRunner
from pipelines import (
    RandomBaselinePipeline,
    ML816Pipeline,
)

PIPELINE_CLASSES = [
    RandomBaselinePipeline,
    ML816Pipeline,
]


# --------------------------------------------------------------------------- #
# Successful runs                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_runner_returns_pipeline_result(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert isinstance(result, PipelineResult)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_has_correct_split_sizes(cls, synthetic_csv_path):
    """Test split should be ~ test_size fraction of raw data."""
    runner = PipelineRunner(test_size=0.2)
    result = runner.run(cls(), str(synthetic_csv_path))

    n_raw = len(result.raw_data)
    n_test = len(result.X_test)
    actual_fraction = n_test / n_raw
    assert abs(actual_fraction - 0.2) < 0.05, (
        f"Test fraction {actual_fraction:.2f} deviates too much from 0.20"
    )


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_train_test_indices_disjoint(cls, synthetic_csv_path):
    """Training and test indices must be disjoint."""
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))

    train_idx = set(result.X_train.index)
    test_idx = set(result.X_test.index)
    assert train_idx.isdisjoint(test_idx), "Training and test sets share row indices."


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_predictions_length(cls, synthetic_csv_path):
    """y_pred must have the same length as X_test."""
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert len(result.y_pred) == len(result.X_test)


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_metrics_contain_accuracy(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert "accuracy" in result.metrics
    assert 0.0 <= result.metrics["accuracy"] <= 1.0


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_metrics_contain_f1(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    assert "f1" in result.metrics


@pytest.mark.parametrize("cls", PIPELINE_CLASSES)
def test_result_X_index_subset_of_raw(cls, synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(cls(), str(synthetic_csv_path))
    # X covers train (possibly deduplicated) + test; test rows are always all present
    assert set(result.X.index).issubset(set(result.raw_data.index))
    assert set(result.X_test.index).issubset(set(result.raw_data.index))


# --------------------------------------------------------------------------- #
# PipelineResult derived properties                                             #
# --------------------------------------------------------------------------- #


def test_result_name_stored(synthetic_csv_path):
    runner = PipelineRunner()
    result = runner.run(
        ML816Pipeline(), str(synthetic_csv_path), name="TestPipeline"
    )
    assert result.name == "TestPipeline"


# --------------------------------------------------------------------------- #
# Reproducibility                                                               #
# --------------------------------------------------------------------------- #


def test_runner_is_reproducible(synthetic_csv_path):
    """Two runs with the same seed must produce identical predictions."""
    runner = PipelineRunner(random_state=99)
    r1 = runner.run(ML816Pipeline(), str(synthetic_csv_path))
    r2 = runner.run(ML816Pipeline(), str(synthetic_csv_path))
    np.testing.assert_array_equal(r1.y_pred, r2.y_pred)


# --------------------------------------------------------------------------- #
# Split, filtering and encoding                                                 #
# --------------------------------------------------------------------------- #


class RecordingPipeline(ML816Pipeline):
    """ML816Pipeline that records which rows each stage receives."""

    def __init__(self) -> None:
        super().__init__()
        self.filter_calls: list[pd.Index] = []
        self.encode_train_index: pd.Index | None = None
        self.encode_test_index: pd.Index | None = None

    def filter_train(self, data):
        self.filter_calls.append(data.index)
        return super().filter_train(data)

    def encode_train(self, data):
        self.encode_train_index = data.index
        return super().encode_train(data)

    def encode_test(self, data):
        self.encode_test_index = data.index
        return super().encode_test(data)


def run_recording(synthetic_csv_path, **runner_kwargs):
    pipeline = RecordingPipeline()
    result = PipelineRunner(**runner_kwargs).run(pipeline, str(synthetic_csv_path))
    return pipeline, result


def test_test_split_has_requested_size(synthetic_csv_path):
    """The test split is test_size of *all* rows (nothing is filtered before splitting)."""
    _, result = run_recording(synthetic_csv_path, test_size=0.2)
    assert len(result.X_test) == math.ceil(0.2 * len(result.raw_data))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_split_is_stratified(seed, synthetic_csv_path):
    """The share of high earners in the test split matches the full data."""
    pipeline, result = run_recording(synthetic_csv_path, random_state=seed)
    y_all = pipeline.compute_target(result.raw_data)
    assert abs(result.y_test.mean() - y_all.mean()) <= 1 / len(result.y_test)


def test_filter_train_only_sees_training_slice(synthetic_csv_path):
    """filter_train is called exactly once, on all rows that are not in the test split."""
    pipeline, result = run_recording(synthetic_csv_path)
    assert len(pipeline.filter_calls) == 1, "filter_train must be called exactly once"
    expected = set(result.raw_data.index) - set(result.X_test.index)
    assert set(pipeline.filter_calls[0]) == expected


def test_training_uses_filtered_rows(synthetic_csv_path):
    """encode_train and X_train contain exactly the rows that survived filter_train."""
    pipeline, result = run_recording(synthetic_csv_path)
    training_slice = result.raw_data.loc[pipeline.filter_calls[0]]
    expected = set(training_slice.drop_duplicates().index)
    assert len(expected) < len(training_slice), "fixture should contain duplicates in the training slice"
    assert set(pipeline.encode_train_index) == expected
    assert set(result.X_train.index) == expected
    assert result.n_train_after_filter < result.n_train_before_filter


def test_test_split_is_not_filtered(synthetic_csv_path):
    """encode_test and X_test contain every row that is not in the training slice."""
    pipeline, result = run_recording(synthetic_csv_path)
    expected = set(result.raw_data.index) - set(pipeline.filter_calls[0])
    assert set(pipeline.encode_test_index) == expected
    assert set(result.X_test.index) == expected


def test_labels_match_target(synthetic_csv_path):
    """y_train and y_test are the target values of the corresponding rows."""
    pipeline, result = run_recording(synthetic_csv_path)
    y_all = pipeline.compute_target(result.raw_data)
    np.testing.assert_array_equal(result.y_train.to_numpy(), y_all.loc[result.y_train.index].to_numpy())
    np.testing.assert_array_equal(result.y_test.to_numpy(), y_all.loc[result.y_test.index].to_numpy())


def test_X_and_y_cover_train_and_test(synthetic_csv_path):
    """result.X and result.y contain the training and the test rows."""
    _, result = run_recording(synthetic_csv_path)
    expected = set(result.X_train.index) | set(result.X_test.index)
    assert len(result.X) == len(result.X_train) + len(result.X_test)
    assert set(result.X.index) == expected
    assert set(result.y.index) == expected


def test_metrics_contain_all_scores(synthetic_csv_path):
    _, result = run_recording(synthetic_csv_path)
    for key in ["accuracy", "f1", "precision", "recall", "roc_auc", "n_test"]:
        assert key in result.metrics, f"metric '{key}' missing"
