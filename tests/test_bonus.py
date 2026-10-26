"""
Tests for the optional bonus pipeline (pipelines/pipeline_fair.py).

All tests are skipped until you start implementing FairPipeline.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

import pipelines.pipeline_fair as fair_module
from pipeline_auditing.base import MLPipeline
from pipeline_auditing.runner import PipelineRunner
from pipelines import FairPipeline

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adult.csv"

MIN_ACCURACY = 0.80
MIN_F1 = 0.50

# The only methods of MLPipeline your pipeline may define itself
YOUR_METHODS = {"__init__", "encode_train", "encode_test", "get_model"}


def _attempted() -> bool:
    try:
        FairPipeline()
    except NotImplementedError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _attempted(), reason="optional bonus not attempted")


# --------------------------------------------------------------------------- #
# Rules                                                                         #
# --------------------------------------------------------------------------- #


def test_only_allowed_methods_are_overridden():
    """All provided MLPipeline methods (filter_train, predict, ...) stay unchanged."""
    for name in vars(MLPipeline):
        if name.startswith("_") or name in YOUR_METHODS:
            continue
        assert getattr(FairPipeline, name) is getattr(MLPipeline, name), (
            f"FairPipeline must not override '{name}'"
        )


def test_no_fairness_library():
    source = inspect.getsource(fair_module).lower()
    for library in ["fairlearn", "aif360"]:
        assert library not in source, f"fairness library '{library}' is not allowed"


# --------------------------------------------------------------------------- #
# Pipeline contract (as for the Step 1 pipelines)                               #
# --------------------------------------------------------------------------- #


def test_encoding_contract(synthetic_csv_path):
    p = FairPipeline()
    filtered = p.filter_train(p.load_data(str(synthetic_csv_path)))
    train_data, test_data = train_test_split(filtered, test_size=0.2, random_state=42)

    X_train, y_train = p.prepare_train(train_data)
    X_test, _ = p.prepare_test(test_data)
    assert X_train.select_dtypes(exclude="number").columns.tolist() == []
    pd.testing.assert_index_equal(X_train.index, train_data.index)
    pd.testing.assert_index_equal(X_test.index, test_data.index)
    assert list(X_test.columns) == list(X_train.columns)

    X_few, _ = p.prepare_test(test_data.iloc[:5])
    pd.testing.assert_frame_equal(X_few, X_test.iloc[:5], check_dtype=False)

    p.train(X_train, y_train)
    assert len(p.predict(X_test)) == len(X_test)


# --------------------------------------------------------------------------- #
# Minimum quality on the real data                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/adult.csv not found")
def test_minimum_accuracy_and_f1():
    """A fairer pipeline must stay useful (predicting '<=50K' for everyone is 'fair' but useless)."""
    metrics = PipelineRunner().run(FairPipeline(), str(DATA_PATH)).metrics
    assert metrics["accuracy"] >= MIN_ACCURACY, f"accuracy {metrics['accuracy']:.4f} < {MIN_ACCURACY}"
    assert not math.isnan(metrics["f1"]) and metrics["f1"] >= MIN_F1, f"F1 {metrics['f1']:.4f} < {MIN_F1}"
