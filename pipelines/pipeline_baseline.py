"""
Pipeline 0 – Random Baseline

A minimal pipeline implementation intended as a starting point for students
writing their own pipeline.  It shows the least amount of code needed to
satisfy the MLPipeline interface.

The model is sklearn's DummyClassifier with strategy="uniform", which
ignores all features and produces random predictions drawn uniformly over
the label set.  Because it learns nothing from the data, its fairness
metrics reflect what a purely random classifier looks like — a useful
baseline: if a real model is *less* fair than this one, that is a strong
signal of a problem.

What this pipeline deliberately leaves out (and why):
- No feature engineering   — encode_train/encode_test return the raw numeric columns as-is
- No categorical encoding  — the categorical (string) columns are simply dropped
- No missing-value handling — "?" only occurs in the dropped categorical columns
- No scaling               — not needed by DummyClassifier
- No custom filter_train   — uses the base-class default (drop_duplicates)
"""
from __future__ import annotations

import pandas as pd
from sklearn.dummy import DummyClassifier

from pipeline_auditing.base import MLPipeline

_FEATURE_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]


class RandomBaselinePipeline(MLPipeline):
    """
    Minimal pipeline that predicts at random.

    Serves as a reference baseline and as a worked example of the smallest
    possible MLPipeline implementation.
    """

    # ------------------------------------------------------------------ #

    def encode_train(self, data: pd.DataFrame):
        return data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

    def encode_test(self, data: pd.DataFrame):
        return data[[c for c in _FEATURE_COLS if c in data.columns]].copy()

    def get_model(self) -> DummyClassifier:
        return DummyClassifier(strategy="uniform", random_state=0)
