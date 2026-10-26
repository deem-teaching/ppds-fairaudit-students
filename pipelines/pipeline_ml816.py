"""
Pipeline 1 – ML816 (Logistic Regression + Custom Threshold)

Based on the adult-income-prediction-ml-81-6.ipynb Kaggle notebook.

Uses a numeric-only feature subset (matching the notebook's choice of purely
continuous columns) and a custom prediction threshold of 0.6 to adjust the
precision/recall trade-off.

Feature selection:
- Included: age, fnlwgt, education.num, capital.gain, capital.loss,
            hours.per.week (exactly the notebook's feature list)
- Dropped:  all categorical (string) columns, mirroring the notebook's intent
            to use only purely numeric features
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline_auditing.base import MLPipeline

_FEATURE_COLS = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]


class ML816Pipeline(MLPipeline):
    """
    Logistic Regression pipeline with a custom 0.6 probability threshold.

    All selected features are numeric and are standardised before fitting.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scaler_: StandardScaler | None = None
        self.threshold: float = 0.6

    def encode_train(self, data: pd.DataFrame):
        cols = [c for c in _FEATURE_COLS if c in data.columns]
        self.scaler_ = StandardScaler()
        return pd.DataFrame(self.scaler_.fit_transform(data[cols].astype(float)), columns=cols, index=data.index)

    def encode_test(self, data: pd.DataFrame):
        cols = [c for c in _FEATURE_COLS if c in data.columns]
        return pd.DataFrame(self.scaler_.transform(data[cols].astype(float)), columns=cols, index=data.index)

    def get_model(self) -> LogisticRegression:
        return LogisticRegression(max_iter=32561, random_state=42)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Apply custom threshold to predicted probabilities."""
        if self._model is None:
            raise RuntimeError(
                "The pipeline has not been trained yet. Call train() before predict()."
            )
        proba = self._model.predict_proba(X)[:, 1]
        return (proba > self.threshold).astype(int)
