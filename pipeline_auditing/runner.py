"""
Pipeline runner: load → split → filter_train → encode → train → evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .base import MLPipeline
from .metrics import accuracy, f1, precision, true_positive_rate


# --------------------------------------------------------------------------- #
# Result container                                                              #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineResult:
    """
    Container for every artefact produced during a single pipeline run.

    Attributes
    ----------
    raw_data : pd.DataFrame
        Data as returned by ``pipeline.load_data``.
    X : pd.DataFrame
        Encoded feature matrix covering the post-filter training rows
        (after ``filter_train``) plus all test rows. Reconstructed as
        ``concat([X_train, X_test])``.
    y : pd.Series
        Target vector aligned to ``X``. Reconstructed as
        ``concat([y_train, y_test])``.
    X_train, X_test : pd.DataFrame
        Train and test feature splits (index-preserving).
    y_train, y_test : pd.Series
        Train and test target splits (index-preserving).
    y_pred : np.ndarray
        Model predictions on the **test** split.
    metrics : dict
        Evaluation metrics computed on the test split.
    pipeline : MLPipeline
        The fitted pipeline instance.
    name : str
        Optional display name for logging / reporting.
    """

    raw_data: pd.DataFrame

    X: pd.DataFrame
    y: pd.Series

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    y_pred: np.ndarray

    pipeline: MLPipeline = field(repr=False)
    metrics: Dict[str, Any] = field(default_factory=dict)
    name: str = ""
    n_train_before_filter: int = 0
    n_train_after_filter: int = 0


# --------------------------------------------------------------------------- #
# Runner                                                                        #
# --------------------------------------------------------------------------- #


class PipelineRunner:
    """
    Orchestrates a complete ML pipeline run.

    The runner is responsible for splitting the data, calling the pipeline's
    stages in the correct order, and computing evaluation metrics.  It is
    intentionally kept separate from the pipeline itself so that the same
    runner can evaluate multiple pipelines under identical conditions.

    Crucially, ``filter_train`` is called **only on the training slice**;
    the test set always contains every row that was not selected for training.

    Parameters
    ----------
    test_size : float
        Fraction of all rows held out for evaluation (default 0.2).
    random_state : int
        Random seed used for the train/test split (default 42).
    stratify : bool
        Whether to stratify the split by the target label (default ``True``).
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def run(
        self,
        pipeline: MLPipeline,
        filepath: str,
        name: Optional[str] = None,
    ) -> PipelineResult:
        """
        Execute *pipeline* end-to-end and return a :class:`PipelineResult`.

        Parameters
        ----------
        pipeline : MLPipeline
            A *fresh* (not yet trained) pipeline instance.
        filepath : str
            Path to the raw ``adult.csv`` data file.
        name : str, optional
            Human-readable label stored in the result.

        Returns
        -------
        PipelineResult
        """
        run_name = name or type(pipeline).__name__

        # 1. Load
        raw_data = ...

        # 2. Compute target on full data (pure threshold, no fitting).
        #    Used only for stratified splitting; no feature encoders are touched.
        y_full = ...

        if len(y_full) != len(raw_data):
            raise ValueError(
                "compute_target must return a Series with the same number of "
                f"rows as raw_data (expected {len(raw_data)}, got {len(y_full)})."
            )

        # 3. Split data indices BEFORE encoding to avoid leakage.
        stratify_labels = ...
        train_idx, test_idx = ...

        # 4. Apply training-only filters (e.g. drop_duplicates) to the train
        #    slice only.  The test slice is never filtered.
        train_slice =  ...
        train_data = ...
        test_data = ...

        # 5. Encode training data: fits all encoders/scalers, returns (X_train, y_train).
        X_train, y_train = ...

        # 6. Encode test data: applies encoders fitted in step 5, no refitting.
        X_test, y_test = ...

        # Reconstruct full X and y for the result object.
        X = ...
        y = ...

        # 7. Train
        ...

        # 8. Predict
        y_pred = ...

        # 9. Evaluate
        metrics = ...

        return PipelineResult(
            raw_data=raw_data,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            metrics=metrics,
            pipeline=pipeline,
            name=run_name,
            n_train_before_filter=int(len(train_slice)),
            n_train_after_filter=int(len(train_data)),
        )


    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_metrics(
        y_test: pd.Series,
        y_pred: np.ndarray,
        pipeline: MLPipeline,
        X_test: pd.DataFrame,
    ) -> Dict[str, Any]:
        # Metric functions from Step 2a (pipeline_auditing/metrics.py);
        # undefined metrics are NaN.
        y_true = np.asarray(y_test)
        y_pred = np.asarray(y_pred)

        metrics: Dict[str, Any] = {
            "accuracy": float(accuracy(y_true, y_pred)),
            "f1": float(f1(y_true, y_pred)),
            "precision": float(precision(y_true, y_pred)),
            "recall": float(true_positive_rate(y_true, y_pred)),
            "n_test": int(len(y_test)),
        }

        # ROC-AUC (from scikit-learn; requires predict_proba support and
        # both classes in the test set)
        try:
            proba = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except (NotImplementedError, AttributeError, ValueError):
            pass  # not all models support probability estimates

        return metrics
