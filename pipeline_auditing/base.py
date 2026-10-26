"""Abstract base class for all ML pipelines in this practical."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator


class MLPipeline(ABC):
    """
    Abstract base class representing a complete ML pipeline for the
    Adult Income dataset.

    Subclasses implement the three abstract methods ``encode_train``,
    ``encode_test`` and ``get_model``. All other methods are already
    implemented and should **not** be overridden (exception: ``predict``,
    for post-processing such as a custom threshold).

    Typical usage (orchestrated by :class:`~pipeline_auditing.runner.PipelineRunner`)::

        pipeline = MyPipeline()
        raw      = pipeline.load_data("data/adult.csv")
        # ... train/test split performed by the runner ...
        train    = pipeline.filter_train(train_slice)
        X_train, y_train = pipeline.prepare_train(train)
        X_test,  y_test  = pipeline.prepare_test(test_slice)
        pipeline.train(X_train, y_train)
        y_pred   = pipeline.predict(X_test)
    """

    def __init__(self) -> None:
        self._model: BaseEstimator | None = None

    # ------------------------------------------------------------------ #
    # Abstract interface – must be implemented by every subclass         #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def encode_train(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit all encoders/scalers on training features and return X.

        Called by prepare_train. `data` has the income/target column already
        removed. Fit every transformer here and store it as an instance
        attribute so encode_test can reuse it without refitting.
        Returns a numeric feature DataFrame with the same index as `data`.
        """

    @abstractmethod
    def encode_test(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform test features using encoders fitted in encode_train.

        Called by prepare_test. `data` has the income/target column already
        removed. Must NOT call fit or fit_transform — doing so leaks test
        statistics into encoding.
        Returns a numeric feature DataFrame with the same index as `data`.
        """

    @abstractmethod
    def get_model(self) -> BaseEstimator:
        """
        Return a fresh, **untrained** scikit-learn estimator.

        Called by :meth:`train` immediately before fitting.  Instantiate
        the estimator with all desired hyper-parameters here so that every
        call to ``train`` starts from a clean slate.

        Returns
        -------
        BaseEstimator
            An unfitted scikit-learn-compatible estimator.
        """


    # ------------------------------------------------------------------ #
    # Concrete helpers – already implemented, no need to override        #
    # ------------------------------------------------------------------ #

    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load the raw dataset from *filepath* and return it as a DataFrame.

        Reads the file with ``pandas.read_csv`` and resets the index to a
        clean ``RangeIndex``, so every row keeps a unique integer index
        through all later steps (the matcher relies on this). The data is
        not modified.

        Parameters
        ----------
        filepath : str
            Path to the raw ``adult.csv`` file.

        Returns
        -------
        pd.DataFrame
            Raw data with a clean ``RangeIndex``.
        """
        return pd.read_csv(filepath).reset_index(drop=True)

    def compute_target(self, data: pd.DataFrame) -> pd.Series:
        """
        Return the shared binary target: ``income == ">50K"``.

        All pipelines use this definition so that fairness comparisons are
        made on an identical task.  The method intentionally has no
        side-effects and must not fit any encoders or scalers.

        Parameters
        ----------
        data : pd.DataFrame
            Data as returned by :meth:`load_data`.  Must contain an
            ``"income"`` column.

        Returns
        -------
        pd.Series
            Binary (0/1) integer Series with the same index as *data*,
            named ``"high_income"``.
        """
        y = (data["income"] == ">50K").astype(int)
        y.name = "high_income"
        return y
    
    def filter_train(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows that should not be used for **training**.

        The default implementation drops exact duplicate rows.

        This method is called by the runner **only on the training slice**
        after the train/test split; it is never applied to test data.

        Parameters
        ----------
        data : pd.DataFrame
            Training slice of the dataset.

        Returns
        -------
        pd.DataFrame
            Filtered training data (index is a subset of ``data.index``).
        """
        return data.drop_duplicates()

    def prepare_train(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract target, strip the income column, then encode training features.

        Calls compute_target on the full data slice, removes the income column,
        then delegates to encode_train for pure feature engineering.
        Called by the runner on the filtered training split.
        """
        y = self.compute_target(data)
        X = self.encode_train(data.drop(columns=["income"], errors="ignore"))
        return X, y

    def prepare_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract target, strip the income column, then encode test features.

        Calls compute_target on the full data slice, removes the income column,
        then delegates to encode_test for pure feature engineering.
        Called by the runner on the test split.
        """
        y = self.compute_target(data)
        X = self.encode_test(data.drop(columns=["income"], errors="ignore"))
        return X, y


    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> BaseEstimator:
        """
        Obtain a fresh model via :meth:`get_model`, fit it on
        ``(X_train, y_train)``, store it, and return it.

        Parameters
        ----------
        X_train : pd.DataFrame
        y_train : pd.Series

        Returns
        -------
        BaseEstimator
            The fitted estimator (also stored as ``self._model``).
        """
        self._model = self.get_model()
        self._model.fit(X_train, y_train)
        return self._model

    
    # this function is overridden by the example pipeline_ml816.py, it does not 
    # need to be overridden for any other pipeline
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return hard-label predictions for *X*.

        Raises
        ------
        RuntimeError
            If the pipeline has not been trained yet.
        """
        if self._model is None:
            raise RuntimeError(
                "The pipeline has not been trained yet. Call train() before predict()."
            )
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return class-probability estimates for *X*.

        Raises
        ------
        RuntimeError
            If the pipeline has not been trained yet.
        NotImplementedError
            If the underlying model does not support ``predict_proba``.
        """
        if self._model is None:
            raise RuntimeError(
                "The pipeline has not been trained yet. "
                "Call train() before predict_proba()."
            )
        if not hasattr(self._model, "predict_proba"):
            raise NotImplementedError(
                f"{type(self._model).__name__} does not support predict_proba."
            )
        return self._model.predict_proba(X)

    @property
    def model(self) -> BaseEstimator:
        """The fitted estimator (available after :meth:`train`)."""
        if self._model is None:
            raise RuntimeError("Pipeline has not been trained yet. Call train() first.")
        return self._model

    def __repr__(self) -> str:  # pragma: no cover
        trained = self._model is not None
        return f"{type(self).__name__}(trained={trained})"
