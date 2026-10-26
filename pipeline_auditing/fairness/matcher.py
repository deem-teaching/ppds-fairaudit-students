"""
Prediction matcher: maps test-set predictions back to the original raw data.

After a pipeline run, three disjoint subsets of the raw data exist:

1. **Filtered out** – removed by ``filter_train`` (e.g. duplicate rows);
   neither a ground-truth label nor a prediction is available for these rows.
2. **Training set** – used to train the model; no test-set prediction exists
   (unless ``include_train=True`` is set).
3. **Test set** – held out for evaluation; both the ground-truth label and
   the model prediction are available.

This class constructs a unified DataFrame that annotates each row of the raw
data with its membership in the subsets above, enabling the
:class:`~pipeline_auditing.fairness.auditor.FairnessAuditor` to slice by any
original feature (including protected attributes) and compute group-level
fairness metrics.

Usage
-----
::

    matcher = PredictionMatcher()
    matched_df = matcher.match(result)

    # Restrict to the auditable subset (test rows with both y_true and y_pred)
    auditable = PredictionMatcher.get_auditable_subset(matched_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline_auditing.runner import PipelineResult

# Column name constants (imported by the auditor as well)
COL_Y_TRUE = "y_true"
COL_Y_PRED = "y_pred"
COL_IN_MODEL = "in_model"
COL_IN_TEST = "in_test"
COL_IN_TRAIN = "in_train"


class PredictionMatcher:
    """
    Merges pipeline predictions back onto the original raw dataset.
    """

    def match(
        self,
        result: PipelineResult,
        include_train: bool = False,
    ) -> pd.DataFrame:
        """
        Return ``result.raw_data`` augmented with prediction metadata.

        Parameters
        ----------
        result : PipelineResult
            The output of a completed
            :class:`~pipeline_auditing.runner.PipelineRunner` run.
        include_train : bool
            If ``True``, the model is also called on ``X_train`` and the
            resulting predictions are stored in the ``y_pred`` column for
            training-set rows.  Useful for coverage analysis but note that
            training-set predictions are optimistically biased (the model
            has seen these examples during fitting).

        Returns
        -------
        pd.DataFrame
            A copy of ``result.raw_data`` with the following additional
            columns appended:

            * ``y_true``   – ground-truth label; ``NaN`` for rows removed
              by ``filter_train`` (e.g. duplicates).
            * ``y_pred``   – model prediction; ``NaN`` for rows not in
              the test set (unless ``include_train=True``).
            * ``in_model`` – ``True`` if the row appears in either the
              training or test split (i.e. was not removed by
              ``filter_train``).
            * ``in_test``  – ``True`` if the row is in the test split.
            * ``in_train`` – ``True`` if the row is in the training split.
        """
        df = result.raw_data.copy()

        # TODO: implement the matching that appends the 5 additional columns

        return df

    @staticmethod
    def get_auditable_subset(matched_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter *matched_df* to the rows that are usable for fairness auditing.

        A row is "auditable" when it is in the test split **and** both its
        ground-truth label and its prediction are non-missing.

        Parameters
        ----------
        matched_df : pd.DataFrame
            Output of :meth:`match`.

        Returns
        -------
        pd.DataFrame
            The subset of ``matched_df`` where auditing is possible.
        """
        raise NotImplementedError("TODO")
