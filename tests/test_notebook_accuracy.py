"""
Sanity check for Step 1 on the real data: every refactored pipeline should
reach about the accuracy reported in its notebook (within about one percentage
point, see the README). Runs on the full dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline_auditing.runner import PipelineRunner
from pipelines import (
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adult.csv"

TOLERANCE = 0.01

# Accuracy as printed in each notebook's output
NOTEBOOK_ACCURACY = [
    # adult-income-prediction-with-logistic-regression.ipynb
    (LogisticRegressionPipeline, 0.8474),
    # classification-analysis-with-xgboost.ipynb (printed rounded)
    (XGBoostPipeline, 0.87),
    # eda-logistic-regression-pca.ipynb (logistic regression on all features)
    (LogisticPCAPipeline, 0.8218),
    # effective-adult-income-classification-pipeline.ipynb (Random Forest)
    (EffectivePipeline, 0.8486),
]


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/adult.csv not found")
@pytest.mark.parametrize(
    "cls, notebook_accuracy", NOTEBOOK_ACCURACY, ids=[c.__name__ for c, _ in NOTEBOOK_ACCURACY]
)
def test_accuracy_close_to_notebook(cls, notebook_accuracy):
    result = PipelineRunner().run(cls(), str(DATA_PATH))
    accuracy = result.metrics["accuracy"]
    assert abs(accuracy - notebook_accuracy) <= TOLERANCE, (
        f"{cls.__name__}: accuracy {accuracy:.4f} differs from the notebook's "
        f"{notebook_accuracy:.4f} by more than {TOLERANCE:.2f}"
    )
