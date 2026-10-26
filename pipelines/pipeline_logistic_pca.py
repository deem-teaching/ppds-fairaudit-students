"""
Pipeline 4 – Logistic Regression + PCA

Based on the eda-logistic-regression-pca.ipynb Kaggle notebook.

"""
from __future__ import annotations
import pandas as pd
from pipeline_auditing.base import MLPipeline


class LogisticPCAPipeline(MLPipeline):
    """
    TODO: implement the abstract classes encode_train, encode_test, and get_model
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO")

    def encode_train(self, data: pd.DataFrame):
        raise NotImplementedError("TODO")

    def encode_test(self, data: pd.DataFrame):
        raise NotImplementedError("TODO")

    def get_model(self):
        raise NotImplementedError("TODO")
