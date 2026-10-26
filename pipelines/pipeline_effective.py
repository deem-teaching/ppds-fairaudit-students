"""
Pipeline 5 – Effective Pipeline (Random Forest)

Based on the effective-adult-income-classification-pipeline.ipynb Kaggle notebook.

"""
from __future__ import annotations
import pandas as pd
from pipeline_auditing.base import MLPipeline


class EffectivePipeline(MLPipeline):
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
