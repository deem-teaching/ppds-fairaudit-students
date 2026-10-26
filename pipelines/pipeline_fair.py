"""
Pipeline 6 – Your own fairer pipeline (optional bonus)

Rules (see the README): implement only encode_train, encode_test and
get_model (helper methods and model classes of your own are fine).

"""
from __future__ import annotations

import pandas as pd
from pipeline_auditing.base import MLPipeline


class FairPipeline(MLPipeline):
    """
    TODO (optional bonus): implement the abstract classes encode_train, encode_test, and get_model
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO")

    def encode_train(self, data: pd.DataFrame):
        raise NotImplementedError("TODO")

    def encode_test(self, data: pd.DataFrame):
        raise NotImplementedError("TODO")

    def get_model(self):
        raise NotImplementedError("TODO")
