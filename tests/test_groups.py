"""Tests for pipeline_auditing.fairness.groups."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pipeline_auditing.fairness.groups import AdultIncomeGroups
from pipeline_auditing.fairness.auditor import FairnessAuditor

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "adult.csv"


# --------------------------------------------------------------------------- #
# Group values                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not DATA_PATH.exists(), reason="data/adult.csv not found")
def test_adult_groups_match_dataset():
    """Every attribute must be a column and every value must occur in it.

    Catches typos such as "female" instead of "Female", which would silently
    produce an empty unprivileged group.
    """
    df = pd.read_csv(DATA_PATH)
    groups = AdultIncomeGroups().get_groups()
    assert groups, "get_groups() must declare at least one protected attribute"
    for attr, values in groups.items():
        assert attr in df.columns, f"'{attr}' is not a column of the dataset"
        values = values if isinstance(values, list) else [values]
        present = set(df[attr].unique())
        missing = [v for v in values if v not in present]
        assert not missing, f"values {missing} do not occur in column '{attr}'"


# --------------------------------------------------------------------------- #
# Integration: callable instance produces same audit results as raw dict       #
# --------------------------------------------------------------------------- #


def test_callable_instance_matches_raw_dict(binary_matched_df):
    auditor = FairnessAuditor(positive_label=1)

    # Compare against the instance's own dict, so any group choice passes.
    raw_dict = AdultIncomeGroups().get_groups()
    attrs = list(raw_dict)

    report_dict = auditor.audit(
        data=binary_matched_df,
        protected_attributes=attrs,
        pipeline_name="test_dict",
        unprivileged_groups=raw_dict,
    )
    report_callable = auditor.audit(
        data=binary_matched_df,
        protected_attributes=attrs,
        pipeline_name="test_callable",
        unprivileged_groups=AdultIncomeGroups(),
    )

    for attr in attrs:
        r_dict = report_dict.attribute_results[attr]
        r_call = report_callable.attribute_results[attr]
        assert r_dict.unprivileged_group == r_call.unprivileged_group
        assert r_dict.privileged_group == r_call.privileged_group
        # nan_ok: values that are absent from the synthetic data give NaN metrics
        assert r_dict.demographic_parity_difference == pytest.approx(
            r_call.demographic_parity_difference, nan_ok=True
        )
        assert r_dict.equal_opportunity_difference == pytest.approx(
            r_call.equal_opportunity_difference, nan_ok=True
        )
        assert r_dict.predictive_parity_difference == pytest.approx(
            r_call.predictive_parity_difference, nan_ok=True
        )
