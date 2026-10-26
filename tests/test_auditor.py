"""
Unit tests for compute_group_metrics, FairnessAuditor and AuditReport (Step 4b).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from pipeline_auditing.fairness.auditor import (
    AuditReport,
    FairnessAuditor,
)
from pipeline_auditing.metrics import compute_group_metrics


# =========================================================================== #
# Per-group metrics                                                             #
# =========================================================================== #


class TestComputeGroupMetrics:
    def test_group_metrics(self):
        # TP = 2, FN = 1, FP = 1, TN = 4
        y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])
        gm = compute_group_metrics(
            y_true, y_pred, group_value="A", positive_label=1
        )
        assert gm.group_value == "A"
        assert gm.n_samples == 8
        assert gm.positive_rate == pytest.approx(3 / 8)
        assert gm.tpr == pytest.approx(2 / 3)
        assert gm.fpr == pytest.approx(1 / 5)
        assert gm.prec == pytest.approx(2 / 3)
        assert gm.acc == pytest.approx(6 / 8)


# =========================================================================== #
# Shared unprivileged-group definitions used by the auditor tests              #
# =========================================================================== #

# sex: arbitrary
# marital.status: pick "Married-civ-spouse" as unprivileged (arbitrary for synthetic data)
# race: "White" as unprivileged (arbitrary for synthetic data)
UNPRIV_GROUPS_BINARY = {
    "sex": "Male",
    "marital.status": "Married-civ-spouse",
    "race": "White",
}


# =========================================================================== #
# FairnessAuditor tests                                                         #
# =========================================================================== #


class TestFairnessAuditorBinary:
    """Tests using a synthetic binary classification scenario."""

    def test_returns_audit_report(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            data=binary_matched_df,
            protected_attributes=["sex"],
            pipeline_name="test",
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        assert isinstance(report, AuditReport)

    def test_report_contains_requested_attributes(self, binary_matched_df):
        auditor = FairnessAuditor()
        attrs = ["sex", "marital.status"]
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=attrs,
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        assert set(report.attribute_results.keys()) == set(attrs)


    def test_group_metrics_has_two_groups(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        result = report.attribute_results["sex"]
        assert len(result.group_metrics) == 2
        assert result.unprivileged_group in result.group_metrics
        assert result.privileged_group in result.group_metrics

    def test_group_metrics_n_samples_sum_to_total(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        result = report.attribute_results["sex"]
        total = sum(gm.n_samples for gm in result.group_metrics.values())
        assert total == len(binary_matched_df)

    def test_positive_rates_in_unit_interval(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        for gm in report.attribute_results["sex"].group_metrics.values():
            assert 0.0 <= gm.positive_rate <= 1.0

    def test_binary_metrics_are_not_nan(self, binary_matched_df):
        """For a binary task, DPD should be numeric (not NaN)."""
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        res = report.attribute_results["sex"]
        val = res.demographic_parity_difference
        assert not math.isnan(val), (
            "demographic_parity_difference should not be NaN for binary task"
        )

    def test_demographic_parity_difference_range(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        dpd = report.attribute_results["sex"].demographic_parity_difference
        assert -1.0 <= dpd <= 1.0

    def test_dpd_equals_unpriv_minus_priv(self, binary_matched_df):
        """DPD = positive_rate(unprivileged) - positive_rate(privileged)."""
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        res = report.attribute_results["sex"]
        expected_dpd = (
            res.group_metrics[res.unprivileged_group].positive_rate
            - res.group_metrics[res.privileged_group].positive_rate
        )
        assert res.demographic_parity_difference == pytest.approx(
            expected_dpd, abs=1e-9
        )

    def test_perfect_fairness_gives_zero_dpd(self):
        """When both groups have equal positive rates, DPD should be ~0."""
        n = 200
        df = pd.DataFrame(
            {
                "group": ["A"] * (n // 2) + ["B"] * (n // 2),
                "y_true": [1, 0] * (n // 2),
                "y_pred": [1, 0] * (n // 2),
                "in_test": [True] * n,
                "in_model": [True] * n,
                "in_train": [False] * n,
            }
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            protected_attributes=["group"],
            unprivileged_groups={"group": "B"},
        )
        dpd = report.attribute_results["group"].demographic_parity_difference
        assert abs(dpd) < 1e-9

    def test_known_dpd_value(self):
        """Manually constructed scenario with known DPD."""
        # Group A: 8/10 predicted positive → rate = 0.8
        # Group B: 2/10 predicted positive → rate = 0.2
        # B is unprivileged → DPD = 0.2 - 0.8 = -0.6
        df = pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "y_true": [1] * 10 + [1] * 10,
                "y_pred": [1] * 8 + [0] * 2 + [1] * 2 + [0] * 8,
                "in_test": [True] * 20,
                "in_model": [True] * 20,
                "in_train": [False] * 20,
            }
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            protected_attributes=["group"],
            unprivileged_groups={"group": "B"},
        )
        dpd = report.attribute_results["group"].demographic_parity_difference
        assert dpd == pytest.approx(-0.6, abs=1e-9)

    def test_known_eod_value(self):
        """Manually constructed scenario with known EOD."""
        # Group A: TP=8, FN=2  → TPR = 0.8
        # Group B: TP=4, FN=6  → TPR = 0.4
        # B unprivileged → EOD = 0.4 - 0.8 = -0.4
        df = pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "y_true": [1] * 10 + [1] * 10,
                "y_pred": [1] * 8 + [0] * 2 + [1] * 4 + [0] * 6,
                "in_test": [True] * 20,
                "in_model": [True] * 20,
                "in_train": [False] * 20,
            }
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            protected_attributes=["group"],
            unprivileged_groups={"group": "B"},
        )
        eod = report.attribute_results["group"].equal_opportunity_difference
        assert eod == pytest.approx(-0.4, abs=1e-9)

    def test_known_apd_value(self):
        """Manually constructed scenario with known APD."""
        # Group A: 9/10 correct → accuracy = 0.9
        # Group B: 5/10 correct → accuracy = 0.5
        # B unprivileged → APD = 0.5 - 0.9 = -0.4
        df = pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "y_true": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                "y_pred": [1, 1, 1, 1, 1, 0, 0, 0, 0, 1] + [1, 1, 0, 0, 0, 1, 1, 0, 0, 0],
                "in_test": [True] * 20,
                "in_model": [True] * 20,
                "in_train": [False] * 20,
            }
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            protected_attributes=["group"],
            unprivileged_groups={"group": "B"},
        )
        res = report.attribute_results["group"]
        assert res.unprivileged_group == "B"
        apd = res.accuracy_parity_difference
        assert apd == pytest.approx(-0.4, abs=1e-9)

    def test_known_ppd_value(self):
        """Manually constructed scenario with known PPD."""
        # Group A: 5 predicted positive, 4 of them correct → precision = 0.8
        # Group B: 4 predicted positive, 2 of them correct → precision = 0.5
        # B unprivileged → PPD = 0.5 - 0.8 = -0.3
        df = pd.DataFrame(
            {
                "group": ["A"] * 10 + ["B"] * 10,
                "y_true": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0] + [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                "y_pred": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                "in_test": [True] * 20,
                "in_model": [True] * 20,
                "in_train": [False] * 20,
            }
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            protected_attributes=["group"],
            unprivileged_groups={"group": "B"},
        )
        ppd = report.attribute_results["group"].predictive_parity_difference
        assert ppd == pytest.approx(-0.3, abs=1e-9)

    def test_missing_attribute_raises(self, binary_matched_df):
        auditor = FairnessAuditor()
        with pytest.raises(KeyError, match="not a column"):
            auditor.audit(
                binary_matched_df,
                protected_attributes=["nonexistent_column"],
                unprivileged_groups={"nonexistent_column": "X"},
            )

    def test_multiple_attributes(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex", "marital.status", "race"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        assert len(report.attribute_results) == 3

    def test_unprivileged_groups_required(self, binary_matched_df):
        """audit() raises ValueError when unprivileged_groups is not provided."""
        auditor = FairnessAuditor()
        with pytest.raises(ValueError, match="unprivileged_groups must be provided"):
            auditor.audit(
                binary_matched_df,
                protected_attributes=["sex"],
            )


# =========================================================================== #
# AuditReport presentation tests                                                #
# =========================================================================== #


class TestAuditReport:
    def test_summary_table_has_correct_columns(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex", "marital.status"],
            pipeline_name="TestPipeline",
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        df = report.summary_table()
        assert isinstance(df, pd.DataFrame)
        for col in ["DPD", "EOD", "PPD", "APD"]:
            assert col in df.columns

    def test_summary_table_index_matches_attributes(self, binary_matched_df):
        auditor = FairnessAuditor()
        attrs = ["sex", "marital.status"]
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=attrs,
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        df = report.summary_table()
        assert set(df.index) == set(attrs)

    def test_group_table_returns_dataframe(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        gt = report.group_table("sex")
        assert isinstance(gt, pd.DataFrame)

    def test_group_table_invalid_attribute_raises(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        with pytest.raises(KeyError):
            report.group_table("nonexistent")

    def test_n_samples_audited_matches_data_length(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        assert report.n_samples_audited == len(binary_matched_df)

    def test_pipeline_name_stored(self, binary_matched_df):
        auditor = FairnessAuditor()
        report = auditor.audit(
            binary_matched_df,
            protected_attributes=["sex"],
            pipeline_name="MyPipeline",
            unprivileged_groups=UNPRIV_GROUPS_BINARY,
        )
        assert report.pipeline_name == "MyPipeline"


# =========================================================================== #
# Unprivileged group tests                                                      #
# =========================================================================== #


class TestUnprivilegedGroups:
    def _make_df(self, col, groups_and_preds):
        """Helper: build a minimal auditable DataFrame."""
        rows = []
        for group, preds in groups_and_preds.items():
            for p in preds:
                rows.append(
                    {
                        col: group,
                        "y_true": 1,
                        "y_pred": p,
                        "in_test": True,
                        "in_model": True,
                        "in_train": False,
                    }
                )
        return pd.DataFrame(rows)

    def test_single_value_unprivileged(self):
        """A single-value unprivileged group is respected even if it has higher pos_rate."""
        # A: 8/10 positive (higher) — but we declare A as unprivileged
        # B: 2/10 positive
        df = self._make_df("group", {"A": [1] * 8 + [0] * 2, "B": [1] * 2 + [0] * 8})
        auditor = FairnessAuditor()
        report = auditor.audit(df, ["group"], unprivileged_groups={"group": "A"})
        res = report.attribute_results["group"]
        assert res.unprivileged_group == "A"
        assert res.privileged_group == "B"

    def test_list_unprivileged_combines_groups(self):
        """With a list, all listed groups are combined into one unprivileged group."""
        # X: 3/10, Y: 2/10 → combined: 5/20 = 0.25; Z: 7/10 → privileged
        df = self._make_df(
            "group",
            {
                "X": [1] * 3 + [0] * 7,
                "Y": [1] * 2 + [0] * 8,
                "Z": [1] * 7 + [0] * 3,
            },
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            ["group"],
            unprivileged_groups={"group": ["X", "Y"]},
        )
        res = report.attribute_results["group"]
        assert isinstance(res.unprivileged_group, tuple)
        assert set(res.unprivileged_group) == {"X", "Y"}
        assert res.privileged_group == "Z"
        combined_gm = res.group_metrics[res.unprivileged_group]
        assert abs(combined_gm.positive_rate - 0.25) < 1e-9

    def test_list_unprivileged_privileged_excludes_all_unprivileged(self):
        """Privileged group must not come from the unprivileged list."""
        df = self._make_df(
            "group",
            {
                "X": [1] * 3 + [0] * 7,
                "Y": [1] * 5 + [0] * 5,
                "Z": [1] * 8 + [0] * 2,
            },
        )
        auditor = FairnessAuditor()
        report = auditor.audit(
            df,
            ["group"],
            unprivileged_groups={"group": ["X", "Y"]},
        )
        res = report.attribute_results["group"]
        assert res.privileged_group == "Z"
        assert isinstance(res.unprivileged_group, tuple)
        assert set(res.unprivileged_group) == {"X", "Y"}
