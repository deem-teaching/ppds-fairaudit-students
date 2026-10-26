"""
End-to-end tests: runner → matcher → auditor, and main.py, on the synthetic data.
"""

from __future__ import annotations

import numpy as np
import pytest

import main
from pipeline_auditing.fairness.auditor import FairnessAuditor
from pipeline_auditing.fairness.matcher import PredictionMatcher
from pipeline_auditing.runner import PipelineRunner
from pipelines import ML816Pipeline

# Arbitrary group definitions, only used to exercise the auditor
GROUPS = {"sex": "Male", "race": ["White", "Other"]}


@pytest.fixture(scope="module")
def audited(synthetic_csv_path):
    result = PipelineRunner().run(ML816Pipeline(), str(synthetic_csv_path))
    matched = PredictionMatcher().match(result)
    auditable = PredictionMatcher.get_auditable_subset(matched)
    report = FairnessAuditor().audit(
        auditable, protected_attributes=list(GROUPS), unprivileged_groups=GROUPS
    )
    return result, report


def test_audit_covers_the_test_split(audited):
    """Every test row is audited, and each row is in exactly one group."""
    result, report = audited
    assert report.n_samples_audited == len(result.X_test)
    for attr in GROUPS:
        sizes = [gm.n_samples for gm in report.attribute_results[attr].group_metrics.values()]
        assert sum(sizes) == len(result.X_test)


def test_audit_of_matched_data_equals_direct_audit(audited):
    """Auditing the matcher's output gives the same result as auditing the test rows directly."""
    result, report = audited
    direct_data = result.raw_data.loc[result.y_test.index].assign(
        y_true=result.y_test.to_numpy(), y_pred=result.y_pred
    )
    direct = FairnessAuditor().audit(
        direct_data, protected_attributes=list(GROUPS), unprivileged_groups=GROUPS
    )
    for attr in GROUPS:
        got, expected = report.attribute_results[attr], direct.attribute_results[attr]
        assert got.unprivileged_group == expected.unprivileged_group
        assert got.privileged_group == expected.privileged_group
        for metric in [
            "demographic_parity_difference",
            "equal_opportunity_difference",
            "predictive_parity_difference",
            "accuracy_parity_difference",
        ]:
            assert getattr(got, metric) == pytest.approx(getattr(expected, metric), nan_ok=True)


def test_group_results_agree_with_runner(audited):
    """Weighted over both groups, accuracy and positive rate equal the runner's overall values."""
    result, report = audited
    n = len(result.X_test)
    for attr in GROUPS:
        groups = report.attribute_results[attr].group_metrics.values()
        accuracy = sum(gm.acc * gm.n_samples for gm in groups if gm.n_samples) / n
        pos_rate = sum(gm.positive_rate * gm.n_samples for gm in groups if gm.n_samples) / n
        assert accuracy == pytest.approx(result.metrics["accuracy"])
        assert pos_rate == pytest.approx(np.mean(result.y_pred == 1))


def test_main_runs_end_to_end(synthetic_csv_path, monkeypatch, capsys):
    """main.py runs all pipelines and the fairness audit without anything missing."""
    monkeypatch.setattr(main, "DATA_PATH", str(synthetic_csv_path))
    main.main()
    out = capsys.readouterr().out
    assert "not implemented yet" not in out
    assert "Cross-Pipeline Fairness Comparison" in out
