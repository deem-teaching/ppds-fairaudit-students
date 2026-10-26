"""
main.py – Full fairness auditing workflow demonstration.

Run this script after ensuring the dataset is present:

    python main.py

The script:
  1. Runs all pipelines on the dataset.
  2. Prints a performance summary for each pipeline.
  3. Matches predictions back to the raw data.
  4. Audits each pipeline for group fairness across protected attributes.
  5. Prints a fairness comparison table across all pipelines.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline_auditing.runner import PipelineRunner
from pipelines import (
    RandomBaselinePipeline,
    ML816Pipeline,
    LogisticRegressionPipeline,
    XGBoostPipeline,
    LogisticPCAPipeline,
    EffectivePipeline,
    FairPipeline,
)
from pipeline_auditing.fairness.matcher import PredictionMatcher
from pipeline_auditing.fairness.auditor import FairnessAuditor, AuditReport
from pipeline_auditing.fairness.groups import AdultIncomeGroups

PIPELINE_CONFIGS = [
    ("Pipeline 0 – Random Baseline",               RandomBaselinePipeline),
    ("Pipeline 1 – ML816 (Logistic+Threshold)",    ML816Pipeline),
    ("Pipeline 2 – Logistic Regression (OHE)",     LogisticRegressionPipeline),
    ("Pipeline 3 – XGBoost",                       XGBoostPipeline),
    ("Pipeline 4 – Logistic + PCA",                LogisticPCAPipeline),
    ("Pipeline 5 – Effective (Random Forest)",     EffectivePipeline),
    ("Pipeline 6 – Your fair pipeline (bonus)",    FairPipeline),
]

# Pipelines that are an optional bonus task
OPTIONAL_PIPELINES = {FairPipeline}


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _print_table(df: pd.DataFrame) -> None:
    try:
        from tabulate import tabulate
        print(tabulate(df, headers="keys", tablefmt="simple", floatfmt="+.4f"))
    except ImportError:
        print(df.to_string())


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_performance(result) -> None:
    m = result.metrics
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  F1        : {m['f1']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    if "roc_auc" in m:
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
    print(f"  Test rows : {m['n_test']}")


def build_comparison_table(
    reports: dict[str, AuditReport],
    attribute: str,
) -> pd.DataFrame:
    """Build a fairness comparison table across all pipelines for one attribute."""
    rows = []
    for pipeline_name, report in reports.items():
        if attribute not in report.attribute_results:
            continue
        res = report.attribute_results[attribute]
        rows.append({
            "Pipeline": pipeline_name,
            "DPD": round(res.demographic_parity_difference, 4),
            "EOD": round(res.equal_opportunity_difference, 4),
            "PPD": round(res.predictive_parity_difference, 4),
            "APD": round(res.accuracy_parity_difference, 4),
            "Unprivileged": AuditReport._fmt_group(res.unprivileged_group),
            "Privileged": AuditReport._fmt_group(res.privileged_group),
        })
    df = pd.DataFrame(rows).set_index("Pipeline")
    return df


# --------------------------------------------------------------------------- #
# Main                                                                          #
# --------------------------------------------------------------------------- #

DATA_PATH = "data/adult.csv"


def main() -> None:
    runner  = PipelineRunner(test_size=0.2, random_state=42)
    matcher = PredictionMatcher()
    auditor = FairnessAuditor(positive_label=1)

    # The fairness audit needs the unprivileged groups from Step 4a. Until
    # they are implemented, only the performance of the pipelines is shown.
    unprivileged_groups = AdultIncomeGroups()
    try:
        protected_attributes = list(unprivileged_groups.get_groups().keys())
    except NotImplementedError:
        protected_attributes = []
        print("AdultIncomeGroups not implemented yet: skipping the fairness audit.")

    reports: dict[str, AuditReport] = {}

    for display_name, pipeline_cls in PIPELINE_CONFIGS:
        print_section(display_name)

        # ── run pipeline ─────────────────────────────────────────────────
        try:
            pipeline = pipeline_cls()
            result = runner.run(pipeline, DATA_PATH, name=display_name)
        except (NotImplementedError, TypeError):
            if pipeline_cls in OPTIONAL_PIPELINES:
                print(f"  {display_name!r} is an optional bonus and was not attempted.")
            else:
                print(f"  Could not run {display_name!r}: pipeline or runner not implemented yet.")
            continue

        print("\n  ── Performance ─────────────────────────────────────")
        print_performance(result)

        if not protected_attributes:
            continue

        # ── match + audit ────────────────────────────────────────────────
        try:
            matched_df   = matcher.match(result)
            auditable_df = PredictionMatcher.get_auditable_subset(matched_df)

            # Filter to attributes that exist in the auditable data
            available_attrs = [
                a for a in protected_attributes if a in auditable_df.columns
            ]

            report = auditor.audit(
                data=auditable_df,
                protected_attributes=available_attrs,
                pipeline_name=display_name,
                unprivileged_groups=unprivileged_groups,
            )
            reports[display_name] = report

            print("\n  ── Fairness (summary) ──────────────────────────────")
            _print_table(report.summary_table())

        except (NotImplementedError, TypeError):
            print("  Fairness audit not implemented yet.")

    if not reports:
        return

    # ── Cross-pipeline comparison ────────────────────────────────────────
    print_section("Cross-Pipeline Fairness Comparison")
    for attr in protected_attributes:
        print(f"\n  Protected attribute: {attr}")
        cmp_table = build_comparison_table(reports, attr)
        _print_table(cmp_table)

    print()
    print("Fairness metric legend:")
    print("  DPD   Demographic Parity Difference     (ideal: 0)")
    print("  EOD   Equal Opportunity Difference       (ideal: 0)")
    print("  PPD   Predictive Parity Difference       (ideal: 0)")
    print("  APD   Accuracy Parity Difference         (ideal: 0)")
    print()
    print("Negative DPD / EOD / PPD / APD = unprivileged group is disadvantaged.")


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if not Path(DATA_PATH).exists():
        print(
            f"ERROR: data file not found at '{DATA_PATH}'.\n"
            "The dataset should already be present at data/adult.csv.\n"
            "If missing, download adult.csv from "
            "https://www.kaggle.com/datasets/uciml/adult-census-income",
            file=sys.stderr,
        )
        sys.exit(1)

    main()
