"""
Shared pytest fixtures for all test modules.

The fixtures provide a small, deterministic synthetic dataset that mirrors
the structure of the Adult Income dataset, allowing tests to run quickly and
without requiring the actual data file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


# --------------------------------------------------------------------------- #
# Constants mirroring the real dataset                                          #
# --------------------------------------------------------------------------- #

N_SAMPLES = 300

# education → education.num, as in the real dataset
EDUCATION_LEVELS = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4, "9th": 5,
    "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9, "Some-college": 10,
    "Assoc-voc": 11, "Assoc-acdm": 12, "Bachelors": 13, "Masters": 14,
    "Prof-school": 15, "Doctorate": 16,
}

# Fraction of "?" (missing) values in workclass, occupation and native.country
MISSING_RATE = 0.05

# Number of exact duplicate rows appended (the real data has duplicates too),
# so that the default filter_train actually removes rows
N_DUPLICATES = 30


# --------------------------------------------------------------------------- #
# Synthetic DataFrame                                                           #
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    """
    A synthetic DataFrame (N_SAMPLES + N_DUPLICATES rows) with the same schema
    as the Adult Income dataset.

    Categorical columns use string values from the real dataset, including
    "?" for missing values in workclass, occupation and native.country.
    The last N_DUPLICATES rows are exact duplicates of earlier rows.
    """
    rng = np.random.default_rng(seed=0)

    education = rng.choice(list(EDUCATION_LEVELS), N_SAMPLES)

    df = pd.DataFrame(
        {
            "age": rng.integers(17, 91, N_SAMPLES),
            "workclass": rng.choice(
                ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                 "Local-gov", "State-gov"],
                N_SAMPLES,
            ),
            "fnlwgt": rng.integers(12_000, 1_500_000, N_SAMPLES),
            "education": education,
            "education.num": [EDUCATION_LEVELS[e] for e in education],
            "marital.status": rng.choice(
                ["Married-civ-spouse", "Never-married", "Divorced", "Separated",
                 "Widowed"],
                N_SAMPLES,
            ),
            "occupation": rng.choice(
                ["Prof-specialty", "Craft-repair", "Exec-managerial",
                 "Adm-clerical", "Sales", "Other-service"],
                N_SAMPLES,
            ),
            "relationship": rng.choice(
                ["Husband", "Not-in-family", "Own-child", "Unmarried", "Wife"],
                N_SAMPLES,
            ),
            "race": rng.choice(
                ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo",
                 "Other"],
                N_SAMPLES,
            ),
            "sex": rng.choice(["Male", "Female"], N_SAMPLES),
            "capital.gain": np.where(
                rng.random(N_SAMPLES) < 0.1, rng.integers(100, 100_000, N_SAMPLES), 0
            ),
            "capital.loss": np.where(
                rng.random(N_SAMPLES) < 0.05, rng.integers(100, 4_400, N_SAMPLES), 0
            ),
            "hours.per.week": rng.integers(1, 100, N_SAMPLES),
            "native.country": rng.choice(
                ["United-States", "Mexico", "Philippines", "Germany", "India"],
                N_SAMPLES,
            ),
            "income": rng.choice([">50K", "<=50K"], N_SAMPLES),
        }
    )

    for col in ["workclass", "occupation", "native.country"]:
        df.loc[rng.random(N_SAMPLES) < MISSING_RATE, col] = "?"

    # A category that occurs only once and ends up in the test split of the
    # pipeline tests (as "Holand-Netherlands" does in the real data): encoders
    # must cope with values unseen during fitting.
    _, test_idx = train_test_split(df.index, test_size=0.2, random_state=42)
    df.loc[test_idx[0], "native.country"] = "Holand-Netherlands"

    duplicates = df.drop(index=test_idx[0]).iloc[:N_DUPLICATES]
    return pd.concat([df, duplicates], ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_csv_path(synthetic_df: pd.DataFrame, tmp_path_factory) -> Path:
    """
    Write *synthetic_df* to a temporary CSV file and return its path.

    This fixture allows pipeline tests to call ``load_data(filepath)`` with
    a real file path without depending on the actual data file.
    """
    tmp_dir = tmp_path_factory.mktemp("data")
    filepath = tmp_dir / "adult.csv"
    synthetic_df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture(scope="session")
def binary_matched_df(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """
    A minimal matched DataFrame with synthetic y_true / y_pred columns
    suitable for testing the FairnessAuditor with a binary task.
    """
    rng = np.random.default_rng(seed=1)
    df = synthetic_df.copy()

    n = len(df)
    df["y_true"] = rng.integers(0, 2, n)
    # Introduce a slight disparity: 20% of predictions for sex=="Female" are wrong
    base_pred = df["y_true"].copy()
    flip_mask = (df["sex"] == "Female") & (rng.random(n) < 0.2)
    base_pred[flip_mask] = 1 - base_pred[flip_mask]
    df["y_pred"] = base_pred.values

    df["in_model"] = True
    df["in_test"] = True
    df["in_train"] = False
    return df
