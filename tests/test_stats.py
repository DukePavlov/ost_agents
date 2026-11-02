"""Fast offline unit tests; do not require Ollama, internet or the UCI CSV."""
from pathlib import Path

import pandas as pd
import pytest

from stats import (
    load_data,
    get_cohort_summary,
    compare_group_means,
    get_feature_statistics,
)


@pytest.fixture
def sample():
    # Tiny SYNTHETIC fixture for testing, NOT real patient records.
    return pd.DataFrame({
        "Age": [40, 60, 50, 70],
        "BMI": [20.0, 30.0, 22.0, 32.0],
        "Glucose": [90, 100, 130, 150],
        "Insulin": [4, 5, 8, 9],
        "Classification": [1, 1, 2, 2],
    })


def test_cohort_summary(sample):
    assert get_cohort_summary(sample) == {
        "total_participants": 4, "healthy_controls": 2, "breast_cancer_patients": 2,
    }


def test_group_comparison(sample):
    result = compare_group_means(sample, "bmi")
    assert result["healthy_mean"] == 25.0
    assert result["cancer_mean"] == 27.0
    assert result["difference_cancer_minus_healthy"] == 2.0


def test_feature_statistics(sample):
    result = get_feature_statistics(sample, "glucose", "cancer")
    assert result["n"] == 2
    assert result["mean"] == 140.0
    assert result["median"] == 140.0
    assert result["min"] == 130.0
    assert result["max"] == 150.0


def test_bad_feature(sample):
    with pytest.raises(ValueError, match="Unsupported feature"):
        compare_group_means(sample, "patient_id")


def test_bad_group(sample):
    with pytest.raises(ValueError, match="Group must"):
        get_feature_statistics(sample, "BMI", "unsupported")


def test_csv_roundtrip(tmp_path: Path, sample):
    path = tmp_path / "test_data.csv"
    sample.to_csv(path, index=False)
    actual = load_data(path)
    assert actual.shape == sample.shape
    assert sorted(actual["Classification"].unique()) == [1, 2]
