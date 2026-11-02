"""Deterministic analysis functions for the Breast Cancer Coimbra dataset.

No LLM or LangChain dependency here, so calculations can be tested independently.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "dataR2.csv"
ALLOWED_FEATURES = ("Age", "BMI", "Glucose", "Insulin")
FEATURE_UNITS = {"Age": "years", "BMI": "kg/m^2", "Glucose": "mg/dL", "Insulin": "uU/mL"}
GROUP_LABELS = {1: "healthy", 2: "cancer"}


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the teaching dataset, checking the fields/labels used in this lab."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python download_data.py"
        )
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.strip()
    required = set(ALLOWED_FEATURES) | {"Classification"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("The dataset is empty.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Null values in required dataset columns.")
    if not frame["Classification"].isin(GROUP_LABELS).all():
        raise ValueError("Expected Classification=1 (healthy) or 2 (cancer).")
    return frame


def _feature_name(feature: str) -> str:
    for name in ALLOWED_FEATURES:
        if name.lower() == feature.strip().lower():
            return name
    raise ValueError(f"Unsupported feature {feature!r}; choose from {ALLOWED_FEATURES}.")


def _group_code(group: str) -> int | None:
    normalized = group.strip().lower()
    if normalized == "all":
        return None
    for code, label in GROUP_LABELS.items():
        if normalized == label:
            return code
    raise ValueError("Group must be 'all', 'healthy', or 'cancer'.")


def get_cohort_summary(frame: pd.DataFrame) -> dict:
    """Number of observations in the dataset and in each labeled group."""
    counts = frame["Classification"].value_counts()
    return {
        "total_participants": int(len(frame)),
        "healthy_controls": int(counts.get(1, 0)),
        "breast_cancer_patients": int(counts.get(2, 0)),
    }


def compare_group_means(frame: pd.DataFrame, feature: str) -> dict:
    """Compare one feature's average across label groups (descriptive only)."""
    feature = _feature_name(feature)
    means = frame.groupby("Classification")[feature].mean()
    if 1 not in means.index or 2 not in means.index:
        raise ValueError("Both study groups are required for comparison.")
    healthy, cancer = float(means[1]), float(means[2])
    return {
        "feature": feature,
        "unit": FEATURE_UNITS[feature],
        "healthy_mean": round(healthy, 2),
        "cancer_mean": round(cancer, 2),
        "difference_cancer_minus_healthy": round(cancer - healthy, 2),
        "note": "Descriptive group comparison; not causal or diagnostic evidence.",
    }


def get_feature_statistics(frame: pd.DataFrame, feature: str, group: str = "all") -> dict:
    """Descriptive summary of a feature, optionally within a labeled group."""
    feature = _feature_name(feature)
    code = _group_code(group)
    subset = frame if code is None else frame.loc[frame["Classification"] == code]
    if subset.empty:
        raise ValueError(f"No observations for group {group!r}.")
    s = subset[feature]
    return {
        "feature": feature,
        "group": group.lower(),
        "unit": FEATURE_UNITS[feature],
        "n": int(s.size),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
        "std": round(float(s.std(ddof=1)), 2) if s.size > 1 else None,
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
    }
