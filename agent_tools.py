"""Three read-only LangChain tools exposed to the LLM.

The model selects a tool. Python/pandas computes the results.
"""
from langchain.tools import tool

from stats import (
    ALLOWED_FEATURES,
    get_cohort_summary as calculate_cohort_summary,
    compare_group_means as calculate_group_means,
    get_feature_statistics as calculate_feature_statistics,
    load_data,
)


@tool
def get_cohort_summary() -> dict:
    """Count all participants and each labeled group: healthy controls or breast cancer patients."""
    return calculate_cohort_summary(load_data())


@tool
def compare_group_means(feature: str) -> dict:
    """Compare the mean of Age, BMI, Glucose, or Insulin between healthy and cancer groups.

    Args:
        feature: One of Age, BMI, Glucose, Insulin (case insensitive).
    """
    try:
        return calculate_group_means(load_data(), feature)
    except ValueError as exc:
        return {"error": str(exc), "supported_features": list(ALLOWED_FEATURES)}


@tool
def get_feature_statistics(feature: str, group: str = "all") -> dict:
    """Get count, mean, median, std, min, and max for Age, BMI, Glucose, or Insulin.

    Args:
        feature: One of Age, BMI, Glucose, Insulin (case insensitive).
        group: 'all', 'healthy', or 'cancer'.
    """
    try:
        return calculate_feature_statistics(load_data(), feature, group)
    except ValueError as exc:
        return {"error": str(exc), "supported_features": list(ALLOWED_FEATURES),
                "supported_groups": ["all", "healthy", "cancer"]}


TOOLS = [get_cohort_summary, compare_group_means, get_feature_statistics]
