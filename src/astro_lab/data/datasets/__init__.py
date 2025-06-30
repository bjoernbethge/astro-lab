"""
AstroLab Datasets
================

Provides datasets for astronomical data processing and graph building.
"""

from astro_lab.config import get_available_surveys

from .survey_graph_dataset import SurveyGraphDataset
from .point_cloud_dataset import AstroPointCloudDataset


def validate_survey(survey: str) -> bool:
    """Validate if survey is supported."""
    return survey in get_available_surveys()


def get_supported_surveys() -> list[str]:
    """Get list of supported surveys."""
    return get_available_surveys()


# For backward compatibility
SUPPORTED_SURVEYS = get_available_surveys()

__all__ = [
    "SurveyGraphDataset",
    "AstroPointCloudDataset",
    "SUPPORTED_SURVEYS",
    "validate_survey",
    "get_supported_surveys",
]
