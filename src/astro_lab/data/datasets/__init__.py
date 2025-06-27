"""
AstroLab Datasets
================

Provides datasets for astronomical data processing and graph building.
"""

from .survey_graph_dataset import SurveyGraphDataset

# Define supported surveys
SUPPORTED_SURVEYS = ["gaia", "sdss", "2mass", "wise", "des", "euclid", "panstarrs", "nsa", "exoplanet"]

__all__ = [
    "SurveyGraphDataset",
    "SUPPORTED_SURVEYS",
]
