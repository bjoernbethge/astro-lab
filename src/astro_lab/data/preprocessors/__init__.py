"""
Survey Data Preprocessors
========================

Factory pattern for survey preprocessors with proper TensorDict integration.
"""

from typing import Type

from astro_lab.config.registry import PREPROCESSOR_REGISTRY

from .base import BaseSurveyProcessor
from .des import DESPreprocessor
from .euclid import EuclidPreprocessor
from .exoplanet import ExoplanetPreprocessor
from .gaia import GaiaPreprocessor
from .linear import LinearPreprocessor
from .nsa import NSAPreprocessor
from .panstarrs import PanSTARRSPreprocessor
from .rrlyrae import RRLyraePreprocessor
from .sdss import SDSSPreprocessor
from .tng50 import TNG50Preprocessor
from .twomass import TwoMASSPreprocessor
from .wise import WISEPreprocessor

# Register preprocessors in the global registry
PREPROCESSOR_REGISTRY["gaia"] = GaiaPreprocessor
PREPROCESSOR_REGISTRY["sdss"] = SDSSPreprocessor
PREPROCESSOR_REGISTRY["nsa"] = NSAPreprocessor
PREPROCESSOR_REGISTRY["tng50"] = TNG50Preprocessor
PREPROCESSOR_REGISTRY["exoplanet"] = ExoplanetPreprocessor
PREPROCESSOR_REGISTRY["twomass"] = TwoMASSPreprocessor
PREPROCESSOR_REGISTRY["wise"] = WISEPreprocessor
PREPROCESSOR_REGISTRY["panstarrs"] = PanSTARRSPreprocessor
PREPROCESSOR_REGISTRY["des"] = DESPreprocessor
PREPROCESSOR_REGISTRY["euclid"] = EuclidPreprocessor
PREPROCESSOR_REGISTRY["linear"] = LinearPreprocessor
PREPROCESSOR_REGISTRY["rrlyrae"] = RRLyraePreprocessor


def get_preprocessor(survey: str, **kwargs) -> BaseSurveyProcessor:
    """
    Get preprocessor instance for a given survey.

    Args:
        survey: Survey name
        **kwargs: Additional configuration for the preprocessor

    Returns:
        Preprocessor instance
    """
    if survey not in PREPROCESSOR_REGISTRY:
        available = ", ".join(sorted(PREPROCESSOR_REGISTRY.keys()))
        raise ValueError(f"Unknown survey: {survey}. Available: {available}")

    preprocessor_class = PREPROCESSOR_REGISTRY[survey]
    return preprocessor_class(survey, **kwargs)


def get_available_preprocessors() -> list[str]:
    """Get list of available survey preprocessors."""
    return sorted(PREPROCESSOR_REGISTRY.keys())


def register_preprocessor(survey: str, preprocessor_class: Type[BaseSurveyProcessor]):
    """Register a new preprocessor for a survey."""
    PREPROCESSOR_REGISTRY[survey] = preprocessor_class


__all__ = [
    "get_preprocessor",
    "get_available_preprocessors",
    "register_preprocessor",
    "BaseSurveyProcessor",
    "PREPROCESSOR_REGISTRY",
    # Individual preprocessors
    "GaiaPreprocessor",
    "SDSSPreprocessor",
    "NSAPreprocessor",
    "TNG50Preprocessor",
    "ExoplanetPreprocessor",
    "TwoMASSPreprocessor",
    "WISEPreprocessor",
    "PanSTARRSPreprocessor",
    "DESPreprocessor",
    "EuclidPreprocessor",
    "LinearPreprocessor",
    "RRLyraePreprocessor",
]
