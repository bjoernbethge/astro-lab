"""Astronomical survey preprocessors.

Unified preprocessing pipeline for different astronomical surveys.
Each preprocessor implements filtering, transformation, and feature extraction
optimized for machine learning applications.
"""

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)
from .des import DESPreprocessor
from .exoplanet import ExoplanetPreprocessor
from .gaia import GaiaPreprocessor
from .linear import LINEARPreprocessor
from .nsa import NSAPreprocessor
from .rrlyrae import RRLyraePreprocessor
from .sdss import SDSSPreprocessor
from .tng50 import TNG50Preprocessor
from .twomass import TwoMASSPreprocessor
from .wise import WISEPreprocessor

# Survey preprocessor registry
PREPROCESSOR_REGISTRY = {
    "gaia": GaiaPreprocessor,
    "sdss": SDSSPreprocessor,
    "nsa": NSAPreprocessor,
    "tng50": TNG50Preprocessor,
    "exoplanet": ExoplanetPreprocessor,
    "twomass": TwoMASSPreprocessor,
    "wise": WISEPreprocessor,
    "des": DESPreprocessor,
    "linear": LINEARPreprocessor,
    "rrlyrae": RRLyraePreprocessor,
}


def get_preprocessor(survey_name: str, config=None):
    """Get preprocessor for specified survey.

    Args:
        survey_name: Name of survey ('gaia', 'sdss', 'nsa', etc.)
        config: Optional configuration dict

    Returns:
        Initialized preprocessor instance

    Raises:
        ValueError: If survey not supported
    """
    if survey_name not in PREPROCESSOR_REGISTRY:
        available = list(PREPROCESSOR_REGISTRY.keys())
        raise ValueError(
            f"Survey '{survey_name}' not supported. Available: {available}"
        )

    preprocessor_class = PREPROCESSOR_REGISTRY[survey_name]
    return preprocessor_class(config=config)


def list_available_surveys():
    """List all available survey preprocessors.

    Returns:
        List of survey names
    """
    return list(PREPROCESSOR_REGISTRY.keys())


__all__ = [
    "AstroLabDataPreprocessor",
    "AstronomicalPreprocessorMixin",
    "StatisticalPreprocessorMixin",
    "GaiaPreprocessor",
    "SDSSPreprocessor",
    "NSAPreprocessor",
    "TNG50Preprocessor",
    "ExoplanetPreprocessor",
    "TwoMASSPreprocessor",
    "WISEPreprocessor",
    "DESPreprocessor",
    "LINEARPreprocessor",
    "RRLyraePreprocessor",
    "get_preprocessor",
    "list_available_surveys",
    "PREPROCESSOR_REGISTRY",
]
