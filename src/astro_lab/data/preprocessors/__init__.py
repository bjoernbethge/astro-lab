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
]
