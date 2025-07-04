"""
Survey Data Collectors
=====================

Factory pattern for survey data collectors with proper download and caching.
"""

from typing import Type

from .base import BaseSurveyCollector
from .des import DESCollector
from .euclid import EuclidCollector
from .exoplanet import ExoplanetCollector
from .gaia import GaiaCollector
from .linear import LinearCollector
from .nsa import NSACollector
from .panstarrs import PanSTARRSCollector
from .rrlyrae import RRLyraeCollector
from .sdss import SDSSCollector
from .tng50 import TNG50Collector
from .twomass import TwoMASSCollector
from .wise import WISECollector

COLLECTOR_REGISTRY = {}

# Register collectors in the global registry
COLLECTOR_REGISTRY["linear"] = LinearCollector
COLLECTOR_REGISTRY["rrlyrae"] = RRLyraeCollector
COLLECTOR_REGISTRY["sdss"] = SDSSCollector
COLLECTOR_REGISTRY["gaia"] = GaiaCollector
COLLECTOR_REGISTRY["exoplanet"] = ExoplanetCollector
COLLECTOR_REGISTRY["nsa"] = NSACollector
COLLECTOR_REGISTRY["tng50"] = TNG50Collector
COLLECTOR_REGISTRY["twomass"] = TwoMASSCollector
COLLECTOR_REGISTRY["wise"] = WISECollector
COLLECTOR_REGISTRY["panstarrs"] = PanSTARRSCollector
COLLECTOR_REGISTRY["des"] = DESCollector
COLLECTOR_REGISTRY["euclid"] = EuclidCollector


def get_collector(survey: str, **kwargs) -> BaseSurveyCollector:
    """
    Get collector instance for a given survey.

    Args:
        survey: Survey name
        **kwargs: Additional configuration for the collector

    Returns:
        Collector instance
    """
    if survey not in COLLECTOR_REGISTRY:
        available = ", ".join(sorted(COLLECTOR_REGISTRY.keys()))
        raise ValueError(f"Unknown survey collector: {survey}. Available: {available}")

    collector_class = COLLECTOR_REGISTRY[survey]
    return collector_class(survey, **kwargs)


def get_available_collectors() -> list[str]:
    """Get list of available survey collectors."""
    return sorted(COLLECTOR_REGISTRY.keys())


def register_collector(survey: str, collector_class: Type[BaseSurveyCollector]):
    """Register a new collector for a survey."""
    COLLECTOR_REGISTRY[survey] = collector_class


__all__ = [
    "get_collector",
    "get_available_collectors",
    "register_collector",
    "BaseSurveyCollector",
    "COLLECTOR_REGISTRY",
    # Individual collectors
    "LinearCollector",
    "RRLyraeCollector",
    "SDSSCollector",
    "GaiaCollector",
    "ExoplanetCollector",
    "NSACollector",
    "TNG50Collector",
    "TwoMASSCollector",
    "WISECollector",
    "PanSTARRSCollector",
    "DESCollector",
    "EuclidCollector",
]
