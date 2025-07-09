"""
Survey Data Collectors
=====================

Collection of survey data collectors with proper download and caching.
"""

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

__all__ = [
    "BaseSurveyCollector",
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
