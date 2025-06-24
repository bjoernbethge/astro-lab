"""
AstroLab Models - Modern Graph Neural Networks for Astronomical Data
"""

# Core models - import only what users actually need
from .core import (
    AstroSurveyGNN,
    AstroPhotGNN,
    TemporalGCN,
    ALCDEFTemporalGNN,
)

# Factory functions for common use cases
from .factories import (
    create_gaia_classifier,
    create_sdss_galaxy_model,
    create_lsst_transient_detector,
    create_asteroid_period_detector,
)

# Configuration (simplified)
from .config import ModelConfig, get_predefined_config

__version__ = "0.4.0"

# Only export what users actually need
__all__ = [
    "AstroSurveyGNN",
    "AstroPhotGNN", 
    "TemporalGCN",
    "ALCDEFTemporalGNN",
    "create_gaia_classifier",
    "create_sdss_galaxy_model",
    "create_lsst_transient_detector",
    "create_asteroid_period_detector",
    "ModelConfig",
    "get_predefined_config",
]
