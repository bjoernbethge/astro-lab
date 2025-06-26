"""
AstroLab Models - Modern Graph Neural Networks for Astronomical Data
"""

# Core models - import only what users actually need
# Configuration (simplified)
# Lightning module integrations
from . import lightning
from .config import ModelConfig, get_predefined_config
from .core import (
    ALCDEFTemporalGNN,
    AstroPhotGNN,
    AstroSurveyGNN,
    TemporalGCN,
)

# Factory functions for common use cases
from .factories import (
    create_asteroid_period_detector,
    create_gaia_classifier,
    create_lsst_transient_detector,
    create_model,  # Generic model factory
    create_sdss_galaxy_model,
)

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
    "create_model",  # Generic model factory
    "ModelConfig",
    "get_predefined_config",
    "lightning",  # Lightning module access
]
