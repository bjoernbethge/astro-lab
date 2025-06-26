"""
Lightning Module Wrappers for AstroLab Models
============================================

Provides Lightning module wrappers for all AstroLab models while keeping
the original models unchanged. This allows both APIs to coexist.
"""

from .base import AstroLabLightningMixin
from .factory import (
    LIGHTNING_MODELS,
    create_lightning_model,
    create_preset_model,
    list_lightning_models,
    list_presets,
)
from .wrappers import (
    LightningALCDEFTemporalGNN,
    LightningAsteroidPeriodDetector,
    LightningAstroPhotGNN,
    LightningAstroSurveyGNN,
    LightningGaiaClassifier,
    LightningGalaxyModeler,
    LightningTemporalGCN,
    LightningTransientClassifier,
)

__all__ = [
    # Base classes
    "AstroLabLightningMixin",
    # Lightning wrapped models
    "LightningAstroSurveyGNN",
    "LightningAstroPhotGNN",
    "LightningTemporalGCN",
    "LightningALCDEFTemporalGNN",
    "LightningGaiaClassifier",
    "LightningGalaxyModeler",
    "LightningAsteroidPeriodDetector",
    "LightningTransientClassifier",
    # Factory
    "create_lightning_model",
    "create_preset_model",
    "list_lightning_models",
    "list_presets",
    "LIGHTNING_MODELS",
]
