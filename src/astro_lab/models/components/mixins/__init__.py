"""
Model Mixins for AstroLab
========================

Organized collection of mixins providing reusable functionality for astronomical models.
"""

# Core functionality mixins
from .metrics import MetricsMixin
from .optimization import OptimizationMixin
from .visualization import VisualizationMixin
from .explainability import ExplainabilityMixin
from .mlflow_mixin import MLflowMixin

# HPO specific mixins
from .hpo import (
    HPOResetMixin,
    HPOMemoryMixin,
    EfficientTrainingMixin,
    ArchitectureSearchMixin,
)

# Astronomical domain mixins
from .astronomical import AstronomicalAugmentationMixin, AstronomicalLossMixin

# Combined mixins for common use cases
from .combined import (
    StandardModelMixin,
    HPOModelMixin,
    AstronomicalModelMixin,
    FullAstronomicalModelMixin,
    LightweightModelMixin,
    ResearchModelMixin,
    ExplainableModelMixin,
)

__all__ = [
    # Core functionality
    "MetricsMixin",
    "OptimizationMixin",
    "VisualizationMixin",
    "ExplainabilityMixin",
    "MLflowMixin",
    # HPO specific
    "HPOResetMixin",
    "HPOMemoryMixin",
    "EfficientTrainingMixin",
    "ArchitectureSearchMixin",
    # Astronomical domain
    "AstronomicalAugmentationMixin",
    "AstronomicalLossMixin",
    # Combined mixins
    "StandardModelMixin",
    "HPOModelMixin",
    "AstronomicalModelMixin",
    "FullAstronomicalModelMixin",
    "LightweightModelMixin",
    "ResearchModelMixin",
    "ExplainableModelMixin",
]
