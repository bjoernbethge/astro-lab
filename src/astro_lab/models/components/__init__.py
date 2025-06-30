"""
Model Components for AstroLab
============================

Modular components for astronomical neural networks.
"""

# Encoders
from .encoders import (
    MultiModalFusion,
    PhotometricEncoder,
    SpatialEncoder,
    SpectralEncoder,
    TemporalEncoder,
)

# Layers
from .layers import (
    BaseAttentionLayer,
    # Base layers
    BaseGraphLayer,
    BasePoolingLayer,
    TensorDictLayer,
)
from .layers.convolution import (
    AstronomicalGraphConv,
    DistanceEncoder,
    FlexibleGraphConv,
)
from .layers.normalization import (
    AdaptiveNormalization,
    AstronomicalFeatureNorm,
    RobustNormalization,
)
from .layers.pooling import (
    AdaptivePooling,
    AttentivePooling,
    HierarchicalPooling,
    MultiScalePooling,
    StatisticalPooling,
)

# Mixins
from .mixins import (
    # Astronomical domain
    AstronomicalAugmentationMixin,
    AstronomicalLossMixin,
    # HPO specific
    HPOResetMixin,
    # Core functionality
    MetricsMixin,
    OptimizationMixin,
    VisualizationMixin,
)

# Output heads
from .output_heads import (
    ClassificationHead,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)

__all__ = [
    # Encoders
    "PhotometricEncoder",
    "SpatialEncoder",
    "SpectralEncoder",
    "TemporalEncoder",
    "MultiModalFusion",
    # Base layers
    "BaseGraphLayer",
    "BasePoolingLayer",
    "BaseAttentionLayer",
    "TensorDictLayer",
    # Convolution layers
    "FlexibleGraphConv",
    "AstronomicalGraphConv",
    "DistanceEncoder",
    # Pooling layers
    "MultiScalePooling",
    "AttentivePooling",
    "HierarchicalPooling",
    "StatisticalPooling",
    "AdaptivePooling",
    # Normalization layers
    "AdaptiveNormalization",
    "AstronomicalFeatureNorm",
    "RobustNormalization",
    # Mixins
    "MetricsMixin",
    "OptimizationMixin",
    "VisualizationMixin",
    "HPOResetMixin",
    "AstronomicalAugmentationMixin",
    "AstronomicalLossMixin",
    # Output heads
    "ClassificationHead",
    "RegressionHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "create_output_head",
]
