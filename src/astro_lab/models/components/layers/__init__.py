"""
Layers for AstroLab Models
=========================

Modular layers for astronomical neural networks.
"""

# Base layers
from .base import (
    BaseGraphLayer,
    BasePoolingLayer,
    BaseAttentionLayer,
    TensorDictLayer,
)

# Convolution layers
from .convolution import (
    FlexibleGraphConv,
    AstronomicalGraphConv,
    DistanceEncoder,
)

# Pooling layers
from .pooling import (
    MultiScalePooling,
    AttentivePooling,
    HierarchicalPooling,
    StatisticalPooling,
    AdaptivePooling,
)

# Normalization layers
from .normalization import (
    AdaptiveNormalization,
    AstronomicalFeatureNorm,
    RobustNormalization,
)

__all__ = [
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
]
