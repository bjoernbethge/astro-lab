"""
Layers for AstroLab Models
=========================

Neural network layers for astronomical data processing.
"""

# Base layers
from .base import (
    BaseGraphLayer,
    BaseLayer,
    BasePoolingLayer,
)

# Convolution layers
from .convolution import (
    AstronomicalGraphConv,
    FlexibleGraphConv,
)

# Graph layers
from .graph import GraphPooling

# Heterogeneous layers
from .hetero import HeteroGNNLayer

# Normalization layers
from .normalization import (
    BatchNorm,
    GraphNorm,
    InstanceNorm,
    LayerNorm,
)

# Point cloud layers
from .point_cloud import (
    AdaptivePointCloudLayer,
    AstroPointCloudLayer,
    MultiScalePointCloudEncoder,
)

# Pooling layers
from .pooling import (
    AdaptivePooling,
    AttentivePooling,
    HierarchicalPooling,
    LambdaPooling,
    MultiScalePooling,
    StatisticalPooling,
)

__all__ = [
    # Base layers
    "BaseLayer",
    "BaseGraphLayer",
    "BasePoolingLayer",
    # Convolution layers
    "FlexibleGraphConv",
    "AstronomicalGraphConv",
    # Graph layers
    "GraphPooling",
    # Heterogeneous layers
    "HeteroGNNLayer",
    # Normalization layers
    "GraphNorm",
    "LayerNorm",
    "BatchNorm",
    "InstanceNorm",
    # Point cloud layers
    "AstroPointCloudLayer",
    "MultiScalePointCloudEncoder",
    "AdaptivePointCloudLayer",
    # Pooling layers
    "AttentivePooling",
    "MultiScalePooling",
    "HierarchicalPooling",
    "StatisticalPooling",
    "AdaptivePooling",
    "LambdaPooling",
]
