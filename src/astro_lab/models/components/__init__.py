"""
Reusable components for AstroLab models.
========================================

Core components used by the consolidated 4-model architecture:
- Base components for device management and graph processing
- Layer utilities for creating MLPs and activation functions
- Output heads for different task types
- Lightning mixins for training functionality
"""

from .base import DeviceMixin, GraphProcessor, PoolingModule, TensorDictFeatureProcessor
from .layers import ResidualBlock, create_conv_layer, create_mlp, get_activation
from .mixins import (
    AstroLightningMixin,
    LossMixin,
    MetricsMixin,
    OptimizerMixin,
    TrainingMixin,
)
from .mixins import (
    DeviceMixin as MixinDeviceMixin,
)
from .output_heads import (
    OUTPUT_HEADS,
    ClassificationHead,
    RegressionHead,
    create_output_head,
)

__all__ = [
    # Base components
    "DeviceMixin",
    "GraphProcessor",
    "TensorDictFeatureProcessor",
    "PoolingModule",
    # Layer functions
    "create_conv_layer",
    "create_mlp",
    "get_activation",
    "ResidualBlock",
    # Output heads
    "ClassificationHead",
    "RegressionHead",
    "create_output_head",
    "OUTPUT_HEADS",
    # Lightning mixins
    "AstroLightningMixin",
    "TrainingMixin",
    "OptimizerMixin",
    "LossMixin",
    "MetricsMixin",
    "MixinDeviceMixin",
]
