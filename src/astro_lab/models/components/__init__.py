"""Reusable components for AstroLab models."""

from .base import DeviceMixin, GraphProcessor, PoolingModule, TensorDictFeatureProcessor
from .layers import ResidualBlock, create_conv_layer, create_mlp, get_activation
from .output_heads import (
    OUTPUT_HEADS,
    ClassificationHead,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)

__all__ = [
    # Base components
    "DeviceMixin",
    "GraphProcessor",
    "FeatureProcessor",
    "PoolingModule",
    # Layer functions
    "create_conv_layer",
    "create_mlp",
    "get_activation",
    "ResidualBlock",
    # Output heads
    "ClassificationHead",
    "RegressionHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "create_output_head",
    "OUTPUT_HEADS",
]
