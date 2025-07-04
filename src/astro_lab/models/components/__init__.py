"""
Model Components for AstroLab
============================

Modular components for astronomical neural networks.
"""

# Base components and factory functions
from .base import (
    AdvancedTemporalEncoder,
    EnhancedMLPBlock,
    GraphPooling,
    ModernGraphEncoder,
    PointNetEncoder,
    TaskSpecificHead,
    create_encoder,
    create_graph_layer,
)

# Output heads and their factory function
from .outputs import (
    ClassificationHead,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)

__all__ = [
    # Base components
    "EnhancedMLPBlock",
    "ModernGraphEncoder",
    "AdvancedTemporalEncoder",
    "PointNetEncoder",
    "TaskSpecificHead",
    "GraphPooling",
    # Factory functions
    "create_encoder",
    "create_graph_layer",
    # Output heads
    "ClassificationHead",
    "RegressionHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "create_output_head",
]
