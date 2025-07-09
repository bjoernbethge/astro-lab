"""
Output Heads for AstroLab Models
===============================

Task-specific output heads for different prediction tasks.
"""

from .classification import ClassificationHead
from .regression import RegressionHead
from .specialized import PeriodDetectionHead, ShapeModelingHead

__all__ = [
    "ClassificationHead",
    "RegressionHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
]
