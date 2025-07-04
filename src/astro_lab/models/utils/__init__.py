"""
Model Utilities
==============

Utility functions for astronomical neural networks with TensorDict support.
"""

from .analysis import count_parameters, get_model_summary
from .benchmarking import benchmark_model_performance
from .ensemble import create_model_ensemble
from .tensordict_utils import (
    convert_model_to_tensordict,
    validate_tensordict_compatibility,
)

__all__ = [
    # Model analysis
    "count_parameters",
    "get_model_summary",
    # TensorDict utilities
    "validate_tensordict_compatibility",
    "convert_model_to_tensordict",
    # Ensemble methods
    "create_model_ensemble",
    # Performance benchmarking
    "benchmark_model_performance",
]
