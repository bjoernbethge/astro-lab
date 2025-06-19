"""
Schema definitions for automatic documentation generation.

This package contains Pydantic schemas for all AstroLab modules,
organized by functional area.
"""

from .data_schemas import *
from .tensor_schemas import *
from .model_schemas import *
from .training_schemas import *
from .utils_schemas import *

__all__ = [
    # Data schemas
    "DatasetConfigSchema",
    "DataLoaderConfigSchema", 
    "ProcessingConfigSchema",
    
    # Tensor schemas
    "TensorConfigSchema",
    "SpatialTensorConfigSchema",
    "PhotometricTensorConfigSchema",
    
    # Model schemas
    "ModelConfigSchema",
    "GNNConfigSchema",
    
    # Training schemas
    "TrainingConfigSchema",
    "OptunaConfigSchema",
    
    # Utility schemas
    "BlenderConfigSchema",
    "VisualizationConfigSchema"
] 