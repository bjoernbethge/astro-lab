"""Astronomical datasets with unified TensorDict and PyG integration.

This module provides the central AstroLabInMemoryDataset class for all astronomical data types.
- Stores data internally as TensorDict for domain-specific operations
- Converts to PyG Data/HeteroData for graph operations
- Supports both TensorDict and PyG transforms in the pipeline
"""

from .astrolab import AstroLabInMemoryDataset
from .lightning import AstroLabDataModule

__all__ = [
    "AstroLabInMemoryDataset",
    "AstroLabDataModule",
]
