"""
Data Loaders for AstroLab
========================

Data loading utilities for astronomical datasets.

Note: The ProcessedDataLoader has been deprecated in favor of the new API.
Use `astro_lab.data.get_preprocessor()` for loading raw data or
`astro_lab.data.create_datamodule()` for creating PyTorch Lightning data modules.
"""

# Import from the processed module for backward compatibility
from .processed import ProcessedDataLoader

__all__ = ["ProcessedDataLoader"]
