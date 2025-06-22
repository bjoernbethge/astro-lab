"""
AstroLab Data Module - High-Performance Astronomical Data Processing

Modern data loading and processing for astronomical surveys using Polars, PyTorch,
and specialized astronomical tensors. Perfect for prototyping and modern astronomical ML pipelines.

Quick Start:
    from astro_lab.data import load_gaia_data
    dataset = load_gaia_data(max_samples=5000)  # Done!
"""

import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# 🌟 PURE CLEAN API - Polars-First Only
# 🔧 CONFIGURATION SYSTEM
from ..datasets import AstroDataset
from .config import (
    DataConfig,
    data_config,
    get_data_dir,
    get_processed_dir,
    get_raw_dir,
    get_survey_paths,
)

# Clean separated imports
from .datamodule import AstroDataModule


# Factory function
def create_astro_datamodule(survey: str, **kwargs) -> AstroDataModule:
    """Create AstroDataModule for given survey."""
    return AstroDataModule(survey=survey, **kwargs)


# 🔧 MANAGER SUPPORT (for CLI compatibility)
from .manager import (
    AstroDataManager,
    data_manager,
    download_bright_all_sky,
    download_gaia,
    import_fits,
    import_tng50,
    list_catalogs,
    load_catalog,
    load_gaia_bright_stars,
    process_for_ml,
)

# 🛠️ PREPROCESSING FUNCTIONS (moved from CLI)
from .preprocessing import (
    preprocess_catalog as preprocess_catalog_new,
)

# 🛠️ UTILITY FUNCTIONS (for preprocessing CLI)
from .utils import (
    check_astroquery_available,
    create_training_splits,
    get_data_dir,
    get_data_statistics,
    get_fits_info,
    load_fits_optimized,
    load_fits_table_optimized,
    load_splits_from_parquet,
    preprocess_catalog,
    save_splits_to_parquet,
)

# Clean exports
__all__ = [
    # 🎯 CORE CLASSES
    "AstroDataset",  # Universal dataset for all surveys
    "AstroDataModule",  # Lightning integration
    # 🏭 FACTORY FUNCTIONS
    "create_astro_dataloader",  # Universal loader factory
    "create_astro_datamodule",  # Universal datamodule factory
    # 🚀 CONVENIENCE FUNCTIONS (Most Common)
    "load_gaia_data",  # One-liner for Gaia
    "load_sdss_data",  # One-liner for SDSS
    "load_nsa_data",  # One-liner for NSA
    "load_lightcurve_data",  # One-liner for lightcurves
    "load_tng50_data",
    "load_tng50_temporal_data",  # 🌟 NEW: TNG50 Temporal loader
    # 🔗 GRAPH CREATION FUNCTIONS - NEW!
    "create_graph_from_dataframe",  # Create graph from DataFrame
    "detect_survey_type",  # Auto-detect survey type
    # 🔧 CONFIGURATION
    "DataConfig",  # New centralized config system
    "data_config",  # Global config instance
    "get_survey_paths",  # Get all paths for a survey
    # 🔧 MANAGER SUPPORT (for CLI)
    "AstroDataManager",
    "data_manager",
    "download_bright_all_sky",
    "download_gaia",
    "list_catalogs",
    "load_catalog",
    "load_gaia_bright_stars",
    "import_fits",
    "import_tng50",
    "process_for_ml",
    # 🛠️ UTILITY FUNCTIONS (for preprocessing CLI)
    "create_training_splits",
    "get_data_statistics",
    "load_splits_from_parquet",
    "preprocess_catalog",
    "save_splits_to_parquet",
    "get_data_dir",
    "check_astroquery_available",
    "load_fits_optimized",
    "load_fits_table_optimized",
    "get_fits_info",
    # 🛠️ PREPROCESSING FUNCTIONS (moved from CLI)
    "preprocess_catalog_new",
]

# Feature flags
HAS_CLEAN_API = True
HAS_MANAGER_API = True  # Supporting data manager for CLI

# Supported surveys
SUPPORTED_SURVEYS = ["gaia", "sdss", "nsa", "linear", "tng50"]

# API version
__version__ = "2.0.0-clean"
