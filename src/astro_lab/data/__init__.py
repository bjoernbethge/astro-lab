"""
AstroLab Data Module - Pure Clean API 🌟
========================================

Saubere Polars-First Implementation ohne Legacy-Wrapper.
Perfekt für Prototyping und moderne astronomische ML-Pipelines.

Quick Start:
    from astro_lab.data import load_gaia_data
    dataset = load_gaia_data(max_samples=5000)  # Done!
"""

# 🌟 PURE CLEAN API - Polars-First Only
# 🔧 CONFIGURATION SYSTEM
from .config import (
    DataConfig,
    data_config,
    get_data_dir,
    get_processed_dir,
    get_raw_dir,
    get_survey_paths,
)
from .core import (
    # Survey Configuration (for advanced users)
    SURVEY_CONFIGS,
    AstroDataModule,
    # Core Classes
    AstroDataset,
    # Factory Functions
    create_astro_dataloader,
    create_astro_datamodule,
    create_graph_datasets_from_splits,
    # 🔗 GRAPH CREATION FUNCTIONS - NEW!
    create_graph_from_dataframe,
    detect_survey_type,
    # Convenience Functions for Common Surveys
    load_gaia_data,  # Stellar catalogs with astrometry
    load_lightcurve_data,  # Variable stars/lightcurves
    load_nsa_data,  # NASA Sloan Atlas galaxies
    load_sdss_data,  # Galaxy photometry & spectroscopy
    load_tng50_data,
    load_tng50_temporal_data,  # 🌟 NEW: TNG50 Temporal loader
)

# 🔧 LEGACY MANAGER SUPPORT (for CLI compatibility)
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

# 🛠️ PREPROCESSING FUNCTIONS (moved from CLI)
from .preprocessing import (
    create_graph_datasets_from_splits,
    create_graph_from_dataframe,
    preprocess_catalog as preprocess_catalog_new,
)

# Clean exports - no legacy bloat
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
    "create_graph_datasets_from_splits",  # Create graphs from splits
    "detect_survey_type",  # Auto-detect survey type
    # 🔧 CONFIGURATION
    "SURVEY_CONFIGS",  # Survey definitions (DRY)
    "DataConfig",  # New centralized config system
    "data_config",  # Global config instance
    "get_survey_paths",  # Get all paths for a survey
    # 🔧 LEGACY MANAGER SUPPORT (for CLI)
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
    "create_graph_datasets_from_splits",
    "create_graph_from_dataframe", 
    "preprocess_catalog_new",
]

# Feature flags
HAS_CLEAN_API = True
HAS_LEGACY_API = True  # Now supporting legacy manager for CLI

# Supported surveys
SUPPORTED_SURVEYS = list(SURVEY_CONFIGS.keys())

# API version
__version__ = "2.0.0-clean"
