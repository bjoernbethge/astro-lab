"""
AstroLab Data Module - High-Performance Astronomical Data Processing

Clean, unified data loading and processing for astronomical surveys using Polars, PyTorch,
and specialized astronomical tensors.

Quick Start:
    from astro_lab.data import load_survey_catalog, preprocess_survey
    df = load_survey_catalog("gaia", max_samples=5000)
    processed_path = preprocess_survey("gaia")
"""

import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# 🔧 CONFIGURATION SYSTEM
from .config import (
    DataConfig,
    data_config,
    get_data_dir,
    get_processed_dir,
    get_raw_dir,
    get_survey_paths,
)

# 🎯 CORE CLASSES
from .datamodule import AstroDataModule


# 🏭 FACTORY FUNCTIONS
def create_astro_datamodule(survey: str, **kwargs) -> AstroDataModule:
    """Create AstroDataModule for given survey."""
    return AstroDataModule(survey=survey, **kwargs)


# 🚀 LOADERS (unified loading functions)
from .loaders import (
    download_2mass,
    download_pan_starrs,
    download_sdss,
    download_survey,
    download_wise,
    import_fits,
    import_tng50,
    list_available_catalogs,
    load_catalog,
    load_survey_catalog,
)

# 🛠️ PROCESSORS (unified preprocessing functions)
from .processors import (
    create_survey_tensordict,
    create_training_splits,
    preprocess_survey,
)

# 🌌 COSMIC WEB ANALYSIS
from .cosmic_web import (
    CosmicWebAnalyzer,
    analyze_gaia_cosmic_web,
    analyze_nsa_cosmic_web,
    analyze_exoplanet_cosmic_web,
)

# 🛠️ UTILITY FUNCTIONS
from .utils import (
    check_astroquery_available,
    detect_survey_type,
    get_data_statistics,
    get_fits_info,
    load_fits_optimized,
    load_fits_table_optimized,
    load_splits_from_parquet,
    save_splits_to_parquet,
)

# Clean exports
__all__ = [
    # 🎯 CORE CLASSES
    "AstroDataModule",
    # 🏭 FACTORY FUNCTIONS
    "create_astro_datamodule",
    # 🚀 LOADERS
    "load_catalog",
    "load_survey_catalog",
    "download_survey",
    "download_sdss",
    "download_2mass",
    "download_wise",
    "download_pan_starrs",
    "import_fits",
    "import_tng50",
    "list_available_catalogs",
    # 🛠️ PROCESSORS
    "preprocess_survey",
    "create_survey_tensordict",
    "create_training_splits",
    # 🌌 COSMIC WEB ANALYSIS
    "CosmicWebAnalyzer",
    "analyze_gaia_cosmic_web",
    "analyze_nsa_cosmic_web",
    "analyze_exoplanet_cosmic_web",
    # 🔧 CONFIGURATION
    "DataConfig",
    "data_config",
    "get_survey_paths",
    "get_data_dir",
    "get_processed_dir",
    "get_raw_dir",
    # 🛠️ UTILITY FUNCTIONS
    "check_astroquery_available",
    "get_data_statistics",
    "get_fits_info",
    "load_fits_optimized",
    "load_fits_table_optimized",
    "load_splits_from_parquet",
    "save_splits_to_parquet",
    "detect_survey_type",
]

# Supported surveys
SUPPORTED_SURVEYS = ["gaia", "sdss", "nsa", "linear", "tng50"]

# API version
__version__ = "3.0.0"
