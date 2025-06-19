"""
AstroLab Data Submodule
======================

Modern PyTorch Geometric datasets for astronomical data:
- InMemoryDataset implementations for Gaia DR3 and TNG50
- Native PyTorch Geometric DataLoaders
- Astronomical coordinate transforms
- Efficient data management and processing
"""

# Modern PyTorch Geometric Datasets
from .datasets import (
    AstroLabDataset,
    AstroPhotDataset,
    ExoplanetGraphDataset,
    GaiaGraphDataset,
    LINEARLightcurveDataset,
    NSAGraphDataset,
    RRLyraeDataset,
    SatelliteOrbitDataset,
    SDSSSpectralDataset,
    TNG50GraphDataset,
)

# PyTorch Geometric DataLoaders
from .loaders import (
    create_astro_dataloader,
    create_astrophot_dataloader,
    create_exoplanet_dataloader,
    create_gaia_dataloader,
    create_linear_lightcurve_dataloader,
    create_nsa_dataloader,
    create_rrlyrae_dataloader,
    create_satellite_orbit_dataloader,
    create_sdss_spectral_dataloader,
    create_tng50_dataloader,
    get_default_transforms,
)

# Data Management
from .manager import (
    AstroDataManager,
    data_manager,
    download_bright_all_sky,
    download_gaia,
    import_fits,
    import_tng50,
    list_catalogs,
    load_bright_stars,
    load_catalog,
    load_gaia_bright_stars,
    process_for_ml,
)

# Tensor Processing
from .processing import AstroTensorProcessor, ProcessingConfig

# Simple Dataset (migrated from data_alt)
# AstroLabDataset is now in datasets.py
# Astronomical Transforms
from .transforms import (
    AddAstronomicalColors,
    AddDistanceFeatures,
    AddRedshiftFeatures,
    CoordinateSystemTransform,
    NormalizeAstronomicalFeatures,
    get_default_astro_transforms,
    get_exoplanet_transforms,
    get_galaxy_transforms,
    get_stellar_transforms,
)

# Essential Data Utilities
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

# Mark that enhanced functions are available
HAS_ENHANCED_FEATURES = True

__all__ = [
    # Modern Datasets (PyTorch Geometric)
    "AstroPhotDataset",
    "ExoplanetGraphDataset",
    "GaiaGraphDataset",
    "LINEARLightcurveDataset",
    "NSAGraphDataset",
    "RRLyraeDataset",
    "SatelliteOrbitDataset",
    "SDSSSpectralDataset",
    "TNG50GraphDataset",
    # DataLoaders
    "create_astro_dataloader",
    "create_astrophot_dataloader",
    "create_exoplanet_dataloader",
    "create_gaia_dataloader",
    "create_linear_lightcurve_dataloader",
    "create_nsa_dataloader",
    "create_rrlyrae_dataloader",
    "create_satellite_orbit_dataloader",
    "create_sdss_spectral_dataloader",
    "create_tng50_dataloader",
    "get_default_transforms",
    # Astronomical Transforms
    "AddAstronomicalColors",
    "AddDistanceFeatures",
    "AddRedshiftFeatures",
    "CoordinateSystemTransform",
    "NormalizeAstronomicalFeatures",
    "get_default_astro_transforms",
    "get_exoplanet_transforms",
    "get_galaxy_transforms",
    "get_stellar_transforms",
    # Data Management
    "AstroDataManager",
    "data_manager",
    "download_gaia",
    "download_bright_all_sky",
    "import_fits",
    "import_tng50",
    "list_catalogs",
    "load_catalog",
    "process_for_ml",
    "load_gaia_bright_stars",
    "load_bright_stars",
    # Tensor Processing
    "AstroTensorProcessor",
    "ProcessingConfig",
    # Enhanced Features
    "AstroLabDataset",
    "create_training_splits",
    "get_data_statistics",
    "load_fits_optimized",
    "load_fits_table_optimized",
    "load_splits_from_parquet",
    "get_fits_info",
    "check_astroquery_available",
    "get_data_dir",
    "preprocess_catalog",
    "save_splits_to_parquet",
    # Feature availability
    "HAS_ENHANCED_FEATURES",
]
