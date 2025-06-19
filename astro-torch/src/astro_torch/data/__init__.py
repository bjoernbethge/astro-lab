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
]
