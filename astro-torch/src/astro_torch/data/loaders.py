"""
PyTorch Geometric DataLoader Functions for Astronomical Data
==========================================================

Create DataLoaders for various astronomical datasets:
- Gaia DR3 stellar catalogs
- TNG50 simulation data
- AstroPhot galaxy fitting
- NSA (NASA Sloan Atlas) galaxies
- NASA Exoplanet Archive data
"""

from pathlib import Path
from typing import List, Optional, Union

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

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
from .transforms import (
    get_default_astro_transforms,
    get_exoplanet_transforms,
    get_galaxy_transforms,
    get_stellar_transforms,
)


def create_gaia_dataloader(
    magnitude_limit: float = 12.0,
    k_neighbors: int = 8,
    max_distance: float = 1.0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_stellar_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for Gaia DR3 stellar data.

    Parameters
    ----------
    magnitude_limit : float, default 12.0
        Magnitude limit for catalog selection
    k_neighbors : int, default 8
        Number of nearest neighbors for graph construction
    max_distance : float, default 1.0
        Maximum distance for connections (degrees)
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_stellar_transforms : bool, default True
        Whether to apply stellar-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for Gaia data
    """
    # Create transform pipeline
    transform = get_stellar_transforms() if use_stellar_transforms else None

    # Create dataset
    dataset = GaiaGraphDataset(
        magnitude_limit=magnitude_limit,
        k_neighbors=k_neighbors,
        max_distance=max_distance,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_tng50_dataloader(
    particle_type: str = "PartType0",
    radius: float = 1.0,
    max_particles: int = 10000,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_galaxy_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for TNG50 simulation data.

    Parameters
    ----------
    particle_type : str, default "PartType0"
        Type of particles to load
    radius : float, default 1.0
        Connection radius in simulation units
    max_particles : int, default 10000
        Maximum number of particles to load
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_galaxy_transforms : bool, default True
        Whether to apply galaxy-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for TNG50 data
    """
    # Create transform pipeline
    transform = get_galaxy_transforms() if use_galaxy_transforms else None

    # Create dataset
    dataset = TNG50GraphDataset(
        particle_type=particle_type,
        radius=radius,
        max_particles=max_particles,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_astrophot_dataloader(
    catalog_path: str,
    cutout_size: int = 128,
    pixel_scale: float = 0.262,
    magnitude_range: tuple = (10.0, 18.0),
    k_neighbors: int = 5,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_galaxy_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for AstroPhot galaxy fitting data.

    Parameters
    ----------
    catalog_path : str
        Path to galaxy catalog
    cutout_size : int, default 128
        Size of image cutouts
    pixel_scale : float, default 0.262
        Pixel scale in arcsec/pixel
    magnitude_range : tuple, default (10.0, 18.0)
        Magnitude range for catalog selection
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_galaxy_transforms : bool, default True
        Whether to apply galaxy-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for AstroPhot data
    """
    # Create transform pipeline
    transform = get_galaxy_transforms() if use_galaxy_transforms else None

    # Create dataset
    dataset = AstroPhotDataset(
        catalog_path=catalog_path,
        cutout_size=cutout_size,
        pixel_scale=pixel_scale,
        magnitude_range=magnitude_range,
        k_neighbors=k_neighbors,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_nsa_dataloader(
    max_galaxies: int = 10000,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_galaxy_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for NSA (NASA Sloan Atlas) galaxy data.

    Parameters
    ----------
    max_galaxies : int, default 10000
        Maximum number of galaxies to include
    k_neighbors : int, default 8
        Number of nearest neighbors for graph construction
    distance_threshold : float, default 50.0
        Maximum distance for connections (Mpc)
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_galaxy_transforms : bool, default True
        Whether to apply galaxy-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for NSA data
    """
    # Create transform pipeline
    transform = get_galaxy_transforms() if use_galaxy_transforms else None

    # Create dataset
    dataset = NSAGraphDataset(
        max_galaxies=max_galaxies,
        k_neighbors=k_neighbors,
        distance_threshold=distance_threshold,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_exoplanet_dataloader(
    max_planets: int = 5000,
    k_neighbors: int = 5,
    max_distance: float = 100.0,  # parsecs
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_exoplanet_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for NASA Exoplanet Archive data.

    Parameters
    ----------
    max_planets : int, default 5000
        Maximum number of planets to include
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    max_distance : float, default 100.0
        Maximum distance for connections (parsecs)
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_exoplanet_transforms : bool, default True
        Whether to apply exoplanet-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for exoplanet data
    """
    # Create transform pipeline
    transform = get_exoplanet_transforms() if use_exoplanet_transforms else None

    # Create dataset
    dataset = ExoplanetGraphDataset(
        max_planets=max_planets,
        k_neighbors=k_neighbors,
        max_distance=max_distance,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_sdss_spectral_dataloader(
    max_spectra: int = 1000,
    k_neighbors: int = 5,
    spectral_similarity_threshold: float = 0.8,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_stellar_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for SDSS spectral data.

    Parameters
    ----------
    max_spectra : int, default 1000
        Maximum number of spectra to include
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    spectral_similarity_threshold : float, default 0.8
        Threshold for spectral similarity connections
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_stellar_transforms : bool, default True
        Whether to apply stellar-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for SDSS spectral data
    """
    # Create transform pipeline
    transform = get_stellar_transforms() if use_stellar_transforms else None

    # Create dataset
    dataset = SDSSSpectralDataset(
        max_spectra=max_spectra,
        k_neighbors=k_neighbors,
        spectral_similarity_threshold=spectral_similarity_threshold,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_linear_lightcurve_dataloader(
    max_objects: int = 500,
    k_neighbors: int = 5,
    min_observations: int = 50,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_default_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for LINEAR lightcurve data.

    Parameters
    ----------
    max_objects : int, default 500
        Maximum number of objects to include
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    min_observations : int, default 50
        Minimum number of observations per object
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_default_transforms : bool, default True
        Whether to apply default transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for LINEAR lightcurve data
    """
    # Create transform pipeline
    transform = get_default_astro_transforms() if use_default_transforms else None

    # Create dataset
    dataset = LINEARLightcurveDataset(
        max_objects=max_objects,
        k_neighbors=k_neighbors,
        min_observations=min_observations,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_rrlyrae_dataloader(
    max_stars: int = 300,
    k_neighbors: int = 5,
    period_similarity_threshold: float = 0.1,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_stellar_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for RR Lyrae variable star data.

    Parameters
    ----------
    max_stars : int, default 300
        Maximum number of RR Lyrae stars to include
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    period_similarity_threshold : float, default 0.1
        Threshold for period similarity connections (days)
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_stellar_transforms : bool, default True
        Whether to apply stellar-optimized transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for RR Lyrae data
    """
    # Create transform pipeline
    transform = get_stellar_transforms() if use_stellar_transforms else None

    # Create dataset
    dataset = RRLyraeDataset(
        max_stars=max_stars,
        k_neighbors=k_neighbors,
        period_similarity_threshold=period_similarity_threshold,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_satellite_orbit_dataloader(
    max_satellites: int = 100,
    k_neighbors: int = 5,
    orbital_similarity_threshold: float = 1000.0,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    use_default_transforms: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create DataLoader for satellite orbit data.

    Parameters
    ----------
    max_satellites : int, default 100
        Maximum number of satellites to include
    k_neighbors : int, default 5
        Number of nearest neighbors for graph construction
    orbital_similarity_threshold : float, default 1000.0
        Threshold for orbital similarity connections (km)
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    use_default_transforms : bool, default True
        Whether to apply default transforms
    **kwargs
        Additional arguments for DataLoader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader for satellite orbit data
    """
    # Create transform pipeline
    transform = get_default_astro_transforms() if use_default_transforms else None

    # Create dataset
    dataset = SatelliteOrbitDataset(
        max_satellites=max_satellites,
        k_neighbors=k_neighbors,
        orbital_similarity_threshold=orbital_similarity_threshold,
        transform=transform,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs,
    )


def create_astro_dataloader(
    dataset_name: str,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = True,
    **dataset_kwargs,
) -> DataLoader:
    """
    Create a general astronomical DataLoader with automatic dataset selection.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ("gaia", "tng50", "nsa", "exoplanet", "astrophot")
    batch_size : int, default 1
        Batch size for DataLoader
    shuffle : bool, default False
        Whether to shuffle the dataset
    num_workers : int, default 0
        Number of worker processes for data loading
    pin_memory : bool, default True
        Whether to pin memory for GPU transfer
    **dataset_kwargs
        Additional arguments passed to the specific dataset loader

    Returns
    -------
    DataLoader
        PyTorch Geometric DataLoader

    Raises
    ------
    ValueError
        If dataset_name is not recognized
    """
    loaders = {
        "gaia": create_gaia_dataloader,
        "tng50": create_tng50_dataloader,
        "nsa": create_nsa_dataloader,
        "exoplanet": create_exoplanet_dataloader,
        "astrophot": create_astrophot_dataloader,
    }

    if dataset_name not in loaders:
        available = list(loaders.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    return loaders[dataset_name](
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **dataset_kwargs,
    )


# === Convenience Functions ===


def get_default_transforms(dataset_type: str = "general") -> Optional[Compose]:
    """
    Get default transforms for a given dataset type.

    Parameters
    ----------
    dataset_type : str, default "general"
        Type of astronomical data ("general", "stellar", "galaxy", "exoplanet")

    Returns
    -------
    Compose or None
        Composed transform pipeline
    """
    transforms = {
        "general": get_default_astro_transforms(),
        "stellar": get_stellar_transforms(),
        "galaxy": get_galaxy_transforms(),
        "exoplanet": get_exoplanet_transforms(),
    }

    return transforms.get(dataset_type)
