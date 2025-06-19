# Loaders Module

Auto-generated documentation for `astro_lab.data.loaders`

## Functions

### create_astro_dataloader(dataset_name: str, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, **dataset_kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_astrophot_dataloader(catalog_path: str, cutout_size: int = 128, pixel_scale: float = 0.262, magnitude_range: tuple = (10.0, 18.0), k_neighbors: int = 5, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_galaxy_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_exoplanet_dataloader(k_neighbors: int = 5, max_distance: float = 100.0, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_exoplanet_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

Create DataLoader for NASA Exoplanet Archive data.

Parameters
----------
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

### create_gaia_dataloader(magnitude_limit: float = 12.0, k_neighbors: int = 8, max_distance: float = 1.0, batch_size: int = 1, shuffle: bool = False, num_workers: int = 4, pin_memory: bool = True, use_stellar_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_linear_lightcurve_dataloader(max_objects: int = 500, k_neighbors: int = 5, min_observations: int = 50, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_default_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_nsa_dataloader(max_galaxies: int = 10000, k_neighbors: int = 8, distance_threshold: float = 50.0, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_galaxy_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_rrlyrae_dataloader(max_stars: int = 300, k_neighbors: int = 5, period_similarity_threshold: float = 0.1, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_stellar_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_satellite_orbit_dataloader(max_satellites: int = 100, k_neighbors: int = 5, orbital_similarity_threshold: float = 1000.0, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_default_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_sdss_spectral_dataloader(max_spectra: int = 1000, k_neighbors: int = 5, spectral_similarity_threshold: float = 0.8, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_stellar_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### create_tng50_dataloader(particle_type: str = 'PartType0', radius: float = 1.0, max_particles: int = 10000, batch_size: int = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = True, use_galaxy_transforms: bool = True, **kwargs) -> torch_geometric.loader.dataloader.DataLoader

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

### get_default_transforms(dataset_type: str = 'general') -> Optional[torch_geometric.transforms.compose.Compose]

Get default transforms for a given dataset type.

Parameters
----------
dataset_type : str, default "general"
    Type of astronomical data ("general", "stellar", "galaxy", "exoplanet")

Returns
-------
Compose or None
    Composed transform pipeline

### optimize_dataloader_for_gpu()

Optimize PyTorch settings for GPU usage.
