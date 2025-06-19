"""
PyTorch Geometric Dataset Classes for Astronomical Data
======================================================

Organized dataset implementations using PyTorch Geometric's InMemoryDataset.
Each dataset type is organized into logical submodules:

- `astronomical`: Stellar and galaxy datasets (Gaia, NSA, TNG50)
- `exoplanets`: Exoplanet datasets from NASA Exoplanet Archive  
- `spectroscopy`: Spectroscopic datasets (SDSS)
- `time_series`: Time-series datasets (LINEAR, RR Lyrae)
- `satellites`: Satellite orbital datasets
- `base`: Base classes and utilities
"""

# Import base utilities
from .base import get_device, to_device, gpu_knn_graph, AstroLabDataset

# Import astronomical datasets
from .astronomical import GaiaGraphDataset, NSAGraphDataset, TNG50GraphDataset, AstroPhotDataset

# Import exoplanet datasets  
from .exoplanets import ExoplanetGraphDataset

# Import spectroscopic datasets
from .spectroscopy import SDSSSpectralDataset

# Import time-series datasets
from .time_series import LINEARLightcurveDataset, RRLyraeDataset

# Import satellite datasets
from .satellites import SatelliteOrbitDataset

__all__ = [
    # Base utilities
    "get_device",
    "to_device", 
    "gpu_knn_graph",
    "AstroLabDataset",
    
    # Astronomical datasets
    "GaiaGraphDataset",
    "NSAGraphDataset", 
    "TNG50GraphDataset",
    "AstroPhotDataset",
    
    # Exoplanet datasets
    "ExoplanetGraphDataset",
    
    # Spectroscopic datasets
    "SDSSSpectralDataset",
    
    # Time-series datasets
    "LINEARLightcurveDataset",
    "RRLyraeDataset",
    
    # Satellite datasets
    "SatelliteOrbitDataset",
] 