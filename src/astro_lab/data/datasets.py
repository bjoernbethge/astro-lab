"""
PyTorch Geometric Dataset Classes for Astronomical Data
======================================================

Unified access to all astronomical dataset classes.
"""

# Import all dataset classes from the structured submodules
from .datasets import *

# Re-export everything
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

# Direct imports available - prototyping phase 