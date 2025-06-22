"""
Astronomical Tensor System
=========================

Tensor-based astronomical data processing with visualization integration.

This module provides specialized tensors for astronomical data types:
- Spatial coordinates with coordinate transformations
- Photometric measurements across multiple bands
- Spectroscopic data with wavelength operations
- Time series and lightcurve analysis
- Survey data coordination and management
- Orbital mechanics and satellite tracking
- Cosmological simulation data

All tensors include:
- Direct PyVista mesh conversion (to_pyvista())
- Direct Blender object conversion (to_blender())
- Memory-efficient data exchange
- Zero-copy operations where possible
- Astronomical metadata preservation
"""

import datetime
from typing import Any, Dict, List, Optional

# Pydantic configurations for tensor types
from pydantic import BaseModel, ConfigDict, Field

# Import base class first
from .base import AstroTensorBase, transfer_direct
from .clustering import ClusteringTensor

# Import refactored components
from .constants import ASTRO, CONSTANTS, GRAVITY, PHOTOMETRY, SPECTROSCOPY
from .crossmatch import CrossMatchTensor

# Import simple tensor classes (no dependencies)
# Import complex tensors that depend on others (after their dependencies)
from .earth_satellite import EarthSatelliteTensor
from .factory import TensorFactory

# Import data processing tensors
from .feature import FeatureTensor
from .lightcurve import LightcurveTensor
from .orbital import ManeuverTensor, OrbitTensor
from .photometric import PhotometricTensor
from .simulation import CosmologyCalculator, SimulationTensor

# Import spatial tensors (minimal dependencies)
from .spatial_3d import Spatial3DTensor
from .spectral import SpectralTensor
from .statistics import StatisticsTensor

# Import coordinator tensor last (depends on most others)
from .survey import SurveyTensor
from .tensor_types import (
    PhotometricTensorProtocol,
    Spatial3DTensorProtocol,
    SurveyTensorProtocol,
    TensorProtocol,
)
from .transformations import TransformationRegistry, apply_transformation

class SpatialTensorConfig(BaseModel):
    """Configuration for Spatial3DTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    coordinate_system: str = Field(
        default="icrs", description="Coordinate reference system"
    )
    units: Dict[str, str] = Field(
        default_factory=lambda: {"ra": "degrees", "dec": "degrees", "distance": "kpc"}
    )
    epoch: Optional[str] = Field(default="J2000.0", description="Coordinate epoch")

class PhotometricTensorConfig(BaseModel):
    """Configuration for PhotometricTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    bands: List[str] = Field(default_factory=lambda: ["u", "g", "r", "i", "z"])
    magnitude_system: str = Field(default="AB", description="Magnitude system")
    zeropoints: Optional[Dict[str, float]] = Field(default=None)

class SpectralTensorConfig(BaseModel):
    """Configuration for SpectralTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    wavelength_unit: str = Field(default="angstrom", description="Wavelength units")
    flux_unit: str = Field(default="erg/s/cm2/A", description="Flux units")
    spectral_resolution: Optional[float] = Field(default=None, description="R = λ/Δλ")

class LightcurveTensorConfig(BaseModel):
    """Configuration for LightcurveTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    time_format: str = Field(default="mjd", description="Time format")
    time_scale: str = Field(default="utc", description="Time scale")
    bands: List[str] = Field(default_factory=lambda: ["V", "I"])

class OrbitTensorConfig(BaseModel):
    """Configuration for OrbitTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    frame: str = Field(default="ecliptic", description="Reference frame")
    units: Dict[str, str] = Field(
        default_factory=lambda: {"a": "au", "e": "dimensionless", "i": "degrees"}
    )

class SurveyTensorConfig(BaseModel):
    """Configuration for SurveyTensor."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")

    survey_name: str = Field(..., description="Name of the survey")
    data_release: Optional[str] = Field(
        default=None, description="Data release version"
    )
    selection_function: Optional[str] = Field(
        default=None, description="Selection function applied"
    )
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())

__all__ = [
    # Base class
    "AstroTensorBase",
    "transfer_direct",
    # Tensor classes
    "Spatial3DTensor",
    "PhotometricTensor",
    "SpectralTensor",
    "LightcurveTensor",
    "OrbitTensor",
    "ManeuverTensor",
    "EarthSatelliteTensor",
    "SurveyTensor",
    "SimulationTensor",
    "CosmologyCalculator",
    # Data processing tensors
    "FeatureTensor",
    "ClusteringTensor",
    "StatisticsTensor",
    "CrossMatchTensor",
    # Pydantic configuration classes
    "SpatialTensorConfig",
    "PhotometricTensorConfig",
    "SpectralTensorConfig",
    "LightcurveTensorConfig",
    "OrbitTensorConfig",
    "SurveyTensorConfig",
    # Factory functions
    "from_astrometric_data",
    "from_lightcurve_data",
    "from_orbital_elements",
]

# Add refactored modules
__all__.extend(
    [
        # Constants
        "ASTRO",
        "CONSTANTS",
        "GRAVITY",
        "PHOTOMETRY",
        "SPECTROSCOPY",
        # Protocols
        "PhotometricTensorProtocol",
        "Spatial3DTensorProtocol",
        "SurveyTensorProtocol",
        "TensorProtocol",
        # Transformations
        "TransformationRegistry",
        "apply_transformation",
        # Factory
        "TensorFactory",
    ]
)

# Version info
__version__ = "0.3.0"
__author__ = "astro-lab"
__description__ = "Astronomical tensor classes with composition-based architecture"

# Architecture Summary
TENSOR_ARCHITECTURE = {
    "base": "AstroTensorBase - composition-based foundation",
    "spatial": "Spatial3DTensor - 3D coordinates, astrometry, BVH indexing",
    "photometric": "PhotometricTensor - multi-band photometry",
    "spectral": "SpectralTensor - spectroscopic data",
    "temporal": "LightcurveTensor - time series data",
    "orbital": "OrbitTensor + ManeuverTensor - celestial mechanics",
    "earth_satellite": "EarthSatelliteTensor - Earth satellites",
    "coordinator": "SurveyTensor - main coordinator tensor",
}

# Convenience factory functions
def create_spatial_tensor(*args, **kwargs):
    """Create spatial tensor."""
    if len(args) == 3:  # x, y, z
        data = torch.stack(args, dim=-1)
        return Spatial3DTensor(data=data, **kwargs)
    elif len(args) == 1:  # data tensor
        return Spatial3DTensor(data=args[0], **kwargs)
    else:
        raise ValueError("Spatial3DTensor requires either 3 coordinates (x,y,z) or 1 data tensor")

def create_photometric_tensor(*args, **kwargs):
    """Create photometric tensor."""
    return PhotometricTensor(*args, **kwargs)

def create_survey_tensor(*args, **kwargs):
    """Create survey tensor."""
    return SurveyTensor(*args, **kwargs)

def create_simulation_tensor(positions, features=None, **kwargs):
    """Create simulation tensor from TNG50/Illustris data."""
    return SimulationTensor(positions, features=features, **kwargs)

# Factory methods for tensor creation
def from_astrometric_data(ra, dec, parallax=None, pmra=None, pmdec=None, **kwargs):
    """Create Spatial3DTensor from astrometric data."""
    import torch

    # Basic coordinate tensor from RA/Dec
    if isinstance(ra, (list, tuple)):
        ra = torch.tensor(ra, dtype=torch.float32)
    if isinstance(dec, (list, tuple)):
        dec = torch.tensor(dec, dtype=torch.float32)

    # Convert to 3D coordinates (simplified)
    coordinates = torch.stack([ra, dec, torch.zeros_like(ra)], dim=1)

    return Spatial3DTensor(
        data=coordinates,
        ra=ra,
        dec=dec,
        parallax=parallax,
        pmra=pmra,
        pmdec=pmdec,
        **kwargs,
    )

def from_lightcurve_data(times, magnitudes, errors=None, **kwargs):
    """Create LightcurveTensor from time series data."""
    return LightcurveTensor(times=times, magnitudes=magnitudes, errors=errors, **kwargs)

def from_orbital_elements(elements, element_type="keplerian", **kwargs):
    """Create OrbitTensor from orbital elements."""
    return OrbitTensor(data=elements, element_type=element_type, **kwargs)

# Removed migration helpers - prototyping phase

import logging

logger = logging.getLogger(__name__)
logger.info("🧭 Astronomical tensors loaded with integrated visualization support")
