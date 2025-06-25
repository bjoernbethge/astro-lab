"""
Astronomical TensorDict System
=============================

Modern tensor-based astronomical data processing using TensorDict architecture.
Fully modernized for PyTorch 2.0+ and Lightning integration.

This module provides specialized TensorDicts for astronomical data types:
- Spatial coordinates with coordinate transformations
- Photometric measurements across multiple bands
- Spectroscopic data with wavelength operations
- Time series and lightcurve analysis
- Survey data coordination and management
- Orbital mechanics and satellite tracking
- Cosmological simulation data

All TensorDicts include:
- Native PyTorch integration
- Memory-efficient hierarchical data structures
- Zero-copy operations where possible
- Astronomical metadata preservation
- Lightning DataModule compatibility
"""

import datetime
from typing import Any, Dict, List, Optional

# Pydantic configurations for tensor types
from pydantic import BaseModel, ConfigDict, Field

# Import TensorDict classes from specific modules
from .crossmatch_tensordict import CrossMatchTensorDict

# Import factory functions from existing modules
from .factories import (
    create_2mass_survey,
    create_asteroid_population,
    create_cosmology_sample,
    create_crossmatch_example,
    create_gaia_survey,
    create_generic_survey,
    create_kepler_lightcurves,
    create_kepler_orbits,
    create_nbody_simulation,
    create_pan_starrs_survey,
    create_sdss_survey,
    create_wise_survey,
    merge_surveys,
)
from .feature_tensordict import (
    ClusteringTensorDict,
    FeatureTensorDict,
    StatisticsTensorDict,
)
from .orbital_tensordict import ManeuverTensorDict, OrbitTensorDict
from .satellite_tensordict import EarthSatelliteTensorDict
from .simulation_tensordict import CosmologyTensorDict, SimulationTensorDict

# Core TensorDict classes - import from the actual modules
from .tensordict_astro import (
    AstroTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)


class SpatialTensorConfig(BaseModel):
    """Configuration for SpatialTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    coordinate_system: str = Field(
        default="icrs", description="Coordinate reference system"
    )
    units: Dict[str, str] = Field(
        default_factory=lambda: {"ra": "degrees", "dec": "degrees", "distance": "kpc"}
    )
    epoch: Optional[str] = Field(default="J2000.0", description="Coordinate epoch")


class PhotometricTensorConfig(BaseModel):
    """Configuration for PhotometricTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    bands: List[str] = Field(default_factory=lambda: ["u", "g", "r", "i", "z"])
    magnitude_system: str = Field(default="AB", description="Magnitude system")
    zeropoints: Optional[Dict[str, float]] = Field(default=None)


class SpectralTensorConfig(BaseModel):
    """Configuration for SpectralTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    wavelength_unit: str = Field(default="angstrom", description="Wavelength units")
    flux_unit: str = Field(default="erg/s/cm2/A", description="Flux units")
    spectral_resolution: Optional[float] = Field(default=None, description="R = λ/Δλ")


class LightcurveTensorConfig(BaseModel):
    """Configuration for LightcurveTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    time_format: str = Field(default="mjd", description="Time format")
    time_scale: str = Field(default="utc", description="Time scale")
    bands: List[str] = Field(default_factory=lambda: ["V", "I"])


class OrbitTensorConfig(BaseModel):
    """Configuration for OrbitTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    frame: str = Field(default="ecliptic", description="Reference frame")
    units: Dict[str, str] = Field(
        default_factory=lambda: {"a": "au", "e": "dimensionless", "i": "degrees"}
    )


class SurveyTensorConfig(BaseModel):
    """Configuration for SurveyTensorDict."""

    model_config = ConfigDict(validate_assignment=True, extra="allow")
    survey_name: str = Field(..., description="Name of the survey")
    data_release: Optional[str] = Field(
        default=None, description="Data release version"
    )
    selection_function: Optional[str] = Field(
        default=None, description="Selection function applied"
    )
    created_at: str = Field(default_factory=lambda: datetime.datetime.now().isoformat())


# Modern TensorDict-only exports
__all__ = [
    # Core TensorDict classes
    "AstroTensorDict",
    "SpatialTensorDict",
    "PhotometricTensorDict",
    "SpectralTensorDict",
    "LightcurveTensorDict",
    "SurveyTensorDict",
    # Specialized TensorDict classes
    "OrbitTensorDict",
    "ManeuverTensorDict",
    "EarthSatelliteTensorDict",
    "SimulationTensorDict",
    "CosmologyTensorDict",
    "FeatureTensorDict",
    "StatisticsTensorDict",
    "ClusteringTensorDict",
    "CrossMatchTensorDict",
    # Configuration classes
    "SpatialTensorConfig",
    "PhotometricTensorConfig",
    "SpectralTensorConfig",
    "LightcurveTensorConfig",
    "OrbitTensorConfig",
    "SurveyTensorConfig",
    # Factory functions
    "create_gaia_survey",
    "create_sdss_survey",
    "create_2mass_survey",
    "create_pan_starrs_survey",
    "create_wise_survey",
    "create_generic_survey",
    "create_kepler_lightcurves",
    "create_kepler_orbits",
    "create_asteroid_population",
    "create_nbody_simulation",
    "create_cosmology_sample",
    "create_crossmatch_example",
    "merge_surveys",
    # Utility functions
    "create_spatial_tensor",
    "create_photometric_tensor",
    "create_survey_tensor",
    "create_simulation_tensor",
    "from_astrometric_data",
    "from_lightcurve_data",
    "from_orbital_elements",
]


# Factory functions using TensorDict architecture
def create_spatial_tensor(coordinates, coordinate_system="icrs", **kwargs):
    """Create SpatialTensorDict from coordinates."""
    import torch

    if not isinstance(coordinates, torch.Tensor):
        coordinates = torch.tensor(coordinates, dtype=torch.float32)

    return SpatialTensorDict(coordinates, coordinate_system=coordinate_system, **kwargs)


def create_photometric_tensor(magnitudes, bands, **kwargs):
    """Create PhotometricTensorDict from magnitude data."""
    import torch

    if not isinstance(magnitudes, torch.Tensor):
        magnitudes = torch.tensor(magnitudes, dtype=torch.float32)

    return PhotometricTensorDict(magnitudes, bands, **kwargs)


def create_survey_tensor(spatial, photometric, survey_name, **kwargs):
    """Create SurveyTensorDict from components."""
    return SurveyTensorDict(
        spatial=spatial, photometric=photometric, survey_name=survey_name, **kwargs
    )


def create_simulation_tensor(positions, features=None, **kwargs):
    """Create SimulationTensorDict for N-body data."""
    return SimulationTensorDict(
        positions=positions,
        velocities=kwargs.get("velocities", positions * 0),
        masses=kwargs.get("masses", positions.new_ones(positions.shape[0])),
        **kwargs,
    )


def from_astrometric_data(ra, dec, parallax=None, pmra=None, pmdec=None, **kwargs):
    """Create SpatialTensorDict from astrometric measurements."""
    import torch

    # Convert to tensors
    if not isinstance(ra, torch.Tensor):
        ra = torch.tensor(ra, dtype=torch.float32)
    if not isinstance(dec, torch.Tensor):
        dec = torch.tensor(dec, dtype=torch.float32)

    # Create coordinates tensor
    coords = torch.stack([ra, dec, torch.zeros_like(ra)], dim=-1)

    # Add distance from parallax if available
    if parallax is not None:
        if not isinstance(parallax, torch.Tensor):
            parallax = torch.tensor(parallax, dtype=torch.float32)
        distance = 1000.0 / (torch.abs(parallax) + 1e-6)  # mas to parsec
        coords[..., 2] = distance

    return SpatialTensorDict(coords, coordinate_system="icrs", **kwargs)


def from_lightcurve_data(times, magnitudes, errors=None, **kwargs):
    """Create LightcurveTensorDict from lightcurve data."""
    import torch

    if not isinstance(times, torch.Tensor):
        times = torch.tensor(times, dtype=torch.float32)
    if not isinstance(magnitudes, torch.Tensor):
        magnitudes = torch.tensor(magnitudes, dtype=torch.float32)

    # Ensure proper shape for LightcurveTensorDict
    if magnitudes.dim() == 2:
        magnitudes = magnitudes.unsqueeze(-1)  # Add band dimension

    if errors is not None:
        if not isinstance(errors, torch.Tensor):
            errors = torch.tensor(errors, dtype=torch.float32)
        if errors.dim() == 2:
            errors = errors.unsqueeze(-1)

    return LightcurveTensorDict(
        times=times, magnitudes=magnitudes, bands=["V"], errors=errors, **kwargs
    )


def from_orbital_elements(elements, element_type="keplerian", **kwargs):
    """Create OrbitTensorDict from orbital elements."""
    import torch

    if not isinstance(elements, torch.Tensor):
        elements = torch.tensor(elements, dtype=torch.float32)

    return OrbitTensorDict(elements=elements, **kwargs)
