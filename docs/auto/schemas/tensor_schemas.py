"""
Pydantic schemas for tensor configurations.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TensorConfigSchema(BaseModel):
    """Base configuration schema for astronomical tensors."""
    
    dtype: str = Field(
        default="float32",
        description="Data type for tensor operations"
    )
    device: str = Field(
        default="auto",
        description="Device for tensor storage (auto, cpu, cuda, mps)"
    )


class SpatialTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for spatial tensors."""
    
    coordinate_system: str = Field(
        default="icrs",
        description="Coordinate system (icrs, galactic, ecliptic)"
    )
    units: Dict[str, str] = Field(
        default_factory=lambda: {"position": "kpc", "velocity": "km/s"},
        description="Units for spatial coordinates"
    )
    epoch: Optional[str] = Field(
        default="J2000.0",
        description="Coordinate epoch"
    )


class PhotometricTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for photometric tensors."""
    
    bands: List[str] = Field(
        default_factory=lambda: ["u", "g", "r", "i", "z"],
        description="Photometric bands"
    )
    magnitude_system: str = Field(
        default="AB",
        description="Magnitude system (AB, Vega, ST)"
    )
    zeropoints: Optional[Dict[str, float]] = Field(
        default=None,
        description="Zeropoints for each band"
    )


class SpectralTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for spectral tensors."""
    
    wavelength_unit: str = Field(
        default="angstrom",
        description="Wavelength units"
    )
    flux_unit: str = Field(
        default="erg/s/cm2/A",
        description="Flux units"
    )
    spectral_resolution: Optional[float] = Field(
        default=None,
        description="Spectral resolution R = λ/Δλ"
    )


class LightcurveTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for lightcurve tensors."""
    
    time_format: str = Field(
        default="mjd",
        description="Time format (mjd, jd, isot)"
    )
    time_scale: str = Field(
        default="utc",
        description="Time scale (utc, tai, tt)"
    )
    bands: List[str] = Field(
        default_factory=lambda: ["V", "I"],
        description="Photometric bands for lightcurve"
    )


class OrbitTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for orbital tensors."""
    
    frame: str = Field(
        default="ecliptic",
        description="Reference frame (ecliptic, equatorial)"
    )
    units: Dict[str, str] = Field(
        default_factory=lambda: {
            "position": "AU",
            "velocity": "AU/day",
            "time": "days"
        },
        description="Units for orbital elements"
    )


class SurveyTensorConfigSchema(TensorConfigSchema):
    """Configuration schema for survey tensors."""
    
    survey_name: str = Field(
        ...,
        description="Name of the survey"
    )
    data_release: Optional[str] = Field(
        default=None,
        description="Data release version"
    )
    selection_function: Optional[str] = Field(
        default=None,
        description="Selection function description"
    ) 