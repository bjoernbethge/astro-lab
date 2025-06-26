"""
TensorDict-based refactoring of astro_lab.tensors
================================================

This implementation uses TensorDict for better performance,
native PyTorch Integration and hierarchical data structures.
"""

from __future__ import annotations

import gc
import math
import weakref
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from ..memory import register_for_cleanup


class AstroTensorDict(TensorDict):
    """
    Basis-class for all astronomical TensorDicts.

    Extends TensorDict with astronomical-specific functionality
    while maintaining native PyTorch performance.
    """

    def __init__(self, data: Dict[str, Any], **kwargs):
        """Initialize TensorDictAstro with metadata handling."""
        # Handle metadata separately to avoid batch dimension issues
        metadata = data.pop("meta", {}) if "meta" in data else {}

        super().__init__(data, **kwargs)

        # Store metadata as instance variable to avoid batch dimension issues
        self._metadata = metadata if metadata else {}

        # Add tensor type information
        if "tensor_type" not in self._metadata:
            self._metadata["tensor_type"] = self.__class__.__name__

        # Register for memory management
        register_for_cleanup(self)

    def _cleanup_metadata(self):
        """Clean up metadata containing large objects."""
        if hasattr(self, "_metadata"):
            for key in list(self._metadata.keys()):
                if isinstance(self._metadata[key], (torch.Tensor, np.ndarray)):
                    del self._metadata[key]

    def clear_temp_tensors(self):
        """Clear temporary tensors to prevent memory leaks."""
        temp_keys = [key for key in self.keys() if key.startswith("_temp_")]
        for key in temp_keys:
            del self[key]

    def optimize_memory(self):
        """Optimize memory usage by clearing temporary data and forcing GC if needed."""
        self.clear_temp_tensors()

        # Force garbage collection if memory usage is high
        if self.memory_info()["total_mb"] > 100:  # > 100MB
            gc.collect()

    def cleanup(self):
        """Explicit cleanup method."""
        try:
            # Clear all data
            self.clear()

            # Clear metadata
            self._cleanup_metadata()

            # Force garbage collection
            gc.collect()

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during destruction

    @property
    def meta(self) -> Dict[str, Any]:
        """Access metadata without batch dimension issues."""
        return self._metadata

    @property
    def n_objects(self) -> int:
        """Number of objects (batch size)."""
        return self.batch_size[0] if self.batch_size else 0

    def add_history(self, operation: str, **details) -> AstroTensorDict:
        """Adds entry to operations history."""
        if "history" not in self._metadata:
            self._metadata["history"] = []

        # Convert to list if it's a tensor (for compatibility)
        history = self._metadata["history"]
        if isinstance(history, torch.Tensor):
            history = []

        history.append({"operation": operation, "details": details})
        self._metadata["history"] = history
        return self

    def memory_info(self) -> Dict[str, Any]:
        """Memory information for all tensors."""
        info = {
            "total_bytes": 0,
            "n_tensors": 0,
            "device": str(self.device) if hasattr(self, "device") else "mixed",
            "batch_size": self.batch_size,
        }

        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                info["total_bytes"] += value.element_size() * value.nelement()
                info["n_tensors"] += 1
                info[f"{key}_shape"] = tuple(value.shape)

        info["total_mb"] = info["total_bytes"] / (1024 * 1024)
        return info


class SpatialTensorDict(AstroTensorDict):
    """
    TensorDict for 3D space coordinates.

    Structure:
    {
        "coordinates": Tensor[N, 3],  # x, y, z or ra, dec, distance
        "meta": {
            "coordinate_system": str,
            "unit": str,
            "epoch": float,
        }
    }
    """

    def __init__(
        self,
        coordinates: torch.Tensor,
        coordinate_system: str = "icrs",
        unit: str = "parsec",
        epoch: float = 2000.0,
        **kwargs,
    ):
        """
        Initialize SpatialTensorDict.

        Args:
            coordinates: [N, 3] Tensor with coordinates
            coordinate_system: Coordinate system
            unit: Unit of coordinates
            epoch: Epoch of coordinates
        """
        if coordinates.shape[-1] != 3:
            raise ValueError(
                f"Coordinates must have shape [..., 3], got {coordinates.shape}"
            )

        data = {
            "coordinates": coordinates,
            "meta": {
                "coordinate_system": coordinate_system,
                "unit": unit,
                "epoch": epoch,
            },
        }

        super().__init__(data, batch_size=coordinates.shape[:-1], **kwargs)

    @property
    def x(self) -> torch.Tensor:
        return self["coordinates"][..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self["coordinates"][..., 1]

    @property
    def z(self) -> torch.Tensor:
        return self["coordinates"][..., 2]

    @property
    def coordinate_system(self) -> str:
        return self._metadata["coordinate_system"]

    def to_spherical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Converts to spherical coordinates (RA, Dec, Distance)."""
        x, y, z = self.x, self.y, self.z

        # Distance
        distance = torch.norm(self["coordinates"], dim=-1)

        # RA (longitude)
        ra = torch.atan2(y, x) * 180 / math.pi
        ra = torch.where(ra < 0, ra + 360, ra)

        # Dec (latitude)
        dec = torch.asin(torch.clamp(z / distance, -1, 1)) * 180 / math.pi

        return ra, dec, distance

    def angular_separation(self, other: SpatialTensorDict) -> torch.Tensor:
        """Calculates angular separation to other coordinates."""
        ra1, dec1, _ = self.to_spherical()
        ra2, dec2, _ = other.to_spherical()

        # Convert to radians
        ra1_rad, dec1_rad = ra1 * math.pi / 180, dec1 * math.pi / 180
        ra2_rad, dec2_rad = ra2 * math.pi / 180, dec2 * math.pi / 180

        # Spherical trigonometry
        cos_sep = torch.sin(dec1_rad) * torch.sin(dec2_rad) + torch.cos(
            dec1_rad
        ) * torch.cos(dec2_rad) * torch.cos(ra1_rad - ra2_rad)

        return torch.acos(torch.clamp(cos_sep, -1, 1)) * 180 / math.pi

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """Finds objects within a cone."""
        center_spatial = SpatialTensorDict(center.unsqueeze(0))
        separations = self.angular_separation(center_spatial)
        return torch.where(separations <= radius_deg)[0]


class PhotometricTensorDict(AstroTensorDict):
    """
    TensorDict for photometric data.

    Structure:
    {
        "magnitudes": Tensor[N, B],  # B = Number of bands
        "errors": Tensor[N, B],      # Optional
        "meta": {
            "bands": List[str],
            "filter_system": str,
            "is_magnitude": bool,
        }
    }
    """

    def __init__(
        self,
        magnitudes: torch.Tensor,
        bands: List[str],
        errors: Optional[torch.Tensor] = None,
        filter_system: str = "AB",
        is_magnitude: bool = True,
        **kwargs,
    ):
        """
        Initialize PhotometricTensorDict.

        Args:
            magnitudes: [N, B] Tensor with magnitudes/fluxes
            bands: List of band names
            errors: Optional errors
            filter_system: Filtersystem
            is_magnitude: True for magnitudes, False for fluxes
        """
        if magnitudes.shape[-1] != len(bands):
            raise ValueError(
                f"Number of bands ({len(bands)}) doesn't match data columns ({magnitudes.shape[-1]})"
            )

        # Einheitlich: meta als dict, nicht TensorDict!
        data = {
            "magnitudes": magnitudes,
            "meta": {
                "bands": bands,
                "filter_system": filter_system,
                "is_magnitude": is_magnitude,
                "n_bands": len(bands),
            },
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=magnitudes.shape[:-1], **kwargs)

    @property
    def bands(self) -> List[str]:
        return self._metadata["bands"]

    @property
    def n_bands(self) -> int:
        return self._metadata["n_bands"]

    @property
    def is_magnitude(self) -> bool:
        return self._metadata["is_magnitude"]

    @property
    def filter_system(self) -> str:
        return self._metadata["filter_system"]

    def get_band(self, band: str) -> torch.Tensor:
        """Extracts data for a specific band."""
        try:
            band_idx = self.bands.index(band)
            return self["magnitudes"][..., band_idx]
        except ValueError:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

    def compute_colors(self, band_pairs: List[Tuple[str, str]]) -> TensorDict:
        """Calculates color indices for band pairs."""
        if not self.is_magnitude:
            raise ValueError("Color computation requires magnitude data")

        colors = {}
        for band1, band2 in band_pairs:
            color_name = f"{band1}_{band2}"
            colors[color_name] = self.get_band(band1) - self.get_band(band2)

        return TensorDict(colors, batch_size=self.batch_size)

    def to_flux(self) -> PhotometricTensorDict:
        """Converts magnitudes to fluxes."""
        if not self.is_magnitude:
            return self.clone()

        # F = 10^(-0.4 * M)
        flux_data = torch.pow(10, -0.4 * self["magnitudes"])

        # Error propagation
        new_errors = None
        if "errors" in self:
            new_errors = (
                flux_data * self["errors"] * (torch.log(torch.tensor(10.0)) / 2.5)
            )

        result = PhotometricTensorDict(
            magnitudes=flux_data,
            bands=self.bands,
            errors=new_errors,
            filter_system=self._metadata["filter_system"],
            is_magnitude=False,
        )
        result.add_history("to_flux")
        return result

    def to_magnitude(self) -> PhotometricTensorDict:
        """Converts fluxes to magnitudes."""
        if self.is_magnitude:
            return self.clone()

        # M = -2.5 * log10(F)
        mag_data = -2.5 * torch.log10(self["magnitudes"])

        # Error propagation
        new_errors = None
        if "errors" in self:
            new_errors = (2.5 / torch.log(torch.tensor(10.0))) * (
                self["errors"] / self["magnitudes"]
            )

        result = PhotometricTensorDict(
            magnitudes=mag_data,
            bands=self.bands,
            errors=new_errors,
            filter_system=self._metadata["filter_system"],
            is_magnitude=True,
        )
        result.add_history("to_magnitude")
        return result


class SpectralTensorDict(AstroTensorDict):
    """
    TensorDict for spectroscopic data.

    Structure:
    {
        "flux": Tensor[N, W],       # W = Number of wavelengths
        "wavelengths": Tensor[W],    # Wavelength grid
        "errors": Tensor[N, W],      # Optional
        "meta": {
            "redshift": float,
            "flux_units": str,
            "wavelength_units": str,
        }
    }
    """

    def __init__(
        self,
        flux: torch.Tensor,
        wavelengths: torch.Tensor,
        errors: Optional[torch.Tensor] = None,
        redshift: float = 0.0,
        flux_units: str = "erg/s/cm2/A",
        wavelength_units: str = "Angstrom",
        **kwargs,
    ):
        if flux.shape[-1] != len(wavelengths):
            raise ValueError(
                f"Flux shape {flux.shape} incompatible with wavelengths length {len(wavelengths)}"
            )

        data = {
            "flux": flux,
            "wavelengths": wavelengths,
            "meta": {
                "redshift": redshift,
                "flux_units": flux_units,
                "wavelength_units": wavelength_units,
                "n_wavelengths": len(wavelengths),
            },
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=flux.shape[:-1], **kwargs)

    @property
    def redshift(self) -> float:
        return self._metadata["redshift"]

    @property
    def rest_wavelengths(self) -> torch.Tensor:
        """Rest-frame wavelengths."""
        return self["wavelengths"] / (1 + self.redshift)

    def apply_redshift(self, z: float) -> SpectralTensorDict:
        """Applies redshift."""
        new_wavelengths = self["wavelengths"] * (1 + z)

        result = SpectralTensorDict(
            flux=self["flux"],
            wavelengths=new_wavelengths,
            errors=self.get("errors"),
            redshift=self.redshift + z,
            flux_units=self._metadata["flux_units"],
            wavelength_units=self._metadata["wavelength_units"],
        )
        result.add_history("apply_redshift", z=z)
        return result

    def normalize(self, wavelength: float) -> SpectralTensorDict:
        """Normalizes to flux at given wavelength."""
        # Find nearest wavelength
        idx = torch.argmin(torch.abs(self["wavelengths"] - wavelength))
        norm_flux = self["flux"][..., idx].unsqueeze(-1)

        normalized_flux = self["flux"] / (norm_flux + 1e-9)

        result = SpectralTensorDict(
            flux=normalized_flux,
            wavelengths=self["wavelengths"],
            errors=self.get("errors"),
            redshift=self.redshift,
            flux_units="normalized",
            wavelength_units=self._metadata["wavelength_units"],
        )
        result.add_history("normalize", wavelength=wavelength)
        return result


class LightcurveTensorDict(AstroTensorDict):
    """
    TensorDict for light curves.

    Structure:
    {
        "times": Tensor[N, T],       # T = Number of time points
        "magnitudes": Tensor[N, T, B], # B = Number of bands
        "errors": Tensor[N, T, B],    # Optional
        "meta": {
            "time_format": str,
            "bands": List[str],
        }
    }
    """

    def __init__(
        self,
        times: torch.Tensor,
        magnitudes: torch.Tensor,
        bands: List[str],
        errors: Optional[torch.Tensor] = None,
        time_format: str = "mjd",
        **kwargs,
    ):
        # Consistency checks
        if times.shape != magnitudes.shape[:-1]:
            raise ValueError(
                f"Times shape {times.shape} incompatible with magnitudes shape {magnitudes.shape}"
            )

        data = {
            "times": times,
            "magnitudes": magnitudes,
            "meta": {
                "time_format": time_format,
                "bands": bands,
                "n_bands": len(bands),
                "n_times": times.shape[-1],
            },
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=times.shape[:-1], **kwargs)

    @property
    def time_span(self) -> torch.Tensor:
        """Time span for each object."""
        return self["times"].max(dim=-1)[0] - self["times"].min(dim=-1)[0]

    def phase_fold(
        self, period: torch.Tensor, epoch: Optional[torch.Tensor] = None
    ) -> LightcurveTensorDict:
        """Phase-folded light curve."""
        if epoch is None:
            epoch = torch.zeros_like(period)

        # Broadcasting for batch operation
        times_expanded = self["times"].unsqueeze(-1)  # [N, T, 1]
        period_expanded = period.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
        epoch_expanded = epoch.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]

        phases = ((times_expanded - epoch_expanded) % period_expanded) / period_expanded
        phases = phases.squeeze(-1)  # [N, T]

        # Sort by phase
        sorted_indices = torch.argsort(phases, dim=-1)

        # Apply sorting to all time-series data
        sorted_times = torch.gather(phases, -1, sorted_indices)
        sorted_magnitudes = torch.gather(
            self["magnitudes"],
            -2,
            sorted_indices.unsqueeze(-1).expand(-1, -1, self["magnitudes"].shape[-1]),
        )

        sorted_errors = None
        if "errors" in self:
            sorted_errors = torch.gather(
                self["errors"],
                -2,
                sorted_indices.unsqueeze(-1).expand(-1, -1, self["errors"].shape[-1]),
            )

        result = LightcurveTensorDict(
            times=sorted_times,
            magnitudes=sorted_magnitudes,
            bands=self._metadata["bands"],
            errors=sorted_errors,
            time_format=self._metadata["time_format"],
        )
        result.add_history("phase_fold", period=period.tolist())
        return result


class SurveyTensorDict(AstroTensorDict):
    """
    Main TensorDict for survey data - coordinates all other tensors.

    Structure:
    {
        "spatial": SpatialTensorDict,
        "photometric": PhotometricTensorDict,
        "spectral": SpectralTensorDict,      # Optional
        "lightcurves": LightcurveTensorDict, # Optional
        "features": Tensor[N, F],            # Additional features
        "meta": {
            "survey_name": str,
            "data_release": str,
            "filter_system": str,
        }
    }
    """

    def __init__(
        self,
        spatial: SpatialTensorDict,
        photometric: PhotometricTensorDict,
        survey_name: str,
        data_release: str = "unknown",
        spectral: Optional[SpectralTensorDict] = None,
        lightcurves: Optional[LightcurveTensorDict] = None,
        features: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Consistency check: All components must have same batch size
        batch_size = spatial.batch_size
        if photometric.batch_size != batch_size:
            raise ValueError("Spatial and photometric data must have same batch size")

        data = {
            "spatial": spatial,
            "photometric": photometric,
            "meta": {
                "survey_name": survey_name,
                "data_release": data_release,
                "filter_system": photometric.filter_system,
                "n_objects": batch_size[0] if batch_size else 0,
            },
        }

        if spectral is not None:
            if spectral.batch_size != batch_size:
                raise ValueError("Spectral data must have same batch size")
            data["spectral"] = spectral

        if lightcurves is not None:
            if lightcurves.batch_size != batch_size:
                raise ValueError("Lightcurve data must have same batch size")
            data["lightcurves"] = lightcurves

        if features is not None:
            data["features"] = features

        super().__init__(data, batch_size=batch_size, **kwargs)

    @property
    def survey_name(self) -> str:
        return self._metadata["survey_name"]

    @property
    def n_objects(self) -> int:
        return self._metadata["n_objects"]

    def cross_match(
        self, other: SurveyTensorDict, tolerance: float = 1.0
    ) -> Tuple[torch.Tensor, ...]:
        """Cross-Match with other survey."""
        # Use spatial coordinates for Cross-Match
        separations = self["spatial"].angular_separation(other["spatial"])
        matches = torch.where(separations <= tolerance)
        return matches

    def compute_colors(self, band_pairs: List[Tuple[str, str]]) -> TensorDict:
        """Delegates to photometric component."""
        return self["photometric"].compute_colors(band_pairs)

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """Delegates to spatial component."""
        return self["spatial"].cone_search(center, radius_deg)

    def query_region(
        self, ra_range: Tuple[float, float], dec_range: Tuple[float, float]
    ) -> SurveyTensorDict:
        """Filters data by RA/Dec region."""
        ra, dec, _ = self["spatial"].to_spherical()

        mask = (
            (ra >= ra_range[0])
            & (ra <= ra_range[1])
            & (dec >= dec_range[0])
            & (dec <= dec_range[1])
        )

        # Apply mask to all components
        filtered_spatial = SpatialTensorDict(
            self["spatial"]["coordinates"][mask],
            coordinate_system=self["spatial"].coordinate_system,
            unit=self["spatial"]["meta"]["unit"],
            epoch=self["spatial"]["meta"]["epoch"],
        )

        filtered_photometric = PhotometricTensorDict(
            self["photometric"]["magnitudes"][mask],
            bands=self["photometric"].bands,
            errors=self["photometric"].get("errors")[mask]
            if "errors" in self["photometric"]
            else None,
            filter_system=self["photometric"]["meta"]["filter_system"],
            is_magnitude=self["photometric"].is_magnitude,
        )

        result = SurveyTensorDict(
            spatial=filtered_spatial,
            photometric=filtered_photometric,
            survey_name=self.survey_name,
            data_release=self._metadata["data_release"],
        )
        result.add_history("query_region", ra_range=ra_range, dec_range=dec_range)
        return result
