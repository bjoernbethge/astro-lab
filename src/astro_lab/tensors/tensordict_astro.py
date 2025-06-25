"""
TensorDict-basierte Refaktorierung der astro_lab.tensors
======================================================

Diese Implementierung nutzt TensorDict für bessere Performance,
native PyTorch Integration und hierarchische Datenstrukturen.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
import numpy as np


class AstroTensorDict(TensorDict):
    """
    Basis-Klasse für alle astronomischen TensorDicts.

    Erweitert TensorDict um astronomie-spezifische Funktionalität
    während die native PyTorch Performance beibehalten wird.
    """

    def __init__(self, data: Dict[str, torch.Tensor], **kwargs):
        """
        Initialisiert AstroTensorDict.

        Args:
            data: Dictionary mit Tensoren und Metadaten
            **kwargs: Zusätzliche TensorDict Parameter
        """
        # Ensure all tensors have consistent batch size
        if data:
            batch_sizes = [v.shape[0] for v in data.values() if isinstance(v, torch.Tensor) and v.dim() > 0]
            if batch_sizes and not all(b == batch_sizes[0] for b in batch_sizes):
                raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

        super().__init__(data, **kwargs)

        # Add tensor type information
        if "meta" not in self:
            self["meta"] = TensorDict({}, batch_size=self.batch_size)

        if "tensor_type" not in self["meta"]:
            self["meta", "tensor_type"] = self.__class__.__name__

    @property
    def n_objects(self) -> int:
        """Anzahl der Objekte (Batch-Größe)."""
        return self.batch_size[0] if self.batch_size else 0

    def add_history(self, operation: str, **details) -> AstroTensorDict:
        """Fügt Eintrag zur Operationshistorie hinzu."""
        if "history" not in self["meta"]:
            self["meta", "history"] = []

        # Convert to list if it's a tensor (for compatibility)
        history = self["meta", "history"]
        if isinstance(history, torch.Tensor):
            history = []

        history.append({"operation": operation, "details": details})
        self["meta", "history"] = history
        return self

    def memory_info(self) -> Dict[str, Any]:
        """Speicher-Informationen für alle Tensoren."""
        info = {
            "total_bytes": 0,
            "n_tensors": 0,
            "device": str(self.device) if hasattr(self, 'device') else "mixed",
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
    TensorDict für 3D-Raumkoordinaten.

    Struktur:
    {
        "coordinates": Tensor[N, 3],  # x, y, z oder ra, dec, distance
        "meta": {
            "coordinate_system": str,
            "unit": str,
            "epoch": float,
        }
    }
    """

    def __init__(self, coordinates: torch.Tensor, coordinate_system: str = "icrs", 
                 unit: str = "parsec", epoch: float = 2000.0, **kwargs):
        """
        Initialisiert SpatialTensorDict.

        Args:
            coordinates: [N, 3] Tensor mit Koordinaten
            coordinate_system: Koordinatensystem
            unit: Einheit der Koordinaten
            epoch: Epoche der Koordinaten
        """
        if coordinates.shape[-1] != 3:
            raise ValueError(f"Coordinates must have shape [..., 3], got {coordinates.shape}")

        data = {
            "coordinates": coordinates,
            "meta": TensorDict({
                "coordinate_system": coordinate_system,
                "unit": unit,
                "epoch": epoch,
            }, batch_size=coordinates.shape[:-1])
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
        return self["meta", "coordinate_system"]

    def to_spherical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Konvertiert zu sphärischen Koordinaten (RA, Dec, Distanz)."""
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
        """Berechnet Winkeltrennung zu anderen Koordinaten."""
        ra1, dec1, _ = self.to_spherical()
        ra2, dec2, _ = other.to_spherical()

        # Konvertiere zu Radians
        ra1_rad, dec1_rad = ra1 * math.pi / 180, dec1 * math.pi / 180
        ra2_rad, dec2_rad = ra2 * math.pi / 180, dec2 * math.pi / 180

        # Sphärische Trigonometrie
        cos_sep = (torch.sin(dec1_rad) * torch.sin(dec2_rad) + 
                   torch.cos(dec1_rad) * torch.cos(dec2_rad) * 
                   torch.cos(ra1_rad - ra2_rad))

        return torch.acos(torch.clamp(cos_sep, -1, 1)) * 180 / math.pi

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """Findet Objekte innerhalb eines Kegels."""
        center_spatial = SpatialTensorDict(center.unsqueeze(0))
        separations = self.angular_separation(center_spatial)
        return torch.where(separations <= radius_deg)[0]


class PhotometricTensorDict(AstroTensorDict):
    """
    TensorDict für photometrische Daten.

    Struktur:
    {
        "magnitudes": Tensor[N, B],  # B = Anzahl Bänder
        "errors": Tensor[N, B],      # Optional
        "meta": {
            "bands": List[str],
            "filter_system": str,
            "is_magnitude": bool,
        }
    }
    """

    def __init__(self, magnitudes: torch.Tensor, bands: List[str], 
                 errors: Optional[torch.Tensor] = None, filter_system: str = "AB",
                 is_magnitude: bool = True, **kwargs):
        """
        Initialisiert PhotometricTensorDict.

        Args:
            magnitudes: [N, B] Tensor mit Magnituden/Flüssen
            bands: Liste der Bandnamen
            errors: Optionale Fehler
            filter_system: Filtersystem
            is_magnitude: True für Magnituden, False für Flüsse
        """
        if magnitudes.shape[-1] != len(bands):
            raise ValueError(f"Number of bands ({len(bands)}) doesn't match data columns ({magnitudes.shape[-1]})")

        data = {
            "magnitudes": magnitudes,
            "meta": TensorDict({
                "bands": bands,
                "filter_system": filter_system,
                "is_magnitude": is_magnitude,
                "n_bands": len(bands),
            }, batch_size=magnitudes.shape[:-1])
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=magnitudes.shape[:-1], **kwargs)

    @property
    def bands(self) -> List[str]:
        return self["meta", "bands"]

    @property
    def n_bands(self) -> int:
        return self["meta", "n_bands"]

    @property
    def is_magnitude(self) -> bool:
        return self["meta", "is_magnitude"]

    def get_band(self, band: str) -> torch.Tensor:
        """Extrahiert Daten für ein spezifisches Band."""
        try:
            band_idx = self.bands.index(band)
            return self["magnitudes"][..., band_idx]
        except ValueError:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

    def compute_colors(self, band_pairs: List[Tuple[str, str]]) -> TensorDict:
        """Berechnet Farbindizes für Bandpaare."""
        if not self.is_magnitude:
            raise ValueError("Color computation requires magnitude data")

        colors = {}
        for band1, band2 in band_pairs:
            color_name = f"{band1}_{band2}"
            colors[color_name] = self.get_band(band1) - self.get_band(band2)

        return TensorDict(colors, batch_size=self.batch_size)

    def to_flux(self) -> PhotometricTensorDict:
        """Konvertiert Magnituden zu Flüssen."""
        if not self.is_magnitude:
            return self.clone()

        # F = 10^(-0.4 * M)
        flux_data = torch.pow(10, -0.4 * self["magnitudes"])

        # Fehler-Propagation
        new_errors = None
        if "errors" in self:
            new_errors = flux_data * self["errors"] * (torch.log(torch.tensor(10.0)) / 2.5)

        return PhotometricTensorDict(
            magnitudes=flux_data,
            bands=self.bands,
            errors=new_errors,
            filter_system=self["meta", "filter_system"],
            is_magnitude=False
        ).add_history("to_flux")

    def to_magnitude(self) -> PhotometricTensorDict:
        """Konvertiert Flüsse zu Magnituden."""
        if self.is_magnitude:
            return self.clone()

        # M = -2.5 * log10(F)
        mag_data = -2.5 * torch.log10(self["magnitudes"])

        # Fehler-Propagation
        new_errors = None
        if "errors" in self:
            new_errors = (2.5 / torch.log(torch.tensor(10.0))) * (self["errors"] / self["magnitudes"])

        return PhotometricTensorDict(
            magnitudes=mag_data,
            bands=self.bands,
            errors=new_errors,
            filter_system=self["meta", "filter_system"],
            is_magnitude=True
        ).add_history("to_magnitude")


class SpectralTensorDict(AstroTensorDict):
    """
    TensorDict für spektroskopische Daten.

    Struktur:
    {
        "flux": Tensor[N, W],       # W = Anzahl Wellenlängen
        "wavelengths": Tensor[W],    # Wellenlängen-Grid
        "errors": Tensor[N, W],      # Optional
        "meta": {
            "redshift": float,
            "flux_units": str,
            "wavelength_units": str,
        }
    }
    """

    def __init__(self, flux: torch.Tensor, wavelengths: torch.Tensor,
                 errors: Optional[torch.Tensor] = None, redshift: float = 0.0,
                 flux_units: str = "erg/s/cm2/A", wavelength_units: str = "Angstrom", **kwargs):

        if flux.shape[-1] != len(wavelengths):
            raise ValueError(f"Flux shape {flux.shape} incompatible with wavelengths length {len(wavelengths)}")

        data = {
            "flux": flux,
            "wavelengths": wavelengths,
            "meta": TensorDict({
                "redshift": redshift,
                "flux_units": flux_units,
                "wavelength_units": wavelength_units,
                "n_wavelengths": len(wavelengths),
            }, batch_size=flux.shape[:-1])
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=flux.shape[:-1], **kwargs)

    @property
    def redshift(self) -> float:
        return self["meta", "redshift"]

    @property
    def rest_wavelengths(self) -> torch.Tensor:
        """Rest-frame Wellenlängen."""
        return self["wavelengths"] / (1 + self.redshift)

    def apply_redshift(self, z: float) -> SpectralTensorDict:
        """Wendet Rotverschiebung an."""
        new_wavelengths = self["wavelengths"] * (1 + z)

        return SpectralTensorDict(
            flux=self["flux"],
            wavelengths=new_wavelengths,
            errors=self.get("errors"),
            redshift=self.redshift + z,
            flux_units=self["meta", "flux_units"],
            wavelength_units=self["meta", "wavelength_units"]
        ).add_history("apply_redshift", z=z)

    def normalize(self, wavelength: float) -> SpectralTensorDict:
        """Normalisiert auf Fluss bei gegebener Wellenlänge."""
        # Finde nächste Wellenlänge
        idx = torch.argmin(torch.abs(self["wavelengths"] - wavelength))
        norm_flux = self["flux"][..., idx].unsqueeze(-1)

        normalized_flux = self["flux"] / (norm_flux + 1e-9)

        return SpectralTensorDict(
            flux=normalized_flux,
            wavelengths=self["wavelengths"],
            errors=self.get("errors"),
            redshift=self.redshift,
            flux_units="normalized",
            wavelength_units=self["meta", "wavelength_units"]
        ).add_history("normalize", wavelength=wavelength)


class LightcurveTensorDict(AstroTensorDict):
    """
    TensorDict für Lichtkurven.

    Struktur:
    {
        "times": Tensor[N, T],       # T = Anzahl Zeitpunkte
        "magnitudes": Tensor[N, T, B], # B = Anzahl Bänder
        "errors": Tensor[N, T, B],    # Optional
        "meta": {
            "time_format": str,
            "bands": List[str],
        }
    }
    """

    def __init__(self, times: torch.Tensor, magnitudes: torch.Tensor,
                 bands: List[str], errors: Optional[torch.Tensor] = None,
                 time_format: str = "mjd", **kwargs):

        # Konsistenz-Checks
        if times.shape != magnitudes.shape[:-1]:
            raise ValueError(f"Times shape {times.shape} incompatible with magnitudes shape {magnitudes.shape}")

        data = {
            "times": times,
            "magnitudes": magnitudes,
            "meta": TensorDict({
                "time_format": time_format,
                "bands": bands,
                "n_bands": len(bands),
                "n_times": times.shape[-1],
            }, batch_size=times.shape[:-1])
        }

        if errors is not None:
            data["errors"] = errors

        super().__init__(data, batch_size=times.shape[:-1], **kwargs)

    @property
    def time_span(self) -> torch.Tensor:
        """Zeitspanne für jedes Objekt."""
        return self["times"].max(dim=-1)[0] - self["times"].min(dim=-1)[0]

    def phase_fold(self, period: torch.Tensor, epoch: torch.Tensor = None) -> LightcurveTensorDict:
        """Phasen-gefaltete Lichtkurve."""
        if epoch is None:
            epoch = torch.zeros_like(period)

        # Broadcasting für Batch-Operation
        times_expanded = self["times"].unsqueeze(-1)  # [N, T, 1]
        period_expanded = period.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1] 
        epoch_expanded = epoch.unsqueeze(-1).unsqueeze(-1)    # [N, 1, 1]

        phases = ((times_expanded - epoch_expanded) % period_expanded) / period_expanded
        phases = phases.squeeze(-1)  # [N, T]

        # Sortiere nach Phase
        sorted_indices = torch.argsort(phases, dim=-1)

        # Apply sorting to all time-series data
        sorted_times = torch.gather(phases, -1, sorted_indices)
        sorted_magnitudes = torch.gather(
            self["magnitudes"], -2, 
            sorted_indices.unsqueeze(-1).expand(-1, -1, self["magnitudes"].shape[-1])
        )

        sorted_errors = None
        if "errors" in self:
            sorted_errors = torch.gather(
                self["errors"], -2,
                sorted_indices.unsqueeze(-1).expand(-1, -1, self["errors"].shape[-1])
            )

        return LightcurveTensorDict(
            times=sorted_times,
            magnitudes=sorted_magnitudes,
            bands=self["meta", "bands"],
            errors=sorted_errors,
            time_format="phase"
        ).add_history("phase_fold", period=period.tolist())


class SurveyTensorDict(AstroTensorDict):
    """
    Haupt-TensorDict für Survey-Daten - koordiniert alle anderen Tensoren.

    Struktur:
    {
        "spatial": SpatialTensorDict,
        "photometric": PhotometricTensorDict,
        "spectral": SpectralTensorDict,      # Optional
        "lightcurves": LightcurveTensorDict, # Optional
        "features": Tensor[N, F],            # Zusätzliche Features
        "meta": {
            "survey_name": str,
            "data_release": str,
            "filter_system": str,
        }
    }
    """

    def __init__(self, spatial: SpatialTensorDict, photometric: PhotometricTensorDict,
                 survey_name: str, data_release: str = "unknown",
                 spectral: Optional[SpectralTensorDict] = None,
                 lightcurves: Optional[LightcurveTensorDict] = None,
                 features: Optional[torch.Tensor] = None, **kwargs):

        # Konsistenz-Check: Alle Komponenten müssen gleiche Batch-Größe haben
        batch_size = spatial.batch_size
        if photometric.batch_size != batch_size:
            raise ValueError("Spatial and photometric data must have same batch size")

        data = {
            "spatial": spatial,
            "photometric": photometric,
            "meta": TensorDict({
                "survey_name": survey_name,
                "data_release": data_release,
                "filter_system": photometric["meta", "filter_system"],
                "n_objects": batch_size[0] if batch_size else 0,
            }, batch_size=batch_size)
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
        return self["meta", "survey_name"]

    @property
    def n_objects(self) -> int:
        return self["meta", "n_objects"]

    def cross_match(self, other: SurveyTensorDict, tolerance: float = 1.0) -> torch.Tensor:
        """Cross-Match mit anderem Survey."""
        # Nutze räumliche Koordinaten für Cross-Match
        separations = self["spatial"].angular_separation(other["spatial"])
        matches = torch.where(separations <= tolerance)
        return matches

    def compute_colors(self, band_pairs: List[Tuple[str, str]]) -> TensorDict:
        """Delegiert an photometrische Komponente."""
        return self["photometric"].compute_colors(band_pairs)

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """Delegiert an räumliche Komponente."""
        return self["spatial"].cone_search(center, radius_deg)

    def query_region(self, ra_range: Tuple[float, float], 
                     dec_range: Tuple[float, float]) -> SurveyTensorDict:
        """Filtert Daten nach RA/Dec Region."""
        ra, dec, _ = self["spatial"].to_spherical()

        mask = ((ra >= ra_range[0]) & (ra <= ra_range[1]) & 
                (dec >= dec_range[0]) & (dec <= dec_range[1]))

        # Wende Maske auf alle Komponenten an
        filtered_spatial = SpatialTensorDict(
            self["spatial"]["coordinates"][mask],
            coordinate_system=self["spatial"].coordinate_system,
            unit=self["spatial"]["meta", "unit"],
            epoch=self["spatial"]["meta", "epoch"]
        )

        filtered_photometric = PhotometricTensorDict(
            self["photometric"]["magnitudes"][mask],
            bands=self["photometric"].bands,
            errors=self["photometric"].get("errors")[mask] if "errors" in self["photometric"] else None,
            filter_system=self["photometric"]["meta", "filter_system"],
            is_magnitude=self["photometric"].is_magnitude
        )

        return SurveyTensorDict(
            spatial=filtered_spatial,
            photometric=filtered_photometric,
            survey_name=self.survey_name,
            data_release=self["meta", "data_release"]
        ).add_history("query_region", ra_range=ra_range, dec_range=dec_range)
