"""
Photometric tensor for multi-band astronomical measurements.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typing_extensions import Self

from .base import AstroTensorBase

class PhotometricTensor(AstroTensorBase):
    """
    Represents photometric data for a set of celestial objects, including
    magnitudes or fluxes in different bands, along with associated metadata.
    """

    def __init__(
        self,
        data: Union[torch.Tensor, List, Any],
        bands: List[str],
        measurement_errors: Optional[torch.Tensor] = None,
        extinction_coefficients: Optional[Dict[str, float]] = None,
        photometric_system: str = "AB",
        is_magnitude: bool = True,
        zero_points: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Initializes the PhotometricTensor.

        Args:
            data: Photometric measurements, shape [..., n_bands].
            bands: A list of band names corresponding to the last dimension of data.
            measurement_errors: Measurement uncertainties, same shape as data.
            extinction_coefficients: Extinction coefficients per band.
            photometric_system: The photometric system (e.g., "AB", "Vega").
            is_magnitude: True if the data is in magnitudes, False if in flux.
            zero_points: Zero point magnitudes per band.
            **kwargs: Additional metadata to be stored.
        """
        # Consolidate all photometry-specific metadata
        photometry_meta = {
            "bands": bands,
            "measurement_errors": measurement_errors,
            "extinction_coefficients": extinction_coefficients or {},
            "photometric_system": photometric_system,
            "is_magnitude": is_magnitude,
            "zero_points": zero_points or {},
        }
        # Combine with any other metadata passed in kwargs
        kwargs.update(photometry_meta)
        
        super().__init__(data=data, **kwargs)
        self._validate()

    def _validate(self) -> None:
        """Validates the integrity of the photometric tensor data and metadata."""
        if self.data.dim() < 1:
            raise ValueError("PhotometricTensor requires at least 1D data.")

        bands = self.meta.get("bands", [])
        if not bands:
            raise ValueError("PhotometricTensor requires a list of band names.")

        if self.data.shape[-1] != len(bands):
            raise ValueError(
                f"Data's last dimension ({self.data.shape[-1]}) does not match "
                f"the number of bands ({len(bands)})."
            )

        errors = self.meta.get("measurement_errors")
        if errors is not None:
            if not isinstance(errors, torch.Tensor):
                raise TypeError("measurement_errors must be a torch.Tensor.")
            if errors.shape != self.data.shape:
                raise ValueError(
                    f"measurement_errors shape ({errors.shape}) does not match "
                    f"data shape ({self.data.shape})."
                )

    @property
    def bands(self) -> List[str]:
        return self.meta.get("bands", [])

    @property
    def measurement_errors(self) -> Optional[torch.Tensor]:
        return self.meta.get("measurement_errors")

    @property
    def is_magnitude(self) -> bool:
        return self.meta.get("is_magnitude", True)

    def get_band_data(self, band: str) -> torch.Tensor:
        """Extracts data for a specific band by name."""
        try:
            band_idx = self.bands.index(band)
            return self.data[..., band_idx]
        except ValueError:
            raise ValueError(f"Band '{band}' not found in tensor bands: {self.bands}")

    def get_colors(self, band_pairs: List[Tuple[str, str]]) -> torch.Tensor:
        """Computes colors (magnitude differences) for pairs of bands."""
        if not self.is_magnitude:
            raise ValueError("Color computation requires data to be in magnitudes.")
        
        band1_indices = [self.bands.index(p[0]) for p in band_pairs]
        band2_indices = [self.bands.index(p[1]) for p in band_pairs]

        return self.data[..., band1_indices] - self.data[..., band2_indices]

    def to_flux(self) -> Self:
        """Converts magnitude data to flux."""
        if not self.is_magnitude:
            return self.copy()

        # Flux conversion formula: F = 10^(-0.4 * M)
        flux_data = 10 ** (-0.4 * self.data)

        new_errors = None
        if self.measurement_errors is not None:
            # Error propagation: dF = F * dM * ln(10) / 2.5
            new_errors = flux_data * self.measurement_errors * (torch.log(torch.tensor(10.0)) / 2.5)
        
        new_meta = self.meta.copy()
        new_meta.update(is_magnitude=False, measurement_errors=new_errors)
        return self.__class__(data=flux_data, **new_meta)

    def to_magnitude(self) -> Self:
        """Converts flux data to magnitude."""
        if self.is_magnitude:
            return self.copy()

        # Magnitude conversion formula: M = -2.5 * log10(F)
        mag_data = -2.5 * torch.log10(self.data)

        new_errors = None
        if self.measurement_errors is not None:
            # Error propagation: dM = (2.5 / ln(10)) * (dF / F)
            new_errors = (2.5 / torch.log(torch.tensor(10.0))) * (self.measurement_errors / self.data)
            
        new_meta = self.meta.copy()
        new_meta.update(is_magnitude=True, measurement_errors=new_errors)
        return self.__class__(data=mag_data, **new_meta)

    def filter_bands(self, bands_to_keep: List[str]) -> Self:
        """Creates a new tensor containing only the specified bands."""
        indices = [i for i, b in enumerate(self.bands) if b in bands_to_keep]
        if len(indices) != len(bands_to_keep):
            missing = set(bands_to_keep) - set(self.bands)
            raise ValueError(f"Bands not found: {missing}")
            
        filtered_data = self.data[..., indices]
        
        new_meta = self.meta.copy()
        new_meta["bands"] = [self.bands[i] for i in indices]
        
        if self.measurement_errors is not None:
            new_meta["measurement_errors"] = self.measurement_errors[..., indices]
            
        return self.__class__(data=filtered_data, **new_meta)

    def apply_extinction(
        self, extinction_values: Union[Dict[str, float], torch.Tensor]
    ) -> "PhotometricTensor":
        """
        Apply extinction correction to photometric data.

        Args:
            extinction_values: Extinction values (A_V or per-band)

        Returns:
            Extinction-corrected PhotometricTensor
        """
        if not self.is_magnitude:
            raise ValueError("Extinction correction requires magnitude data")

        corrected_data = self.data.clone()

        if isinstance(extinction_values, dict):
            for i, band in enumerate(self.bands):
                if band in extinction_values:
                    corrected_data[..., i] -= extinction_values[band]
        elif isinstance(extinction_values, torch.Tensor):
            corrected_data -= extinction_values
        else:
            raise TypeError("extinction_values must be a dict or torch.Tensor")

        return self._create_new_instance(new_data=corrected_data)

    def compute_synthetic_colors(self) -> Dict[str, torch.Tensor]:
        """Compute common synthetic colors from available bands."""
        if not self.is_magnitude:
            raise ValueError("Color computation requires magnitude data")

        colors = {}

        # Common color combinations
        color_pairs = [
            ("u", "g"),
            ("g", "r"),
            ("r", "i"),
            ("i", "z"),
            ("B", "V"),
            ("V", "R"),
            ("V", "I"),
            ("J", "H"),
            ("H", "K"),
        ]

        for band1, band2 in color_pairs:
            if band1 in self.bands and band2 in self.bands:
                color_name = f"{band1}_{band2}"
                colors[color_name] = self.get_colors([(band1, band2)])

        return colors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()

        # Convert tensor metadata to serializable format
        if self.measurement_errors is not None:
            result["measurement_errors"] = (
                self.measurement_errors.detach().cpu().numpy()
            )

        return result

    @classmethod  # type: ignore
    def from_dict(cls, data_dict: Dict[str, Any]) -> "PhotometricTensor":
        """Create PhotometricTensor from dictionary."""
        data = torch.tensor(data_dict["data"])

        # Reconstruct metadata
        metadata = data_dict.copy()
        del metadata["data"]
        del metadata["dtype"]
        del metadata["shape"]
        del metadata["device"]

        # Convert measurement_errors back to tensor if present
        if (
            "measurement_errors" in metadata
            and metadata["measurement_errors"] is not None
        ):
            metadata["measurement_errors"] = torch.tensor(
                metadata["measurement_errors"]
            )

        return cls(data, **metadata)
