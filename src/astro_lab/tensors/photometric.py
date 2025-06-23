"""
Photometric tensor for multi-band astronomical measurements.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import Field
from typing_extensions import Self

from .base import AstroTensorBase

class PhotometricTensor(AstroTensorBase):
    """
    Represents photometric data for a set of celestial objects, including
    magnitudes or fluxes in different bands, along with associated metadata.
    """
    bands: List[str] = Field(..., description="A list of band names corresponding to the last dimension of data.")
    measurement_errors: Optional[torch.Tensor] = Field(default=None, description="Measurement uncertainties, same shape as data.")
    photometric_system: str = Field(default="AB", description="The photometric system (e.g., 'AB', 'Vega').")
    is_magnitude: bool = Field(default=True, description="True if the data is in magnitudes, False if in flux.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._validate()

    def _validate(self) -> None:
        """Validates the integrity of the photometric tensor data and metadata."""
        if self.data.dim() < 1:
            raise ValueError("PhotometricTensor requires at least 1D data.")

        if not self.bands:
            raise ValueError("PhotometricTensor requires a list of band names.")

        if self.data.shape[-1] != len(self.bands):
            raise ValueError(
                f"Data's last dimension ({self.data.shape[-1]}) does not match "
                f"the number of bands ({len(self.bands)})."
            )

        if self.measurement_errors is not None:
            if not isinstance(self.measurement_errors, torch.Tensor):
                raise TypeError("measurement_errors must be a torch.Tensor.")
            if self.measurement_errors.shape != self.data.shape:
                raise ValueError(
                    f"measurement_errors shape ({self.measurement_errors.shape}) does not match "
                    f"data shape ({self.data.shape})."
                )

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
        
        return self._create_new_instance(
            new_data=flux_data, 
            is_magnitude=False, 
            measurement_errors=new_errors
        )

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
            
        return self._create_new_instance(
            new_data=mag_data, 
            is_magnitude=True, 
            measurement_errors=new_errors
        )

    def filter_bands(self, bands_to_keep: List[str]) -> Self:
        """Creates a new tensor containing only the specified bands."""
        indices = [i for i, b in enumerate(self.bands) if b in bands_to_keep]
        if len(indices) != len(bands_to_keep):
            missing = set(bands_to_keep) - set(self.bands)
            raise ValueError(f"Bands not found: {missing}")
            
        filtered_data = self.data[..., indices]
        new_bands = [self.bands[i] for i in indices]
        
        new_errors = None
        if self.measurement_errors is not None:
            new_errors = self.measurement_errors[..., indices]
            
        return self._create_new_instance(
            new_data=filtered_data, 
            bands=new_bands, 
            measurement_errors=new_errors
        )

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
