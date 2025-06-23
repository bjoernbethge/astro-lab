"""
Photometric tensor for multi-band astronomical measurements.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from pydantic import Field
from typing_extensions import Self
import numpy as np

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

    def __init__(self, data: Union[torch.Tensor, np.ndarray], bands: Optional[List[str]] = None, **kwargs):
        """
        Initialize PhotometricTensor.
        
        Args:
            data: Photometric measurements tensor [N, num_bands]
            bands: List of photometric band names (e.g., ["u", "g", "r", "i", "z"])
            **kwargs: Additional arguments passed to parent
        """
        # Set default bands if not provided
        if bands is None:
            if isinstance(data, torch.Tensor):
                num_bands = data.shape[1] if data.dim() > 1 else 1
            else:
                num_bands = data.shape[1] if data.ndim > 1 else 1
            bands = [f"band_{i}" for i in range(num_bands)]
            
        # Convert data to tensor
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)
            
        # Pass bands through kwargs to avoid direct assignment
        kwargs['bands'] = bands
        super().__init__(data=data, **kwargs)

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

    def compute_colors(self, color_pairs: Optional[List[Tuple[str, str]]] = None) -> Dict[str, torch.Tensor]:
        """Compute color indices from photometric bands."""
        if color_pairs is None:
            # Default color combinations
            color_pairs = [
                ("B", "V"), ("V", "R"), ("R", "I"), 
                ("g", "r"), ("r", "i"), ("i", "z")
            ]
        
        colors = {}
        for band1, band2 in color_pairs:
            if band1 in self.bands and band2 in self.bands:
                idx1 = self.bands.index(band1)
                idx2 = self.bands.index(band2)
                color_name = f"{band1}-{band2}"
                colors[color_name] = self.data[:, idx1] - self.data[:, idx2]
        
        return colors

    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary for serialization."""
        return {
            "data": self.data.cpu().numpy().tolist(),
            "bands": self.bands,
            "meta": self.meta,
            "tensor_type": "photometric",
            "n_bands": self.n_bands,
            "extinction_coefficients": self.extinction_coefficients
        }

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

    @property
    def n_bands(self) -> int:
        """Return the number of photometric bands."""
        return len(self.bands)
    
    @property
    def extinction_coefficients(self) -> Dict[str, float]:
        """Return extinction coefficients for each band."""
        # Default extinction coefficients for common bands
        default_extinctions = {
            'u': 4.239, 'g': 3.303, 'r': 2.285, 'i': 1.698, 'z': 1.263,
            'U': 4.968, 'B': 4.215, 'V': 3.240, 'R': 2.634, 'I': 1.905,
            'J': 0.902, 'H': 0.576, 'K': 0.367
        }
        return {band: default_extinctions.get(band, 1.0) for band in self.bands}

    def _validate(self) -> None:
        """Validate the photometric tensor data and bands."""
        super()._validate()
        
        # Check if number of bands matches data columns
        if self.data.dim() > 1 and len(self.bands) != self.data.shape[1]:
            raise ValueError(f"Number of bands ({len(self.bands)}) doesn't match data columns ({self.data.shape[1]})")
        
        # Validate band names
        valid_bands = {'u', 'g', 'r', 'i', 'z', 'U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'y'}
        invalid_bands = set(self.bands) - valid_bands
        if invalid_bands and not any(band.startswith('band_') for band in invalid_bands):
            # Only warn for non-generic band names
            pass  # Allow custom band names for flexibility
