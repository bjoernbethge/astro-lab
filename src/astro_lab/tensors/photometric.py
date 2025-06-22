"""
Photometric tensor for multi-band astronomical measurements.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from .base import AstroTensorBase


class PhotometricTensor(AstroTensorBase):
    """
    Tensor for multi-band photometric measurements using composition.

    Stores magnitudes/fluxes across multiple bands with associated metadata
    like measurement errors, extinction coefficients, and zero points.
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
        Initialize photometric tensor.

        Args:
            data: Photometric measurements [..., n_bands]
            bands: List of band names
            measurement_errors: Measurement uncertainties
            extinction_coefficients: Extinction coefficients per band
            photometric_system: Photometric system (AB, Vega, etc.)
            is_magnitude: Whether data is in magnitudes (vs flux)
            zero_points: Zero point magnitudes per band
        """
        # Store photometry-specific metadata
        metadata = {
            "bands": bands,
            "measurement_errors": measurement_errors,
            "extinction_coefficients": extinction_coefficients or {},
            "photometric_system": photometric_system,
            "is_magnitude": is_magnitude,
            "zero_points": zero_points or {},
        }
        metadata.update(kwargs)

        super().__init__(data, **metadata, tensor_type="photometric")
        self._validate()  # Call validation after initialization

    def _validate(self) -> None:
        """Validate photometric tensor data."""
        if self._data.dim() < 1:
            raise ValueError("PhotometricTensor requires at least 1D data")

        bands = self._metadata.get("bands", [])
        if not bands:
            raise ValueError("PhotometricTensor requires band names")

        # Check that last dimension matches number of bands
        if self._data.shape[-1] != len(bands):
            raise ValueError(
                f"Data shape {self._data.shape} doesn't match number of bands {len(bands)}"
            )

        # Validate measurement errors if provided
        errors = self._metadata.get("measurement_errors")
        if errors is not None:
            if not isinstance(errors, torch.Tensor):
                raise ValueError("measurement_errors must be a torch.Tensor")
            if errors.shape != self._data.shape:
                raise ValueError(
                    f"measurement_errors shape {errors.shape} doesn't match data shape {self._data.shape}"
                )

    @property
    def bands(self) -> List[str]:
        """Band names."""
        return self._metadata["bands"]

    @property
    def n_bands(self) -> int:
        """Number of bands."""
        return len(self.bands)

    @property
    def measurement_errors(self) -> Optional[torch.Tensor]:
        """Measurement uncertainties."""
        return self._metadata.get("measurement_errors")

    @property
    def extinction_coefficients(self) -> Dict[str, float]:
        """Extinction coefficients per band."""
        return self._metadata.get("extinction_coefficients", {})

    @property
    def photometric_system(self) -> str:
        """Photometric system."""
        return self._metadata.get("photometric_system", "AB")

    @property
    def is_magnitude(self) -> bool:
        """Whether data is in magnitudes."""
        return self._metadata.get("is_magnitude", True)

    @property
    def zero_points(self) -> Dict[str, float]:
        """Zero point magnitudes per band."""
        return self._metadata.get("zero_points", {})

    def get_band_data(self, band: str) -> torch.Tensor:
        """Get data for a specific band."""
        if band not in self.bands:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

        band_idx = self.bands.index(band)
        return self._data[..., band_idx]

    def get_band_error(self, band: str) -> Optional[torch.Tensor]:
        """Get measurement error for a specific band."""
        if self.measurement_errors is None:
            return None

        if band not in self.bands:
            raise ValueError(f"Band '{band}' not found in {self.bands}")

        band_idx = self.bands.index(band)
        return self.measurement_errors[..., band_idx]

    def compute_colors(self, band1: str, band2: str) -> torch.Tensor:
        """Compute color (magnitude difference) between two bands."""
        if not self.is_magnitude:
            raise ValueError("Color computation requires magnitude data")

        mag1 = self.get_band_data(band1)
        mag2 = self.get_band_data(band2)

        return mag1 - mag2

    def to_flux(self, band: Optional[str] = None) -> "PhotometricTensor":
        """
        Convert magnitudes to flux units.

        Args:
            band: Specific band to convert (None for all bands)

        Returns:
            PhotometricTensor with flux data
        """
        if not self.is_magnitude:
            raise ValueError("Data is already in flux units")

        if band is not None:
            # Convert specific band
            band_idx = self.bands.index(band)
            flux_data = self._data.clone()
            flux_data[..., band_idx] = 10 ** (-0.4 * self._data[..., band_idx])
        else:
            # Convert all bands
            flux_data = 10 ** (-0.4 * self._data)

        # Convert errors if available
        new_errors = None
        if self.measurement_errors is not None:
            # Error propagation for magnitude to flux conversion
            flux_val = 10 ** (-0.4 * self._data)
            new_errors = (
                flux_val * self.measurement_errors * 0.4 * torch.log(torch.tensor(10.0))
            )

        # Create new metadata
        new_metadata = self._metadata.copy()
        new_metadata["is_magnitude"] = False
        new_metadata["measurement_errors"] = new_errors

        return PhotometricTensor(flux_data, **new_metadata)

    def to_magnitude(self, band: Optional[str] = None) -> "PhotometricTensor":
        """
        Convert flux to magnitude units.

        Args:
            band: Specific band to convert (None for all bands)

        Returns:
            PhotometricTensor with magnitude data
        """
        if self.is_magnitude:
            raise ValueError("Data is already in magnitude units")

        if band is not None:
            # Convert specific band
            band_idx = self.bands.index(band)
            mag_data = self._data.clone()
            mag_data[..., band_idx] = -2.5 * torch.log10(self._data[..., band_idx])
        else:
            # Convert all bands
            mag_data = -2.5 * torch.log10(self._data)

        # Convert errors if available
        new_errors = None
        if self.measurement_errors is not None:
            # Error propagation for flux to magnitude conversion
            new_errors = (
                2.5
                * self.measurement_errors
                / (self._data * torch.log(torch.tensor(10.0)))
            )

        # Create new metadata
        new_metadata = self._metadata.copy()
        new_metadata["is_magnitude"] = True
        new_metadata["measurement_errors"] = new_errors

        return PhotometricTensor(mag_data, **new_metadata)

    def apply_extinction_correction(
        self, extinction_values: Union[float, torch.Tensor, Dict[str, float]]
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

        corrected_data = self._data.clone()

        if isinstance(extinction_values, dict):
            # Band-specific extinction values
            for band, ext_val in extinction_values.items():
                if band in self.bands:
                    band_idx = self.bands.index(band)
                    corrected_data[..., band_idx] -= ext_val
        else:
            # Apply extinction using extinction coefficients
            if isinstance(extinction_values, (int, float)):
                a_v = torch.tensor(extinction_values)
            else:
                a_v = extinction_values

            for i, band in enumerate(self.bands):
                if band in self.extinction_coefficients:
                    coeff = self.extinction_coefficients[band]
                    corrected_data[..., i] -= a_v * coeff

        # Create new tensor with corrected data
        metadata_dict = self._metadata.copy()
        return PhotometricTensor(corrected_data, **metadata_dict)

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
                colors[color_name] = self.compute_colors(band1, band2)

        return colors

    def filter_by_bands(self, selected_bands: List[str]) -> "PhotometricTensor":
        """Create new PhotometricTensor with only selected bands."""
        # Find indices of selected bands
        band_indices = []
        for band in selected_bands:
            if band not in self.bands:
                raise ValueError(
                    f"Band '{band}' not found in available bands {self.bands}"
                )
            band_indices.append(self.bands.index(band))

        # Select data for chosen bands
        filtered_data = self._data[..., band_indices]

        # Update metadata
        new_metadata = self._metadata.copy()
        new_metadata["bands"] = selected_bands

        # Filter other band-specific metadata
        if self.measurement_errors is not None:
            new_metadata["measurement_errors"] = self.measurement_errors[
                ..., band_indices
            ]

        new_metadata["extinction_coefficients"] = {
            band: coeff
            for band, coeff in self.extinction_coefficients.items()
            if band in selected_bands
        }

        new_metadata["zero_points"] = {
            band: zp for band, zp in self.zero_points.items() if band in selected_bands
        }

        return PhotometricTensor(filtered_data, **new_metadata)

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
