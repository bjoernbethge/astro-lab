"""
Tensor Factory for Astronomical Tensors
=======================================

Factory pattern for creating astronomical tensors following the refactoring guide.
Consolidates inconsistent factory methods and provides unified creation interface.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .constants import ASTRO, GRAVITY

logger = logging.getLogger(__name__)

class TensorFactory:
    """
    Factory for creating astronomical tensors.

    Provides unified interface for tensor creation with automatic
    type detection and validation.
    """

    @staticmethod
    def create_spatial(
        ra: Union[torch.Tensor, np.ndarray, List],
        dec: Union[torch.Tensor, np.ndarray, List],
        distance: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        coordinate_system: str = "icrs",
        unit: str = "degree",
        **kwargs,
    ):
        """
        Create spatial tensor from coordinates.

        Args:
            ra: Right ascension
            dec: Declination
            distance: Distance (parallax will be computed if None)
            coordinate_system: Coordinate system name
            unit: Angular unit
            **kwargs: Additional metadata

        Returns:
            Spatial3DTensor instance
        """
        try:
            from .spatial_3d import Spatial3DTensor
        except ImportError:
            raise ImportError("Spatial3DTensor not available")

        # Convert inputs to tensors
        ra = torch.as_tensor(ra, dtype=torch.float32)
        dec = torch.as_tensor(dec, dtype=torch.float32)

        if distance is None:
            # Create 2D spatial data (spherical coordinates)
            data = torch.stack([ra, dec], dim=1)
            logger.info(f"Created 2D spatial tensor with {len(ra)} objects")
        else:
            distance = torch.as_tensor(distance, dtype=torch.float32)
            data = torch.stack([ra, dec, distance], dim=1)
            logger.info(f"Created 3D spatial tensor with {len(ra)} objects")

        return Spatial3DTensor(
            data=data, coordinate_system=coordinate_system, unit=unit, **kwargs
        )

    @staticmethod
    def create_photometric(
        magnitudes: Union[torch.Tensor, np.ndarray, Dict[str, Any]],
        bands: List[str],
        errors: Optional[Union[torch.Tensor, np.ndarray]] = None,
        is_magnitude: bool = True,
        filter_system: str = "unknown",
        **kwargs,
    ):
        """
        Create photometric tensor from magnitude/flux data.

        Args:
            magnitudes: Photometric measurements
            bands: Band names (e.g., ['g', 'r', 'i'])
            errors: Measurement errors
            is_magnitude: True for magnitudes, False for fluxes
            filter_system: Filter system name
            **kwargs: Additional metadata

        Returns:
            PhotometricTensor instance
        """
        try:
            from .photometric import PhotometricTensor
        except ImportError:
            raise ImportError("PhotometricTensor not available")

        # Handle dictionary input (band_name -> values)
        if isinstance(magnitudes, dict):
            if not bands:
                bands = list(magnitudes.keys())

            # Stack dictionary values in band order
            mag_data = torch.stack(
                [
                    torch.as_tensor(magnitudes[band], dtype=torch.float32)
                    for band in bands
                ],
                dim=1,
            )
        else:
            mag_data = torch.as_tensor(magnitudes, dtype=torch.float32)

            # Ensure correct shape
            if mag_data.dim() == 1:
                mag_data = mag_data.unsqueeze(1)

        # Validate band count
        if mag_data.shape[1] != len(bands):
            raise ValueError(
                f"Number of bands ({len(bands)}) doesn't match data columns ({mag_data.shape[1]})"
            )

        logger.info(
            f"Created photometric tensor: {len(bands)} bands, {mag_data.shape[0]} objects"
        )

        return PhotometricTensor(
            data=mag_data,
            bands=bands,
            errors=errors,
            is_magnitude=is_magnitude,
            filter_system=filter_system,
            **kwargs,
        )

    @staticmethod
    def create_lightcurve(
        times: Union[torch.Tensor, np.ndarray, List],
        magnitudes: Union[torch.Tensor, np.ndarray, List],
        bands: Optional[List[str]] = None,
        errors: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        object_ids: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        time_format: str = "mjd",
        **kwargs,
    ):
        """
        Create lightcurve tensor from time series data.

        Args:
            times: Observation times
            magnitudes: Magnitude measurements
            bands: Band names (if multi-band)
            errors: Measurement errors
            object_ids: Object identifiers
            time_format: Time format (mjd, jd, etc.)
            **kwargs: Additional metadata

        Returns:
            LightcurveTensor instance
        """
        try:
            from .lightcurve import LightcurveTensor
        except ImportError:
            raise ImportError("LightcurveTensor not available")

        # Convert to tensors
        times = torch.as_tensor(times, dtype=torch.float32)
        magnitudes = torch.as_tensor(magnitudes, dtype=torch.float32)

        # Handle errors
        if errors is not None:
            errors = torch.as_tensor(errors, dtype=torch.float32)

        # Default band names
        if bands is None:
            if magnitudes.dim() == 1:
                bands = ["band_0"]
            else:
                bands = [f"band_{i}" for i in range(magnitudes.shape[1])]

        logger.info(
            f"Created lightcurve tensor: {len(bands)} bands, {len(times)} observations"
        )

        return LightcurveTensor(
            times=times,
            magnitudes=magnitudes,
            bands=bands,
            errors=errors,
            object_ids=object_ids,
            time_format=time_format,
            **kwargs,
        )

    @staticmethod
    def create_spectral(
        wavelengths: Union[torch.Tensor, np.ndarray, List],
        fluxes: Union[torch.Tensor, np.ndarray, List],
        errors: Optional[Union[torch.Tensor, np.ndarray, List]] = None,
        wavelength_unit: str = "angstrom",
        flux_unit: str = "erg/s/cm2/A",
        redshift: float = 0.0,
        **kwargs,
    ):
        """
        Create spectral tensor from wavelength and flux data.

        Args:
            wavelengths: Wavelength array
            fluxes: Flux measurements
            errors: Flux errors
            wavelength_unit: Wavelength unit
            flux_unit: Flux unit
            redshift: Object redshift
            **kwargs: Additional metadata

        Returns:
            SpectralTensor instance
        """
        try:
            from .spectral import SpectralTensor
        except ImportError:
            raise ImportError("SpectralTensor not available")

        # Convert to tensors
        wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32)
        fluxes = torch.as_tensor(fluxes, dtype=torch.float32)

        # Combine wavelengths and fluxes
        if fluxes.dim() == 1:
            # Single spectrum
            data = torch.stack([wavelengths, fluxes], dim=1)
        else:
            # Multiple spectra - prepend wavelengths
            data = torch.cat(
                [
                    wavelengths.unsqueeze(0).expand(fluxes.shape[0], -1).unsqueeze(-1),
                    fluxes.unsqueeze(-1),
                ],
                dim=-1,
            )

        logger.info(f"Created spectral tensor: {len(wavelengths)} wavelength points")

        return SpectralTensor(
            data=data,
            wavelengths=wavelengths,
            errors=errors,
            wavelength_unit=wavelength_unit,
            flux_unit=flux_unit,
            redshift=redshift,
            **kwargs,
        )

    @staticmethod
    def create_orbital(
        elements: Union[torch.Tensor, np.ndarray, Dict[str, float]],
        element_type: str = "keplerian",
        epoch: Optional[float] = None,
        central_body: str = "earth",
        **kwargs,
    ):
        """
        Create orbital tensor from orbital elements.

        Args:
            elements: Orbital elements (6 or 7 element array or dict)
            element_type: Type of elements (keplerian, cartesian, tle)
            epoch: Epoch time (MJD)
            central_body: Central body name
            **kwargs: Additional metadata

        Returns:
            OrbitTensor instance
        """
        try:
            from .orbital import OrbitTensor
        except ImportError:
            raise ImportError("OrbitTensor not available")

        # Handle dictionary input for Keplerian elements
        if isinstance(elements, dict):
            if element_type == "keplerian":
                # Standard order: a, e, i, Ω, ω, M
                element_order = ["a", "e", "i", "raan", "argp", "mean_anomaly"]
                elements_array = [elements.get(key, 0.0) for key in element_order]
            else:
                # For other types, convert dict values to list
                elements_array = list(elements.values())

            elements = torch.tensor(elements_array, dtype=torch.float32)
        else:
            elements = torch.as_tensor(elements, dtype=torch.float32)

        # Get gravitational parameter for central body
        mu = GRAVITY.get(central_body.upper(), GRAVITY.EARTH)

        # Default epoch to J2000 if not provided
        if epoch is None:
            epoch = 51544.5  # J2000.0 in MJD

        logger.info(
            f"Created orbital tensor: {element_type} elements for {central_body}"
        )

        return OrbitTensor(
            data=elements,
            element_type=element_type,
            epoch=epoch,
            central_body=central_body,
            mu=mu,
            **kwargs,
        )

    @staticmethod
    def create_survey(
        data: Union[torch.Tensor, Dict[str, Any]], survey_name: str, **kwargs
    ):
        """Create survey tensor."""
        try:
            from .survey import SurveyTensor
        except ImportError:
            raise ImportError("SurveyTensor not available")

        if isinstance(data, dict):
            data_tensor = torch.stack(
                [
                    torch.as_tensor(data[col], dtype=torch.float32)
                    for col in data.keys()
                ],
                dim=1,
            )
        else:
            data_tensor = torch.as_tensor(data, dtype=torch.float32)

        return SurveyTensor(data=data_tensor, survey_name=survey_name, **kwargs)

    @staticmethod
    def from_astropy_table(table, tensor_type: str = "survey", **kwargs):
        """
        Create tensor from Astropy Table.

        Args:
            table: Astropy Table object
            tensor_type: Type of tensor to create
            **kwargs: Additional arguments for tensor creation

        Returns:
            Appropriate tensor instance
        """
        try:
            import astropy.table
        except ImportError:
            raise ImportError("Astropy required for table conversion")

        if not isinstance(table, astropy.table.Table):
            raise TypeError("Input must be an Astropy Table")

        # Convert table to dictionary
        data_dict = {col: table[col].data for col in table.colnames}

        # Create appropriate tensor based on type
        if tensor_type == "survey":
            return TensorFactory.create_survey(data_dict, **kwargs)
        elif tensor_type == "photometric":
            # Auto-detect photometric columns
            mag_columns = [col for col in table.colnames if "mag" in col.lower()]
            return TensorFactory.create_photometric(
                data_dict, bands=mag_columns, **kwargs
            )
        elif tensor_type == "spatial":
            # Look for coordinate columns
            ra_col = next(
                (col for col in table.colnames if col.lower() in ["ra", "ra_deg"]), None
            )
            dec_col = next(
                (col for col in table.colnames if col.lower() in ["dec", "dec_deg"]),
                None,
            )

            if ra_col and dec_col:
                distance_col = next(
                    (
                        col
                        for col in table.colnames
                        if col.lower() in ["distance", "dist", "parallax"]
                    ),
                    None,
                )
                distance = data_dict[distance_col] if distance_col else None

                return TensorFactory.create_spatial(
                    data_dict[ra_col], data_dict[dec_col], distance, **kwargs
                )
            else:
                raise ValueError("No coordinate columns found for spatial tensor")
        else:
            raise ValueError(f"Unknown tensor type: {tensor_type}")

    @staticmethod
    def auto_detect_tensor_type(data: Union[torch.Tensor, Dict, Any]) -> str:
        """
        Automatically detect the most appropriate tensor type for data.

        Args:
            data: Input data

        Returns:
            Suggested tensor type
        """
        if isinstance(data, dict):
            columns = list(data.keys())
        elif hasattr(data, "colnames"):  # Astropy table
            columns = data.colnames
        else:
            # For raw tensors, default to survey
            return "survey"

        columns_lower = [col.lower() for col in columns]

        # Check for spatial data
        if any(col in columns_lower for col in ["ra", "dec", "ra_deg", "dec_deg"]):
            return "spatial"

        # Check for photometric data
        if any("mag" in col or "flux" in col for col in columns_lower):
            return "photometric"

        # Check for time series data
        if any(col in columns_lower for col in ["time", "mjd", "jd", "date"]):
            return "lightcurve"

        # Check for spectroscopic data
        if any(col in columns_lower for col in ["wavelength", "wave", "lambda"]):
            return "spectral"

        # Check for orbital data
        if any(col in columns_lower for col in ["a", "e", "i", "sma", "ecc", "inc"]):
            return "orbital"

        # Default to survey
        return "survey"

# Convenience functions for common use cases
def create_gaia_tensor(data: Dict[str, Any], **kwargs):
    """Create survey tensor optimized for Gaia data."""
    return TensorFactory.create_survey(data=data, survey_name="gaia", **kwargs)

def create_sdss_tensor(data: Dict[str, Any], **kwargs):
    """Create survey tensor optimized for SDSS data."""
    return TensorFactory.create_survey(data=data, survey_name="sdss", **kwargs)

def create_jwst_tensor(data: Dict[str, Any], **kwargs):
    """Create survey tensor optimized for JWST data."""
    return TensorFactory.create_survey(data=data, survey_name="jwst", **kwargs)
