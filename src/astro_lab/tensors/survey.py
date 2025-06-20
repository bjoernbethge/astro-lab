"""
Refactored Survey Tensor - Main Coordinator Tensor
=================================================

This tensor acts as the main coordinator for all specialized astronomical tensors,
providing unified access to photometry, astrometry, lightcurves, spectroscopy, etc.

Improvements:
- Uses Protocol types instead of string literals
- Centralized constants
- Cleaner imports
- Better type safety
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import AstroTensorBase, ValidationMixin
from .constants import ASTRO, PHOTOMETRY, SPECTROSCOPY
from .tensor_types import (
    LightcurveTensorProtocol,
    PhotometricTensorProtocol,
    Spatial3DTensorProtocol,
    SpectralTensorProtocol,
    SurveyTensorProtocol,
)


class SurveyTensor(AstroTensorBase, ValidationMixin):
    """
    Main coordinator tensor for astronomical survey data using composition.

    Improvements:
    - Uses Protocol types for type safety
    - Cleaner validation patterns
    - Centralized constants usage
    - Better memory management
    """

    def __init__(
        self,
        data: Union[torch.Tensor, List, Any],
        survey_name: str,
        data_release: Optional[str] = None,
        filter_system: Optional[str] = None,
        column_mapping: Optional[Dict[str, int]] = None,
        survey_metadata: Optional[Dict[str, Any]] = None,
        transformations: Optional[Dict[str, Callable]] = None,
        **kwargs,
    ):
        """
        Initialize survey tensor.

        Args:
            data: Survey data tensor
            survey_name: Name of the astronomical survey
            data_release: Data release version
            filter_system: Photometric filter system
            column_mapping: Mapping of column names to indices
            survey_metadata: Additional survey metadata
            transformations: Survey transformation functions
        """
        # Store survey-specific metadata
        metadata = {
            "survey_name": survey_name,
            "data_release": data_release,
            "filter_system": filter_system
            or SurveyTensor._get_default_filter_system(survey_name),
            "column_mapping": column_mapping
            or SurveyTensor._get_default_columns(survey_name),
            "survey_metadata": survey_metadata or {},
            "transformations": transformations or {},
            "tensor_type": "survey",
        }
        metadata.update(kwargs)

        super().__init__(data, **metadata)

    def _validate(self) -> None:
        """Validate survey tensor data with enhanced checks."""
        super()._validate()  # Run base validation

        # Survey-specific validation
        self.validate_shape(expected_dims=1)  # At least 1D

        survey_name = self.get_metadata("survey_name")
        if not survey_name:
            raise ValueError("SurveyTensor requires survey_name")

        # Validate known survey names
        if survey_name.lower() not in self._get_known_surveys():
            import warnings

            warnings.warn(f"Unknown survey: {survey_name}")

    # =========================================================================
    # Survey configuration and defaults - using centralized constants
    # =========================================================================

    @staticmethod
    def _get_default_filter_system(survey_name: str) -> str:
        """Get default filter system for known surveys."""
        filter_systems = {
            "gaia": "gaia_dr3",
            "sdss": "sdss_ugriz",
            "lsst": "lsst_ugrizy",
            "euclid": "euclid_nisp",
            "des": "des_ugrizY",
            "ps1": "ps1_grizy",
            "2mass": "2mass_jhk",
            "wise": "wise_w1234",
            "hst": "hst_wfc3",
            "jwst": "jwst_nircam",
        }
        return filter_systems.get(survey_name.lower(), "unknown")

    @staticmethod
    def _get_known_surveys() -> List[str]:
        """Get list of known surveys with modern additions."""
        return [
            "gaia",
            "sdss",
            "nsa",  # NASA-Sloan Atlas
            "lsst",
            "euclid",
            "des",
            "ps1",
            "2mass",
            "wise",
            "hst",
            "jwst",
            "tess",
            "kepler",
            "roman",
            "vera_rubin",
        ]

    @staticmethod
    def _get_default_columns(survey_name: str) -> Dict[str, int]:
        """Get default column mapping for known surveys."""
        default_mappings = {
            "gaia": {
                "ra": 0,
                "dec": 1,
                "parallax": 2,
                "pmra": 3,
                "pmdec": 4,
                "phot_g_mean_mag": 5,
                "phot_bp_mean_mag": 6,
                "phot_rp_mean_mag": 7,
                "radial_velocity": 8,
                "teff_gspphot": 9,
            },
            "sdss": {
                "ra": 0,
                "dec": 1,
                "u": 2,
                "g": 3,
                "r": 4,
                "i": 5,
                "z": 6,
                "redshift": 7,
                "petroMag_r": 8,
                "modelMag_i": 9,
            },
            "lsst": {
                "ra": 0,
                "dec": 1,
                "u": 2,
                "g": 3,
                "r": 4,
                "i": 5,
                "z": 6,
                "y": 7,
                "redshift": 8,
                "stellar_mass": 9,
            },
            "jwst": {
                "ra": 0,
                "dec": 1,
                "f090w": 2,
                "f115w": 3,
                "f150w": 4,
                "f200w": 5,
                "f277w": 6,
                "f356w": 7,
                "f444w": 8,
                "redshift": 9,
            },
        }
        return default_mappings.get(survey_name.lower(), {})

    # =========================================================================
    # Consistent property access (eliminating getattr inconsistencies)
    # =========================================================================

    @property
    def survey_name(self) -> str:
        """Survey name with consistent default."""
        return self.get_metadata("survey_name", "unknown")

    @property
    def data_release(self) -> Optional[str]:
        """Data release version."""
        return self.get_metadata("data_release")

    @property
    def filter_system(self) -> str:
        """Filter system with consistent default."""
        return self.get_metadata("filter_system", "unknown")

    @property
    def column_mapping(self) -> Dict[str, int]:
        """Column mapping with consistent default."""
        return self.get_metadata("column_mapping", {})

    # =========================================================================
    # Specialized tensor creation using Protocols
    # =========================================================================

    def get_photometric_tensor(
        self, band_columns: Optional[List[str]] = None, force_recreate: bool = False
    ) -> PhotometricTensorProtocol:
        """
        Get or create PhotometricTensor for this survey.

        Args:
            band_columns: Specific bands to extract (auto-detect if None)
            force_recreate: Force recreation even if cached

        Returns:
            PhotometricTensor following the protocol
        """
        # Import at runtime to avoid circular dependencies
        try:
            from .photometric import PhotometricTensor
        except ImportError:
            raise ImportError(
                "PhotometricTensor not available - install full tensor package"
            )

        # Check cache first
        if not force_recreate and self.has_metadata("_photometric_tensor"):
            cached = self.get_metadata("_photometric_tensor")
            if cached is not None:
                return cached

        # Auto-detect bands if not specified
        if band_columns is None:
            band_columns = self._detect_photometric_columns()

        if not band_columns:
            raise ValueError("No photometric columns found in survey data")

        # Extract photometric data using column mapping
        band_indices = [
            self.column_mapping[col]
            for col in band_columns
            if col in self.column_mapping
        ]

        if not band_indices:
            raise ValueError(f"No valid photometric columns found: {band_columns}")

        # Create photometric tensor
        photometric_data = self._data[:, band_indices]

        # Use centralized constants for magnitude systems
        is_magnitude = self._is_magnitude_system(self.filter_system)

        photometric_tensor = PhotometricTensor(
            data=photometric_data,
            bands=band_columns,
            is_magnitude=is_magnitude,
            filter_system=self.filter_system,
            survey_name=self.survey_name,
        )

        # Cache the tensor
        self.update_metadata(_photometric_tensor=photometric_tensor)
        return photometric_tensor

    def get_spatial_tensor(
        self, include_distances: bool = True, force_recreate: bool = False
    ) -> Spatial3DTensorProtocol:
        """
        Get or create Spatial3DTensor for this survey.

        Args:
            include_distances: Include distance/parallax information
            force_recreate: Force recreation even if cached

        Returns:
            Spatial3DTensor following the protocol
        """
        try:
            from .spatial_3d import Spatial3DTensor
        except ImportError:
            raise ImportError("Spatial3DTensor not available")

        # Check cache
        if not force_recreate and self.has_metadata("_spatial_tensor"):
            cached = self.get_metadata("_spatial_tensor")
            if cached is not None:
                return cached

        # Extract spatial coordinates
        coord_columns = self._detect_spatial_columns(include_distances)
        if not coord_columns:
            raise ValueError("No spatial coordinate columns found")

        coord_indices = [
            self.column_mapping[col]
            for col in coord_columns
            if col in self.column_mapping
        ]

        spatial_data = self._data[:, coord_indices]

        # Use astronomical constants for coordinate systems
        spatial_tensor = Spatial3DTensor(
            data=spatial_data,
            coordinate_system="icrs",  # Default to ICRS
            unit="degree" if "ra" in coord_columns else "radian",
            survey_name=self.survey_name,
        )

        self.update_metadata(_spatial_tensor=spatial_tensor)
        return spatial_tensor

    def create_lightcurve_tensor(
        self,
        time_column: str,
        magnitude_columns: List[str],
        error_columns: Optional[List[str]] = None,
        object_id_column: Optional[str] = None,
    ) -> LightcurveTensorProtocol:
        """
        Create LightcurveTensor from survey data.

        Args:
            time_column: Column containing time data
            magnitude_columns: Columns containing magnitude measurements
            error_columns: Columns containing error estimates
            object_id_column: Column containing object identifiers

        Returns:
            LightcurveTensor following the protocol
        """
        try:
            from .lightcurve import LightcurveTensor
        except ImportError:
            raise ImportError("LightcurveTensor not available")

        # Validate required columns exist
        required_columns = [time_column] + magnitude_columns
        missing_columns = [
            col for col in required_columns if col not in self.column_mapping
        ]
        if missing_columns:
            raise ValueError(f"Missing columns for lightcurve: {missing_columns}")

        # Extract data
        time_data = self.get_column(time_column)
        mag_data = torch.stack(
            [self.get_column(col) for col in magnitude_columns], dim=1
        )

        error_data = None
        if error_columns:
            error_data = torch.stack(
                [self.get_column(col) for col in error_columns], dim=1
            )

        # Create lightcurve tensor
        lightcurve_tensor = LightcurveTensor(
            times=time_data,
            magnitudes=mag_data,
            errors=error_data,
            bands=magnitude_columns,
            survey_name=self.survey_name,
        )

        return lightcurve_tensor

    # =========================================================================
    # Utility methods with improved error handling
    # =========================================================================

    def _detect_photometric_columns(self) -> List[str]:
        """Detect photometric columns based on survey type and column names."""
        survey_name = self.survey_name.lower()

        # Survey-specific band detection
        if survey_name == "gaia":
            return ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]
        elif survey_name == "sdss":
            return ["u", "g", "r", "i", "z"]
        elif survey_name == "lsst":
            return ["u", "g", "r", "i", "z", "y"]
        elif survey_name == "jwst":
            return ["f090w", "f115w", "f150w", "f200w", "f277w", "f356w", "f444w"]
        else:
            # Generic detection based on common patterns
            photometric_patterns = ["mag", "flux", "_g", "_r", "_i", "_z", "_u", "_y"]
            return [
                col
                for col in self.column_mapping.keys()
                if any(pattern in col.lower() for pattern in photometric_patterns)
            ]

    def _detect_spatial_columns(self, include_distances: bool = True) -> List[str]:
        """Detect spatial coordinate columns."""
        base_columns = ["ra", "dec"]

        if include_distances:
            # Add distance-related columns if available
            distance_columns = ["parallax", "distance", "dist"]
            for col in distance_columns:
                if col in self.column_mapping:
                    base_columns.append(col)
                    break

        return [col for col in base_columns if col in self.column_mapping]

    def _is_magnitude_system(self, filter_system: str) -> bool:
        """Determine if filter system uses magnitude or flux units."""
        # Most optical surveys use magnitudes
        magnitude_systems = ["sdss", "gaia", "lsst", "des", "ps1", "hst"]
        flux_systems = ["wise", "2mass", "spitzer"]

        system_lower = filter_system.lower()

        if any(mag_sys in system_lower for mag_sys in magnitude_systems):
            return True
        elif any(flux_sys in system_lower for flux_sys in flux_systems):
            return False
        else:
            # Default to magnitude for unknown systems
            return True

    def get_column(self, column_name: str) -> torch.Tensor:
        """
        Get data for a specific column with improved error handling.

        Args:
            column_name: Name of the column to extract

        Returns:
            Tensor containing column data

        Raises:
            KeyError: If column not found in mapping
            IndexError: If column index is out of bounds
        """
        if column_name not in self.column_mapping:
            available_columns = list(self.column_mapping.keys())
            raise KeyError(
                f"Column '{column_name}' not found. "
                f"Available columns: {available_columns}"
            )

        column_index = self.column_mapping[column_name]

        if column_index >= self._data.shape[-1]:
            raise IndexError(
                f"Column index {column_index} out of bounds for data shape {self._data.shape}"
            )

        return self._data[:, column_index]

    def __repr__(self) -> str:
        """Improved string representation."""
        n_objects = len(self)
        n_columns = self._data.shape[-1] if self._data.dim() > 1 else 1

        return (
            f"SurveyTensor(survey='{self.survey_name}', "
            f"release='{self.data_release or 'unknown'}', "
            f"objects={n_objects}, columns={n_columns}, "
            f"filter_system='{self.filter_system}')"
        )
