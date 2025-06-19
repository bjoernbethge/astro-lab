"""
Survey tensor for multi-survey astronomical data - Main Coordinator Tensor.

This tensor acts as the main coordinator for all specialized astronomical tensors,
providing unified access to photometry, astrometry, lightcurves, spectroscopy, etc.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch

from .base import AstroTensorBase

# Forward references for type hints to avoid circular imports
if TYPE_CHECKING:
    from .photometric import PhotometricTensor
    from .lightcurve import LightcurveTensor
    from .spatial_3d import Spatial3DTensor
    from .spectral import SpectralTensor

# All type annotations use string literals to avoid runtime imports


class SurveyTensor(AstroTensorBase):
    """
    Main coordinator tensor for astronomical survey data using composition.

    Coordinates all specialized tensors and provides unified access to:
    - Photometric measurements (via PhotometricTensor)
    - Time series data (via LightcurveTensor)
    - 3D spatial coordinates (via Spatial3DTensor)
    - Spectroscopic data (via SpectralTensor)
    - Astrometric measurements (via AstrometricTensor)
    - Survey transformations and metadata
    """

    _metadata_fields = [
        "survey_name",
        "data_release",
        "filter_system",
        "column_mapping",
        "survey_metadata",
        "transformations",
        # Specialized tensor references
        "_photometric_tensor",
        "_lightcurve_tensor",
        "_spatial_tensor",
        "_spectral_tensor",
        "_astrometric_tensor",
    ]

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
            or self._get_default_filter_system(survey_name),
            "column_mapping": column_mapping or self._get_default_columns(survey_name),
            "survey_metadata": survey_metadata or {},
            "transformations": transformations or {},
        }
        metadata.update(kwargs)

        super().__init__(data, **metadata, tensor_type="survey")

    def _validate(self) -> None:
        """Validate survey tensor data."""
        if self._data.dim() == 0:
            raise ValueError("SurveyTensor requires at least 1D data")

        survey_name = self._metadata.get("survey_name")
        if not survey_name:
            raise ValueError("SurveyTensor requires survey_name")

    @staticmethod  # type: ignore
    def _get_default_filter_system(survey_name: str) -> str:
        """Get default filter system for known surveys."""
        filter_systems = {
            "gaia": "gaia",
            "sdss": "sdss",
            "lsst": "lsst",
            "euclid": "euclid",
            "des": "des",
            "ps1": "ps1",
            "2mass": "2mass",
            "wise": "wise",
        }
        return filter_systems.get(survey_name.lower(), "unknown")

    @staticmethod
    def _get_known_surveys() -> List[str]:
        """Get list of known surveys."""
        return ["gaia", "sdss", "lsst", "euclid", "des", "ps1", "2mass", "wise"]

    @staticmethod  # type: ignore
    def _get_default_columns(survey_name: str) -> Dict[str, int]:
        """Get default column mapping for known surveys."""
        # This would be populated with actual survey schemas
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
            },
        }
        return default_mappings.get(survey_name.lower(), {})

    @property
    def survey_name(self) -> str:
        """Survey name."""
        return self._metadata.get("survey_name", "unknown")

    @property
    def data_release(self) -> Optional[str]:
        """Data release version."""
        return self._metadata.get("data_release")

    @property
    def filter_system(self) -> str:
        """Filter system."""
        return self._metadata.get("filter_system", "unknown")

    @property
    def column_mapping(self) -> Dict[str, int]:
        """Column mapping."""
        return self._metadata.get("column_mapping", {})

    def get_photometric_tensor(
        self, band_columns: Optional[List[str]] = None, force_recreate: bool = False
    ) -> "PhotometricTensor":
        """
        Get or create PhotometricTensor for this survey.

        Args:
            band_columns: Specific bands to extract (auto-detect if None)
            force_recreate: Force recreation even if cached

        Returns:
            PhotometricTensor with photometric measurements
        """
        # Import at runtime to avoid circular dependencies
        from .photometric import PhotometricTensor

        if not force_recreate and self._metadata.get("_photometric_tensor") is not None:
            return self._metadata["_photometric_tensor"]

        if band_columns is None:
            band_columns = self._detect_photometric_columns()

        if not band_columns:
            raise ValueError("No photometric columns found in survey data")

        # Extract photometric data
        band_indices = [
            self.column_mapping[col]
            for col in band_columns
            if col in self.column_mapping
        ]

        if not band_indices:
            raise ValueError(
                f"None of the bands {band_columns} found in column mapping"
            )

        phot_data = self._data[..., band_indices]

        # Get extinction coefficients for survey
        extinction_coeffs = self._get_extinction_coefficients()

        phot_tensor = PhotometricTensor(
            data=phot_data,
            bands=band_columns,
            measurement_errors=None,
            extinction_coefficients=extinction_coeffs,
            photometric_system="AB",  # Most surveys use AB
            is_magnitude=True,
            zero_points={},
        )

        # Cache the tensor
        self.update_metadata(_photometric_tensor=phot_tensor)
        return phot_tensor

    def get_spatial_tensor(
        self, include_distances: bool = True, force_recreate: bool = False
    ) -> "Spatial3DTensor":
        """
        Get or create Spatial3DTensor for coordinate analysis.

        Args:
            include_distances: Whether to include distance information
            force_recreate: Force recreation even if cached

        Returns:
            Spatial3DTensor with 3D coordinates
        """
        # Import at runtime to avoid circular dependencies
        from .spatial_3d import Spatial3DTensor

        if not force_recreate and self._metadata.get("_spatial_tensor") is not None:
            return self._metadata["_spatial_tensor"]

        # Extract coordinates
        if "ra" not in self.column_mapping or "dec" not in self.column_mapping:
            raise ValueError("RA and DEC columns required for spatial tensor")

        ra = self.get_column("ra")
        dec = self.get_column("dec")

        # Try to get distances
        distance = None
        if include_distances:
            for dist_col in ["distance", "dist", "r_dist", "zdist"]:
                if dist_col in self.column_mapping:
                    distance = self.get_column(dist_col)
                    break

            # If no direct distance, try to derive from redshift
            if distance is None and "redshift" in self.column_mapping:
                z = self.get_column("redshift")
                # Simple distance calculation (can be enhanced)
                distance = z * 3000.0  # Very approximate Mpc

        if distance is None:
            # Use unit distances for pure angular analysis
            distance = torch.ones_like(ra)

        spatial_tensor = Spatial3DTensor(
            data=torch.stack([ra, dec, distance], dim=-1),
            coordinate_system="equatorial",
            unit="deg",
            distance_unit="Mpc",
        )

        # Cache the tensor
        self.update_metadata(_spatial_tensor=spatial_tensor)
        return spatial_tensor

    def get_astrometric_tensor(
        self, include_radial_velocity: bool = True, force_recreate: bool = False
    ) -> "Spatial3DTensor":
        """
        Get or create Spatial3DTensor with astrometric data for astrometric analysis.

        Args:
            include_radial_velocity: Whether to include radial velocity if available
            force_recreate: Force recreation even if cached

        Returns:
            Spatial3DTensor with astrometric data
        """
        # Import at runtime to avoid circular dependencies
        from .spatial_3d import Spatial3DTensor

        if not force_recreate and self._metadata.get("_astrometric_tensor") is not None:
            return self._metadata["_astrometric_tensor"]

        column_mapping = self.column_mapping

        # Required columns for astrometry
        required_cols = ["ra", "dec", "parallax", "pmra", "pmdec"]
        missing_cols = [col for col in required_cols if col not in column_mapping]

        if missing_cols:
            raise ValueError(f"Missing required astrometric columns: {missing_cols}")

        # Extract coordinate and astrometric data
        ra = self.get_column("ra")
        dec = self.get_column("dec")
        parallax = self.get_column("parallax")
        pmra = self.get_column("pmra")
        pmdec = self.get_column("pmdec")

        # Optional radial velocity
        radial_velocity = None
        if include_radial_velocity and "radial_velocity" in column_mapping:
            radial_velocity = self.get_column("radial_velocity")

        # Create Spatial3DTensor from spherical coordinates
        # Convert parallax to distance (1000/parallax_mas = distance_pc)
        distance_pc = 1000.0 / torch.clamp(
            parallax, min=0.001
        )  # Avoid division by zero
        distance_mpc = distance_pc / 1e6  # Convert to Mpc

        astrometric_tensor = Spatial3DTensor.from_spherical(
            ra=ra,
            dec=dec,
            distance=distance_mpc,
            coordinate_system="icrs",
            unit="Mpc",
            # Add astrometric metadata
            parallax=parallax,
            proper_motion_ra=pmra,
            proper_motion_dec=pmdec,
            radial_velocity=radial_velocity,
            epoch=2016.0,  # Gaia DR3 epoch
            astrometric_source="gaia",
        )

        # Cache the tensor
        self.update_metadata(_astrometric_tensor=astrometric_tensor)
        return astrometric_tensor

    def create_lightcurve_tensor(
        self,
        time_column: str,
        magnitude_columns: List[str],
        error_columns: Optional[List[str]] = None,
        object_id_column: Optional[str] = None,
    ) -> "LightcurveTensor":
        """
        Create LightcurveTensor from time-series survey data.

        Args:
            time_column: Name of time/date column
            magnitude_columns: Names of magnitude columns
            error_columns: Names of error columns (optional)
            object_id_column: Name of object ID column (optional)

        Returns:
            LightcurveTensor with time-series data
        """
        # Import at runtime to avoid circular dependencies
        from .lightcurve import LightcurveTensor

        if time_column not in self.column_mapping:
            raise ValueError(f"Time column '{time_column}' not found")

        missing_mags = [
            col for col in magnitude_columns if col not in self.column_mapping
        ]
        if missing_mags:
            raise ValueError(f"Missing magnitude columns: {missing_mags}")

        # Extract time series data
        times = self.get_column(time_column)

        # Stack magnitude data
        mag_data = []
        for mag_col in magnitude_columns:
            mag_data.append(self.get_column(mag_col))
        magnitudes = torch.stack(mag_data, dim=-1)

        # Extract errors if provided
        errors = None
        if error_columns:
            available_errs = [
                col for col in error_columns if col in self.column_mapping
            ]
            if available_errs:
                err_data = []
                for err_col in available_errs:
                    err_data.append(self.get_column(err_col))
                errors = torch.stack(err_data, dim=-1)

        # Extract object IDs if provided
        object_ids = None
        if object_id_column and object_id_column in self.column_mapping:
            object_ids = self.get_column(object_id_column)

        lightcurve_tensor = LightcurveTensor(
            times=times,
            magnitudes=magnitudes,
            errors=errors,
            object_ids=object_ids,
            bands=magnitude_columns,
            time_unit="days",  # Could be determined from survey metadata
            magnitude_system="AB",
        )

        # Cache the tensor
        self.update_metadata(_lightcurve_tensor=lightcurve_tensor)
        return lightcurve_tensor

    def get_unified_catalog(self) -> Dict[str, Any]:
        """
        Get unified catalog with all specialized tensor data.

        Returns:
            Dictionary with all available tensor data
        """
        unified = {
            "survey_info": {
                "name": self.survey_name,
                "data_release": self.data_release,
                "filter_system": self.filter_system,
                "n_objects": len(self),
                "n_columns": self.shape[-1] if self.dim() > 0 else 0,
            },
            "raw_data": self._data,
            "column_mapping": self.column_mapping,
        }

        # Add specialized tensor data if available
        try:
            unified["photometry"] = self.get_photometric_tensor().to_dict()
        except (ValueError, ImportError):
            unified["photometry"] = None

        try:
            unified["spatial"] = self.get_spatial_tensor().to_dict()
        except (ValueError, ImportError):
            unified["spatial"] = None

        try:
            unified["astrometry"] = self.get_astrometric_tensor().to_dict()
        except (ValueError, ImportError):
            unified["astrometry"] = None

        return unified

    def _detect_photometric_columns(self) -> List[str]:
        """Auto-detect photometric columns."""
        photometric_patterns = {
            "gaia": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
            "sdss": ["u", "g", "r", "i", "z"],
            "lsst": ["u", "g", "r", "i", "z", "y"],
            "des": ["g", "r", "i", "z", "Y"],
            "ps1": ["g", "r", "i", "z", "y"],
            "2mass": ["J", "H", "K"],
            "wise": ["W1", "W2", "W3", "W4"],
        }

        if self.survey_name.lower() in photometric_patterns:
            pattern = photometric_patterns[self.survey_name.lower()]
            return [col for col in pattern if col in self.column_mapping]

        # Generic detection
        phot_cols = []
        for col in self.column_mapping:
            if any(keyword in col.lower() for keyword in ["mag", "flux", "phot"]):
                phot_cols.append(col)

        return phot_cols

    def _get_extinction_coefficients(self) -> Dict[str, float]:
        """Get extinction coefficients for survey bands."""
        # A_lambda / A_V coefficients for common surveys
        extinction_coeffs = {
            "gaia": {
                "phot_g_mean_mag": 0.789,
                "phot_bp_mean_mag": 1.002,
                "phot_rp_mean_mag": 0.634,
            },
            "sdss": {"u": 1.579, "g": 1.161, "r": 0.871, "i": 0.672, "z": 0.487},
            "lsst": {
                "u": 1.569,
                "g": 1.181,
                "r": 0.876,
                "i": 0.671,
                "z": 0.486,
                "y": 0.395,
            },
        }

        return extinction_coeffs.get(self.survey_name.lower(), {})

    def transform_to_survey(
        self, target_survey: str, target_release: Optional[str] = None
    ) -> "SurveyTensor":
        """
        Transform data to another survey system.

        Args:
            target_survey: Target survey name
            target_release: Target data release

        Returns:
            Transformed SurveyTensor
        """
        survey_name = self.survey_name
        transform_key = f"{survey_name}_to_{target_survey}"

        # Check if direct transformation exists
        transformations = self._metadata.get("transformations", {})
        if transform_key in transformations:
            transform_func = transformations[transform_key]
            transformed_data = transform_func(self)
        else:
            # Try to find transformation path
            transformed_data = self._apply_transformation(target_survey)

        # Create new survey tensor
        return SurveyTensor(
            transformed_data,
            survey_name=target_survey,
            data_release=target_release,
            filter_system=self._get_default_filter_system(target_survey),
            column_mapping={},
            survey_metadata={},
            transformations=transformations,
        )

    def _apply_transformation(self, target_survey: str) -> torch.Tensor:
        """Apply survey transformation."""
        survey_name = self.survey_name
        # Implement common transformations
        if survey_name == "gaia" and target_survey == "sdss":
            return self._gaia_to_sdss()
        elif survey_name == "sdss" and target_survey == "gaia":
            return self._sdss_to_gaia()
        elif survey_name == "gaia" and target_survey == "lsst":
            return self._gaia_to_lsst()
        else:
            raise NotImplementedError(
                f"Transformation from {survey_name} to {target_survey} not implemented"
            )

    def _gaia_to_sdss(self) -> torch.Tensor:
        """Transform Gaia photometry to SDSS system."""
        # Get Gaia photometry
        g = self.get_column("phot_g_mean_mag")
        bp = self.get_column("phot_bp_mean_mag")
        rp = self.get_column("phot_rp_mean_mag")

        # Color indices
        bp_rp = bp - rp
        g_rp = g - rp

        # Transformation equations (simplified, from Gaia DR3 documentation)
        # These are approximate transformations
        sdss_g = (
            g - 0.13518 - 0.46245 * bp_rp - 0.25171 * bp_rp**2 + 0.021349 * bp_rp**3
        )
        sdss_r = (
            g - 0.12879 + 0.24662 * bp_rp - 0.027464 * bp_rp**2 - 0.049465 * bp_rp**3
        )
        sdss_i = g - 0.29676 + 0.64728 * bp_rp - 0.10141 * bp_rp**2

        # Approximate u and z from other bands
        sdss_u = sdss_g + 1.5 * (sdss_g - sdss_r)  # Very approximate
        sdss_z = sdss_i - 0.5 * (sdss_r - sdss_i)  # Very approximate

        # Combine into new data array
        n_objects = len(self)
        transformed = torch.zeros((n_objects, 7))  # ra, dec, u, g, r, i, z

        # Copy coordinates
        transformed[:, 0] = self.get_column("ra")
        transformed[:, 1] = self.get_column("dec")

        # Add photometry
        transformed[:, 2] = sdss_u
        transformed[:, 3] = sdss_g
        transformed[:, 4] = sdss_r
        transformed[:, 5] = sdss_i
        transformed[:, 6] = sdss_z

        return transformed

    def _sdss_to_gaia(self) -> torch.Tensor:
        """Transform SDSS photometry to Gaia system."""
        # Get SDSS photometry
        g = self.get_column("g")
        r = self.get_column("r")
        i = self.get_column("i")

        # Inverse transformation (approximate)
        g_r = g - r
        r_i = r - i

        # Gaia G magnitude
        gaia_g = (
            r
            + 0.12879
            - 0.24662 * (g_r)
            + 0.027464 * (g_r) ** 2
            + 0.049465 * (g_r) ** 3
        )

        # Gaia BP and RP (approximate)
        gaia_bp = g + 0.2
        gaia_rp = r - 0.1

        # Combine into new data array
        n_objects = len(self)
        transformed = torch.zeros(
            (n_objects, 8)
        )  # ra, dec, parallax, pmra, pmdec, G, BP, RP

        # Copy coordinates and astrometry if available
        transformed[:, 0] = self.get_column("ra")
        transformed[:, 1] = self.get_column("dec")

        # Add photometry
        transformed[:, 5] = gaia_g
        transformed[:, 6] = gaia_bp
        transformed[:, 7] = gaia_rp

        return transformed

    def _gaia_to_lsst(self) -> torch.Tensor:
        """Transform Gaia photometry to LSST system."""
        # First transform to SDSS as intermediate
        sdss_data = self._gaia_to_sdss()

        # LSST bands are similar to SDSS but with slight differences
        # These transformations are approximate
        lsst_u = sdss_data[:, 2] - 0.241 * (sdss_data[:, 2] - sdss_data[:, 3])
        lsst_g = sdss_data[:, 3] + 0.013 + 0.145 * (sdss_data[:, 3] - sdss_data[:, 4])
        lsst_r = sdss_data[:, 4] + 0.001 + 0.004 * (sdss_data[:, 3] - sdss_data[:, 4])
        lsst_i = sdss_data[:, 5] + 0.002 - 0.002 * (sdss_data[:, 4] - sdss_data[:, 5])
        lsst_z = sdss_data[:, 6] - 0.005 - 0.013 * (sdss_data[:, 5] - sdss_data[:, 6])

        # LSST y-band (approximate from z and i)
        lsst_y = lsst_z - 0.6 * (lsst_i - lsst_z)

        # Combine into new data array
        n_objects = len(self)
        transformed = torch.zeros((n_objects, 8))  # ra, dec, u, g, r, i, z, y

        transformed[:, 0] = self.get_column("ra")
        transformed[:, 1] = self.get_column("dec")
        transformed[:, 2] = lsst_u
        transformed[:, 3] = lsst_g
        transformed[:, 4] = lsst_r
        transformed[:, 5] = lsst_i
        transformed[:, 6] = lsst_z
        transformed[:, 7] = lsst_y

        return transformed

    def register_transformation(
        self, source_survey: str, target_survey: str, transform_func: Callable
    ) -> None:
        """
        Register a custom transformation function.

        Args:
            source_survey: Source survey name
            target_survey: Target survey name
            transform_func: Transformation function
        """
        transformations = self._metadata.get("transformations", {})
        key = f"{source_survey}_to_{target_survey}"
        transformations[key] = transform_func

    def apply_quality_cuts(
        self, criteria: Dict[str, Tuple[Optional[float], Optional[float]]]
    ) -> "SurveyTensor":
        """
        Apply quality cuts based on survey-specific criteria.

        Args:
            criteria: Dictionary mapping column names to (min, max) values

        Returns:
            Filtered SurveyTensor
        """
        mask = torch.ones(len(self), dtype=torch.bool)

        column_mapping = self.column_mapping
        for col_name, (min_val, max_val) in criteria.items():
            if col_name not in column_mapping:
                continue

            col_data = self.get_column(col_name)

            if min_val is not None:
                mask &= col_data >= min_val
            if max_val is not None:
                mask &= col_data <= max_val

        # Create filtered tensor with same metadata
        filtered_data = self._data[mask]
        return SurveyTensor(
            filtered_data,
            survey_name=self.survey_name,
            data_release=self.data_release,
            filter_system=self.filter_system,
            column_mapping=self.column_mapping,
            survey_metadata=self._metadata.get("survey_metadata", {}),
            transformations=self._metadata.get("transformations", {}),
        )

    def get_column(self, column_name: str) -> torch.Tensor:
        """Get values for a specific column."""
        if column_name not in self.column_mapping:
            raise ValueError(f"Column '{column_name}' not found in {self.survey_name}")

        idx = self.column_mapping[column_name]
        return self._data[..., idx]

    def add_derived_columns(self, derived: Dict[str, torch.Tensor]) -> "SurveyTensor":
        """
        Add derived columns to the survey data.

        Args:
            derived: Dictionary mapping column names to tensors

        Returns:
            New SurveyTensor with added columns
        """
        # Current data
        current_n_cols = self.shape[-1]

        # Add new columns
        new_data = torch.cat(
            [self._data, torch.stack(list(derived.values()), dim=-1)], dim=-1
        )

        # Update column mapping
        column_mapping = self.column_mapping.copy()
        new_mapping = column_mapping.copy()
        for i, col_name in enumerate(derived.keys()):
            new_mapping[col_name] = current_n_cols + i

        return SurveyTensor(
            new_data,
            survey_name=self.survey_name,
            data_release=self.data_release,
            filter_system=self.filter_system,
            column_mapping=new_mapping,
            survey_metadata=self._metadata.get("survey_metadata", {}),
            transformations=self._metadata.get("transformations", {}),
        )

    def get_catalog_data(self) -> Dict[str, Any]:
        """
        Get catalog data in standardized format.

        Returns:
            Dictionary with standardized catalog fields
        """
        survey_name = self.survey_name
        data_release = self.data_release
        column_mapping = self.column_mapping

        catalog_data = {
            "survey_name": survey_name,
            "data_release": data_release,
            "raw_data": self._data,
            "column_mapping": column_mapping,
            "n_objects": len(self),
            "n_columns": self.shape[-1] if self.dim() > 0 else 0,
        }

        # Extract coordinates if available
        if "ra" in column_mapping and "dec" in column_mapping:
            ra = self.get_column("ra")
            dec = self.get_column("dec")
            catalog_data["coordinates"] = torch.stack([ra, dec], dim=-1)
            catalog_data["ra"] = ra
            catalog_data["dec"] = dec

        # Extract photometry
        phot_cols = self._detect_photometric_columns()
        if phot_cols:
            phot_data = torch.stack([self.get_column(col) for col in phot_cols], dim=-1)
            catalog_data["photometry"] = phot_data
            catalog_data["bands"] = phot_cols

        return catalog_data

    def compute_survey_statistics(self) -> Dict[str, Any]:
        """Compute survey-specific statistics."""
        survey_name = self.survey_name
        data_release = self.data_release
        column_mapping = self.column_mapping
        stats = {
            "survey": survey_name,
            "data_release": data_release,
            "n_objects": len(self),
            "n_columns": len(column_mapping),
        }

        # Photometric statistics
        phot_cols = self._detect_photometric_columns()
        if phot_cols:
            phot_stats = {}
            for col in phot_cols:
                col_data = self.get_column(col)
                phot_stats[col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "median": float(torch.median(col_data)),
                }
            stats["photometry"] = phot_stats

        # Coordinate coverage
        if "ra" in column_mapping and "dec" in column_mapping:
            ra = self.get_column("ra")
            dec = self.get_column("dec")
            stats["sky_coverage"] = {
                "ra_range": [float(ra.min()), float(ra.max())],
                "dec_range": [float(dec.min()), float(dec.max())],
                "area_sq_deg": float(
                    (ra.max() - ra.min())
                    * (dec.max() - dec.min())
                    * torch.cos(torch.deg2rad(dec.mean()))
                ),
            }

        return stats

    def match_to_reference(
        self, reference: "SurveyTensor", radius: float = 1.0, unit: str = "arcsec"
    ) -> Dict[str, torch.Tensor]:
        """
        Match to a reference survey catalog using spatial coordinates.

        Args:
            reference: Reference SurveyTensor
            radius: Matching radius
            unit: Unit of radius

        Returns:
            Match results with indices and separations
        """
        # Get coordinate data from both surveys
        self_data = self.get_catalog_data()
        ref_data = reference.get_catalog_data()

        if "coordinates" not in self_data or "coordinates" not in ref_data:
            raise ValueError("Both surveys must have coordinate data for matching")

        # Simple nearest neighbor matching (can be enhanced later)
        self_coords = self_data["coordinates"]
        ref_coords = ref_data["coordinates"]

        # Calculate angular separations (simplified)
        # This is a basic implementation - could be enhanced with proper spherical geometry
        separations = torch.cdist(self_coords, ref_coords)

        # Convert radius to degrees if needed
        if unit == "arcsec":
            radius_deg = radius / 3600.0
        elif unit == "arcmin":
            radius_deg = radius / 60.0
        else:
            radius_deg = radius

        # Find matches within radius
        min_seps, indices = torch.min(separations, dim=1)
        mask = min_seps <= radius_deg

        return {
            "match_indices": indices[mask],
            "separations": min_seps[mask],
            "mask": mask,
        }

    def __repr__(self) -> str:
        """String representation."""
        data_release = self.data_release
        survey_name = self.survey_name
        column_mapping = self.column_mapping
        filter_system = self.filter_system

        dr_str = f"_{data_release}" if data_release else ""
        n_objects = self.shape[0] if self.dim() >= 1 else 1
        n_columns = len(column_mapping)

        return (
            f"SurveyTensor(survey='{survey_name}{dr_str}', "
            f"n_objects={n_objects}, n_columns={n_columns}, "
            f"filter_system='{filter_system}')"
        )

    # ========== Exoplanet Analysis Features (consolidated from EphemerisTensor) ==========

    def exoplanet_habitability_score(self) -> Optional[torch.Tensor]:
        """
        Calculate habitability score for exoplanets in the survey.

        Returns:
            Habitability score [0, 1] or None if no exoplanet data
        """
        column_mapping = self.column_mapping

        # Check if this contains exoplanet data
        exoplanet_columns = [
            "planet_radius",
            "planet_mass",
            "equilibrium_temperature",
            "stellar_flux",
        ]
        available_columns = [col for col in exoplanet_columns if col in column_mapping]

        if not available_columns:
            return None

        # Initialize score
        score = torch.ones_like(self.get_column(available_columns[0]))

        # Temperature factor (assume Earth-like optimal at 288K)
        if "equilibrium_temperature" in column_mapping:
            temp = self.get_column("equilibrium_temperature")
            temp_optimal = 288.0  # K
            temp_factor = torch.exp(-(((temp - temp_optimal) / 50.0) ** 2))
            score *= temp_factor

        # Size factor (Earth-like optimal)
        if "planet_radius" in column_mapping:
            radius = self.get_column("planet_radius")  # Earth radii
            size_factor = torch.exp(-(((radius - 1.0) / 0.5) ** 2))
            score *= size_factor

        # Mass factor (affects gravity)
        if "planet_mass" in column_mapping:
            mass = self.get_column("planet_mass")  # Earth masses
            mass_factor = torch.exp(-(((mass - 1.0) / 2.0) ** 2))
            score *= mass_factor

        # Stellar flux factor
        if "stellar_flux" in column_mapping:
            flux = self.get_column("stellar_flux")  # Earth flux units
            flux_factor = torch.exp(-(((flux - 1.0) / 0.5) ** 2))
            score *= flux_factor

        return torch.clamp(score, 0.0, 1.0)

    def atmospheric_escape_rate(self) -> Optional[torch.Tensor]:
        """
        Estimate atmospheric escape rate for exoplanets.

        Returns:
            Escape rate in kg/s (logarithmic scale) or None if no data
        """
        column_mapping = self.column_mapping

        # Required columns for escape rate calculation
        required_cols = ["planet_mass", "planet_radius", "equilibrium_temperature"]
        missing_cols = [col for col in required_cols if col not in column_mapping]

        if missing_cols:
            return None

        # Get planet properties
        mass = self.get_column("planet_mass")  # Earth masses
        radius = self.get_column("planet_radius")  # Earth radii
        temp = self.get_column("equilibrium_temperature")  # K

        # Stellar flux (optional)
        if "stellar_flux" in column_mapping:
            stellar_flux = self.get_column("stellar_flux")
        else:
            stellar_flux = torch.ones_like(mass)  # Earth flux units

        # Escape velocity calculation
        earth_escape_vel = 11.2  # km/s
        escape_vel = earth_escape_vel * torch.sqrt(mass / radius)

        # Jeans escape (simplified)
        # Higher temperature and lower gravity increase escape
        escape_rate_log = torch.log10(stellar_flux * temp / (mass / radius**2))

        return escape_rate_log

    def biosignature_potential(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Assess potential for detecting biosignatures in exoplanet survey.

        Returns:
            Dictionary with biosignature detection metrics or None if no data
        """
        column_mapping = self.column_mapping

        # Check for distance data
        if "distance" not in column_mapping:
            return None

        distance_ly = self.get_column("distance")

        # Convert distance units if needed
        if getattr(self, "distance_unit", "pc") == "pc":
            distance_ly = distance_ly * 3.26156  # pc to ly
        elif getattr(self, "distance_unit", "pc") == "Mpc":
            distance_ly = distance_ly * 3.26156e6  # Mpc to ly

        # Transit probability (optional)
        if "transit_probability" in column_mapping:
            transit_prob = self.get_column("transit_probability")
        else:
            transit_prob = torch.full_like(distance_ly, 0.01)  # Default 1%

        # Atmospheric retention (from escape rate if available)
        escape_rate = self.atmospheric_escape_rate()
        if escape_rate is not None:
            retention_factor = torch.sigmoid(-escape_rate + 5)  # Sigmoid function
        else:
            retention_factor = torch.full_like(distance_ly, 0.5)  # Default

        # Distance factor (closer = easier to observe)
        distance_factor = torch.exp(
            -distance_ly / 50.0
        )  # 50 ly characteristic distance

        # Host star brightness (affects SNR)
        if "host_star_magnitude" in column_mapping:
            stellar_magnitude = self.get_column("host_star_magnitude")
        else:
            stellar_magnitude = torch.full_like(distance_ly, 10.0)  # Default magnitude

        brightness_factor = torch.exp(-(stellar_magnitude - 5.0) / 5.0)

        # Overall potential
        overall = transit_prob * retention_factor * distance_factor * brightness_factor

        return {
            "overall_potential": overall,
            "transit_probability": transit_prob,
            "atmospheric_retention": retention_factor,
            "distance_factor": distance_factor,
            "brightness_factor": brightness_factor,
        }

    def filter_habitable_exoplanets(
        self,
        min_habitability_score: float = 0.5,
        max_distance_ly: float = 100.0,
        min_biosignature_potential: float = 0.1,
    ) -> "SurveyTensor":
        """
        Filter survey to only include potentially habitable exoplanets.

        Args:
            min_habitability_score: Minimum habitability score
            max_distance_ly: Maximum distance in light-years
            min_biosignature_potential: Minimum biosignature detection potential

        Returns:
            Filtered SurveyTensor with habitable exoplanets
        """
        # Calculate habitability metrics
        hab_score = self.exoplanet_habitability_score()
        biosig_potential = self.biosignature_potential()

        if hab_score is None or biosig_potential is None:
            raise ValueError("Exoplanet data not available for habitability filtering")

        # Distance filter
        distance_ly = biosig_potential["distance_factor"]  # Already converted

        # Create filter mask
        mask = (
            (hab_score >= min_habitability_score)
            & (distance_ly <= max_distance_ly)
            & (biosig_potential["overall_potential"] >= min_biosignature_potential)
        )

        # Apply filter
        filtered_data = self._data[mask]

        # Create new filtered tensor
        filtered_tensor = SurveyTensor(
            filtered_data,
            survey_name=self.survey_name,
            data_release=self.data_release,
            filter_system=self.filter_system,
            column_mapping=self.column_mapping,
            survey_metadata=self._metadata.get("survey_metadata", {}),
            transformations=self._metadata.get("transformations", {}),
        )

        return filtered_tensor

    def exoplanet_summary_statistics(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Calculate summary statistics for exoplanet survey data.

        Returns:
            Dictionary with summary statistics or None if no exoplanet data
        """
        column_mapping = self.column_mapping

        exoplanet_columns = [
            "planet_radius",
            "planet_mass",
            "equilibrium_temperature",
            "stellar_flux",
            "host_star_magnitude",
            "distance",
        ]

        available_columns = [col for col in exoplanet_columns if col in column_mapping]

        if not available_columns:
            return None

        stats = {}

        for col in available_columns:
            data = self.get_column(col)
            stats[f"{col}_mean"] = torch.mean(data)
            stats[f"{col}_std"] = torch.std(data)
            stats[f"{col}_min"] = torch.min(data)
            stats[f"{col}_max"] = torch.max(data)
            stats[f"{col}_median"] = torch.median(data)

        # Special calculations
        hab_score = self.exoplanet_habitability_score()
        if hab_score is not None:
            stats["habitability_score_mean"] = torch.mean(hab_score)
            stats["num_potentially_habitable"] = torch.sum(hab_score > 0.5)

        biosig_potential = self.biosignature_potential()
        if biosig_potential is not None:
            stats["biosignature_potential_mean"] = torch.mean(
                biosig_potential["overall_potential"]
            )
            stats["num_high_biosignature_potential"] = torch.sum(
                biosig_potential["overall_potential"] > 0.1
            )

        stats["total_exoplanets"] = torch.tensor(self.shape[0])

        return stats

    def dim(self) -> int:
        """Number of dimensions."""
        return self._data.dim()
