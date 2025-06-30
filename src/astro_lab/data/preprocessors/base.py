"""
Base Survey Processor
====================

Unified data collection and preprocessing interface with proper TensorDict integration.
"""

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from astro_lab.config import get_data_config

# Import our tensor classes
from astro_lab.tensors import (
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
)

logger = logging.getLogger(__name__)


class BaseSurveyProcessor(ABC):
    """
    Base class for survey data processing with enhanced TensorDict integration.

    Features:
    - Unified data loading from multiple formats (FITS, Parquet, CSV, HDF5)
    - Proper astronomical coordinate handling with astropy
    - Multi-modal data extraction (spatial, photometric, spectral, temporal)
    - Data quality filtering and validation
    - Memory-efficient processing for large surveys
    - Standardized output format compatible with PyG and Lightning
    - Comprehensive metadata tracking
    """

    feature_names: Optional[List[str]] = None

    def __init__(self, survey_name: str, data_config: Optional[Dict] = None):
        self.survey_name = survey_name
        self.config = data_config or self._get_default_config()
        self._setup_paths()
        self._setup_coordinate_system()
        self._cache = {}  # For caching frequently used data

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for the survey. Override in subclasses."""
        return {
            "coordinate_system": "icrs",
            "epoch": "J2000",
            "distance_unit": "pc",
            "filter_system": "AB",
        }

    def _setup_paths(self):
        """Setup data paths for raw and processed data."""
        data_config = get_data_config()
        self.raw_dir = data_config.get_survey_raw_dir(self.survey_name)
        self.processed_dir = data_config.get_survey_processed_dir(self.survey_name)

        # Ensure directories exist
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _setup_coordinate_system(self):
        """Setup coordinate system parameters."""
        self.coordinate_system = self.config.get("coordinate_system", "icrs")
        self.epoch = self.config.get("epoch", "J2000")
        self.distance_unit = self.config.get("distance_unit", "pc")
        self.filter_system = self.config.get("filter_system", "AB")

    def get_data_path(self) -> Path:
        """Get path to survey data file."""
        # Look for processed data first
        processed_file = self.processed_dir / f"{self.survey_name}.parquet"
        if processed_file.exists() and processed_file.stat().st_size > 1000:
            return processed_file

        # Fallback to raw data
        raw_file = self.raw_dir / f"{self.survey_name}.parquet"
        if raw_file.exists():
            return raw_file

        # If no file found, return a default path for error handling
        return self.processed_dir / f"{self.survey_name}.parquet"

    def load_data(self) -> pl.DataFrame:
        """Load survey data directly using Polars."""
        logger.info(f"📂 Loading {self.survey_name} data")

        # Get data path
        data_path = self.get_data_path()
        logger.info(f"📂 Loading from: {data_path}")

        # Load based on file type
        if data_path.suffix == ".parquet":
            df = pl.read_parquet(data_path)
        elif data_path.suffix in [".csv", ".tsv"]:
            df = pl.read_csv(data_path)
        elif data_path.suffix in [".fits", ".fit"]:
            # Use astropy for FITS files
            table = Table.read(data_path)
            # Handle multidimensional columns by selecting first dimension or flattening
            processed_cols = {}
            for name in table.colnames:
                col_data = table[name]
                if len(col_data.shape) > 1:
                    # For multidimensional columns, take the first dimension or flatten
                    if col_data.shape[1] <= 5:  # If reasonable number of dimensions
                        # Take first dimension or create separate columns
                        if col_data.shape[1] == 1:
                            processed_cols[name] = col_data[:, 0]
                        else:
                            # Create separate columns for each dimension
                            for i in range(col_data.shape[1]):
                                processed_cols[f"{name}_{i}"] = col_data[:, i]
                    else:
                        # Skip very large multidimensional columns
                        logger.warning(
                            f"Skipping large multidimensional column: {name} with shape {col_data.shape}"
                        )
                else:
                    processed_cols[name] = col_data

            # Create new table with processed columns
            from astropy.table import Table as AstroTable

            processed_table = AstroTable(processed_cols)

            # Convert to pandas first, then to polars
            pdf = processed_table.to_pandas()
            df = pl.from_pandas(pdf)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")

        logger.info(f"📊 Loaded {len(df)} objects with {len(df.columns)} columns")

        # First, calculate 3D coordinates for all data (even "bad" data)
        df = self._add_3d_coordinates(df)

        # Then apply quality filters
        df = self.apply_quality_filters(df)

        return df

    def _add_3d_coordinates(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add 3D coordinates to dataframe before filtering."""
        try:
            # Calculate 3D coordinates
            coords_3d = self.extract_coordinates(df)

            # Add 3D coordinates as new columns
            df = df.with_columns(
                [
                    pl.lit(coords_3d[:, 0].numpy()).alias("x_3d"),
                    pl.lit(coords_3d[:, 1].numpy()).alias("y_3d"),
                    pl.lit(coords_3d[:, 2].numpy()).alias("z_3d"),
                ]
            )

            logger.info(f"✅ Added 3D coordinates for {len(df)} objects")
            return df

        except Exception as e:
            logger.warning(f"Failed to calculate 3D coordinates: {e}")
            # Return original dataframe if 3D calculation fails
            return df

    def apply_quality_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply survey-generic quality filters."""
        initial_count = len(df)

        # 1. Remove null coordinates (essential for spatial analysis)
        coord_cols = self.get_coordinate_columns()
        for col in coord_cols[:2]:  # At least RA/Dec must be valid
            if col in df.columns:
                df = df.filter(pl.col(col).is_not_null() & pl.col(col).is_finite())

        # 2. Remove infinite values in numeric columns (but be more lenient)
        numeric_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Float32, pl.Float64]
            and col not in ["x_3d", "y_3d", "z_3d"]  # Don't filter 3D coords yet
        ]
        for col in numeric_cols:
            if col in df.columns:
                df = df.filter(pl.col(col).is_finite())

        # 3. Remove duplicate sources (if source_id available)
        id_cols = ["source_id", "object_id", "id", "designation"]
        for id_col in id_cols:
            if id_col in df.columns:
                df = df.unique(subset=[id_col])
                break

        # 4. Survey-specific filters (override in subclasses)
        df = self.apply_survey_specific_filters(df)

        # 5. Filter based on 3D coordinates if available
        if all(col in df.columns for col in ["x_3d", "y_3d", "z_3d"]):
            # Remove points with invalid 3D coordinates (NaN, Inf)
            df = df.filter(
                pl.col("x_3d").is_finite()
                & pl.col("y_3d").is_finite()
                & pl.col("z_3d").is_finite()
            )

            # Remove points too far away (e.g., parallax errors)
            distance = (
                pl.col("x_3d") ** 2 + pl.col("y_3d") ** 2 + pl.col("z_3d") ** 2
            ).sqrt()
            df = df.filter(distance < 100000)  # Max 100kpc

            logger.info(f"✅ Filtered to {len(df)} valid 3D coordinate entries")

        # 6. Validate coordinates if present (original 2D coordinates)
        if "ra" in df.columns and "dec" in df.columns:
            # Filter out invalid coordinates (but be more lenient)
            df = df.filter(
                (pl.col("ra").is_not_null())
                & (pl.col("dec").is_not_null())
                & (pl.col("ra") >= 0)
                & (pl.col("ra") <= 360)
                & (pl.col("dec") >= -90)
                & (pl.col("dec") <= 90)
            )
            logger.info(f"✅ Filtered to {len(df)} valid coordinate entries")

        final_count = len(df)
        if final_count < initial_count:
            logger.info(
                f"🔍 Quality filters: {initial_count} → {final_count} objects "
                f"({final_count / initial_count * 100:.1f}% retained)"
            )

        return df

    def apply_survey_specific_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply survey-specific quality filters. Override in subclasses."""
        return df

    @abstractmethod
    def get_coordinate_columns(self) -> List[str]:
        """Return list of coordinate column names for this survey."""
        pass

    @abstractmethod
    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from survey data."""
        pass

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract general feature vector from dataframe."""
        feature_cols = self.config.get("feature_cols", [])

        if not feature_cols:
            # Auto-detect feature columns
            coord_cols = set(self.get_coordinate_columns())
            mag_cols = set(self.get_magnitude_columns())

            feature_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
                and col not in coord_cols
                and col not in mag_cols
                and not col.endswith("_error")  # Exclude error columns
                and not col.startswith("flag_")  # Exclude flag columns
            ]

        if not feature_cols:
            # Fallback: create minimal features from coordinates
            logger.warning(
                f"No feature columns found for {self.survey_name}, "
                f"using coordinate-based features"
            )
            coords = self.extract_coordinates(df)
            # Use distances and angles as features
            distances = torch.norm(coords, dim=1, keepdim=True)
            return distances

        # Extract and stack features
        features = []
        for col in feature_cols:
            if col in df.columns:
                values = df[col].to_numpy()

                # Handle NaN values
                values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

                features.append(torch.tensor(values, dtype=torch.float32))

        if not features:
            n_objects = len(df)
            return torch.zeros(n_objects, 1, dtype=torch.float32)

        return torch.stack(features, dim=1)

    def get_magnitude_columns(self) -> List[str]:
        """Get list of magnitude column names.

        Override in subclasses for survey-specific names.
        """
        return self.config.get("mag_cols", [])

    def extract_photometry(self, df: pl.DataFrame) -> Optional[PhotometricTensorDict]:
        """Extract photometric data with proper band handling."""
        mag_cols = self.get_magnitude_columns()
        error_cols = self.config.get("error_cols", [])

        if not mag_cols:
            return None

        # Extract magnitudes and errors
        mags = []
        bands = []
        errors = []

        for i, col in enumerate(mag_cols):
            if col not in df.columns:
                continue

            mag_values = df[col].to_numpy()

            # Quality filters for magnitudes
            valid_mask = (
                np.isfinite(mag_values)
                & (mag_values > 0)
                & (mag_values < 50)  # Reasonable magnitude range
            )

            if not np.any(valid_mask):
                continue

            mags.append(torch.tensor(mag_values, dtype=torch.float32))

            # Extract band name from column
            band_name = self.extract_band_name(col)
            bands.append(band_name)

            # Extract errors if available
            if i < len(error_cols) and error_cols[i] in df.columns:
                error_values = df[error_cols[i]].to_numpy()
                error_values = np.nan_to_num(error_values, nan=0.1)  # Default error
                errors.append(torch.tensor(error_values, dtype=torch.float32))
            else:
                # Estimate errors if not available (5% of magnitude)
                estimated_errors = np.abs(mag_values) * 0.05 + 0.01
                errors.append(torch.tensor(estimated_errors, dtype=torch.float32))

        if not mags:
            return None

        magnitudes = torch.stack(mags, dim=1)  # [N, B]
        magnitude_errors = torch.stack(errors, dim=1) if errors else None

        return PhotometricTensorDict(
            magnitudes=magnitudes,
            bands=bands,
            magnitude_errors=magnitude_errors,
            filter_system=self.filter_system,
        )

    def extract_band_name(self, column_name: str) -> str:
        """Extract band name from column name with survey-specific patterns."""
        # Standard magnitude mappings
        magnitude_mappings = {
            "phot_g_mean_mag": "G",
            "phot_bp_mean_mag": "BP",
            "phot_rp_mean_mag": "RP",
            "bp_mag": "BP",
            "rp_mag": "RP",
            "modelMag_g": "g",
            "modelMag_r": "r",
            "modelMag_i": "i",
            "modelMag_u": "u",
            "modelMag_z": "z",
        }

        # Direct mapping
        if column_name in magnitude_mappings:
            return magnitude_mappings[column_name]

        # Pattern matching
        col_lower = column_name.lower()
        for pattern, band in magnitude_mappings.items():
            if pattern.lower() in col_lower:
                return band

        # Extract single letter bands
        match = re.search(r"([ugrizyUBVRIJHKGYW]\d?)", column_name)
        if match:
            return match.group(1)

        # Fallback: use column name
        return column_name.replace("_mag", "").replace("mag_", "")

    def extract_spectral(self, df: pl.DataFrame) -> Optional[SpectralTensorDict]:
        """Extract spectral data if available. Override in subclasses with spectroscopy."""
        return None

    def extract_lightcurves(self, df: pl.DataFrame) -> Optional[LightcurveTensorDict]:
        """Extract lightcurve data if available. Override in subclasses with time-series."""
        return None

    def create_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Create SpatialTensorDict with proper coordinate handling."""
        coordinates = self.extract_coordinates(df)

        # Create SkyCoord if we have RA/Dec
        coord_cols = self.get_coordinate_columns()

        if len(coord_cols) >= 2:
            ra_col, dec_col = coord_cols[0], coord_cols[1]
            if ra_col in df.columns and dec_col in df.columns:
                ra = df[ra_col].to_numpy()
                dec = df[dec_col].to_numpy()

                # Handle distance if available
                if len(coord_cols) >= 3 and coord_cols[2] in df.columns:
                    dist = df[coord_cols[2]].to_numpy()
                    # Handle parallax conversion if needed
                    if coord_cols[2] in ["parallax", "plx"]:
                        # Convert parallax to distance (handle negative/small values)
                        dist = np.where(dist > 0.001, 1000.0 / dist, 10000.0)  # pc

                    SkyCoord(
                        ra=ra * u.Unit("deg"),
                        dec=dec * u.Unit("deg"),
                        distance=dist * u.Unit(self.distance_unit),
                        frame=self.coordinate_system,
                    )
                else:
                    # Create SkyCoord for coordinate transformations
                    SkyCoord(
                        ra=ra * u.Unit("deg"),
                        dec=dec * u.Unit("deg"),
                        frame=self.coordinate_system,
                    )

        return SpatialTensorDict(
            coordinates=coordinates,
            coordinate_system=self.coordinate_system,
            unit=u.Unit(self.distance_unit),
            epoch=self.epoch,
        )

    def create_tensordict(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Create comprehensive TensorDict from dataframe."""
        tensors = {}

        # Spatial data (always required)
        tensors["spatial"] = self.create_spatial_tensor(df)

        # Features
        features = self.extract_features(df)
        if features.numel() > 0:
            tensors["features"] = features

        # Photometric data
        photometry = self.extract_photometry(df)
        if photometry is not None:
            tensors["photometric"] = photometry

        # Spectral data
        spectral = self.extract_spectral(df)
        if spectral is not None:
            tensors["spectral"] = spectral

        # Lightcurve data
        lightcurves = self.extract_lightcurves(df)
        if lightcurves is not None:
            tensors["lightcurves"] = lightcurves

        # Add survey statistics
        stats = self.get_survey_statistics(df)

        # Metadata
        tensors["meta"] = {
            "survey_name": self.survey_name,
            "n_objects": len(df),
            "data_release": self.config.get("data_release", "unknown"),
            "coordinate_system": self.coordinate_system,
            "epoch": self.epoch,
            "filter_system": self.filter_system,
            "processing_date": "2024-01-01T00:00:00",  # Use simple string instead of Time.now().iso
            "statistics": stats,
            "feature_names": getattr(self, "feature_names", None),
        }

        return tensors

    def preprocess_and_save(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        force_reprocess: bool = False,
    ) -> Path:
        """Preprocess survey data and save to processed directory."""
        # Use default paths if not provided
        if input_path is None:
            input_path = self.get_data_path()

        # Ensure output_path is always a Path object
        if output_path is None:
            output_path_path = self.processed_dir / f"{self.survey_name}.parquet"
        else:
            output_path_path = Path(output_path)

        # Ensure output directory exists
        output_path_path.parent.mkdir(parents=True, exist_ok=True)

        # Load and preprocess data
        df = self.load_data()
        processed_df = self.preprocess_dataframe(df)

        # Save processed data
        processed_df.write_parquet(str(output_path_path))
        logger.info(f"✅ Saved processed {self.survey_name} data to {output_path_path}")

        return output_path_path

    def preprocess(self, force: bool = False) -> Dict[str, Any]:
        """
        Unified preprocessing method - main interface for all surveys.

        This is the primary method that should be used by all clients.
        It provides a standardized interface for loading, processing, and
        returning survey data as tensors.

        Args:
            force: Force reprocessing even if cached file exists

        Returns:
            Dictionary with standardized format:
            {
                "spatial_tensor": SpatialTensorDict,
                "photometric_tensor": PhotometricTensorDict (optional),
                "metadata": Dict with survey info and statistics
            }
        """
        return self.preprocess_unified(force=force)

    def preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply survey-specific preprocessing to a DataFrame.
        This is the old preprocess(df) method, now renamed for clarity.
        Override in subclasses for survey-specific processing.
        """
        return df

    def preprocess_unified(self, force: bool = False) -> Dict[str, Any]:
        """
        Unified preprocessing method that all surveys must implement.

        This method provides a standardized interface for all survey preprocessors.
        It handles loading, filtering, and tensor creation in a consistent way.

        Args:
            force: Force reprocessing even if cached file exists

        Returns:
            Dictionary with standardized format:
            {
                "spatial_tensor": SpatialTensorDict,
                "photometric_tensor": PhotometricTensorDict (optional),
                "metadata": Dict with survey info and statistics
            }
        """
        logger.info(f"Starting {self.survey_name} preprocessing...")

        # Check if processed file exists and force is False
        processed_file = self.processed_dir / f"{self.survey_name}.parquet"
        if processed_file.exists() and not force:
            logger.info(f"Using existing processed file: {processed_file}")
            return self._load_processed_data(processed_file)

        # Load and process data
        df = self.load_data()
        if len(df) == 0:
            raise ValueError(f"No valid data found for {self.survey_name}")

        # Apply survey-specific preprocessing
        df = self.preprocess_dataframe(df)

        logger.info(f"Processing {len(df):,} sources...")

        # Create tensors
        spatial_tensor = self.create_spatial_tensor(df)
        photometric_tensor = self.extract_photometry(df)

        # Save processed data
        self._save_processed_data(df, processed_file)

        # Create metadata
        metadata = {
            "survey": self.survey_name,
            "n_sources": len(df),
            "processed_file": str(processed_file),
            "coordinate_system": self.coordinate_system,
            "epoch": self.epoch,
            "filter_system": self.filter_system,
            "statistics": self.get_survey_statistics(df),
        }

        result = {
            "spatial_tensor": spatial_tensor,
            "metadata": metadata,
        }

        if photometric_tensor is not None:
            result["photometric_tensor"] = photometric_tensor

        return result

    def _save_processed_data(self, df: pl.DataFrame, output_file: Path):
        """Save processed data to Parquet file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Select key columns for processed file
        key_columns = self._get_key_columns(df)
        processed_df = df.select(key_columns)

        processed_df.write_parquet(output_file)
        logger.info(f"Saved processed data: {output_file}")

    def _get_key_columns(self, df: pl.DataFrame) -> List[str]:
        """Get key columns to save in processed file. Override in subclasses."""
        # Default: save all columns
        return df.columns

    def _load_processed_data(self, processed_file: Path) -> Dict[str, Any]:
        """Load existing processed data and recreate tensors."""
        df = pl.read_parquet(processed_file)

        # Recreate tensors
        spatial_tensor = self.create_spatial_tensor(df)
        photometric_tensor = self.extract_photometry(df)

        result = {
            "spatial_tensor": spatial_tensor,
            "metadata": {
                "survey": self.survey_name,
                "n_sources": len(df),
                "processed_file": str(processed_file),
                "coordinate_system": self.coordinate_system,
                "epoch": self.epoch,
                "filter_system": self.filter_system,
                "statistics": self.get_survey_statistics(df),
            },
        }

        if photometric_tensor is not None:
            result["photometric_tensor"] = photometric_tensor

        return result

    def get_survey_statistics(self, df: pl.DataFrame) -> Dict[str, Any]:
        """Compute survey statistics for data quality assessment."""
        stats = {
            "n_objects": len(df),
            "n_columns": len(df.columns),
            "memory_usage_mb": df.estimated_size() / (1024 * 1024),
            "coordinate_coverage": {},
            "magnitude_statistics": {},
            "data_quality": {},
        }

        # Coordinate coverage
        coord_cols = self.get_coordinate_columns()
        for col in coord_cols:
            if col in df.columns:
                col_data = df[col].drop_nulls()
                if len(col_data) > 0:
                    stats["coordinate_coverage"][col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "mean": float(col_data.mean()),
                        "std": float(col_data.std()),
                        "completeness": len(col_data) / len(df),
                    }

        # Magnitude statistics
        mag_cols = self.get_magnitude_columns()
        for col in mag_cols:
            if col in df.columns:
                col_data = df[col].drop_nulls()
                if len(col_data) > 0:
                    stats["magnitude_statistics"][col] = {
                        "min": float(col_data.min()),
                        "max": float(col_data.max()),
                        "median": float(col_data.median()),
                        "percentile_95": float(col_data.quantile(0.95)),
                        "completeness": len(col_data) / len(df),
                    }

        # Data quality metrics
        null_count = sum(df.null_count())
        total_cells = len(df) * len(df.columns)
        stats["data_quality"]["completeness"] = 1.0 - (null_count / total_cells)

        # Feature richness
        feature_cols = [
            col
            for col in df.columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        stats["data_quality"]["feature_richness"] = len(feature_cols) / len(df.columns)

        return stats

    def validate_data(self, df: pl.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data quality and return validation status and issues."""
        issues = []

        # Check minimum requirements
        if len(df) == 0:
            issues.append("Empty dataframe")
            return False, issues

        # Check coordinate columns
        coord_cols = self.get_coordinate_columns()
        for col in coord_cols[:2]:  # At least RA/Dec required
            if col not in df.columns:
                issues.append(f"Missing required coordinate column: {col}")
            elif df[col].null_count() > len(df) * 0.5:
                issues.append(
                    f"Too many null values in {col}: {df[col].null_count()}/{len(df)}"
                )

        # Check for reasonable coordinate ranges
        if "ra" in df.columns:
            ra_data = df["ra"].drop_nulls()
            if len(ra_data) > 0:
                ra_min = ra_data.min()
                ra_max = ra_data.max()
                if (isinstance(ra_min, (int, float)) and ra_min < 0) or (
                    isinstance(ra_max, (int, float)) and ra_max > 360
                ):
                    issues.append("RA values outside valid range [0, 360]")

        if "dec" in df.columns:
            dec_data = df["dec"].drop_nulls()
            if len(dec_data) > 0:
                dec_min = dec_data.min()
                dec_max = dec_data.max()
                if (isinstance(dec_min, (int, float)) and dec_min < -90) or (
                    isinstance(dec_max, (int, float)) and dec_max > 90
                ):
                    issues.append("Dec values outside valid range [-90, 90]")

        # Check magnitude reasonableness
        mag_cols = self.get_magnitude_columns()
        for col in mag_cols:
            if col in df.columns:
                mag_data = df[col].drop_nulls()
                if len(mag_data) > 0:
                    mag_min = mag_data.min()
                    mag_max = mag_data.max()
                    if (isinstance(mag_min, (int, float)) and mag_min < -5) or (
                        isinstance(mag_max, (int, float)) and mag_max > 50
                    ):
                        issues.append(
                            f"Magnitude {col} outside reasonable range [-5, 50]"
                        )

        return len(issues) == 0, issues
