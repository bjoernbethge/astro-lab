"""
Pan-STARRS Survey Processor
==========================

Handles Pan-STARRS data loading and preprocessing.
"""

import logging

import numpy as np
import polars as pl
import torch

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class PanSTARRSPreprocessor(BaseSurveyProcessor):
    """Processor for Pan-STARRS survey data."""

    def __init__(self):
        super().__init__("panstarrs")

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from Pan-STARRS data."""
        # Pan-STARRS has RA/Dec - convert to 3D with estimated distance
        if all(col in df.columns for col in ["ra", "dec"]):
            return self._ra_dec_to_xyz_estimated(df)

        raise ValueError("No valid coordinate columns found in Pan-STARRS data")

    def _ra_dec_to_xyz_estimated(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert RA/Dec to 3D coordinates with estimated distance."""
        ra = np.deg2rad(df["ra"].to_numpy())
        dec = np.deg2rad(df["dec"].to_numpy())

        # Estimate distance from r magnitude if available
        if "r_mag" in df.columns:
            r_mag = df["r_mag"].to_numpy()
            # Assume absolute r magnitude of ~-20 for typical galaxies
            abs_mag = -20.0
            distance_mpc = 10 ** ((r_mag - abs_mag + 5) / 5)
            distance_pc = distance_mpc * 1e6
        else:
            # Default to 100 Mpc if no magnitude available
            distance_pc = np.full(len(df), 100.0 * 1e6)

        # Convert to Cartesian
        x = distance_pc * np.cos(dec) * np.cos(ra)
        y = distance_pc * np.cos(dec) * np.sin(ra)
        z = distance_pc * np.sin(dec)

        coords = np.stack([x, y, z], axis=1)
        return torch.tensor(coords, dtype=torch.float32)

    def extract_features(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract feature vector from Pan-STARRS data."""
        feature_cols = []

        # Pan-STARRS photometry (g, r, i, z, y bands)
        for band in ["g", "r", "i", "z", "y"]:
            # Try different column naming conventions
            for col_pattern in [f"{band}_mag", f"{band}mag", f"mag_{band}", band]:
                if col_pattern in df.columns:
                    feature_cols.append(col_pattern)
                    break

        # Color indices
        if "g_mag" in df.columns and "r_mag" in df.columns:
            feature_cols.append("g_r")
        if "r_mag" in df.columns and "i_mag" in df.columns:
            feature_cols.append("r_i")
        if "i_mag" in df.columns and "z_mag" in df.columns:
            feature_cols.append("i_z")
        if "z_mag" in df.columns and "y_mag" in df.columns:
            feature_cols.append("z_y")

        # Quality flags
        for col in ["quality_flag", "obj_info_flag", "detect_info_flag"]:
            if col in df.columns:
                feature_cols.append(col)

        # Extinction
        for col in ["g_ext", "r_ext", "i_ext", "z_ext", "y_ext"]:
            if col in df.columns:
                feature_cols.append(col)

        # Variability
        for col in ["g_var", "r_var", "i_var", "z_var", "y_var"]:
            if col in df.columns:
                feature_cols.append(col)

        # Morphology
        for col in ["g_kron_rad", "r_kron_rad", "i_kron_rad"]:
            if col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)
        return torch.tensor(features, dtype=torch.float32)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Pan-STARRS-specific preprocessing."""
        # Calculate colors if we have magnitudes
        mag_bands = []
        for band in ["g", "r", "i", "z", "y"]:
            for col in [f"{band}_mag", f"{band}mag", f"mag_{band}", band]:
                if col in df.columns:
                    mag_bands.append((band, col))
                    break

        # Calculate common colors
        if len(mag_bands) >= 2:
            for i in range(len(mag_bands) - 1):
                band1, col1 = mag_bands[i]
                band2, col2 = mag_bands[i + 1]
                color_name = f"{band1}_{band2}"

                if color_name not in df.columns:
                    df = df.with_columns(
                        (pl.col(col1) - pl.col(col2)).alias(color_name)
                    )

        # Quality filtering
        if "quality_flag" in df.columns:
            # Keep only high-quality detections
            df = df.filter(pl.col("quality_flag") == 0)

        # Handle missing values
        for band in ["g", "r", "i", "z", "y"]:
            mag_col = f"{band}_mag"
            if mag_col in df.columns:
                # Replace extreme values with NaN
                df = df.with_columns(
                    pl.when(pl.col(mag_col) < -999)
                    .then(None)
                    .otherwise(pl.col(mag_col))
                    .alias(mag_col)
                )

        # Add extinction-corrected magnitudes
        for band in ["g", "r", "i", "z", "y"]:
            mag_col = f"{band}_mag"
            ext_col = f"{band}_ext"
            if mag_col in df.columns and ext_col in df.columns:
                df = df.with_columns(
                    (pl.col(mag_col) - pl.col(ext_col)).alias(f"{band}_mag_dered")
                )

        return df
