"""
DES Survey Processor
===================

Handles Dark Energy Survey data loading and preprocessing.
"""

import logging
from typing import List

import numpy as np
import polars as pl
import torch

from .base import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class DESPreprocessor(BaseSurveyProcessor):
    """Processor for DES survey data."""

    def __init__(self, survey_name: str = "des"):
        super().__init__(survey_name)

    def get_coordinate_columns(self) -> List[str]:
        """Get DES coordinate column names."""
        return ["RA_ICRS", "DE_ICRS", "rmag"]  # Use r magnitude for distance estimation

    def get_magnitude_columns(self) -> List[str]:
        """Get DES magnitude column names."""
        return ["gmag", "rmag", "imag", "zmag", "Ymag"]

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from DES data."""
        # DES has RA_ICRS/DE_ICRS - convert to 3D with estimated distance
        if all(col in df.columns for col in ["RA_ICRS", "DE_ICRS"]):
            return self._ra_dec_to_xyz_estimated(df)

        raise ValueError("No valid coordinate columns found in DES data")

    def _ra_dec_to_xyz_estimated(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert RA/Dec to 3D coordinates with estimated distance."""
        ra = np.deg2rad(df["RA_ICRS"].to_numpy())
        dec = np.deg2rad(df["DE_ICRS"].to_numpy())

        # Estimate distance from r magnitude if available
        if "rmag" in df.columns:
            r_mag = df["rmag"].to_numpy()
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
        """Extract feature vector from DES data."""
        feature_cols = []

        # DES photometry (g, r, i, z, Y bands)
        for band in ["g", "r", "i", "z", "y"]:
            # Try different column naming conventions
            for col_pattern in [f"{band}mag", f"{band}_mag", f"mag_{band}", band]:
                if col_pattern in df.columns:
                    feature_cols.append(col_pattern)
                    break

        # Color indices
        if "gmag" in df.columns and "rmag" in df.columns:
            feature_cols.append("g_r")
        if "rmag" in df.columns and "imag" in df.columns:
            feature_cols.append("r_i")
        if "imag" in df.columns and "zmag" in df.columns:
            feature_cols.append("i_z")
        if "zmag" in df.columns and "Ymag" in df.columns:
            feature_cols.append("z_y")

        # Quality flags
        for col in ["gFlag", "rFlag", "iFlag", "zFlag", "yFlag"]:
            if col in df.columns:
                feature_cols.append(col)

        # Magnitude errors
        for col in ["e_gmag", "e_rmag", "e_imag", "e_zmag", "e_Ymag"]:
            if col in df.columns:
                feature_cols.append(col)

        # Morphology
        for col in ["Aimg", "Bimg", "PA"]:
            if col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)
        return torch.tensor(features, dtype=torch.float32)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply DES-specific preprocessing."""
        # Calculate colors if we have magnitudes
        mag_bands = []
        for band in ["g", "r", "i", "z", "y"]:
            for col in [f"{band}mag", f"{band}_mag", f"mag_{band}", band]:
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
        for flag_col in ["gFlag", "rFlag", "iFlag", "zFlag", "yFlag"]:
            if flag_col in df.columns:
                # Keep only high-quality detections (flags = 0)
                df = df.filter(pl.col(flag_col) == 0)

        # Handle missing values
        for band in ["g", "r", "i", "z", "y"]:
            mag_col = f"{band}mag"
            if mag_col in df.columns:
                # Replace extreme values with NaN
                df = df.with_columns(
                    pl.when(pl.col(mag_col) < -999)
                    .then(None)
                    .otherwise(pl.col(mag_col))
                    .alias(mag_col)
                )

        # Add signal-to-noise ratios
        for band in ["g", "r", "i", "z", "y"]:
            mag_col = f"{band}mag"
            err_col = f"e_{band}mag"
            if mag_col in df.columns and err_col in df.columns:
                df = df.with_columns(
                    (pl.col(mag_col) / pl.col(err_col)).alias(f"{band}_snr")
                )

        return df
