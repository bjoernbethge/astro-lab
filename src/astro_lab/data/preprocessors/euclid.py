"""
Euclid Survey Processor
======================

Handles Euclid mission data loading and preprocessing.
"""

import logging

import numpy as np
import polars as pl
import torch

from .astro import BaseSurveyProcessor

logger = logging.getLogger(__name__)


class EuclidPreprocessor(BaseSurveyProcessor):
    """Processor for Euclid mission data."""

    def __init__(self):
        super().__init__("euclid")

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract 3D coordinates from Euclid data."""
        # Euclid has RA/Dec - convert to 3D with estimated distance
        if all(col in df.columns for col in ["ra", "dec"]):
            return self._ra_dec_to_xyz_estimated(df)

        raise ValueError("No valid coordinate columns found in Euclid data")

    def _ra_dec_to_xyz_estimated(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert RA/Dec to 3D coordinates with estimated distance."""
        ra = np.deg2rad(df["ra"].to_numpy())
        dec = np.deg2rad(df["dec"].to_numpy())

        # Estimate distance from VIS magnitude if available
        if "vis_mag" in df.columns:
            vis_mag = df["vis_mag"].to_numpy()
            # Assume absolute VIS magnitude of ~-20 for typical galaxies
            abs_mag = -20.0
            distance_mpc = 10 ** ((vis_mag - abs_mag + 5) / 5)
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
        """Extract feature vector from Euclid data."""
        feature_cols = []

        # Euclid photometry (VIS, Y, J, H bands)
        for band in ["vis", "y", "j", "h"]:
            # Try different column naming conventions
            for col_pattern in [f"{band}_mag", f"{band}mag", f"mag_{band}", band]:
                if col_pattern in df.columns:
                    feature_cols.append(col_pattern)
                    break

        # Color indices
        if "vis_mag" in df.columns and "y_mag" in df.columns:
            feature_cols.append("vis_y")
        if "y_mag" in df.columns and "j_mag" in df.columns:
            feature_cols.append("y_j")
        if "j_mag" in df.columns and "h_mag" in df.columns:
            feature_cols.append("j_h")

        # Quality flags
        for col in ["flags", "quality_flag", "detection_flag"]:
            if col in df.columns:
                feature_cols.append(col)

        # Magnitude errors
        for col in ["vis_mag_err", "y_mag_err", "j_mag_err", "h_mag_err"]:
            if col in df.columns:
                feature_cols.append(col)

        # Morphology
        for col in ["vis_kron_rad", "y_kron_rad", "j_kron_rad", "h_kron_rad"]:
            if col in df.columns:
                feature_cols.append(col)

        # Redshift (if available)
        for col in ["z", "redshift", "spec_z", "phot_z"]:
            if col in df.columns:
                feature_cols.append(col)
                break

        # Shear measurements (for weak lensing)
        for col in ["g1", "g2", "e1", "e2"]:
            if col in df.columns:
                feature_cols.append(col)

        if not feature_cols:
            return torch.zeros(len(df), 1, dtype=torch.float32)

        features = df.select(feature_cols).to_numpy()
        features = np.nan_to_num(features, nan=0.0)
        return torch.tensor(features, dtype=torch.float32)

    def preprocess(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Euclid-specific preprocessing. Expects a DataFrame as input."""
        # Calculate colors if we have magnitudes
        mag_bands = []
        for band in ["vis", "y", "j", "h"]:
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
        if "flags" in df.columns:
            # Keep only high-quality detections (flags = 0)
            df = df.filter(pl.col("flags") == 0)

        # Handle missing values
        for band in ["vis", "y", "j", "h"]:
            mag_col = f"{band}_mag"
            if mag_col in df.columns:
                # Replace extreme values with NaN
                df = df.with_columns(
                    pl.when(pl.col(mag_col) < -999)
                    .then(None)
                    .otherwise(pl.col(mag_col))
                    .alias(mag_col)
                )

        # Add signal-to-noise ratios
        for band in ["vis", "y", "j", "h"]:
            mag_col = f"{band}_mag"
            err_col = f"{band}_mag_err"
            if mag_col in df.columns and err_col in df.columns:
                df = df.with_columns(
                    (pl.col(mag_col) / pl.col(err_col)).alias(f"{band}_snr")
                )

        # Add shear magnitude
        if "g1" in df.columns and "g2" in df.columns:
            df = df.with_columns(
                (pl.col("g1") ** 2 + pl.col("g2") ** 2).sqrt().alias("g_mag")
            )

        # Add ellipticity magnitude
        if "e1" in df.columns and "e2" in df.columns:
            df = df.with_columns(
                (pl.col("e1") ** 2 + pl.col("e2") ** 2).sqrt().alias("e_mag")
            )

        return df

    def load_raw(self, max_samples=None):
        return super().load_raw(max_samples=max_samples)

    def filter(self, df):
        return self.apply_quality_filters(df)

    def create_graph(self, df):
        return super().create_graph(df)

    def load_processed(self):
        return super().load_processed()

    def load_graph(self):
        raise NotImplementedError(
            "load_graph must be implemented for Euclid if needed."
        )

    def apply_quality_filters(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply only central harmonization and filter for valid coordinates."""
        df = self.harmonize_survey_columns(df)
        initial_count = len(df)
        if not ("ra" in df.columns and "dec" in df.columns):
            raise ValueError(
                "Data must contain 'ra' and 'dec' columns after harmonization!"
            )
        df = df.filter(df["ra"].is_not_null() & df["dec"].is_not_null())
        logger.info(
            f"Euclid quality filters: {initial_count:,}  {len(df):,} ({len(df) / initial_count * 100:.1f}% retained)"
        )
        return df
