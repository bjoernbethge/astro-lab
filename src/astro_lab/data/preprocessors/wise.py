"""WISE (Wide-field Infrared Survey Explorer) preprocessor implementation.

Handles WISE infrared survey data preprocessing.
"""

import logging
from typing import Any, Dict, Optional

import polars as pl

from astro_lab.data.transforms.astronomical import spherical_to_cartesian

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class WISEPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for WISE infrared survey data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize WISE preprocessor."""
        default_config = {
            "min_snr": 2.0,  # Minimum signal-to-noise ratio
            "max_chi2": 10.0,  # Maximum chi-squared for PSF fitting
            "agn_w1_w2_min": 0.8,  # AGN color criterion
            "require_w1_w2": True,  # Require W1 and W2 detections
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.required_columns = ["designation"]  # WISE designation is the ID
        self.wise_bands = ["W1", "W2", "W3", "W4"]

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "wise"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "infrared_source"  # WISE detects various IR sources

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply WISE-specific quality filters."""
        initial_count = len(df)

        # 1. Require W1 and W2 detections (most reliable bands)
        if self.config["require_w1_w2"]:
            w1_cols = ["w1mpro", "w1mag", "W1", "w1"]
            w2_cols = ["w2mpro", "w2mag", "W2", "w2"]

            w1_col = None
            w2_col = None

            for col in w1_cols:
                if col in df.columns:
                    w1_col = col
                    break

            for col in w2_cols:
                if col in df.columns:
                    w2_col = col
                    break

            if w1_col and w2_col:
                df = df.filter(
                    pl.col(w1_col).is_not_null()
                    & pl.col(w1_col).is_finite()
                    & pl.col(w2_col).is_not_null()
                    & pl.col(w2_col).is_finite()
                )

        # 2. Signal-to-noise ratio thresholds
        snr_cols = ["w1snr", "w2snr", "w1_snr", "w2_snr"]
        for col in snr_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_null() | (pl.col(col) >= self.config["min_snr"])
                )

        # 3. Chi-squared thresholds for PSF fitting
        chi2_cols = ["w1rchi2", "w2rchi2", "w1_chi2", "w2_chi2"]
        for col in chi2_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_null() | (pl.col(col) <= self.config["max_chi2"])
                )

        # 4. Remove invalid magnitudes
        mag_cols = [
            "w1mpro",
            "w2mpro",
            "w3mpro",
            "w4mpro",
            "w1mag",
            "w2mag",
            "w3mag",
            "w4mag",
        ]
        for col in mag_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_null()
                    | (
                        (pl.col(col) > -10) & (pl.col(col) < 25)
                    )  # Reasonable magnitude range
                )

        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} WISE sources ({((initial_count - final_count) / initial_count * 100):.1f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform WISE data."""
        # Standardize column names
        df = self._standardize_column_names(df)

        # Add infrared colors
        df = self._add_infrared_colors(df)

        # Add classification flags
        df = self._add_classification_flags(df)

        # Convert to cartesian coordinates if RA/Dec available
        ra_cols = ["ra", "RA", "ra_deg"]
        dec_cols = ["dec", "DEC", "dec_deg"]

        ra_col = None
        dec_col = None

        for col in ra_cols:
            if col in df.columns:
                ra_col = col
                break

        for col in dec_cols:
            if col in df.columns:
                dec_col = col
                break

        if ra_col and dec_col:
            # Use simple distance estimation based on W1 magnitude
            w1_cols = ["w1mpro", "w1mag", "W1", "w1"]
            w1_col = None
            for col in w1_cols:
                if col in df.columns:
                    w1_col = col
                    break

            if w1_col:
                # Simple distance estimation
                df = df.with_columns(
                    [
                        (10 ** ((pl.col(w1_col) - 2.5 + 5) / 5))
                        .clip(100, 1e6)
                        .alias("distance_pc")
                    ]
                )
            else:
                # Default distance
                df = df.with_columns(
                    [
                        pl.lit(10000.0).alias("distance_pc")  # 10 kpc default
                    ]
                )

            x, y, z = spherical_to_cartesian(df[ra_col], df[dec_col], df["distance_pc"])
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )

        return df

    def _standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize WISE column names."""
        column_mapping = {
            # Magnitudes
            "w1mag": "w1mpro",
            "w2mag": "w2mpro",
            "w3mag": "w3mpro",
            "w4mag": "w4mpro",
            "W1": "w1mpro",
            "W2": "w2mpro",
            "W3": "w3mpro",
            "W4": "w4mpro",
            # Coordinates
            "RA": "ra",
            "DEC": "dec",
            # ID
            "wise_designation": "designation",
            "WISE_name": "designation",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_infrared_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add infrared color indices."""
        # Standard WISE colors
        color_definitions = [
            ("w1_w2", "w1mpro", "w2mpro"),
            ("w2_w3", "w2mpro", "w3mpro"),
            ("w3_w4", "w3mpro", "w4mpro"),
            ("w1_w3", "w1mpro", "w3mpro"),
            ("w1_w4", "w1mpro", "w4mpro"),
        ]

        for color_name, mag1, mag2 in color_definitions:
            if mag1 in df.columns and mag2 in df.columns:
                df = df.with_columns([(pl.col(mag1) - pl.col(mag2)).alias(color_name)])

        return df

    def _add_classification_flags(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add source classification flags based on WISE colors."""
        # AGN classification (Stern et al. 2012)
        if "w1_w2" in df.columns:
            df = df.with_columns(
                [
                    (pl.col("w1_w2") > self.config["agn_w1_w2_min"]).alias(
                        "agn_candidate"
                    ),
                    (pl.col("w1_w2") > 1.0).alias("qso_candidate"),
                ]
            )

        # YSO classification (Koenig et al. 2012)
        if "w1_w2" in df.columns and "w2_w3" in df.columns:
            df = df.with_columns(
                [
                    ((pl.col("w1_w2") > 0.25) & (pl.col("w2_w3") > 1.0)).alias(
                        "yso_candidate"
                    )
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract WISE infrared features."""
        feature_columns = []

        # ID column
        id_cols = ["designation", "wise_name", "id"]
        id_col = None
        for col in id_cols:
            if col in df.columns:
                id_col = col
                break

        # Position
        if all(col in df.columns for col in ["x", "y", "z"]):
            feature_columns.extend(["x", "y", "z"])
        elif all(col in df.columns for col in ["ra", "dec"]):
            feature_columns.extend(["ra", "dec"])

        # WISE magnitudes
        mag_cols = ["w1mpro", "w2mpro", "w3mpro", "w4mpro"]
        for col in mag_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Infrared colors
        color_cols = ["w1_w2", "w2_w3", "w3_w4", "w1_w3", "w1_w4"]
        for col in color_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Classification flags
        class_cols = ["agn_candidate", "qso_candidate", "yso_candidate"]
        for col in class_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Signal-to-noise ratios
        snr_cols = ["w1snr", "w2snr", "w3snr", "w4snr"]
        for col in snr_cols:
            if col in df.columns:
                feature_columns.append(col)

        # If no standard features, use available numeric columns
        if not feature_columns:
            logger.warning("No standard WISE features found, using numeric columns")
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
            feature_columns = numeric_cols[:15]  # Limit to first 15

        # Keep available features
        available_features = [col for col in feature_columns if col in df.columns]

        # Normalize features
        if available_features:
            # Remove boolean columns from normalization
            numeric_features = []
            boolean_features = []

            for col in available_features:
                if df[col].dtype == pl.Boolean:
                    boolean_features.append(col)
                else:
                    numeric_features.append(col)

            # Normalize numeric features
            if numeric_features:
                df = self.normalize_columns(df, numeric_features, method="standard")

                # Remove _norm suffix
                for col in numeric_features:
                    if f"{col}_norm" in df.columns:
                        df = df.with_columns([pl.col(f"{col}_norm").alias(col)])
                        df = df.drop(f"{col}_norm")

            # Convert boolean features to float
            for col in boolean_features:
                df = df.with_columns([pl.col(col).cast(pl.Float32).alias(col)])

        # Select final columns
        keep_columns = []
        if id_col:
            keep_columns.append(id_col)
        keep_columns.extend(available_features)

        df = df.select([col for col in keep_columns if col in df.columns])

        logger.info(
            f"Extracted {len(available_features)} WISE features: {available_features}"
        )

        return df
