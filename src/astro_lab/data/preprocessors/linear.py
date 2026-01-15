"""LINEAR (Lincoln Near-Earth Asteroid Research) preprocessor implementation.

Handles LINEAR asteroid survey data preprocessing.
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


class LINEARPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for LINEAR asteroid survey data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LINEAR preprocessor."""
        default_config = {
            "magnitude_limit": 18.0,  # Magnitude limit
            "period_limit": [0.01, 100.0],  # Period range in days
            "amplitude_limit": 0.05,  # Minimum amplitude for variability
            "require_period": False,  # Period not always available
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.required_columns = ["id"]  # LINEAR object ID

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "linear"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "asteroid"

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply LINEAR-specific quality filters."""
        initial_count = len(df)

        # 1. Magnitude filter - from SURVEY_CONFIGS, LINEAR has 'r' mag
        mag_cols = ["r", "magnitude", "mag", "V", "v_mag", "mean_mag", "<mL>"]
        mag_col = None
        for col in mag_cols:
            if col in df.columns:
                mag_col = col
                break

        if mag_col:
            df = df.filter(
                pl.col(mag_col).is_not_null()
                & pl.col(mag_col).is_finite()
                & (pl.col(mag_col) > 10)
                & (pl.col(mag_col) < self.config["magnitude_limit"])
            )

        # 2. Period filter (if available and required) - LP1 in SURVEY_CONFIGS
        if self.config["require_period"]:
            period_cols = ["LP1", "period", "period_days", "P", "rotation_period"]
            period_col = None
            for col in period_cols:
                if col in df.columns:
                    period_col = col
                    break

            if period_col:
                min_period, max_period = self.config["period_limit"]
                df = df.filter(
                    (pl.col(period_col) >= min_period)
                    & (pl.col(period_col) <= max_period)
                )

        # 3. Amplitude filter (for variability detection) - from SURVEY_CONFIGS: std, rms
        amp_cols = [
            "std",
            "rms",
            "amplitude",
            "amp",
            "delta_mag",
            "variability_amplitude",
        ]
        amp_col = None
        for col in amp_cols:
            if col in df.columns:
                amp_col = col
                break

        if amp_col:
            df = df.filter(
                pl.col(amp_col).is_null()
                | (pl.col(amp_col) >= self.config["amplitude_limit"])
            )

        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} LINEAR objects ({((initial_count - final_count) / initial_count * 100):.1f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform LINEAR data."""
        # Standardize column names
        df = self._standardize_column_names(df)

        # Add asteroid classification features
        df = self._add_asteroid_features(df)

        # Convert to cartesian coordinates if available - from SURVEY_CONFIGS: raLIN, decLIN
        ra_cols = ["raLIN", "ra", "RA", "ra_deg"]
        dec_cols = ["decLIN", "dec", "DEC", "dec_deg"]

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
            # Distance estimation for asteroids (typically 1-5 AU from Sun)
            # Use magnitude for rough distance estimation
            mag_cols = ["r", "<mL>", "magnitude", "mag", "V", "v_mag", "mean_mag"]
            mag_col = None
            for col in mag_cols:
                if col in df.columns:
                    mag_col = col
                    break

            if mag_col:
                # Simple distance estimation for asteroids
                # Brighter = closer, fainter = farther
                df = df.with_columns(
                    [
                        (2.0 + (pl.col(mag_col) - 15.0) * 0.2)
                        .clip(1.0, 5.0)
                        .alias("distance_au")
                    ]
                )

                # Convert AU to parsecs for consistency
                df = df.with_columns(
                    [
                        (pl.col("distance_au") * 206265).alias(
                            "distance_pc"
                        )  # 1 AU = 206265 pc
                    ]
                )
            else:
                # Default distance (2.5 AU average)
                df = df.with_columns([pl.lit(2.5 * 206265).alias("distance_pc")])

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
        """Standardize LINEAR column names based on SURVEY_CONFIGS."""
        column_mapping = {
            # Coordinates - from SURVEY_CONFIGS
            "raLIN": "ra",
            "decLIN": "dec",
            "RA": "ra",
            "DEC": "dec",
            "ra_deg": "ra",
            "dec_deg": "dec",
            # Magnitudes - from SURVEY_CONFIGS
            "r": "magnitude",
            "<mL>": "mean_magnitude",
            "V": "magnitude",
            "v_mag": "magnitude",
            "mean_mag": "magnitude",
            "mag": "magnitude",
            # Period - from SURVEY_CONFIGS
            "LP1": "period",
            "P": "period",
            "period_days": "period",
            "rotation_period": "period",
            # Amplitude/variability - from SURVEY_CONFIGS
            "std": "amplitude",
            "rms": "rms_variation",
            "amp": "amplitude",
            "delta_mag": "amplitude",
            "variability_amplitude": "amplitude",
            # Colors - from SURVEY_CONFIGS
            "ug": "u_g",
            "gr": "g_r",
            "ri": "r_i",
            "iz": "i_z",
            "JK": "j_k",
            # Other LINEAR-specific - from SURVEY_CONFIGS
            "Lchi2": "chi2",
            "phi1": "phase",
            "S": "stetson_index",
            "prior": "prior_info",
            # ID
            "linear_id": "id",
            "LINEAR_id": "id",
            "object_id": "id",
            "name": "id",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_asteroid_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add asteroid-specific features."""
        # Period-based classification
        if "period" in df.columns:
            df = df.with_columns(
                [
                    # Fast rotators (< 2.2 hours, rubble pile limit)
                    (pl.col("period") < 2.2 / 24.0).alias("fast_rotator"),
                    # Slow rotators (> 12 hours)
                    (pl.col("period") > 0.5).alias("slow_rotator"),
                    # Normal rotators
                    (
                        (pl.col("period") >= 2.2 / 24.0) & (pl.col("period") <= 0.5)
                    ).alias("normal_rotator"),
                ]
            )

        # Amplitude-based features
        if "amplitude" in df.columns:
            df = df.with_columns(
                [
                    # High amplitude variability (elongated shape)
                    (pl.col("amplitude") > 0.5).alias("high_amplitude"),
                    # Low amplitude (spherical shape)
                    (pl.col("amplitude") < 0.1).alias("low_amplitude"),
                    # Shape indicator (log amplitude)
                    pl.col("amplitude").log().alias("log_amplitude"),
                ]
            )

        # Color-based classification using LINEAR colors from SURVEY_CONFIGS
        color_pairs = [("u_g", 0.3, 0.9), ("g_r", 0.2, 0.8), ("r_i", 0.1, 0.6)]
        for color_col, min_val, max_val in color_pairs:
            if color_col in df.columns:
                df = df.with_columns(
                    [
                        # S-type asteroids (bright, moderate color)
                        (
                            (pl.col(color_col) > min_val)
                            & (pl.col(color_col) < max_val)
                        ).alias(f"s_type_{color_col}"),
                        # C-type asteroids (dark, neutral color)
                        (pl.col(color_col) < min_val).alias(f"c_type_{color_col}"),
                    ]
                )
                break  # Use first available color

        # Lightcurve quality indicators - from SURVEY_CONFIGS
        if "chi2" in df.columns:
            df = df.with_columns(
                [
                    # Good fit quality
                    (pl.col("chi2") < 1.5).alias("good_fit"),
                    # Log chi2 for feature
                    pl.col("chi2").log().alias("log_chi2"),
                ]
            )

        # Stetson index for variability
        if "stetson_index" in df.columns:
            df = df.with_columns(
                [
                    # Variable objects
                    (pl.col("stetson_index") > 0.5).alias("variable_object"),
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract LINEAR asteroid features."""
        feature_columns = []

        # ID column
        id_cols = ["id", "linear_id", "LINEAR_id", "object_id", "name"]
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

        # Basic properties - using LINEAR-specific columns
        basic_cols = [
            "magnitude",
            "mean_magnitude",
            "period",
            "amplitude",
            "rms_variation",
        ]
        for col in basic_cols:
            if col in df.columns:
                feature_columns.append(col)

        # LINEAR-specific features from SURVEY_CONFIGS
        linear_cols = ["chi2", "phase", "stetson_index", "prior_info"]
        for col in linear_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Classification features
        class_cols = [
            "fast_rotator",
            "slow_rotator",
            "normal_rotator",
            "high_amplitude",
            "low_amplitude",
            "s_type_u_g",
            "c_type_u_g",
            "s_type_g_r",
            "c_type_g_r",
            "good_fit",
            "variable_object",
        ]
        for col in class_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Derived features
        derived_cols = ["log_amplitude", "log_chi2"]
        for col in derived_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Color information from SURVEY_CONFIGS
        color_cols = ["u_g", "g_r", "r_i", "i_z", "j_k"]
        for col in color_cols:
            if col in df.columns:
                feature_columns.append(col)

        # If no standard features, use all numeric columns
        if not feature_columns:
            logger.warning("No standard LINEAR features found, using numeric columns")
            numeric_cols = [
                col
                for col in df.columns
                if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
            feature_columns = numeric_cols[:10]  # Limit to first 10

        # Keep available features
        available_features = [col for col in feature_columns if col in df.columns]

        # Normalize features
        if available_features:
            # Separate numeric and boolean features
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
            f"Extracted {len(available_features)} LINEAR features: {available_features}"
        )

        return df
