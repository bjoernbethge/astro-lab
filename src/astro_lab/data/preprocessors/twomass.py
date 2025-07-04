"""TwoMASS (Two Micron All-Sky Survey) preprocessor implementation.

Handles TwoMASS near-infrared survey data preprocessing.
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


class TwoMASSPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for TwoMASS near-infrared survey data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TwoMASS preprocessor."""
        default_config = {
            "max_uncertainty": 0.2,  # Maximum photometric uncertainty (mag)
            "giant_jh_limit": 0.75,  # J-H > 0.75 for giants
            "dwarf_jks_limit": 0.95,  # J-Ks < 0.95 for dwarfs
            "require_jhk": True,  # Require all three bands
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.required_columns = ["designation"]  # TwoMASS designation
        self.twomass_bands = ["J", "H", "Ks"]

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "twomass"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "nir_source"  # Near-infrared sources

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply TwoMASS-specific quality filters."""
        initial_count = len(df)

        # 1. Require good photometry in all bands
        if self.config["require_jhk"]:
            # mag_cols = ['j_m', 'h_m', 'k_m']  # Removed unused variable
            # alt_mag_cols = ['j_mag', 'h_mag', 'k_mag', 'J', 'H', 'K', 'Ks']  # Removed unused variable

            # Find available magnitude columns
            available_mags = []
            for band in ["j", "h", "k"]:
                found = False
                for col in df.columns:
                    if col.lower().startswith(band) and (
                        "mag" in col.lower()
                        or col.lower() == band
                        or col.upper() == band.upper()
                    ):
                        available_mags.append(col)
                        found = True
                        break
                if not found:
                    logger.warning(f"No {band.upper()} magnitude column found")

            # Filter for finite magnitudes
            for col in available_mags:
                if col in df.columns:
                    df = df.filter(
                        pl.col(col).is_not_null()
                        & pl.col(col).is_finite()
                        & (pl.col(col) > 0)
                        & (pl.col(col) < 20)  # Reasonable NIR limit
                    )

        # 2. Photometric uncertainty thresholds
        error_cols = ["j_cmsig", "h_cmsig", "k_cmsig", "j_err", "h_err", "k_err"]
        for col in error_cols:
            if col in df.columns:
                df = df.filter(
                    pl.col(col).is_null()
                    | (pl.col(col) <= self.config["max_uncertainty"])
                )

        # 3. Photometric quality flags (if available)
        if "ph_qual" in df.columns:
            # Keep sources with quality A, B, or C
            df = df.filter(
                pl.col("ph_qual").str.slice(0, 1).is_in(["A", "B", "C"])
                | pl.col("ph_qual").str.slice(1, 1).is_in(["A", "B", "C"])
                | pl.col("ph_qual").str.slice(2, 1).is_in(["A", "B", "C"])
            )

        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} TwoMASS sources ({((initial_count - final_count) / initial_count * 100):.1f}%)"
        )

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform TwoMASS data."""
        # Standardize column names
        df = self._standardize_column_names(df)

        # Add NIR colors
        df = self._add_nir_colors(df)

        # Add stellar classification flags
        df = self._add_stellar_classification(df)

        # Convert to cartesian coordinates if available
        ra_cols = ["ra", "RA", "raj2000", "_RAJ2000"]
        dec_cols = ["dec", "DEC", "dej2000", "_DEJ2000"]

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
            # Distance estimation using Ks magnitude
            k_cols = ["k_m", "k_mag", "Ks", "K", "k"]
            k_col = None
            for col in k_cols:
                if col in df.columns:
                    k_col = col
                    break

            if k_col:
                # Simple distance estimation for NIR
                # Assume absolute Ks magnitude based on color
                if "j_k" in df.columns:
                    # Use J-K color for better absolute magnitude estimation
                    df = df.with_columns(
                        [
                            pl.when(pl.col("j_k") < 0.5)
                            .then(pl.lit(-1.0))  # Early type
                            .when(pl.col("j_k") < 1.0)
                            .then(pl.lit(2.0))  # Solar type
                            .when(pl.col("j_k") < 1.5)
                            .then(pl.lit(4.0))  # Red dwarf
                            .otherwise(pl.lit(6.0))  # Very red
                            .alias("abs_k_est")
                        ]
                    )

                    df = df.with_columns(
                        [
                            (10 ** ((pl.col(k_col) - pl.col("abs_k_est") + 5) / 5))
                            .clip(50, 50000)
                            .alias("distance_pc")
                        ]
                    )
                else:
                    # Default distance estimation
                    df = df.with_columns(
                        [
                            (10 ** ((pl.col(k_col) - 3.0 + 5) / 5))
                            .clip(50, 50000)
                            .alias("distance_pc")
                        ]
                    )
            else:
                # Default distance
                df = df.with_columns(
                    [
                        pl.lit(5000.0).alias("distance_pc")  # 5 kpc default
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
        """Standardize TwoMASS column names."""
        column_mapping = {
            # Magnitudes
            "jmag": "j_m",
            "hmag": "h_m",
            "kmag": "k_m",
            "j_mag": "j_m",
            "h_mag": "h_m",
            "k_mag": "k_m",
            "J": "j_m",
            "H": "h_m",
            "K": "k_m",
            "Ks": "k_m",
            # Errors
            "j_err": "j_cmsig",
            "h_err": "h_cmsig",
            "k_err": "k_cmsig",
            "e_jmag": "j_cmsig",
            "e_hmag": "h_cmsig",
            "e_kmag": "k_cmsig",
            # Coordinates
            "RA": "ra",
            "DEC": "dec",
            "_RAJ2000": "ra",
            "_DEJ2000": "dec",
            "raj2000": "ra",
            "dej2000": "dec",
            # ID
            "twomass_name": "designation",
            "name": "designation",
        }

        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_nir_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add near-infrared color indices."""
        # Standard TwoMASS colors
        color_definitions = [
            ("j_h", "j_m", "h_m"),
            ("h_k", "h_m", "k_m"),
            ("j_k", "j_m", "k_m"),
        ]

        for color_name, mag1, mag2 in color_definitions:
            if mag1 in df.columns and mag2 in df.columns:
                df = df.with_columns([(pl.col(mag1) - pl.col(mag2)).alias(color_name)])

        # Infrared excess parameter
        if all(col in df.columns for col in ["j_m", "h_m", "k_m"]):
            df = df.with_columns(
                [
                    (
                        (pl.col("j_m") - pl.col("k_m"))
                        - 2.0 * (pl.col("h_m") - pl.col("k_m"))
                    ).alias("ir_excess")
                ]
            )

        return df

    def _add_stellar_classification(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add stellar classification flags based on NIR colors."""
        # Giant star classification
        if "j_h" in df.columns and "j_k" in df.columns:
            df = df.with_columns(
                [
                    (
                        (pl.col("j_h") > self.config["giant_jh_limit"])
                        & (pl.col("j_k") > 0.8)
                    ).alias("giant_candidate"),
                    (
                        (pl.col("j_k") < self.config["dwarf_jks_limit"])
                        & (pl.col("j_h") < 0.7)
                    ).alias("dwarf_candidate"),
                    (pl.col("j_k") > 1.4).alias("carbon_star_candidate"),
                ]
            )

        # Young Stellar Object classification
        if "ir_excess" in df.columns:
            df = df.with_columns([(pl.col("ir_excess") > 0.3).alias("yso_candidate")])

        # Brown dwarf candidates
        if "j_k" in df.columns and "k_m" in df.columns:
            df = df.with_columns(
                [
                    ((pl.col("j_k") > 1.2) & (pl.col("k_m") > 12)).alias(
                        "brown_dwarf_candidate"
                    )
                ]
            )

        # Temperature indicator from J-Ks color
        if "j_k" in df.columns:
            df = df.with_columns(
                [
                    (3.981 - 0.324 * pl.col("j_k")).alias("log_teff_dwarf"),
                    (10 ** (3.981 - 0.324 * pl.col("j_k")) / 5772).alias(
                        "temperature_indicator"
                    ),  # Normalized to solar
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract TwoMASS NIR features."""
        feature_columns = []

        # ID column
        id_cols = ["designation", "twomass_name", "name", "id"]
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

        # NIR magnitudes
        mag_cols = ["j_m", "h_m", "k_m"]
        for col in mag_cols:
            if col in df.columns:
                feature_columns.append(col)

        # NIR colors
        color_cols = ["j_h", "h_k", "j_k", "ir_excess"]
        for col in color_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Stellar classification flags
        class_cols = [
            "giant_candidate",
            "dwarf_candidate",
            "carbon_star_candidate",
            "yso_candidate",
            "brown_dwarf_candidate",
        ]
        for col in class_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Temperature indicators
        temp_cols = ["temperature_indicator", "log_teff_dwarf"]
        for col in temp_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Photometric uncertainties
        error_cols = ["j_cmsig", "h_cmsig", "k_cmsig"]
        for col in error_cols:
            if col in df.columns:
                feature_columns.append(col)

        # If no standard features, use available numeric columns
        if not feature_columns:
            logger.warning("No standard TwoMASS features found, using numeric columns")
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
            f"Extracted {len(available_features)} TwoMASS features: {available_features}"
        )

        return df
