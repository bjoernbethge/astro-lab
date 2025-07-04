"""DES (Dark Energy Survey) preprocessor implementation.

Handles DES optical survey data preprocessing.
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


class DESPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for DES optical survey data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize DES preprocessor."""
        default_config = {
            "redshift_limit": None,
            "magnitude_limit": None,
            "mass_limit": None,
            "remove_duplicates": False,
            "map_columns": True,
            "require_redshift": False,  # Redshift not always available
            "min_snr": 5.0,  # Minimum signal-to-noise ratio
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        self.required_columns = []  # Make ID optional for DES
        self.des_bands = ["g", "r", "i", "z", "y"]

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "des"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "galaxy"  # DES primarily surveys galaxies

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply DES-specific quality filters."""
        logger.info("Applying DES filters...")
        initial_count = len(df)
        df = self.map_and_select_columns(df)
        # Redshift filter
        if "redshift" in df.columns and self.config["redshift_limit"] is not None:
            df = df.filter(
                (pl.col("redshift") > 0)
                & (pl.col("redshift") < self.config["redshift_limit"])
            )
            logger.info(
                f"Applied redshift filter: 0 < redshift < {self.config['redshift_limit']}"
            )
        # Magnitude filter
        if "mag_r" in df.columns and self.config["magnitude_limit"] is not None:
            df = df.filter(pl.col("mag_r") < self.config["magnitude_limit"])
            logger.info(
                f"Applied magnitude filter: mag_r < {self.config['magnitude_limit']}"
            )
        # Mass filter
        if "mass" in df.columns and self.config["mass_limit"] is not None:
            min_mass, max_mass = self.config["mass_limit"]
            df = df.filter((pl.col("mass") >= min_mass) & (pl.col("mass") <= max_mass))
            logger.info(f"Applied mass filter: {min_mass} < mass < {max_mass}")
        # Remove duplicates
        if self.config["remove_duplicates"] and "objid" in df.columns:
            df = df.unique(subset=["objid"])
            logger.info("Removed duplicate objid entries")
        final_count = len(df)
        logger.info(
            f"Filtered {initial_count - final_count} objects ({100 * (1 - final_count / initial_count):.1f}%)"
        )
        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform DES data."""
        # Standardize column names
        df = self._standardize_column_names(df)

        # Add optical colors
        df = self._add_optical_colors(df)

        # Add classification features
        df = self._add_classification_features(df)

        # Convert to cartesian coordinates
        ra_cols = ["ra", "RA", "alpha_j2000", "ALPHA_J2000"]
        dec_cols = ["dec", "DEC", "delta_j2000", "DELTA_J2000"]

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
            # Distance estimation
            if "redshift" in df.columns:
                # Cosmological distance
                H0 = 70  # km/s/Mpc
                c = 300000  # km/s
                df = df.with_columns(
                    [(pl.col("redshift") * c / H0).alias("distance_mpc")]
                )
            else:
                # Photometric distance estimation using i-band
                i_cols = ["mag_i", "i_mag", "I_MAG", "i"]
                i_col = None
                for col in i_cols:
                    if col in df.columns:
                        i_col = col
                        break

                if i_col:
                    # Simple distance estimation (assuming galaxy)
                    df = df.with_columns(
                        [
                            (
                                pl.when(10 ** ((pl.col(i_col) - (-20.0) + 5) / 5) < 10)
                                .then(10.0)
                                .when(10 ** ((pl.col(i_col) - (-20.0) + 5) / 5) > 1000)
                                .then(1000.0)
                                .otherwise(10 ** ((pl.col(i_col) - (-20.0) + 5) / 5))
                                .alias("distance_mpc")
                            )
                        ]
                    )
                else:
                    # Default distance
                    df = df.with_columns(
                        [
                            pl.lit(500.0).alias("distance_mpc")  # 500 Mpc default
                        ]
                    )

            x, y, z = spherical_to_cartesian(
                df[ra_col], df[dec_col], df["distance_mpc"]
            )
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )

        return df

    def _standardize_column_names(self, df: pl.DataFrame) -> pl.DataFrame:
        """Standardize DES column names."""
        column_mapping = {
            # Coordinates
            "RA": "ra",
            "DEC": "dec",
            "alpha_j2000": "ra",
            "delta_j2000": "dec",
            "ALPHA_J2000": "ra",
            "DELTA_J2000": "dec",
            # Magnitudes
            "g_mag": "mag_g",
            "r_mag": "mag_r",
            "i_mag": "mag_i",
            "z_mag": "mag_z",
            "y_mag": "mag_y",
            "G_MAG": "mag_g",
            "R_MAG": "mag_r",
            "I_MAG": "mag_i",
            "Z_MAG": "mag_z",
            "Y_MAG": "mag_y",
            # ID
            "object_id": "id",
            "des_id": "id",
            "DES_ID": "id",
            "OBJID": "id",
            "objid": "id",
            # Redshift - be more specific to avoid conflicts
            "z_spec": "redshift",
            "z_phot": "redshift",
            "redshift_spec": "redshift",
            "redshift_phot": "redshift",
        }

        # Handle 'z' and 'Z' columns carefully to avoid coordinate conflicts
        # Check if we already have a coordinate 'z' or if this looks like redshift
        has_coord_z = any(col in df.columns for col in ["x", "y"]) and "z" in df.columns

        if "z" in df.columns and "redshift" not in df.columns and not has_coord_z:
            # Only map z to redshift if it looks like redshift data
            try:
                z_sample = df["z"].drop_nulls().head(100)
                if len(z_sample) > 0:
                    z_mean = (
                        float(z_sample.mean()) if z_sample.mean() is not None else None
                    )
                    z_max = (
                        float(z_sample.max()) if z_sample.max() is not None else None
                    )
                    # Redshift is typically 0 < z < 5 for most surveys
                    if z_mean is not None and z_max is not None:
                        if 0 <= z_mean <= 5 and z_max <= 10:
                            column_mapping["z"] = "redshift"
            except:
                pass

        # Apply mappings
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename({old_name: new_name})

        return df

    def _add_optical_colors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add optical color indices."""
        # Standard DES colors
        color_definitions = [
            ("g_r", "mag_g", "mag_r"),
            ("r_i", "mag_r", "mag_i"),
            (
                "i_z_color",
                "mag_i",
                "mag_z",
            ),  # Rename to avoid conflict with i_z coordinate
            ("z_y", "mag_z", "mag_y"),
            ("g_i", "mag_g", "mag_i"),
            ("r_z_color", "mag_r", "mag_z"),  # Rename to avoid conflict
        ]

        for color_name, mag1, mag2 in color_definitions:
            if mag1 in df.columns and mag2 in df.columns:
                df = df.with_columns([(pl.col(mag1) - pl.col(mag2)).alias(color_name)])

        return df

    def _add_classification_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add object classification features based on DES colors."""
        # Star-galaxy separation using colors
        if "g_r" in df.columns and "r_i" in df.columns:
            df = df.with_columns(
                [
                    # Stars are typically bluer
                    ((pl.col("g_r") < 1.0) & (pl.col("r_i") < 0.8)).alias(
                        "star_candidate"
                    ),
                    # Galaxies are typically redder
                    ((pl.col("g_r") > 0.5) & (pl.col("r_i") > 0.3)).alias(
                        "galaxy_candidate"
                    ),
                ]
            )

        # QSO candidates using color selection
        if "g_r" in df.columns and "r_i" in df.columns and "i_z_color" in df.columns:
            df = df.with_columns(
                [
                    # QSO color selection (simplified)
                    (
                        (pl.col("g_r") < 1.2)
                        & (pl.col("r_i") < 0.2)
                        & (pl.col("i_z_color") > -0.2)
                    ).alias("qso_candidate")
                ]
            )

        # Red galaxy selection
        if "g_r" in df.columns and "r_i" in df.columns:
            df = df.with_columns(
                [
                    ((pl.col("g_r") > 1.2) & (pl.col("r_i") > 0.8)).alias(
                        "red_galaxy_candidate"
                    )
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract DES optical features."""
        feature_columns = []

        # ID column
        id_cols = ["id", "object_id", "des_id", "DES_ID", "OBJID", "objid"]
        id_col = None
        for col in id_cols:
            if col in df.columns:
                id_col = col
                break

        # Create a simple ID if none exists
        if not id_col:
            df = df.with_row_index("row_id")
            id_col = "row_id"

        # Position
        if all(col in df.columns for col in ["x", "y", "z"]):
            feature_columns.extend(["x", "y", "z"])
        elif all(col in df.columns for col in ["ra", "dec"]):
            feature_columns.extend(["ra", "dec"])

        # DES magnitudes
        mag_cols = []
        for band in self.des_bands:
            possible_cols = [
                f"mag_{band}",
                f"{band}_mag",
                f"{band.upper()}_MAG",
                f"{band}",
            ]
            for col in possible_cols:
                if col in df.columns:
                    mag_cols.append(col)
                    break

        feature_columns.extend(mag_cols)

        # Optical colors (use renamed versions to avoid conflicts)
        color_cols = ["g_r", "r_i", "i_z_color", "z_y", "g_i", "r_z_color"]
        for col in color_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Classification flags
        class_cols = [
            "star_candidate",
            "galaxy_candidate",
            "qso_candidate",
            "red_galaxy_candidate",
        ]
        for col in class_cols:
            if col in df.columns:
                feature_columns.append(col)

        # Redshift
        if "redshift" in df.columns:
            feature_columns.append("redshift")

        # Signal-to-noise ratios
        snr_cols = []
        for band in self.des_bands:
            possible_snr_cols = [f"snr_{band}", f"{band}_snr", f"{band.upper()}_SNR"]
            for col in possible_snr_cols:
                if col in df.columns:
                    snr_cols.append(col)
                    break

        feature_columns.extend(snr_cols)

        # If no standard features, use available numeric columns
        if not feature_columns:
            logger.warning("No standard DES features found, using numeric columns")
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
                if col in df.columns and df[col].dtype == pl.Boolean:
                    boolean_features.append(col)
                elif col in df.columns:
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

        # Select final columns - ensure no duplicates
        keep_columns = [id_col] + list(set(available_features))

        # Filter to only existing columns
        final_columns = [col for col in keep_columns if col in df.columns]

        df = df.select(final_columns)

        logger.info(
            f"Extracted {len(available_features)} DES features: {available_features}"
        )

        return df
