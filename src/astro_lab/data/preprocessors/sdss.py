"""SDSS survey preprocessor implementation.

Handles SDSS photometric and spectroscopic data preprocessing.
"""

import logging
from typing import Any, Dict, Optional

import polars as pl

from astro_lab.config import get_survey_config
from astro_lab.data.transforms.astronomical import spherical_to_cartesian

from .astro import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
)

logger = logging.getLogger(__name__)


class SDSSPreprocessor(
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    StatisticalPreprocessorMixin,
):
    """Preprocessor for SDSS data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize SDSS preprocessor."""
        default_config = {
            "redshift_limit": None,
            "magnitude_limit": None,
            "mass_limit": None,
            "remove_duplicates": False,
            "map_columns": True,
            "clean_photometry": True,
            "require_spectroscopy": False,
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        # Get survey configuration
        self.survey_config = get_survey_config("sdss")

        # Use actual SDSS column names from survey config
        self.magnitude_columns = self.survey_config[
            "mag_cols"
        ]  # ['modelMag_u', 'modelMag_g', etc.]
        self.coord_columns = self.survey_config["coord_cols"]  # ['ra', 'dec']
        self.extra_columns = self.survey_config["extra_cols"]  # ['z', 'petroRad_r']

        self.required_columns = ["objid"] + self.coord_columns

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "sdss"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "galaxy"  # SDSS is primarily for galaxies/quasars

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply SDSS-specific quality filters."""
        logger.info("Applying SDSS filters...")
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
        """Transform SDSS data for ML."""
        # Calculate colors using actual SDSS column names
        color_pairs = self.survey_config["color_pairs"]

        for mag1, mag2 in color_pairs:
            if mag1 in df.columns and mag2 in df.columns:
                # Create color name like 'g_r' from 'modelMag_g' and 'modelMag_r'
                band1 = mag1.split("_")[-1]  # Extract 'g' from 'modelMag_g'
                band2 = mag2.split("_")[-1]  # Extract 'r' from 'modelMag_r'
                color_name = f"{band1}_{band2}"

                df = df.with_columns([(pl.col(mag1) - pl.col(mag2)).alias(color_name)])

        # Convert redshift to distance (simplified)
        if "z" in df.columns:
            # Hubble constant = 70 km/s/Mpc
            c = 299792.458  # km/s
            H0 = 70.0
            df = df.with_columns([(c * pl.col("z") / H0).alias("distance_mpc")])

            # Convert to cartesian if distance available
            x, y, z = spherical_to_cartesian(df["ra"], df["dec"], df["distance_mpc"])
            df = df.with_columns(
                [
                    pl.Series("x", x),
                    pl.Series("y", y),
                    pl.Series("z", z),
                ]
            )

        # Add galactic coordinates
        if "ra" in df.columns and "dec" in df.columns:
            from astropy import units as u
            from astropy.coordinates import SkyCoord

            degree = u.Unit("deg")

            coords = SkyCoord(
                ra=df["ra"].to_numpy() * degree,
                dec=df["dec"].to_numpy() * degree,
                frame="icrs",
            )
            galactic = coords.galactic

            df = df.with_columns(
                [
                    pl.Series("gal_l", galactic.l.degree),
                    pl.Series("gal_b", galactic.b.degree),
                ]
            )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract ML-ready features from SDSS data."""
        feature_columns = []

        # Object ID
        id_col = None
        if "objid" in df.columns:
            id_col = "objid"
        elif "objID" in df.columns:
            id_col = "objID"

        # Magnitude features
        for mag_col in self.magnitude_columns:
            if mag_col in df.columns:
                feature_columns.append(mag_col)

        # Color features - look for any color columns we created
        for col in df.columns:
            if "_" in col and col.count("_") == 1:  # Simple color like 'g_r'
                parts = col.split("_")
                if len(parts[0]) == 1 and len(parts[1]) == 1:  # Single letter bands
                    feature_columns.append(col)

        # Extra features
        for extra_col in self.extra_columns:
            if extra_col in df.columns:
                feature_columns.append(extra_col)

        # Position features
        if all(col in df.columns for col in ["x", "y", "z"]):
            feature_columns.extend(["x", "y", "z"])
        elif all(col in df.columns for col in ["ra", "dec"]):
            feature_columns.extend(["ra", "dec"])

        # Galactic coordinates
        if all(col in df.columns for col in ["gal_l", "gal_b"]):
            feature_columns.extend(["gal_l", "gal_b"])

        # Remove duplicates
        feature_columns = list(set(feature_columns))

        # Normalize numeric features
        numeric_features = []
        for col in feature_columns:
            if col in df.columns and df[col].dtype in [
                pl.Float32,
                pl.Float64,
                pl.Int32,
                pl.Int64,
            ]:
                numeric_features.append(col)

        if numeric_features:
            df = self.normalize_columns(df, numeric_features, method="standard")

            # Remove _norm suffix and keep normalized values
            for col in numeric_features:
                if f"{col}_norm" in df.columns:
                    df = df.with_columns([pl.col(f"{col}_norm").alias(col)])
                    df = df.drop(f"{col}_norm")

        # Keep ID and features
        keep_columns = []
        if id_col:
            keep_columns.append(id_col)
        keep_columns.extend(feature_columns)

        # Only select columns that exist
        keep_columns = [col for col in keep_columns if col in df.columns]

        df = df.select(keep_columns)

        logger.info(f"Extracted {len(feature_columns)} features: {feature_columns}")

        return df
