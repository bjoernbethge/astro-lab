"""TNG50 simulation preprocessor implementation.

Handles TNG50 cosmological simulation data preprocessing.
"""

import logging
from typing import Any, Dict, Optional

import polars as pl

from .astro import AstroLabDataPreprocessor, StatisticalPreprocessorMixin

logger = logging.getLogger(__name__)


class TNG50Preprocessor(AstroLabDataPreprocessor, StatisticalPreprocessorMixin):
    """Preprocessor for TNG50 simulation data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize TNG50 preprocessor."""
        default_config = {
            "snapshot": 99,  # z=0
            "halo_mass_limit": [10.0, 15.0],  # log solar masses
            "subhalo_only": False,
        }

        if config:
            default_config.update(config)

        super().__init__(default_config)

        # From SURVEY_CONFIGS - TNG50 already uses x,y,z coordinates
        self.required_columns = ["halo_id", "x", "y", "z", "mass"]

    def get_survey_name(self) -> str:
        """Return the survey name."""
        return "tng50"

    def get_object_type(self) -> str:
        """Return the object type for this survey."""
        return "halo"  # Dark matter halos

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply TNG50-specific filters."""
        initial_count = len(df)

        # Halo mass filter
        if "mass" in df.columns and self.config["halo_mass_limit"]:
            min_mass, max_mass = self.config["halo_mass_limit"]
            df = df.filter((pl.col("mass") >= min_mass) & (pl.col("mass") <= max_mass))

        # Subhalo filter
        if self.config["subhalo_only"] and "is_subhalo" in df.columns:
            df = df.filter(pl.col("is_subhalo") == True)

        # Particle type filter - from SURVEY_CONFIGS
        if "particle_type" in df.columns:
            # Could filter by particle type if needed
            pass

        final_count = len(df)
        logger.info(f"Filtered {initial_count - final_count} halos")

        return df

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """Transform TNG50 data."""
        # Positions are already in Cartesian (Mpc/h comoving)
        # Convert to physical units
        h = 0.6774  # TNG50 Hubble parameter

        df = df.with_columns(
            [
                (pl.col("x") / h).alias("x"),
                (pl.col("y") / h).alias("y"),
                (pl.col("z") / h).alias("z"),
            ]
        )

        # Velocity features from SURVEY_CONFIGS
        if all(col in df.columns for col in ["vx", "vy", "vz"]):
            df = df.with_columns(
                [
                    (pl.col("vx") / h).alias("vx"),
                    (pl.col("vy") / h).alias("vy"),
                    (pl.col("vz") / h).alias("vz"),
                    # Total velocity
                    (
                        (pl.col("vx") ** 2 + pl.col("vy") ** 2 + pl.col("vz") ** 2)
                        ** 0.5
                    ).alias("v_total"),
                ]
            )

        # Calculate environmental properties
        # (simplified - would use actual neighbor counts)
        df = df.with_columns(
            [
                pl.col("mass").alias("log_mass"),
                # Placeholder for environmental density
                pl.lit(0.0).alias("env_density"),
            ]
        )

        return df

    def extract_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Extract simulation features."""
        # Basic features from positions and mass
        feature_columns = ["x", "y", "z", "log_mass"]

        # Velocity features if available
        velocity_cols = ["vx", "vy", "vz", "v_total"]
        feature_columns.extend([col for col in velocity_cols if col in df.columns])

        # Additional simulation properties
        extra_cols = ["sfr", "metallicity", "env_density", "particle_type"]
        feature_columns.extend([col for col in extra_cols if col in df.columns])

        # Keep available features
        feature_columns = [col for col in feature_columns if col in df.columns]

        # Normalize
        df = self.normalize_columns(df, feature_columns, method="standard")

        # Remove _norm suffix and keep normalized values
        for col in feature_columns:
            if f"{col}_norm" in df.columns:
                df = df.with_columns([pl.col(f"{col}_norm").alias(col)])
                df = df.drop(f"{col}_norm")

        keep_columns = ["halo_id"] + feature_columns
        df = df.select([col for col in keep_columns if col in df.columns])

        return df
