"""
Cross-Match Utilities
====================

Cross-match different surveys to enrich data (e.g., Gaia coordinates for exoplanets).
"""

import logging
from typing import Optional

import numpy as np
import polars as pl
from astropy import units as u
from astropy.coordinates import SkyCoord

from .preprocessors import get_preprocessor

logger = logging.getLogger(__name__)


class AstroCrossMatch:
    """Cross-match utilities for astronomical surveys."""
    
    @staticmethod
    def cross_match_gaia(
        target_df: pl.DataFrame,
        gaia_df: Optional[pl.DataFrame] = None,
        ra_col: str = "ra",
        dec_col: str = "dec",
        max_sep_arcsec: float = 3.0,
    ) -> pl.DataFrame:
        """
        Cross-match target catalog with Gaia to get parallax/proper motion.

        Args:
            target_df: Target DataFrame with RA/Dec
            gaia_df: Gaia DataFrame (will load if None)
            ra_col: RA column name in target
            dec_col: Dec column name in target
            max_sep_arcsec: Maximum separation in arcseconds

        Returns:
            Target DataFrame enriched with Gaia data
        """
        logger.info(f'Cross-matching with Gaia (max sep: {max_sep_arcsec}")')

        if gaia_df is None:
            # Load minimal Gaia catalog using preprocessor
            logger.info("Loading Gaia data for cross-match...")
            preprocessor = get_preprocessor("gaia")
            gaia_df = preprocessor.load_raw_data(max_samples=1000000)  # Load up to 1M stars for matching
            
            # Select only needed columns
            gaia_columns = ["ra", "dec", "parallax", "parallax_error", "pmra", "pmdec", "phot_g_mean_mag"]
            available_columns = [col for col in gaia_columns if col in gaia_df.columns]
            gaia_df = gaia_df.select(available_columns)

        # Convert to SkyCoord for matching
        target_coords = SkyCoord(
            ra=target_df[ra_col].to_numpy() * u.Unit("deg"),
            dec=target_df[dec_col].to_numpy() * u.Unit("deg"),
        )

        gaia_coords = SkyCoord(
            ra=gaia_df["ra"].to_numpy() * u.Unit("deg"),
            dec=gaia_df["dec"].to_numpy() * u.Unit("deg"),
        )

        # Match catalogs
        idx, sep2d, _ = target_coords.match_to_catalog_sky(gaia_coords)

        # Filter by separation
        mask = sep2d < max_sep_arcsec * u.arcsec

        # Add Gaia columns to targets that matched
        matched_indices = np.where(mask)[0]
        gaia_indices = idx[mask]

        # Initialize new columns with nulls
        n_targets = len(target_df)
        gaia_columns = {
            "gaia_source_id": pl.Series([None] * n_targets, dtype=pl.Int64),
            "parallax": pl.Series([None] * n_targets, dtype=pl.Float32),
            "parallax_error": pl.Series([None] * n_targets, dtype=pl.Float32),
            "pmra": pl.Series([None] * n_targets, dtype=pl.Float32),
            "pmdec": pl.Series([None] * n_targets, dtype=pl.Float32),
            "phot_g_mean_mag": pl.Series([None] * n_targets, dtype=pl.Float32),
        }

        # Fill matched values
        for i, (target_idx, gaia_idx) in enumerate(zip(matched_indices, gaia_indices)):
            for col in ["parallax", "parallax_error", "pmra", "pmdec", "phot_g_mean_mag"]:
                if col in gaia_df.columns:
                    gaia_columns[col][target_idx] = gaia_df[col][gaia_idx]

        # Add columns to DataFrame
        for col_name, col_data in gaia_columns.items():
            target_df = target_df.with_columns(col_data.alias(col_name))

        logger.info(f"Matched {len(matched_indices)}/{len(target_df)} targets with Gaia")

        return target_df

    @staticmethod
    def enrich_exoplanet_with_gaia(exoplanet_df: pl.DataFrame) -> pl.DataFrame:
        """
        Enrich exoplanet data with Gaia host star information.

        Args:
            exoplanet_df: Exoplanet DataFrame

        Returns:
            Enriched DataFrame with Gaia parallax for better distances
        """
        # Cross-match with Gaia
        enriched_df = AstroCrossMatch.cross_match_gaia(exoplanet_df, max_sep_arcsec=1.0)

        # Use Gaia parallax if available and better than existing distance
        if "parallax" in enriched_df.columns:
            # Convert parallax to distance
            gaia_dist_pc = 1000.0 / enriched_df["parallax"]

            # Replace distance if Gaia is available and reasonable
            mask = (
                enriched_df["parallax"].is_not_null()
                & (enriched_df["parallax"] > 0.1)  # > 0.1 mas (< 10 kpc)
                & (
                    enriched_df["parallax_error"] / enriched_df["parallax"] < 0.2
                )  # < 20% error
            )

            enriched_df = enriched_df.with_columns(
                pl.when(mask)
                .then(gaia_dist_pc)
                .otherwise(pl.col("st_dist"))
                .alias("st_dist")
            )

            # Add flag for Gaia distance
            enriched_df = enriched_df.with_columns(mask.alias("has_gaia_distance"))

        return enriched_df


# For backward compatibility
def cross_match_gaia(*args, **kwargs):
    """Backward compatibility wrapper."""
    return AstroCrossMatch.cross_match_gaia(*args, **kwargs)


def enrich_exoplanet_with_gaia(*args, **kwargs):
    """Backward compatibility wrapper."""
    return AstroCrossMatch.enrich_exoplanet_with_gaia(*args, **kwargs)
