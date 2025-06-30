"""
Gaia Survey Collector
====================

Collector for Gaia DR3 bright all-sky data using astroquery.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
from astroquery.gaia import Gaia

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class GaiaCollector(BaseSurveyCollector):
    """
    Collector for Gaia DR3 bright all-sky data.

    Features:
    - Downloads Gaia DR3 data using astroquery
    - Configurable magnitude limit (default: 12.0)
    - Quality filtering for clean data
    - Efficient batch downloading
    """

    def __init__(self, survey_name: str = "gaia", data_config=None):
        super().__init__(survey_name, data_config)
        self.magnitude_limit = 12.0  # Default for bright all-sky
        self.batch_size = 500000  # Maximum sources per query

    def get_target_files(self) -> List[str]:
        """Get target file names based on magnitude limit."""
        return [f"gaia_dr3_bright_all_sky_mag{self.magnitude_limit}.parquet"]

    def get_download_urls(self) -> List[str]:
        """
        Get download URLs for Gaia data.

        For Gaia DR3, we use astroquery instead of direct downloads,
        but this method is required by the base class.
        """
        return []

    def download(self, force: bool = False) -> List[Path]:
        """
        Download Gaia DR3 bright all-sky data using astroquery.

        Args:
            force: Force re-download even if file exists

        Returns:
            List of downloaded file paths
        """
        logger.info("📥 Downloading Gaia DR3 bright all-sky data...")

        # Target file name includes magnitude limit
        target_filename = f"gaia_dr3_bright_all_sky_mag{self.magnitude_limit}.parquet"
        target_path = self.raw_dir / target_filename
        metadata_path = (
            self.raw_dir / f"gaia_dr3_bright_all_sky_mag{self.magnitude_limit}.json"
        )

        if target_path.exists() and not force:
            logger.info(f"✓ Gaia data already exists: {target_path}")
            return [target_path]

        try:
            # Configure Gaia for DR3
            Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"  # type: ignore
            Gaia.ROW_LIMIT = -1  # type: ignore # No limit for async queries

            logger.info(
                f"🔍 Querying Gaia DR3 Archive for bright sources "
                f"(G < {self.magnitude_limit})..."
            )

            # Build query for bright all-sky sources with quality filters
            query = f"""
            SELECT 
                source_id, ra, dec, l, b, 
                parallax, parallax_error, parallax_over_error,
                pmra, pmra_error, pmdec, pmdec_error,
                phot_g_mean_mag, phot_g_mean_flux_over_error,
                phot_bp_mean_mag, phot_bp_mean_flux_over_error,
                phot_rp_mean_mag, phot_rp_mean_flux_over_error,
                bp_rp, g_rp, bp_g,
                radial_velocity, radial_velocity_error,
                teff_gspphot, teff_gspphot_lower, teff_gspphot_upper,
                logg_gspphot, logg_gspphot_lower, logg_gspphot_upper,
                mh_gspphot, mh_gspphot_lower, mh_gspphot_upper,
                distance_gspphot, distance_gspphot_lower, distance_gspphot_upper,
                ruwe, astrometric_excess_noise, astrometric_excess_noise_sig,
                phot_bp_rp_excess_factor,
                visibility_periods_used, astrometric_n_obs_al, astrometric_n_good_obs_al,
                phot_g_n_obs, phot_bp_n_obs, phot_rp_n_obs
            FROM gaiadr3.gaia_source 
            WHERE phot_g_mean_mag < {self.magnitude_limit}
            AND ruwe < 1.4
            AND parallax_over_error > 5
            AND phot_g_mean_flux_over_error > 50
            AND phot_bp_mean_flux_over_error > 20
            AND phot_rp_mean_flux_over_error > 20
            AND phot_bp_rp_excess_factor < 
                1.3 + 0.06 * POWER(phot_bp_mean_mag - phot_rp_mean_mag, 2)
            AND astrometric_excess_noise < 1
            AND visibility_periods_used > 5
            ORDER BY phot_g_mean_mag
            """

            # Execute async query for large datasets
            job = Gaia.launch_job_async(query, dump_to_file=False)
            logger.info(f"⏳ Query submitted, job ID: {job.jobid}")

            # Get results
            result = job.get_results()

            if result is not None and len(result) > 0:
                logger.info(f"✅ Downloaded {len(result):,} Gaia sources")

                # Convert to pandas DataFrame
                df = result.to_pandas()

                # Calculate additional useful columns
                # Handle potential division by zero for parallax
                valid_parallax = df["parallax"] > 0
                df.loc[valid_parallax, "abs_g_mag"] = df.loc[
                    valid_parallax, "phot_g_mean_mag"
                ] + 5 * np.log10(df.loc[valid_parallax, "parallax"] / 100)
                df.loc[valid_parallax, "distance_pc"] = (
                    1000.0 / df.loc[valid_parallax, "parallax"]
                )

                # Calculate velocities
                df.loc[valid_parallax, "vra_km_s"] = (
                    4.74
                    * df.loc[valid_parallax, "pmra"]
                    * df.loc[valid_parallax, "distance_pc"]
                    / 1000
                )
                df.loc[valid_parallax, "vdec_km_s"] = (
                    4.74
                    * df.loc[valid_parallax, "pmdec"]
                    * df.loc[valid_parallax, "distance_pc"]
                    / 1000
                )

                # Total proper motion
                df["pm_total"] = np.sqrt(df["pmra"] ** 2 + df["pmdec"] ** 2)

                # Add metadata columns
                df["survey"] = "gaia"
                df["data_release"] = "DR3"
                df["region"] = "bright_all_sky"
                df["magnitude_limit"] = self.magnitude_limit

                # Convert to polars for efficient storage
                pl_df = pl.from_pandas(df)

                # Save as parquet
                pl_df.write_parquet(target_path)
                logger.info(f"💾 Saved {len(pl_df):,} sources to {target_path}")

                # Save metadata
                metadata = {
                    "source": "Gaia DR3",
                    "region": "bright_all_sky",
                    "magnitude_limit": self.magnitude_limit,
                    "n_sources": len(pl_df),
                    "columns": list(df.columns),
                    "file_size_mb": target_path.stat().st_size / (1024 * 1024),
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                logger.info(f"📄 Saved metadata to {metadata_path}")

                return [target_path]

            else:
                raise ValueError("No data returned from Gaia query")

        except Exception as e:
            logger.error(f"❌ Gaia download failed: {e}")
            raise
