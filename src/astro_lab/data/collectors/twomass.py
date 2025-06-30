"""
TwoMASS Survey Collector
=======================

Collector for 2MASS (Two Micron All Sky Survey) data using astroquery.irsa.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class TwoMASSCollector(BaseSurveyCollector):
    """
    Collector for 2MASS data using astroquery.irsa.
    """

    def __init__(self, survey_name: str = "twomass", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["twomass_sample.tbl"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download 2MASS data using astroquery.irsa and save as Parquet.
        """
        logger.info("📥 Downloading 2MASS data using astroquery.irsa...")
        target_tbl = self.raw_dir / "twomass.tbl"
        target_parquet = self.raw_dir / "twomass.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"✓ 2MASS Parquet data already exists: {target_parquet}")
            return [target_parquet]
        try:
            import astropy.units as u
            import polars as pl
            from astropy.coordinates import SkyCoord
            from astropy.table import Table
            from astroquery.irsa import Irsa

            # Example: small cone search
            coord = SkyCoord(ra=56.75, dec=24.1167, unit=(u.deg, u.deg), frame="icrs")
            table = Irsa.query_region(
                coord, catalog="fp_psc", spatial="Cone", radius=0.1 * u.deg
            )
            table.write(target_tbl, format="ascii.ipac", overwrite=True)
            logger.info(f"✅ 2MASS data downloaded: {target_tbl}")

            # Convert to Parquet
            tbl = Table.read(target_tbl, format="ascii.ipac")
            df = pl.from_pandas(tbl.to_pandas())
            df.write_parquet(target_parquet)
            logger.info(f"✅ 2MASS data converted to Parquet: {target_parquet}")

            # Optionally remove the .tbl file
            target_tbl.unlink(missing_ok=True)
            return [target_parquet]
        except Exception as e:
            logger.error(f"❌ 2MASS download failed: {e}")
            raise
