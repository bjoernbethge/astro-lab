"""
WISE Survey Collector
====================

Collector for WISE (Wide-field Infrared Survey Explorer) data using astroquery.irsa.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class WISECollector(BaseSurveyCollector):
    """
    Collector for WISE data using astroquery.irsa.
    """

    def __init__(self, survey_name: str = "wise", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["wise_sample.tbl"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download WISE data using astroquery.irsa and save as Parquet.
        """
        logger.info("📥 Downloading WISE data using astroquery.irsa...")
        target_tbl = self.raw_dir / "wise.tbl"
        target_parquet = self.raw_dir / "wise.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"✓ WISE Parquet data already exists: {target_parquet}")
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
                coord, catalog="allwise_p3as_psd", spatial="Cone", radius=0.1 * u.deg
            )
            table.write(target_tbl, format="ascii.ipac", overwrite=True)
            logger.info(f"✅ WISE data downloaded: {target_tbl}")

            # Convert to Parquet
            tbl = Table.read(target_tbl, format="ascii.ipac")
            df = pl.from_pandas(tbl.to_pandas())
            df.write_parquet(target_parquet)
            logger.info(f"✅ WISE data converted to Parquet: {target_parquet}")

            # Optionally remove the .tbl file
            target_tbl.unlink(missing_ok=True)
            return [target_parquet]
        except Exception as e:
            logger.error(f"❌ WISE download failed: {e}")
            raise
