"""
Exoplanet Survey Collector
=========================

Collector for NASA Exoplanet Archive data using astroquery.nasa_exoplanet_archive.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class ExoplanetCollector(BaseSurveyCollector):
    """
    Collector for NASA Exoplanet Archive data using astroquery.nasa_exoplanet_archive.
    """

    def __init__(self, survey_name: str = "exoplanet", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["exoplanet_catalog.csv"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download exoplanet data using astroquery.nasa_exoplanet_archive.
        """
        logger.info(
            "📥 Downloading exoplanet data using astroquery.nasa_exoplanet_archive..."
        )
        target_path = self.raw_dir / "exoplanet_catalog.csv"
        if target_path.exists() and not force:
            logger.info(f"✓ Exoplanet data already exists: {target_path}")
            return [target_path]
        try:
            from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive

            table = NasaExoplanetArchive.query_criteria(
                table="exoplanets",
                select="*",
            )
            table.write(target_path, format="csv", overwrite=True)
            logger.info(f"✅ Exoplanet data downloaded: {target_path}")
            return [target_path]
        except Exception as e:
            logger.error(f"❌ Exoplanet download failed: {e}")
            raise
