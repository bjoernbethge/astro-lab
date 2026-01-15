"""
Pan-STARRS Survey Collector
=========================

Collector for Pan-STARRS data using astroquery.mast.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class PanSTARRSCollector(BaseSurveyCollector):
    """
    Collector for Pan-STARRS data using astroquery.mast.
    """

    def __init__(self, survey_name: str = "panstarrs", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["panstarrs_sample.csv"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download Pan-STARRS data using astroquery.mast.Catalogs.
        """
        logger.info("📥 Downloading Pan-STARRS data using astroquery.mast...")
        target_path = self.raw_dir / "panstarrs_sample.csv"
        if target_path.exists() and not force:
            logger.info(f"✓ Pan-STARRS data already exists: {target_path}")
            return [target_path]
        try:
            from astroquery.mast import Catalogs

            # Example: small cone search
            result = Catalogs.query_region("M31", radius=0.02, catalog="Panstarrs")
            result.write(target_path, format="csv", overwrite=True)
            logger.info(f"✅ Pan-STARRS data downloaded: {target_path}")
            return [target_path]
        except Exception as e:
            logger.error(f"❌ Pan-STARRS download failed: {e}")
            raise
