"""
NSA Survey Collector
===================

Collector for NASA-Sloan Atlas (NSA) data using astroquery.vizier.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class NSACollector(BaseSurveyCollector):
    """
    Collector for NASA-Sloan Atlas data using astroquery.vizier.
    """

    def __init__(self, survey_name: str = "nsa", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["nsa_v1_0_1.fits"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download NSA data using astroquery.vizier.
        """
        logger.info("📥 Downloading NSA data using astroquery.vizier...")
        target_path = self.raw_dir / "nsa_v1_0_1.fits"
        if target_path.exists() and not force:
            logger.info(f"✓ NSA data already exists: {target_path}")
            return [target_path]
        try:
            from astroquery.vizier import Vizier

            # Query the NSA catalog
            Vizier.ROW_LIMIT = -1  # Get all rows
            result = Vizier.query_catalog("J/ApJS/221/12/nsa")
            if result:
                result[0].write(target_path, format="fits", overwrite=True)
                logger.info(f"✅ NSA data downloaded: {target_path}")
                return [target_path]
            else:
                raise ValueError("No data returned from VizieR")
        except Exception as e:
            logger.error(f"❌ NSA download failed: {e}")
            raise
