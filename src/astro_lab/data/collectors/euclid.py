"""
Euclid Survey Collector
======================

Collector for ESA Euclid data (placeholder, as public access is limited).
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class EuclidCollector(BaseSurveyCollector):
    """
    Collector for ESA Euclid data (demo placeholder).
    """

    def __init__(self, survey_name: str = "euclid", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Placeholder: no public URLs yet
        return []

    def get_target_files(self) -> List[str]:
        return ["euclid_sample.fits"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download Euclid data (placeholder, as public access is limited).
        """
        logger.info("📥 Downloading Euclid data (placeholder)...")
        target_path = self.raw_dir / "euclid_sample.fits"
        if target_path.exists() and not force:
            logger.info(f"✓ Euclid data already exists: {target_path}")
            return [target_path]
        try:
            # Placeholder: create empty file
            target_path.touch()
            logger.info(f"✅ Euclid data placeholder created: {target_path}")
            return [target_path]
        except Exception as e:
            logger.error(f"❌ Euclid download failed: {e}")
            raise
