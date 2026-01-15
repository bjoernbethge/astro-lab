"""
SDSS Survey Collector
=====================

Collector for SDSS (Sloan Digital Sky Survey) data using sdss-access.
"""

import logging
import subprocess
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class SDSSCollector(BaseSurveyCollector):
    """
    Collector for SDSS data using the official sdss-access tool.
    """

    def __init__(self, survey_name: str = "sdss", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as sdss-access handles download logic
        return []

    def get_target_files(self) -> List[str]:
        # Example: list of expected files (can be extended)
        return ["sdss_dr17_specObj.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download SDSS data using sdss-access CLI.
        """
        logger.info("📥 Downloading SDSS data using sdss-access...")
        target_path = self.raw_dir / "sdss_dr17_specObj.parquet"
        if target_path.exists() and not force:
            logger.info(f"✓ SDSS data already exists: {target_path}")
            return [target_path]
        # Example: Download spectra for a given plate/mjd/fiber (customize as needed)
        # Here, we just show how to call sdss-access for a sample file
        try:
            # You must have sdss-access installed and configured (see https://sdss-access.readthedocs.io/)
            # Example: download all spectra for DR17
            cmd = [
                "sdss-access",
                "fetch",
                "dr17/sdss/spectro/redux/26/spectra/lite/0260/0260-51602-001.par",
            ]
            subprocess.run(cmd, check=True)
            # Move or copy the file to the raw_dir as needed
            # For demo, we just touch the file
            target_path.touch()
            logger.info(f"✅ SDSS data fetched (demo): {target_path}")
            return [target_path]
        except Exception as e:
            logger.error(f"❌ SDSS download failed: {e}")
            raise
