"""
RR Lyrae Survey Collector
=========================

Collector for RR Lyrae variable star survey data.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class RRLyraeCollector(BaseSurveyCollector):
    """
    Collector for RR Lyrae survey data.

    RR Lyrae stars are pulsating variable stars used as standard candles
    for distance measurements in astronomy.
    """

    def __init__(self, survey_name: str = "rrlyrae", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        """Get URLs for RR Lyrae data download."""
        # Example URLs - replace with actual RR Lyrae data sources
        return [
            "https://example.com/rrlyrae/rrlyrae_catalog.fits",  # Replace with real URL
            # Add more URLs if needed
        ]

    def get_target_files(self) -> List[str]:
        """Get target file names for downloaded data."""
        return [
            "rrlyrae_raw.parquet",  # Main catalog
            # Add more target files if needed
        ]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download RR Lyrae survey data.

        For now, this copies existing data from the global raw directory
        to the survey-specific directory, simulating a download.
        """
        logger.info(f"📥 Collecting RR Lyrae data for {self.survey_name}")

        # Check if we have existing data in the global raw directory
        global_raw_path = Path("data/raw/rrlyrae/rrlyrae_raw.parquet")

        if global_raw_path.exists():
            # Copy existing data to survey directory
            target_path = self.raw_dir / "rrlyrae_raw.parquet"

            if target_path.exists() and not force:
                logger.info(f"✓ RR Lyrae data already exists: {target_path}")
                return [target_path]

            logger.info(
                f"📋 Copying RR Lyrae data from {global_raw_path} to {target_path}"
            )
            import shutil

            shutil.copy2(global_raw_path, target_path)

            logger.info(f"✅ RR Lyrae data collected: {target_path}")
            return [target_path]
        else:
            # If no existing data, try actual download
            logger.warning("No existing RR Lyrae data found, attempting download...")
            return super().download(force)
