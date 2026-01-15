"""
LINEAR Survey Collector
======================

Collector for LINEAR (Lincoln Near-Earth Asteroid Research) survey data.
"""

import logging
from pathlib import Path
from typing import List

from astro_lab.config import get_data_paths

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class LinearCollector(BaseSurveyCollector):
    """
    Collector for LINEAR survey data.

    LINEAR (Lincoln Near-Earth Asteroid Research) was a project to detect
    near-Earth asteroids using automated telescopes.
    """

    def __init__(self, survey_name: str = "linear", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        """Get URLs for LINEAR data download."""
        # Example URLs - replace with actual LINEAR data sources
        return [
            "https://example.com/linear/linear_catalog.csv",  # Replace with real URL
            # Add more URLs if needed
        ]

    def get_target_files(self) -> List[str]:
        """Get target file names for downloaded data."""
        return [
            "linear_raw.parquet",  # Main catalog
            # Add more target files if needed
        ]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download LINEAR survey data.

        For now, this copies existing data from the global raw directory
        to the survey-specific directory, simulating a download.
        """
        logger.info(f"📥 Collecting LINEAR data for {self.survey_name}")

        # Check if we have existing data in the global raw directory
        global_raw_path = (
            Path(get_data_paths()["raw_dir"]) / "linear" / "linear_raw.parquet"
        )

        if global_raw_path.exists():
            # Copy existing data to survey directory
            target_path = self.raw_dir / "linear_raw.parquet"

            if target_path.exists() and not force:
                logger.info(f"✓ LINEAR data already exists: {target_path}")
                return [target_path]

            logger.info(
                f"📋 Copying LINEAR data from {global_raw_path} to {target_path}"
            )
            import shutil

            shutil.copy2(global_raw_path, target_path)

            logger.info(f"✅ LINEAR data collected: {target_path}")
            return [target_path]
        else:
            # If no existing data, try actual download
            logger.warning("No existing LINEAR data found, attempting download...")
            return super().download(force)
