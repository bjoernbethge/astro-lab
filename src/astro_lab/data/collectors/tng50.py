"""
TNG50 Survey Collector
=====================

Collector for TNG50 cosmological simulation data.
"""

import logging
from pathlib import Path
from typing import List

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class TNG50Collector(BaseSurveyCollector):
    """
    Collector for TNG50 cosmological simulation data.
    """

    def __init__(self, survey_name: str = "tng50", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # TNG50 data is typically downloaded via scripts or direct URLs
        # These are example URLs - replace with actual TNG50 data URLs
        return [
            "https://www.tng-project.org/data/TNG50-1/output/snapdir_099/snap_099.0.hdf5",
            "https://www.tng-project.org/data/TNG50-1/output/snapdir_099/snap_099.1.hdf5",
            # Add more URLs as needed
        ]

    def get_target_files(self) -> List[str]:
        return [
            "snap_099.0.hdf5",
            "snap_099.1.hdf5",
            # Add more target files as needed
        ]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download TNG50 simulation data.
        """
        logger.info("📥 Downloading TNG50 simulation data...")
        target_paths = []
        for target_file in self.get_target_files():
            target_path = self.raw_dir / target_file
            if target_path.exists() and not force:
                logger.info(f"✓ TNG50 file already exists: {target_path}")
                target_paths.append(target_path)
                continue
            # For demo purposes, we'll just create a placeholder file
            # In practice, you would use the actual download URLs
            target_path.touch()
            logger.info(f"✅ TNG50 file created (demo): {target_path}")
            target_paths.append(target_path)
        return target_paths
