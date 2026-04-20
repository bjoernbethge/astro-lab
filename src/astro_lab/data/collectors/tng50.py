"""
TNG50 Survey Collector
=====================

Collector for TNG50 cosmological simulation data.
Requires manual download via the IllustrisTNG API (authentication required).
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class TNG50Collector(BaseSurveyCollector):
    """Collector for TNG50 cosmological simulation data."""

    def __init__(self, survey_name: str = "tng50", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return []

    def download(self, force: bool = False) -> List[Path]:
        raise NotImplementedError(
            "TNG50 simulation data requires authentication via the IllustrisTNG API "
            "(https://www.tng-project.org/data/). "
            f"Please download HDF5 snapshots manually and place them in: {self.raw_dir}"
        )
