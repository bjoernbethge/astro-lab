"""
Euclid Survey Collector
======================

Collector for ESA Euclid data. Public catalog access is limited;
download via ESA Euclid Archive required.
"""

import logging
from pathlib import Path
from typing import List, Optional

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class EuclidCollector(BaseSurveyCollector):
    """Collector for ESA Euclid data."""

    def __init__(self, survey_name: str = "euclid", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return []

    def download(self, force: bool = False) -> List[Path]:
        raise NotImplementedError(
            "Euclid data download is not yet implemented. "
            "Euclid Early Release Observations are available via the ESA Euclid Archive "
            "(https://easotf.esac.esa.int/). "
            f"Please download data manually and place files in: {self.raw_dir}"
        )
