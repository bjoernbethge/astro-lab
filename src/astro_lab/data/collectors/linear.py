"""
LINEAR Survey Collector
======================

Collector for LINEAR (Lincoln Near-Earth Asteroid Research) variable star
classifications via VizieR (J/AJ/148/21).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class LinearCollector(BaseSurveyCollector):
    """Collector for LINEAR variable star classification catalog via VizieR."""

    VIZIER_CATALOG = "J/AJ/146/101"

    def __init__(self, survey_name: str = "linear", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return ["linear_catalog.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / "linear_catalog.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"LINEAR data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info("Downloading LINEAR classification catalog from VizieR ...")

        viz = Vizier(row_limit=-1, timeout=self.config["download_timeout"])
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"LINEAR: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
