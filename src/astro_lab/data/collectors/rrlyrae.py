"""
RR Lyrae Survey Collector
=========================

Collector for RR Lyrae variable star catalog via VizieR (Gaia DR3 RR Lyrae).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class RRLyraeCollector(BaseSurveyCollector):
    """Collector for Gaia DR3 RR Lyrae catalog via VizieR."""

    VIZIER_CATALOG = "I/358/vrrlyr"

    def __init__(self, survey_name: str = "rrlyrae", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return ["rrlyrae_catalog.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / "rrlyrae_catalog.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"RR Lyrae data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info("Downloading Gaia DR3 RR Lyrae catalog from VizieR ...")

        viz = Vizier(row_limit=-1, timeout=self.config["download_timeout"])
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"RR Lyrae: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
