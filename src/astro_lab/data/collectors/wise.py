"""
WISE Survey Collector
====================

Collector for WISE (Wide-field Infrared Survey Explorer) data via VizieR (AllWISE catalog II/328/allwise).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class WISECollector(BaseSurveyCollector):
    """Collector for AllWISE data via VizieR."""

    VIZIER_CATALOG = "II/328/allwise"

    def __init__(self, survey_name: str = "wise", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)
        if self.magnitude_limit is None:
            self.magnitude_limit = 12.0

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return [f"wise_allwise_mag{self.magnitude_limit}.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / f"wise_allwise_mag{self.magnitude_limit}.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"WISE data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info(f"Downloading AllWISE data from VizieR (W1 < {self.magnitude_limit}) ...")

        viz = Vizier(
            columns=["RAJ2000", "DEJ2000", "W1mag", "W2mag", "W3mag", "W4mag",
                      "e_W1mag", "e_W2mag", "e_W3mag", "e_W4mag",
                      "Jmag", "Hmag", "Kmag", "ccf", "ex", "var"],
            column_filters={"W1mag": f"<{self.magnitude_limit}"},
            row_limit=-1,
            timeout=self.config["download_timeout"],
        )
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"WISE: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
