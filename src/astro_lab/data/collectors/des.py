"""
DES Survey Collector
===================

Collector for DES (Dark Energy Survey) DR2 data via VizieR (II/371/des_dr2).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class DESCollector(BaseSurveyCollector):
    """Collector for DES DR2 data via VizieR."""

    VIZIER_CATALOG = "II/371/des_dr2"

    def __init__(self, survey_name: str = "des", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)
        if self.magnitude_limit is None:
            self.magnitude_limit = 16.0

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return [f"des_dr2_mag{self.magnitude_limit}.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / f"des_dr2_mag{self.magnitude_limit}.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"DES data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info(f"Downloading DES DR2 from VizieR (r < {self.magnitude_limit}) ...")

        viz = Vizier(
            columns=["RAJ2000", "DEJ2000",
                      "gmag", "rmag", "imag", "zmag", "Ymag",
                      "e_gmag", "e_rmag", "e_imag", "e_zmag", "e_Ymag",
                      "SpreadMr"],
            column_filters={"rmag": f"<{self.magnitude_limit}"},
            row_limit=-1,
            timeout=self.config["download_timeout"],
        )
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"DES: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
