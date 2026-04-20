"""
TwoMASS Survey Collector
=======================

Collector for 2MASS (Two Micron All Sky Survey) data via VizieR (Point Source Catalog II/246/out).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class TwoMASSCollector(BaseSurveyCollector):
    """Collector for 2MASS PSC data via VizieR."""

    VIZIER_CATALOG = "II/246/out"

    def __init__(self, survey_name: str = "twomass", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)
        if self.magnitude_limit is None:
            self.magnitude_limit = 12.0

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return [f"twomass_psc_mag{self.magnitude_limit}.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / f"twomass_psc_mag{self.magnitude_limit}.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"2MASS data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info(f"Downloading 2MASS PSC from VizieR (J < {self.magnitude_limit}) ...")

        viz = Vizier(
            columns=["RAJ2000", "DEJ2000", "Jmag", "Hmag", "Kmag",
                      "e_Jmag", "e_Hmag", "e_Kmag", "Qflg", "Rflg", "Bflg", "Cflg"],
            column_filters={"Jmag": f"<{self.magnitude_limit}"},
            row_limit=-1,
            timeout=self.config["download_timeout"],
        )
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"2MASS: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
