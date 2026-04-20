"""
Pan-STARRS Survey Collector
=========================

Collector for Pan-STARRS DR1 data via VizieR (Mean Object catalog II/349/ps1).
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class PanSTARRSCollector(BaseSurveyCollector):
    """Collector for Pan-STARRS DR1 data via VizieR."""

    VIZIER_CATALOG = "II/349/ps1"

    def __init__(self, survey_name: str = "panstarrs", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)
        if self.magnitude_limit is None:
            self.magnitude_limit = 14.0

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return [f"panstarrs_dr1_mag{self.magnitude_limit}.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / f"panstarrs_dr1_mag{self.magnitude_limit}.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"Pan-STARRS data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.vizier import Vizier

        logger.info(f"Downloading Pan-STARRS DR1 from VizieR (r < {self.magnitude_limit}) ...")

        viz = Vizier(
            columns=["RAJ2000", "DEJ2000", "objID",
                      "gmag", "rmag", "imag", "zmag", "ymag",
                      "e_gmag", "e_rmag", "e_imag", "e_zmag", "e_ymag"],
            column_filters={"rmag": f"<{self.magnitude_limit}"},
            row_limit=-1,
            timeout=self.config["download_timeout"],
        )
        result = viz.get_catalogs(self.VIZIER_CATALOG)
        if not result:
            raise ValueError(f"No data returned from VizieR for {self.VIZIER_CATALOG}")

        df = pl.from_pandas(result[0].to_pandas())
        df.write_parquet(target_parquet)
        logger.info(f"Pan-STARRS: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
