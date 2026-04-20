"""
SDSS Survey Collector
=====================

Collector for SDSS DR17 spectrophotometric data via astroquery.sdss SQL queries.
"""

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class SDSSCollector(BaseSurveyCollector):
    """Collector for SDSS DR17 data via CAS SQL queries."""

    def __init__(self, survey_name: str = "sdss", data_config=None,
                 magnitude_limit: Optional[float] = None, region: Optional[str] = None):
        super().__init__(survey_name, data_config, magnitude_limit=magnitude_limit, region=region)
        if self.magnitude_limit is None:
            self.magnitude_limit = 17.77  # SDSS spectroscopic completeness limit

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return [f"sdss_dr17_specphoto_mag{self.magnitude_limit}.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        target_parquet = self.raw_dir / f"sdss_dr17_specphoto_mag{self.magnitude_limit}.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"SDSS data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.sdss import SDSS

        logger.info(f"Downloading SDSS DR17 from CAS (r < {self.magnitude_limit}) ...")

        # SDSS CAS has a row limit per query — chunk by RA ranges
        ra_ranges = [(i * 36.0, (i + 1) * 36.0) for i in range(10)]
        all_frames = []

        for ra_min, ra_max in ra_ranges:
            query = f"""
            SELECT
                s.specObjID, s.ra, s.dec, s.z, s.zErr, s.class, s.subClass,
                p.objID, p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z,
                p.modelMagErr_u, p.modelMagErr_g, p.modelMagErr_r, p.modelMagErr_i, p.modelMagErr_z,
                p.petroRad_r, p.petroR50_r, p.petroR90_r,
                p.extinction_u, p.extinction_g, p.extinction_r, p.extinction_i, p.extinction_z
            FROM SpecObj AS s
            JOIN PhotoObj AS p ON s.bestObjID = p.objID
            WHERE s.zWarning = 0
              AND p.modelMag_r < {self.magnitude_limit}
              AND p.mode = 1
              AND s.class != 'UNKNOWN'
              AND s.ra >= {ra_min} AND s.ra < {ra_max}
            """
            logger.info(f"  Querying RA [{ra_min:.0f}, {ra_max:.0f}) ...")
            result = SDSS.query_sql(query, data_release=17)
            if result is not None and len(result) > 0:
                all_frames.append(pl.from_pandas(result.to_pandas()))
                logger.info(f"    {len(result):,} objects")

        if not all_frames:
            raise ValueError("No data returned from SDSS CAS query")

        df = pl.concat(all_frames)
        df.write_parquet(target_parquet)
        logger.info(f"SDSS: {len(df):,} sources saved to {target_parquet}")
        return [target_parquet]
