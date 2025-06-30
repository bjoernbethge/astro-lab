"""
DES Survey Collector
===================

Collector for DES (Dark Energy Survey) data using astroquery.vizier.
"""

import logging
from pathlib import Path
from typing import List

import astropy.units as u
import polars as pl
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class DESCollector(BaseSurveyCollector):
    """
    Collector for DES data using astroquery.vizier with chunked downloads.
    """

    def __init__(self, survey_name: str = "des", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        # Not used, as astroquery handles download logic
        return []

    def get_target_files(self) -> List[str]:
        return ["des.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download DES data using astroquery.vizier with chunked region queries.
        """
        logger.info("📥 Downloading DES data using astroquery.vizier (chunked)...")
        target_parquet = self.raw_dir / "des.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"✓ DES Parquet data already exists: {target_parquet}")
            return [target_parquet]

        try:
            from astroquery.vizier import Vizier

            # Define multiple sky regions to cover different parts of the DES footprint
            regions = [
                # DES Y3 Gold footprint regions (approximate)
                {"ra": 56.75, "dec": -24.1167, "radius": 2.0},  # Center region
                {"ra": 56.75, "dec": -30.0, "radius": 2.0},  # South region
                {"ra": 56.75, "dec": -18.0, "radius": 2.0},  # North region
                {"ra": 40.0, "dec": -24.1167, "radius": 2.0},  # West region
                {"ra": 73.5, "dec": -24.1167, "radius": 2.0},  # East region
            ]

            all_tables = []
            vizier = Vizier()
            vizier.ROW_LIMIT = 50000  # Reasonable limit per region

            for i, region in enumerate(regions):
                logger.info(
                    f"📡 Querying DES region {i + 1}/{len(regions)}: "
                    f"RA={region['ra']:.1f}°, Dec={region['dec']:.1f}°, "
                    f"Radius={region['radius']:.1f}°"
                )

                try:
                    coord = SkyCoord(
                        ra=region["ra"],
                        dec=region["dec"],
                        unit=(u.deg, u.deg),
                        frame="icrs",
                    )

                    # Query with magnitude filter to reduce data volume
                    vizier = Vizier(
                        catalog="II/371", column_filters={"MAG_AUTO_R": "<23.0"}
                    )

                    result = vizier.query_region(coord, radius=region["radius"] * u.deg)

                    if result and len(result) > 0:
                        all_tables.append(result[0])
                        logger.info(f"✅ Region {i + 1}: {len(result[0])} objects")
                    else:
                        logger.warning(f"⚠️ Region {i + 1}: No data returned")

                except Exception as e:
                    logger.warning(f"⚠️ Region {i + 1} failed: {e}")
                    continue

            if not all_tables:
                raise ValueError("No data returned from any DES region")

            # Combine all tables
            logger.info("🔗 Combining all regions...")
            combined_table = vstack(all_tables)
            logger.info(f"✅ Combined: {len(combined_table)} total objects")

            # Convert to Parquet
            df = pl.from_pandas(combined_table.to_pandas())
            df.write_parquet(target_parquet)
            logger.info(f"✅ DES data saved as Parquet: {target_parquet}")

            return [target_parquet]

        except Exception as e:
            logger.error(f"❌ DES download failed: {e}")
            raise
