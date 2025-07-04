"""
DES Survey Collector
===================

Collector for DES (Dark Energy Survey) data using multiple astroquery sources.
"""

import logging
from pathlib import Path
from typing import List

import astropy.units as u
import polars as pl
from astropy.coordinates import SkyCoord

from .base import BaseSurveyCollector

logger = logging.getLogger(__name__)


class DESCollector(BaseSurveyCollector):
    """
    Collector for DES data using SIMBAD (real data only, no fallback, no synthetic).
    """

    def __init__(self, survey_name: str = "des", data_config=None):
        super().__init__(survey_name, data_config)

    def get_download_urls(self) -> List[str]:
        return []

    def get_target_files(self) -> List[str]:
        return ["des_simbad.parquet"]

    def download(self, force: bool = False) -> List[Path]:
        """
        Download real DES galaxy data from SIMBAD for Cosmic Web analysis.
        """
        logger.info("📥 Downloading DES galaxy data from SIMBAD...")
        target_parquet = self.raw_dir / "des_simbad.parquet"
        if target_parquet.exists() and not force:
            logger.info(f"✓ DES SIMBAD data already exists: {target_parquet}")
            return [target_parquet]

        from astroquery.simbad import Simbad

        # Check available votable fields
        available_fields = set(Simbad.get_votable_fields())
        logger.info(f"Available SIMBAD fields: {sorted(list(available_fields))}")

        # Only add fields that are available
        fields_to_add = []
        for f in ["otype", "redshift", "flux(V)", "flux(B)", "flux(R)", "flux(I)"]:
            if f in available_fields:
                fields_to_add.append(f)
        for f in fields_to_add:
            Simbad.add_votable_fields(f)
        logger.info(f"Querying with fields: {fields_to_add}")

        # DES footprint: RA 40-75°, Dec -35 to -15° (split into tiles for large queries)
        regions = [
            (56.75, -24.1167, 5.0),
            (60.0, -25.0, 3.0),
            (55.0, -20.0, 3.0),
            (50.0, -30.0, 3.0),
            (65.0, -30.0, 3.0),
        ]
        all_results = []
        for ra, dec, radius in regions:
            coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
            logger.info(
                f"📡 Querying SIMBAD region: RA={ra}, Dec={dec}, Radius={radius}°"
            )
            try:
                result = Simbad.query_region(coord, radius=radius * u.deg)
                if result is None or len(result) == 0:
                    logger.warning(f"No objects found in region RA={ra}, Dec={dec}")
                    continue
                # Filter for galaxies (otype G) and with redshift if available
                otype_col = next(
                    (c for c in result.colnames if c.upper() == "OTYPE"), None
                )
                redshift_col = next(
                    (c for c in result.colnames if c.upper() == "REDSHIFT"), None
                )
                if otype_col:
                    mask = [str(val).startswith("G") for val in result[otype_col]]
                    filtered = result[mask]
                else:
                    filtered = result
                if redshift_col:
                    filtered = filtered[[v is not None for v in filtered[redshift_col]]]
                if len(filtered) > 0:
                    df = pl.from_pandas(filtered.to_pandas())
                    df = df.with_columns(
                        pl.lit(f"region_{ra}_{dec}").alias("source_region")
                    )
                    all_results.append(df)
                    logger.info(f"✅ Added {len(df)} galaxies from region {ra}, {dec}")
            except Exception as e:
                logger.error(
                    f"❌ SIMBAD query failed for region RA={ra}, Dec={dec}: {e}"
                )
                continue
        if not all_results:
            raise RuntimeError(
                "No real data could be downloaded from SIMBAD. "
                "Please try again later or check your network."
            )
        combined_df = pl.concat(all_results, how="vertical")
        logger.info(f"✅ Downloaded total {len(combined_df)} galaxies from SIMBAD.")
        combined_df.write_parquet(target_parquet)
        logger.info(f"✅ DES SIMBAD data saved: {target_parquet}")
        return [target_parquet]

    def _try_large_simbad_galaxies(self) -> pl.DataFrame:
        """Download large galaxy dataset from SIMBAD."""
        try:
            from astroquery.simbad import Simbad

            logger.info("📡 Querying large SIMBAD dataset...")

            # Define multiple regions to cover larger area
            regions = [
                # DES footprint regions
                (56.75, -24.1167, 5.0),  # Main DES region
                (60.0, -25.0, 3.0),  # Extended region 1
                (55.0, -20.0, 3.0),  # Extended region 2
                (50.0, -30.0, 3.0),  # Extended region 3
                (65.0, -30.0, 3.0),  # Extended region 4
            ]

            all_results = []

            for ra, dec, radius in regions:
                coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs")
                radius_deg = radius * u.deg

                logger.info(
                    f"📡 Querying SIMBAD region: RA={ra}, Dec={dec}, Radius={radius}°"
                )

                # Configure SIMBAD for galaxies
                Simbad.add_votable_fields(
                    "otype", "z_value", "flux(V)", "flux(B)", "flux(R)", "flux(I)"
                )

                # Query for galaxies in the region
                result = Simbad.query_region(coord, radius=radius_deg)

                if result and len(result) > 0:
                    # Check if OTYPE column exists
                    if "OTYPE" in result.colnames:
                        # Filter for galaxies and objects with redshift
                        galaxy_mask = []
                        for i, obj_type in enumerate(result["OTYPE"]):
                            is_galaxy = obj_type.startswith("G") if obj_type else False
                            has_redshift = result["Z_VALUE"][i] is not None
                            galaxy_mask.append(is_galaxy and has_redshift)

                        filtered_result = result[galaxy_mask]
                    else:
                        # If no OTYPE, use all results with redshift
                        redshift_mask = [
                            result["Z_VALUE"][i] is not None for i in range(len(result))
                        ]
                        filtered_result = result[redshift_mask]

                    if len(filtered_result) > 0:
                        # Convert to Polars DataFrame
                        df = pl.from_pandas(filtered_result.to_pandas())

                        # Add region identifier
                        df = df.with_columns(
                            pl.lit(f"region_{ra}_{dec}").alias("source_region")
                        )
                        all_results.append(df)

                        logger.info(
                            f"✅ Added {len(filtered_result)} galaxies from region {ra}, {dec}"
                        )

            if all_results:
                combined_df = pl.concat(all_results, how="vertical")

                logger.info(f"✅ Total SIMBAD galaxies: {len(combined_df)}")
                return combined_df

            return pl.DataFrame()  # Return empty DataFrame instead of None

        except Exception as e:
            logger.error(f"❌ Large SIMBAD download failed: {e}")
            return pl.DataFrame()  # Return empty DataFrame instead of raising

    def _try_large_vizier_catalogs(self) -> pl.DataFrame:
        """Download large dataset from Vizier catalogs."""
        try:
            from astroquery.vizier import Vizier

            logger.info("📡 Querying large Vizier dataset...")

            # Large catalogs for cosmic web analysis
            catalogs = [
                ("II/349/ps1", "Pan-STARRS1 DR1"),  # Large optical survey
                ("II/328/allwise", "AllWISE Source Catalog"),  # Infrared survey
                ("II/246/out", "2MASS All-Sky Point Source Catalog"),  # Near-infrared
                ("II/311/wise", "WISE All-Sky Data Release"),  # Wide-field infrared
            ]

            all_results = []

            for catalog_id, catalog_name in catalogs:
                try:
                    logger.info(f"📡 Querying {catalog_name}...")

                    # Query large regions
                    regions = [
                        (56.75, -24.1167, 10.0),  # Large DES region
                        (60.0, -25.0, 8.0),  # Extended region
                        (50.0, -30.0, 8.0),  # Another region
                    ]

                    for ra, dec, radius in regions:
                        coord = SkyCoord(
                            ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs"
                        )
                        radius_deg = radius * u.deg

                        Vizier.ROW_LIMIT = 50000  # Large limit
                        result = Vizier.query_region(
                            coord, radius=radius_deg, catalog=catalog_id
                        )

                        if result and len(result) > 0:
                            # Convert to Polars DataFrame
                            df = pl.from_pandas(result[0].to_pandas())

                            # Add source information
                            df = df.with_columns(
                                [
                                    pl.lit(catalog_name).alias("source_catalog"),
                                    pl.lit(f"region_{ra}_{dec}").alias("source_region"),
                                ]
                            )

                            all_results.append(df)
                            logger.info(
                                f"✅ Added {len(df)} objects from {catalog_name} region {ra}, {dec}"
                            )

                except Exception as e:
                    logger.warning(f"⚠️ Catalog {catalog_name} failed: {e}")
                    continue

            if all_results:
                combined_df = pl.concat(all_results, how="vertical")

                logger.info(f"✅ Total Vizier objects: {len(combined_df)}")
                return combined_df

            return pl.DataFrame()  # Return empty DataFrame instead of None

        except Exception as e:
            logger.error(f"❌ Large Vizier download failed: {e}")
            raise

    def _try_large_esasky_catalogs(self) -> pl.DataFrame:
        """Download large dataset from ESASky catalogs."""
        try:
            from astroquery.esasky import ESASky

            logger.info("📡 Querying large ESASky dataset...")

            # ESASky catalogs
            catalogs = ["GAIA", "2MASS", "WISE", "PANSTARRS"]

            all_results = []

            for catalog in catalogs:
                try:
                    logger.info(f"📡 Querying ESASky {catalog}...")

                    # Query large regions
                    regions = [
                        (56.75, -24.1167, 8.0),
                        (60.0, -25.0, 6.0),
                        (50.0, -30.0, 6.0),
                    ]

                    for ra, dec, radius in regions:
                        coord = SkyCoord(
                            ra=ra, dec=dec, unit=(u.deg, u.deg), frame="icrs"
                        )
                        radius_deg = radius * u.deg

                        result = ESASky.query_region_catalogs(
                            coord, radius=radius_deg, catalogs=catalog
                        )

                        if result and len(result) > 0:
                            # Convert to Polars DataFrame
                            df = pl.from_pandas(result[0].to_pandas())

                            # Add source information
                            df = df.with_columns(
                                [
                                    pl.lit(f"ESASky_{catalog}").alias("source_catalog"),
                                    pl.lit(f"region_{ra}_{dec}").alias("source_region"),
                                ]
                            )

                            all_results.append(df)
                            logger.info(
                                f"✅ Added {len(df)} objects from ESASky {catalog} region {ra}, {dec}"
                            )

                except Exception as e:
                    logger.warning(f"⚠️ ESASky catalog {catalog} failed: {e}")
                    continue

            if all_results:
                combined_df = pl.concat(all_results, how="vertical")

                logger.info(f"✅ Total ESASky objects: {len(combined_df)}")
                return combined_df

            return pl.DataFrame()  # Return empty DataFrame instead of None

        except Exception as e:
            logger.error(f"❌ Large ESASky download failed: {e}")
            raise
