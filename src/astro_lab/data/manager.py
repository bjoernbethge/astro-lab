"""
AstroLab Data Manager - Data Management
======================================

Data management system for astronomical datasets.
Handles data loading, processing, and catalog management.
Updated for TensorDict architecture.
"""

import datetime
import gc
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import polars as pl
import psutil
import torch
from astropy.io import fits
from astropy.table import Table
from astroquery.gaia import Gaia
from pydantic import BaseModel, Field

# Use TensorDict classes instead of old tensor classes
from ..tensors import (
    ClusteringTensorDict,
    FeatureTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SimulationTensorDict,
    SpatialTensorDict,
    StatisticsTensorDict,
    SurveyTensorDict,
)

# Removed memory.py - using simple gc instead
from .config import DataConfig, data_config
from .core import AstroDataset
from .preprocessing import preprocess_catalog, preprocess_catalog_lazy
from .utils import load_fits_optimized

logger = logging.getLogger(__name__)


class AstroDataManager:
    """
    data manager with comprehensive memory management.

    Handles loading, preprocessing, and managing astronomical survey data
    with automatic memory cleanup and optimization.
    """

    def __init__(self, config: Optional[Union[str, DataConfig]] = None, **kwargs):
        """Initialize data manager with memory management."""
        # Load configuration
        if isinstance(config, str):
            # Simple config loading without from_yaml
            self.config = DataConfig(config)
        elif isinstance(config, DataConfig):
            self.config = config
        else:
            self.config = DataConfig(**kwargs)

        # Set up paths
        self.data_dir = Path(self.config.base_dir)
        self._processed_dir = self.data_dir / "processed"
        self._processed_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📂 Data manager initialized: {self.data_dir}")

    def setup_directories(self):
        """Create standardized data directory structure using new config."""
        # Use new clean structure from config
        self.config.setup_directories()

    @property
    def base_dir(self) -> Path:
        """Get base data directory."""
        return self.data_dir

    @property
    def raw_dir(self) -> Path:
        return self.config.raw_dir

    @property
    def processed_dir(self) -> Path:
        return self.config.processed_dir

    @property
    def cache_dir(self) -> Path:
        return self.config.cache_dir

    def download_gaia_catalog(
        self,
        magnitude_limit: float = 15.0,
        region: str = "lmc",
        max_sources: int = 1000000,
    ) -> Path:
        """Download Gaia DR3 catalog to raw storage."""

        # astroquery is now a required dependency

        # WARNING: All-sky catalog is MASSIVE
        if region == "all_sky":
            estimated_size_tb = 10.0  # ~10 TB for complete catalog
            print(f"⚠️  WARNING: All-sky Gaia DR3 catalog is ~{estimated_size_tb} TB!")
            print("   This would take days to download and require massive storage.")
            print("   Consider using regional catalogs instead.")

            response = input("Continue with all-sky download? (yes/no): ")
            if response.lower() != "yes":
                raise ValueError("All-sky download cancelled by user")

        # Ensure gaia directories exist
        self.config.ensure_survey_directories("gaia")

        output_file = (
            self.config.get_survey_raw_dir("gaia")
            / f"gaia_dr3_{region}_mag{magnitude_limit:.1f}.parquet"
        )

        if output_file.exists():
            print(f"📂 Gaia catalog exists: {output_file.name}")
            return output_file

        print(f"🌟 Downloading Gaia DR3: {region}, G < {magnitude_limit}")

        # Region-specific queries
        region_queries = {
            "lmc": "l BETWEEN 270 AND 290 AND b BETWEEN -35 AND -15",
            "smc": "l BETWEEN 295 AND 315 AND b BETWEEN -50 AND -35",
            "galactic_plane": "ABS(b) < 10",
            "galactic_poles": "ABS(b) > 60",
            "all_sky": "",  # No spatial filter = entire sky
            "bright_all_sky": "",  # Bright stars from entire sky (G < 12)
        }

        # Special handling for bright all-sky catalog
        if region == "bright_all_sky":
            if magnitude_limit > 12.0:
                print(
                    f"⚠️  Adjusting magnitude limit from {magnitude_limit} to 12.0 for bright all-sky catalog"
                )
                magnitude_limit = 12.0
            print(
                f"🌟 Downloading bright stars from entire sky (G < {magnitude_limit})"
            )
            print("   Expected: ~10-15 million stars, ~1 GB download")

        region_clause = (
            f"AND {region_queries.get(region, '')}"
            if region_queries.get(region)
            else ""
        )

        query = f"""
        SELECT 
            source_id, ra, dec, l, b,
            parallax, parallax_error,
            pmra, pmdec,
            phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
            bp_rp, g_rp,
            teff_gspphot, logg_gspphot
        FROM gaiadr3.gaia_source 
        WHERE phot_g_mean_mag < {magnitude_limit}
        AND parallax IS NOT NULL 
        AND parallax > 0
        {region_clause}
        """

        try:
            job = Gaia.launch_job_async(query)
            results = job.get_results()

            if results is None or len(results) == 0:
                raise ValueError("No Gaia data returned")

            # Convert to Polars with proper types
            df = pl.DataFrame({col: np.array(results[col]) for col in results.colnames})

            # Add derived columns with proper units (step by step to avoid dependency issues)
            # First: Distance in parsecs (1000/parallax)
            df = df.with_columns((1000.0 / pl.col("parallax")).alias("distance_pc"))

            # Second: Add other derived columns that depend on distance_pc
            df = df.with_columns(
                [
                    # Absolute magnitude: M = m + 5*log10(parallax) - 10
                    (
                        pl.col("phot_g_mean_mag") + 5 * pl.col("parallax").log10() - 10
                    ).alias("abs_g_mag"),
                    # Proper motion in mas/yr -> km/s (4.74 * mu * d)
                    (4.74 * pl.col("pmra") * pl.col("distance_pc") / 1000.0).alias(
                        "vra_km_s"
                    ),
                    (4.74 * pl.col("pmdec") * pl.col("distance_pc") / 1000.0).alias(
                        "vdec_km_s"
                    ),
                    # Total proper motion
                    (pl.col("pmra").pow(2) + pl.col("pmdec").pow(2))
                    .sqrt()
                    .alias("pm_total"),
                ]
            )

            # Save as compressed Parquet
            df.write_parquet(output_file, compression="zstd")

            # Save metadata
            metadata = {
                "source": "Gaia DR3",
                "region": region,
                "magnitude_limit": magnitude_limit,
                "n_sources": len(df),
                "columns": df.columns,
                "file_size_mb": output_file.stat().st_size / 1024**2,
            }

            metadata_file = output_file.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"✅ Downloaded {len(df):,} Gaia sources ({metadata['file_size_mb']:.1f} MB)"
            )
            return output_file

        except Exception as e:
            print(f"❌ Gaia download failed: {e}")
            raise

    def import_fits_catalog(
        self,
        fits_file: Union[str, Path],
        catalog_name: str,
        hdu_index: int = 1,
    ) -> Path:
        """Import FITS catalog to raw storage."""

        fits_file = Path(fits_file)
        if not fits_file.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_file}")

        # Ensure sdss directories exist (fits -> sdss)
        self.config.ensure_survey_directories("sdss")

        output_file = self.config.get_survey_raw_dir("sdss") / f"{catalog_name}.parquet"

        if output_file.exists():
            print(f"📂 FITS catalog exists: {output_file.name}")
            return output_file

        print(f"📥 Importing FITS: {fits_file.name} -> {catalog_name}")

        try:
            # Read FITS table
            with fits.open(fits_file) as hdul:
                table = Table(hdul[hdu_index].data)  # type: ignore

            # Convert to Polars
            df = pl.from_pandas(table.to_pandas())

            # Save as compressed Parquet
            df.write_parquet(output_file, compression="zstd")

            # Save metadata
            metadata = {
                "source": f"FITS: {fits_file.name}",
                "catalog_name": catalog_name,
                "hdu_index": hdu_index,
                "n_sources": len(df),
                "columns": df.columns,
                "file_size_mb": output_file.stat().st_size / 1024**2,
                "original_size_mb": fits_file.stat().st_size / 1024**2,
            }

            metadata_file = output_file.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

            print(
                f"✅ Imported {len(df):,} objects ({metadata['file_size_mb']:.1f} MB)"
            )
            return output_file

        except Exception as e:
            print(f"❌ FITS import failed: {e}")
            raise

    def import_tng50_hdf5(
        self,
        hdf5_file: Union[str, Path],
        dataset_name: str = "PartType0",
        max_particles: int = 1000000,
    ) -> Path:
        """Import TNG50 simulation data from HDF5."""

        hdf5_file = Path(hdf5_file)
        if not hdf5_file.exists():
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_file}")

        output_file = self.raw_dir / "tng50" / f"tng50_{dataset_name.lower()}.parquet"
        output_file.parent.mkdir(exist_ok=True)

        if output_file.exists():
            print(f"📂 TNG50 catalog exists: {output_file.name}")
            return output_file

        print(f"🌌 Importing TNG50: {hdf5_file.name} -> {dataset_name}")

        try:
            with h5py.File(hdf5_file, "r") as f:  # type: ignore
                # Get particle data
                if dataset_name not in f:  # type: ignore
                    available = list(f.keys())  # type: ignore
                    raise ValueError(
                        f"Dataset {dataset_name} not found. Available: {available}"
                    )

                group = f[dataset_name]  # type: ignore

                # Common TNG50 fields
                data_dict = {}

                # Coordinates (essential)
                if "Coordinates" in group:  # type: ignore
                    coords = np.array(group["Coordinates"][:])  # type: ignore
                    if len(coords) > max_particles:  # type: ignore
                        indices = np.random.choice(  # type: ignore
                            len(coords),  # type: ignore
                            max_particles,
                            replace=False,
                        )
                        coords = coords[indices]  # type: ignore
                    else:
                        indices = slice(None)

                    data_dict["x"] = coords[:, 0]  # type: ignore
                    data_dict["y"] = coords[:, 1]  # type: ignore
                    data_dict["z"] = coords[:, 2]  # type: ignore

                # Other common fields
                for field in [
                    "Masses",
                    "Velocities",
                    "Density",
                    "Temperature",
                    "Metallicity",
                ]:
                    if field in group:  # type: ignore
                        data = np.array(group[field][:])  # type: ignore
                        if indices is not slice(None):
                            data = data[indices]  # type: ignore

                        # Sanitize field name for column
                        col_name = field.lower()
                        if col_name.endswith("es"):
                            col_name = col_name[:-2]
                        elif col_name.endswith("s"):
                            col_name = col_name[:-1]

                        if data.ndim > 1:  # type: ignore
                            # Vector quantities
                            for i in range(data.shape[1]):  # type: ignore
                                data_dict[f"{col_name}_{i}"] = data[:, i]  # type: ignore
                        else:
                            data_dict[col_name] = data  # type: ignore

                # Convert to Polars
                df = pl.DataFrame(data_dict)

                # Save as compressed Parquet
                df.write_parquet(output_file, compression="zstd")

                # Save metadata
                metadata = {
                    "source": f"TNG50: {hdf5_file.name}",
                    "dataset_name": dataset_name,
                    "n_particles": len(df),
                    "max_particles": max_particles,
                    "columns": df.columns,
                    "file_size_mb": output_file.stat().st_size / 1024**2,
                    "original_size_mb": hdf5_file.stat().st_size / 1024**2,
                }

                metadata_file = output_file.with_suffix(".json")
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=2)

                print(
                    f"✅ Imported {len(df):,} particles ({metadata['file_size_mb']:.1f} MB)"
                )
                return output_file

        except Exception as e:
            print(f"❌ TNG50 import failed: {e}")
            raise

    def process_for_ml(
        self,
        raw_file: Union[str, Path],
        survey: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> Path:
        """Process raw catalog for ML training and save as {survey}.parquet in processed/{survey}/"""
        raw_file = Path(raw_file)
        if survey is None:
            # Survey must be specified explicitly!
            raise ValueError("survey must be specified for ML-Processing")
        processed_dir = self.processed_dir / survey
        processed_dir.mkdir(parents=True, exist_ok=True)
        output_path = processed_dir / f"{survey}.parquet"
        print(f"\U0001f504 Processing {raw_file.name} for ML as {output_path} ...")
        df = pl.read_parquet(raw_file)
        if filters:
            for col, (min_val, max_val) in filters.items():
                if col in df.columns:
                    df = df.filter(pl.col(col).is_between(min_val, max_val))
        critical_cols = ["ra", "dec"]
        if "phot_g_mean_mag" in df.columns:
            critical_cols.append("phot_g_mean_mag")
        elif "psfMag_r" in df.columns:
            critical_cols.append("psfMag_r")
        df = df.drop_nulls(subset=critical_cols)
        df = df.with_columns(
            [
                (pl.col("ra") / 360.0).alias("ra_norm"),
                ((pl.col("dec") + 90) / 180.0).alias("dec_norm"),
            ]
        )
        df.write_parquet(output_path)
        metadata = {
            "source_file": str(raw_file),
            "processing_date": datetime.datetime.now().isoformat(),
            "filters_applied": filters or {},
            "n_sources_input": len(pl.read_parquet(raw_file)),
            "n_sources_output": len(df),
            "columns": df.columns,
        }
        metadata_file = output_path.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"\U00002705 Processed {len(df):,} sources for ML: {output_path}")
        return output_path

    def list_catalogs(self, data_type: str = "all") -> pl.DataFrame:
        """List available catalogs, only survey-based for processed."""
        catalogs = []
        if data_type in ["all", "raw"]:
            for parquet_file in self.raw_dir.rglob("*.parquet"):
                metadata_file = parquet_file.with_suffix(".json")
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                else:
                    metadata = {"source": "Unknown", "n_sources": 0}
                catalogs.append(
                    {
                        "name": parquet_file.name,
                        "type": "raw",
                        "source": metadata.get("source", "Unknown"),
                        "n_sources": metadata.get("n_sources", 0),
                        "size_mb": parquet_file.stat().st_size / 1024**2,
                        "path": str(parquet_file),
                    }
                )
        if data_type in ["all", "processed"]:
            # Only show survey-based
            for survey_dir in (self.processed_dir).iterdir():
                if survey_dir.is_dir():
                    parquet_file = survey_dir / f"{survey_dir.name}.parquet"
                    if parquet_file.exists():
                        metadata_file = parquet_file.with_suffix(".json")
                        if metadata_file.exists():
                            with open(metadata_file) as f:
                                metadata = json.load(f)
                        else:
                            metadata = {"source": "Unknown", "n_sources": 0}
                        catalogs.append(
                            {
                                "name": parquet_file.name,
                                "type": "processed",
                                "source": metadata.get("source_file", "Unknown"),
                                "n_sources": metadata.get("n_sources_output", 0),
                                "size_mb": parquet_file.stat().st_size / 1024**2,
                                "path": str(parquet_file),
                            }
                        )
        if not catalogs:
            return pl.DataFrame(
                {
                    "name": [],
                    "type": [],
                    "source": [],
                    "n_sources": [],
                    "size_mb": [],
                    "path": [],
                }
            )
        return pl.DataFrame(catalogs).sort("size_mb", descending=True)

    def load_catalog(self, catalog_path: Union[str, Path]) -> pl.DataFrame:
        """Load catalog with memory management."""
        catalog_path = Path(catalog_path)

        logger.info(f"📂 Loading catalog: {catalog_path}")

        if catalog_path.suffix.lower() in [".fits", ".fit"]:
            data = load_fits_optimized(catalog_path)
            # Ensure we return a Polars DataFrame
            if data is None:
                raise ValueError(f"Could not load FITS file: {catalog_path}")
            if isinstance(data, pl.DataFrame):
                return data
            else:
                # Convert to Polars DataFrame
                import pandas as pd

                # Handle different data types
                if hasattr(data, "__array__"):
                    # numpy arrays and similar
                    df_pandas = pd.DataFrame(data)
                elif hasattr(data, "to_pandas"):
                    # astropy tables and similar
                    df_pandas = data.to_pandas()
                else:
                    # fallback
                    df_pandas = pd.DataFrame([data])
                return pl.from_pandas(df_pandas)
        elif catalog_path.suffix.lower() in [".parquet", ".pq"]:
            data = pl.read_parquet(catalog_path)
        elif catalog_path.suffix.lower() == ".csv":
            data = pl.read_csv(catalog_path)
        else:
            raise ValueError(f"Unsupported format: {catalog_path.suffix}")

        logger.info(f"✅ Catalog loaded: {len(data)} rows, {len(data.columns)} columns")
        return data

    def convert_to_physical_units(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Convert Gaia catalog to physical units using astropy.

        Parameters
        ----------
        df : pl.DataFrame
            Gaia catalog with parallax, proper motions, etc.

        Returns
        -------
        pl.DataFrame
            Catalog with additional physical unit columns
        """
        # astropy is now a required dependency

        # Add physical unit conversions using astropy constants
        df_with_units = df.with_columns(
            [
                # Distance: parallax (mas) -> distance (pc)
                # Using astropy: 1 / (parallax * u.mas).to(u.arcsec) * u.pc
                (1000.0 / pl.col("parallax")).alias("distance_pc"),
                # Absolute magnitude: M = m + 5*log10(π) - 10
                (pl.col("phot_g_mean_mag") + 5 * pl.col("parallax").log10() - 10).alias(
                    "abs_g_mag"
                ),
                # Tangential velocity using astropy conversion factor
                # v_tan = 4.74047 * μ * d (exact astropy constant)
                (
                    4.74047 * pl.col("pmra") * (1000.0 / pl.col("parallax")) / 1000.0
                ).alias("vra_km_s"),
                (
                    4.74047 * pl.col("pmdec") * (1000.0 / pl.col("parallax")) / 1000.0
                ).alias("vdec_km_s"),
                # Total proper motion (mas/yr) and tangential velocity (km/s)
                (pl.col("pmra").pow(2) + pl.col("pmdec").pow(2))
                .sqrt()
                .alias("pm_total_mas_yr"),
                (
                    4.74047
                    * (pl.col("pmra").pow(2) + pl.col("pmdec").pow(2)).sqrt()
                    * (1000.0 / pl.col("parallax"))
                    / 1000.0
                ).alias("v_tan_km_s"),
                # Galactic coordinates with explicit units
                pl.col("l").alias("gal_lon_deg"),
                pl.col("b").alias("gal_lat_deg"),
            ]
        )

        print("✅ Converted to physical units using astropy constants")

        # Add physical unit conversions
        df_with_units = df.with_columns(
            [
                # Distance: parallax (mas) -> distance (pc)
                (1000.0 / pl.col("parallax")).alias("distance_pc"),
                # Absolute magnitude: apparent + distance modulus
                (pl.col("phot_g_mean_mag") + 5 * pl.col("parallax").log10() - 10).alias(
                    "abs_g_mag"
                ),
                # Tangential velocity: proper motion (mas/yr) -> velocity (km/s)
                # v_tan = 4.74 * μ * d, where μ in mas/yr, d in pc
                (4.74 * pl.col("pmra") * (1000.0 / pl.col("parallax")) / 1000.0).alias(
                    "vra_km_s"
                ),
                (4.74 * pl.col("pmdec") * (1000.0 / pl.col("parallax")) / 1000.0).alias(
                    "vdec_km_s"
                ),
                # Total proper motion and tangential velocity
                (pl.col("pmra").pow(2) + pl.col("pmdec").pow(2))
                .sqrt()
                .alias("pm_total_mas_yr"),
                # Galactic coordinates already in degrees
                pl.col("l").alias("gal_lon_deg"),
                pl.col("b").alias("gal_lat_deg"),
            ]
        )

        return df_with_units

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single file with memory management.

        Args:
            file_path: Path to the file to process

        Returns:
            Processing results
        """
        file_path = Path(file_path)

        logger.info(f"📂 Processing file: {file_path}")

        # Load data with memory optimization
        if file_path.suffix.lower() in [".fits", ".fit"]:
            data = load_fits_optimized(file_path)
        elif file_path.suffix.lower() in [".parquet", ".pq"]:
            data = pl.read_parquet(file_path)
        elif file_path.suffix.lower() == ".csv":
            data = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        # Preprocess data
        # Detect survey type from filename or use generic
        survey_type = "generic"
        if "gaia" in file_path.name.lower():
            survey_type = "gaia"
        elif "sdss" in file_path.name.lower():
            survey_type = "sdss"
        elif "nsa" in file_path.name.lower():
            survey_type = "nsa"
        elif "linear" in file_path.name.lower():
            survey_type = "linear"
        elif "tng" in file_path.name.lower():
            survey_type = "tng50"

        lf_processed = preprocess_catalog_lazy(
            data, survey_type=survey_type, use_streaming=True
        )
        processed_data = lf_processed.collect()

        # Save processed data
        output_path = self.processed_dir / f"{file_path.stem}.parquet"
        processed_data.write_parquet(output_path)

        results = {
            "input_file": str(file_path),
            "output_file": str(output_path),
            "num_rows": len(processed_data),
            "num_columns": len(processed_data.columns),
        }

        logger.info(f"✅ File processed: {len(processed_data)} rows")
        return results

    def process_surveys(self, surveys: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process multiple surveys with batch memory management.

        Args:
            surveys: List of survey names to process

        Returns:
            Batch processing results
        """
        # Default surveys if none specified
        if surveys is None:
            surveys = ["gaia", "sdss", "nsa", "linear", "tng50"]

        logger.info(f"📊 Processing {len(surveys)} surveys")

        results = {
            "surveys_processed": [],
            "total_rows": 0,
        }

        for survey_name in surveys:
            try:
                survey_result = self._process_single_survey(survey_name)
                results["surveys_processed"].append(survey_result)
                results["total_rows"] += survey_result.get("num_rows", 0)

                logger.info(f"✅ Survey {survey_name} processed successfully")

            except Exception as e:
                logger.error(f"❌ Failed to process survey {survey_name}: {e}")
                results["surveys_processed"].append(
                    {"survey": survey_name, "error": str(e), "status": "failed"}
                )

        logger.info(
            f"📊 Batch processing completed: {results['total_rows']} total rows"
        )
        return results

    def _process_single_survey(self, survey_name: str) -> Dict[str, Any]:
        """Process a single survey with memory management."""
        # Find survey data files
        survey_files = list(self.data_dir.glob(f"*{survey_name}*"))
        if not survey_files:
            raise FileNotFoundError(f"No files found for survey: {survey_name}")

        # Process each file
        survey_results = []
        total_rows = 0

        for file_path in survey_files:
            if file_path.suffix.lower() in [".fits", ".parquet", ".csv"]:
                file_result = self.process_file(file_path)
                survey_results.append(file_result)
                total_rows += file_result.get("num_rows", 0)

        return {
            "survey": survey_name,
            "files_processed": len(survey_results),
            "num_rows": total_rows,
            "files": survey_results,
            "status": "completed",
        }

    def create_dataset(
        self, survey_name: str, force_reload: bool = False
    ) -> AstroDataset:
        """
        Create AstroDataset with memory management.

        Args:
            survey_name: Name of the survey
            force_reload: Whether to force reload data

        Returns:
            AstroDataset instance
        """
        logger.info(f"🔄 Creating dataset for survey: {survey_name}")

        dataset = AstroDataset(
            root=str(self.processed_dir),
            survey=survey_name,
        )

        logger.info(f"✅ Dataset created: {len(dataset)} samples")
        return dataset

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024

        stats = {
            "system_memory_mb": current_memory,
            "data_dir": str(self.data_dir),
            "processed_dir": str(self.processed_dir),
        }

        return stats

    def cleanup_temp_files(self):
        """Clean up temporary files with memory management."""
        temp_patterns = ["*.tmp", "*.temp", "*_temp_*"]
        cleaned_files = 0

        for pattern in temp_patterns:
            for temp_file in self.data_dir.rglob(pattern):
                try:
                    temp_file.unlink()
                    cleaned_files += 1
                except Exception as e:
                    logger.warning(f"Failed to delete {temp_file}: {e}")

        logger.info(f"🧹 Cleaned up {cleaned_files} temporary files")

    def download_survey_catalog(
        self,
        survey: str,
        magnitude_limit: float = 15.0,
        region: str = "all_sky",
        max_sources: int = 1000000,
    ) -> Path:
        """Download survey catalog to raw storage. Generic function for all surveys."""

        # Get survey configuration
        from astro_lab.utils.config.surveys import get_survey_config

        survey_config = get_survey_config(survey)

        # Ensure survey directories exist
        self.config.ensure_survey_directories(survey)

        output_file = (
            self.config.get_survey_raw_dir(survey)
            / f"{survey}_{region}_mag{magnitude_limit:.1f}.parquet"
        )

        if output_file.exists():
            print(f"📂 {survey} catalog exists: {output_file.name}")
            return output_file

        print(f"🌟 Downloading {survey}: {region}, magnitude limit < {magnitude_limit}")

        # Survey-specific download logic
        if survey.lower() == "gaia":
            return self.download_gaia_catalog(magnitude_limit, region, max_sources)
        elif survey.lower() == "sdss":
            return self._download_sdss_catalog(magnitude_limit, region, max_sources)
        elif survey.lower() == "2mass":
            return self._download_2mass_catalog(magnitude_limit, region, max_sources)
        elif survey.lower() == "wise":
            return self._download_wise_catalog(magnitude_limit, region, max_sources)
        elif survey.lower() == "pan_starrs":
            return self._download_pan_starrs_catalog(
                magnitude_limit, region, max_sources
            )
        else:
            raise ValueError(f"Download not implemented for survey: {survey}")

    def _download_sdss_catalog(
        self,
        magnitude_limit: float = 15.0,
        region: str = "all_sky",
        max_sources: int = 1000000,
    ) -> Path:
        """Download SDSS catalog."""
        # Placeholder for SDSS download
        print("⚠️ SDSS download not yet implemented")
        raise NotImplementedError("SDSS download not yet implemented")

    def _download_2mass_catalog(
        self,
        magnitude_limit: float = 15.0,
        region: str = "all_sky",
        max_sources: int = 1000000,
    ) -> Path:
        """Download 2MASS catalog."""
        # Placeholder for 2MASS download
        print("⚠️ 2MASS download not yet implemented")
        raise NotImplementedError("2MASS download not yet implemented")

    def _download_wise_catalog(
        self,
        magnitude_limit: float = 15.0,
        region: str = "all_sky",
        max_sources: int = 1000000,
    ) -> Path:
        """Download WISE catalog."""
        # Placeholder for WISE download
        print("⚠️ WISE download not yet implemented")
        raise NotImplementedError("WISE download not yet implemented")

    def _download_pan_starrs_catalog(
        self,
        magnitude_limit: float = 15.0,
        region: str = "all_sky",
        max_sources: int = 1000000,
    ) -> Path:
        """Download Pan-STARRS catalog."""
        # Placeholder for Pan-STARRS download
        print("⚠️ Pan-STARRS download not yet implemented")
        raise NotImplementedError("Pan-STARRS download not yet implemented")


# Global data manager instance
data_manager = AstroDataManager()


# Convenience functions
def download_survey(
    survey: str, region: str = "all_sky", magnitude_limit: float = 15.0
) -> Path:
    """Download any survey catalog."""
    return data_manager.download_survey_catalog(survey, magnitude_limit, region)


def download_sdss(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download SDSS catalog."""
    return data_manager.download_survey_catalog("sdss", magnitude_limit, region)


def download_2mass(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download 2MASS catalog."""
    return data_manager.download_survey_catalog("2mass", magnitude_limit, region)


def download_wise(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download WISE catalog."""
    return data_manager.download_survey_catalog("wise", magnitude_limit, region)


def download_pan_starrs(region: str = "all_sky", magnitude_limit: float = 15.0) -> Path:
    """Download Pan-STARRS catalog."""
    return data_manager.download_survey_catalog("pan_starrs", magnitude_limit, region)


def import_fits(fits_file: Union[str, Path], catalog_name: str) -> Path:
    """Import FITS catalog."""
    return data_manager.import_fits_catalog(fits_file, catalog_name)


def import_tng50(hdf5_file: Union[str, Path], dataset_name: str = "PartType0") -> Path:
    """Convenience function to import a TNG50 HDF5 file."""
    return data_manager.import_tng50_hdf5(hdf5_file, dataset_name)


def list_catalogs() -> pl.DataFrame:
    """Convenience function to list available catalogs."""
    return data_manager.list_catalogs()


def load_catalog(catalog_path: Union[str, Path]) -> pl.DataFrame:
    """Convenience function to load a catalog."""
    return data_manager.load_catalog(catalog_path)


def list_catalog_names() -> list:
    """Convenience function to list available catalog names."""
    df = data_manager.list_catalogs()
    if df.is_empty():
        return []
    return df["name"].to_list()


def process_for_ml(raw_file: Union[str, Path], **kwargs) -> Path:
    """Convenience function to process a raw file for ML."""
    return data_manager.process_for_ml(raw_file, **kwargs)


__all__ = [
    "AstroDataManager",
    "data_manager",
    "download_survey",
    "download_sdss",
    "download_2mass",
    "download_wise",
    "download_pan_starrs",
    "import_fits",
    "import_tng50",
    "list_catalogs",
    "list_catalog_names",
    "process_for_ml",
]
