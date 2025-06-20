"""
AstroLab Data Manager
====================

Modern astronomical data management with proper raw/processed structure:
- Raw data: Original downloads from surveys (Gaia, SDSS, etc.)
- Processed data: Cleaned, filtered, and ML-ready datasets
- HDF5 support for large spectroscopic datasets
- Parquet for efficient columnar storage
"""

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import polars as pl
from astroquery.gaia import Gaia

from .config import DataConfig, data_config


class AstroDataManager:
    """Modern astronomical data management with structured storage."""

    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.config = DataConfig(base_dir)
        self.base_dir = self.config.base_dir
        self.setup_directories()

    def setup_directories(self):
        """Create standardized data directory structure using new config."""
        # Use new clean structure from config
        self.config.setup_directories()

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
            from astropy.io import fits
            from astropy.table import Table

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
            import h5py

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
        output_name: Optional[str] = None,
        filters: Optional[Dict] = None,
    ) -> Path:
        """Process raw catalog for ML training."""

        raw_file = Path(raw_file)

        if output_name is None:
            output_name = f"ml_{raw_file.stem}"

        output_file = self.processed_dir / "ml_ready" / f"{output_name}.parquet"

        print(f"🔄 Processing {raw_file.name} for ML...")

        # Load raw data
        df = pl.read_parquet(raw_file)

        # Apply filters
        if filters:
            for col, (min_val, max_val) in filters.items():
                if col in df.columns:
                    df = df.filter(pl.col(col).is_between(min_val, max_val))

        # Remove rows with missing critical data
        critical_cols = ["ra", "dec"]
        if "phot_g_mean_mag" in df.columns:
            critical_cols.append("phot_g_mean_mag")
        elif "psfMag_r" in df.columns:
            critical_cols.append("psfMag_r")

        df = df.drop_nulls(subset=critical_cols)

        # Normalize coordinates
        df = df.with_columns(
            [
                (pl.col("ra") / 360.0).alias("ra_norm"),
                ((pl.col("dec") + 90) / 180.0).alias("dec_norm"),
            ]
        )

        # Save processed data
        df.write_parquet(output_file, compression="zstd")

        # Save processing metadata
        import datetime

        metadata = {
            "source_file": str(raw_file),
            "processing_date": datetime.datetime.now().isoformat(),
            "filters_applied": filters or {},
            "n_sources_input": len(pl.read_parquet(raw_file)),
            "n_sources_output": len(df),
            "columns": df.columns,
        }

        metadata_file = output_file.with_suffix(".json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Processed {len(df):,} sources for ML")
        return output_file

    def list_catalogs(self, data_type: str = "all") -> pl.DataFrame:
        """List available catalogs."""

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
            for parquet_file in self.processed_dir.rglob("*.parquet"):
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
        """Load catalog from file (supports .parquet, .csv, .fits)."""
        catalog_path = Path(catalog_path)

        if not catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {catalog_path}")

        print(f"📂 Loading: {catalog_path.name}")

        # Handle different file formats
        suffix = catalog_path.suffix.lower()

        if suffix == ".parquet":
            df = pl.read_parquet(catalog_path)
        elif suffix == ".csv":
            df = pl.read_csv(catalog_path)
        elif suffix == ".fits":
            # Use the optimized FITS loader from utils
            from .utils import load_fits_table_optimized

            df = load_fits_table_optimized(catalog_path, as_polars=True)
            if df is None:
                raise ValueError(f"Failed to load FITS file: {catalog_path}")
        else:
            raise ValueError(
                f"Unsupported file format: {suffix}. Supported: .parquet, .csv, .fits"
            )

        # Load metadata if available
        metadata_file = catalog_path.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
                print(f"📊 Source: {metadata.get('source', 'Unknown')}")
                print(f"📊 Objects: {len(df):,}")
                print(f"📊 Columns: {len(df.columns)}")
        else:
            print(f"📊 Objects: {len(df):,}, Columns: {len(df.columns)}")

        return df

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


# Global data manager instance
data_manager = AstroDataManager()


# Convenience functions
def download_gaia(region: str = "lmc", magnitude_limit: float = 15.0) -> Path:
    """Download Gaia catalog."""
    return data_manager.download_gaia_catalog(magnitude_limit, region)


def download_bright_all_sky(magnitude_limit: float = 12.0) -> Path:
    """Download bright all-sky Gaia catalog (~1 GB)."""
    return data_manager.download_gaia_catalog(magnitude_limit, "bright_all_sky")


def import_fits(fits_file: Union[str, Path], catalog_name: str) -> Path:
    """Import FITS catalog."""
    return data_manager.import_fits_catalog(fits_file, catalog_name)


def import_tng50(hdf5_file: Union[str, Path], dataset_name: str = "PartType0") -> Path:
    """Import TNG50 simulation data."""
    return data_manager.import_tng50_hdf5(hdf5_file, dataset_name)


def list_catalogs() -> pl.DataFrame:
    """List all available catalogs."""
    return data_manager.list_catalogs()


def load_catalog(path: Union[str, Path]) -> pl.DataFrame:
    """Load catalog from path."""
    return data_manager.load_catalog(path)


def process_for_ml(raw_file: Union[str, Path], **kwargs) -> Path:
    """Process raw catalog for ML."""
    return data_manager.process_for_ml(raw_file, **kwargs)


def load_gaia_bright_stars(magnitude_limit: float = 12.0) -> pl.DataFrame:
    """
    Load bright Gaia DR3 stars from our real catalogs.

    Parameters
    ----------
    magnitude_limit : float, default 12.0
        Magnitude limit (10.0 or 12.0 available)

    Returns
    -------
    pl.DataFrame
        Gaia star catalog
    """
    try:
        manager = AstroDataManager()

        # Choose appropriate catalog file
        if magnitude_limit <= 10.0:
            catalog_file = Path("data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet")
        else:
            catalog_file = Path("data/raw/gaia/gaia_dr3_bright_all_sky_mag12.0.parquet")

        if catalog_file.exists():
            print(f"📊 Loading {catalog_file.name}...")
            return manager.load_catalog(catalog_file)
        else:
            print(f"❌ Catalog file not found: {catalog_file}")
            return pl.DataFrame()

    except Exception as e:
        print(f"❌ Error loading Gaia catalog: {e}")
        return pl.DataFrame()


def load_bright_stars(limit: Optional[int] = None) -> pl.DataFrame:
    """Load bright stars (alias for load_gaia_bright_stars)"""
    data = load_gaia_bright_stars(12.0)
    if limit and not data.is_empty():
        return data.head(limit)
    return data


__all__ = [
    "AstroDataManager",
    "data_manager",
    "download_gaia",
    "download_bright_all_sky",
    "import_fits",
    "import_tng50",
    "list_catalogs",
    "load_catalog",
    "process_for_ml",
    "load_gaia_bright_stars",
    "load_bright_stars",
]
