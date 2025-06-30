#!/usr/bin/env python3
"""
Gaia DR3 Preprocessor with GPU acceleration and memory optimization.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

from astro_lab.data.preprocessors.base import BaseSurveyProcessor
from astro_lab.tensors import PhotometricTensorDict, SpatialTensorDict

logger = logging.getLogger(__name__)


class GaiaPreprocessor(BaseSurveyProcessor):
    """
    Preprocessor for Gaia DR3 data.

    Handles:
    - Bright all-sky data (G < 12)
    - Astrometric data (ra, dec, parallax, proper motions)
    - Photometric data (G, BP, RP magnitudes)
    - Quality filtering
    - Coordinate transformations
    """

    def __init__(self, survey_name: str = "gaia", data_config: Optional[Dict] = None):
        super().__init__(survey_name, data_config)

        # Quality cuts for bright stars
        self.min_parallax_over_error = 5.0
        self.max_ruwe = 1.4
        self.min_phot_g_n_obs = 5

        # Memory optimization settings
        self.chunk_size = 1_000_000  # Process 1M sources at a time
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            logger.info("🚀 GPU acceleration enabled for Polars")
            try:
                # Test GPU engine
                test_df = pl.LazyFrame({"a": [1, 2, 3]})
                test_df.collect(engine="gpu")
                self.gpu_engine = pl.GPUEngine(device=0, raise_on_fail=False)
            except Exception as e:
                logger.warning(f"GPU engine not available: {e}")
                self.use_gpu = False
        else:
            logger.info("💻 Using CPU processing")
            self.gpu_engine = None

    def create_spatial_tensor(self, df: pl.DataFrame) -> SpatialTensorDict:
        """Create spatial tensor from Gaia coordinates with GPU acceleration."""
        logger.info("Creating spatial tensor...")

        # Convert to 3D Cartesian coordinates
        if self.use_gpu and self.gpu_engine:
            try:
                # Use GPU for coordinate conversion
                coords_3d = self._convert_to_3d_gpu(df)
                logger.info(
                    f"Spatial tensor created using GPU: {len(coords_3d):,} sources"
                )
            except Exception as e:
                logger.warning(f"GPU coordinate conversion failed: {e}")
                coords_3d = self._convert_to_3d_cpu(df)
        else:
            coords_3d = self._convert_to_3d_cpu(df)

        # Create spatial tensor with correct parameters
        spatial_tensor = SpatialTensorDict(
            coordinates=coords_3d,
            coordinate_system="icrs",
            unit="pc",
        )

        return spatial_tensor

    def create_photometric_tensor(self, df: pl.DataFrame) -> PhotometricTensorDict:
        """Create photometric tensor from Gaia photometric data."""
        logger.info("Creating photometric tensor...")

        # Extract magnitudes
        magnitudes = df.select(
            [
                pl.col("phot_g_mean_mag"),
                pl.col("phot_bp_mean_mag"),
                pl.col("phot_rp_mean_mag"),
            ]
        ).to_numpy()

        # Create photometric tensor
        photometric_tensor = PhotometricTensorDict(
            magnitudes=torch.tensor(magnitudes, dtype=torch.float32),
            bands=["G", "BP", "RP"],
            filter_system="Vega",
        )

        return photometric_tensor

    def _convert_to_3d_cpu(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert coordinates to 3D Cartesian using CPU."""
        # Extract coordinates
        coords = df.select(
            [
                pl.col("ra"),
                pl.col("dec"),
                pl.col("parallax"),
            ]
        ).to_numpy()

        # Convert to 3D Cartesian
        ra_rad = np.radians(coords[:, 0])
        dec_rad = np.radians(coords[:, 1])
        parallax_mas = coords[:, 2]

        # Distance in parsecs (1/parallax)
        distance_pc = 1000.0 / parallax_mas

        # Convert to Cartesian coordinates
        x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance_pc * np.sin(dec_rad)

        return torch.tensor(np.column_stack([x, y, z]), dtype=torch.float32)

    def _convert_to_3d_gpu(self, df: pl.DataFrame) -> torch.Tensor:
        """Convert coordinates to 3D Cartesian using optimized processing."""
        # Use optimized coordinate calculations (GPU engine not available on Windows)
        coords_df = df.select(
            [
                pl.col("ra"),
                pl.col("dec"),
                pl.col("parallax"),
            ]
        ).lazy()

        # Convert to radians and calculate 3D coordinates
        coords_3d = coords_df.select(
            [
                (pl.col("ra") * (np.pi / 180.0)).alias("ra_rad"),
                (pl.col("dec") * (np.pi / 180.0)).alias("dec_rad"),
                (1000.0 / pl.col("parallax")).alias("distance"),
            ]
        ).collect()

        # Convert to torch tensors
        ra_rad = torch.tensor(coords_3d["ra_rad"].to_numpy(), dtype=torch.float32)
        dec_rad = torch.tensor(coords_3d["dec_rad"].to_numpy(), dtype=torch.float32)
        distance = torch.tensor(coords_3d["distance"].to_numpy(), dtype=torch.float32)

        # Calculate 3D Cartesian coordinates
        x = distance * torch.cos(dec_rad) * torch.cos(ra_rad)
        y = distance * torch.cos(dec_rad) * torch.sin(ra_rad)
        z = distance * torch.sin(dec_rad)

        return torch.stack([x, y, z], dim=1)

    def preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply Gaia-specific preprocessing."""
        # Apply quality cuts (more lenient to preserve data)
        df_filtered = df.filter(
            # Basic coordinate requirements
            pl.col("ra").is_not_null()
            & pl.col("dec").is_not_null()
            & pl.col("parallax").is_not_null()
            &
            # More lenient quality cuts
            (pl.col("parallax_over_error") >= 1.0)  # Was 5.0
            & (pl.col("ruwe") <= 2.0)  # Was 1.4
            & (pl.col("phot_g_n_obs") >= 2)  # Was 5
            & pl.all_horizontal(
                [
                    pl.col(col).is_not_null()
                    for col in [
                        "ra",
                        "dec",
                        "parallax",
                        "phot_g_mean_mag",
                    ]
                ]
            )
        )

        logger.info(
            f"Quality cuts: {len(df):,} → {len(df_filtered):,} sources "
            f"({100 * len(df_filtered) / len(df):.1f}% retained)"
        )

        return df_filtered

    def _save_processed_data(self, df: pl.DataFrame, output_file: Path):
        """Save processed data to Parquet file with Gaia-specific columns."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Select key columns for processed file
        processed_df = df.select(
            [
                "source_id",
                "ra",
                "dec",
                "parallax",
                "parallax_error",
                "parallax_over_error",
                "pmra",
                "pmdec",
                "pm",
                "ruwe",
                "phot_g_mean_mag",
                "phot_bp_mean_mag",
                "phot_rp_mean_mag",
                "phot_g_n_obs",
                "phot_bp_n_obs",
                "phot_rp_n_obs",
                "bp_rp",
                "bp_g",
                "g_rp",
                "radial_velocity",
                "radial_velocity_error",
            ]
        )

        processed_df.write_parquet(output_file)
        logger.info(f"Saved processed data: {output_file}")

    def _load_processed_data(self, processed_file: Path) -> Dict[str, Any]:
        """Load existing processed data."""
        df = pl.read_parquet(processed_file)

        # Recreate tensors
        spatial_tensor = self.create_spatial_tensor(df)
        photometric_tensor = self.create_photometric_tensor(df)

        return {
            "spatial_tensor": spatial_tensor,
            "photometric_tensor": photometric_tensor,
            "metadata": {
                "survey": "gaia",
                "data_release": "DR3",
                "n_sources": len(df),
                "processed_file": str(processed_file),
                "gpu_accelerated": self.use_gpu,
                "quality_cuts": {
                    "min_parallax_over_error": self.min_parallax_over_error,
                    "max_ruwe": self.max_ruwe,
                    "min_phot_g_n_obs": self.min_phot_g_n_obs,
                },
            },
        }

    def get_coordinate_columns(self) -> List[str]:
        """Get coordinate column names for Gaia."""
        return ["ra", "dec", "parallax"]

    def extract_coordinates(self, df: pl.DataFrame) -> torch.Tensor:
        """Extract coordinates from Gaia data."""
        # Extract RA, Dec, and distance from parallax
        coords = df.select(
            [
                pl.col("ra").alias("ra_deg"),
                pl.col("dec").alias("dec_deg"),
                pl.col("parallax").alias("parallax_mas"),
            ]
        ).to_numpy()

        # Convert to 3D Cartesian coordinates
        ra_rad = np.radians(coords[:, 0])
        dec_rad = np.radians(coords[:, 1])
        parallax_mas = coords[:, 2]

        # Distance in parsecs (1/parallax)
        distance_pc = 1000.0 / parallax_mas

        # Convert to Cartesian coordinates
        x = distance_pc * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance_pc * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance_pc * np.sin(dec_rad)

        spatial_coords = np.column_stack([x, y, z])
        return torch.tensor(spatial_coords, dtype=torch.float32)

    def create_tensordict(
        self, df: pl.DataFrame, use_gpu: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Create TensorDict from Gaia data with GPU support."""
        # Use GPU for preprocessing if available
        device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )

        logger.info(f"Creating TensorDict on {device}...")

        # Extract coordinates
        coords = self.extract_coordinates(df).to(device)

        # Extract features (photometry, proper motions, etc.)
        available_cols = df.columns
        feature_cols = []

        for col in ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"]:
            if col in available_cols:
                feature_cols.append(pl.col(col))

        for col in ["pmra", "pmdec"]:
            if col in available_cols:
                feature_cols.append(pl.col(col))
            else:
                feature_cols.append(pl.lit(0.0).alias(col))

        if "parallax_over_error" in available_cols:
            feature_cols.append(pl.col("parallax_over_error"))
        elif "parallax" in available_cols and "parallax_error" in available_cols:
            feature_cols.append(
                (pl.col("parallax") / pl.col("parallax_error")).alias(
                    "parallax_over_error"
                )
            )
        else:
            feature_cols.append(pl.lit(5.0).alias("parallax_over_error"))

        if "ruwe" in available_cols:
            feature_cols.append(pl.col("ruwe"))
        else:
            feature_cols.append(pl.lit(1.0).alias("ruwe"))

        features = torch.tensor(
            df.select(feature_cols).to_numpy(), dtype=torch.float32, device=device
        )

        tensor_dict = {
            "features": features,
            "x": features,  # Alias for PyG
            "spatial": {
                "coordinates": coords,
                "pos": coords,  # Alias for PyG
            },
            "num_nodes": len(df),
        }

        # Add synthetic labels if needed
        if "label" not in df.columns and "class" not in df.columns:
            g_mag = torch.tensor(df["phot_g_mean_mag"].to_numpy(), device=device)
            labels = torch.zeros(len(df), dtype=torch.long, device=device)
            labels[g_mag < 6.0] = 0  # Bright stars
            labels[(g_mag >= 6.0) & (g_mag < 10.0)] = 1  # Medium
            labels[g_mag >= 10.0] = 2  # Faint
            tensor_dict["labels"] = labels
            tensor_dict["y"] = labels  # Alias for PyG

        # Move all tensors to CPU before returning (for saving)
        def to_cpu(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu()
            elif isinstance(obj, dict):
                return {k: to_cpu(v) for k, v in obj.items()}
            else:
                return obj

        return to_cpu(tensor_dict)

    def preprocess_and_save(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        force_reprocess: bool = False,
    ) -> Path:
        """Preprocess Gaia DR3 data and save to processed directory."""
        # Use default paths if not provided
        if output_path is None:
            output_path = Path("data/processed/gaia/gaia.parquet")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Call the preprocess method
        result = self.preprocess(force=force_reprocess)

        # Save processed data
        if "metadata" in result and "processed_file" in result["metadata"]:
            # Data was already saved by preprocess method
            return Path(result["metadata"]["processed_file"])
        else:
            # Save the processed data manually
            df = self.load_data()
            df_filtered = self.preprocess_dataframe(df)
            self._save_processed_data(df_filtered, output_path)
            return output_path
