"""
AstroLab Data Processors - Unified preprocessing and processing functions.

This module consolidates the core preprocessing logic from the massive
preprocessing.py file (1566 lines) into clean, focused functions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
from torch_geometric.data import Data

from ..tensors import PhotometricTensorDict, SpatialTensorDict, SurveyTensorDict
from .config import data_config

logger = logging.getLogger(__name__)


def preprocess_survey(
    survey: str,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    max_samples: Optional[int] = None,
    **kwargs,
) -> Path:
    """
    Unified survey preprocessing function.
    """
    logger.info(f"Preprocessing survey: {survey}")

    # Load raw data
    if input_path is None:
        from .loaders import load_survey_catalog

        df = load_survey_catalog(survey, max_samples=max_samples)
    else:
        from .loaders import load_catalog

        df = load_catalog(input_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, seed=42)

    # Apply survey-specific preprocessing
    df_processed = _apply_survey_preprocessing(df, survey)

    # Create output path
    if output_path is None:
        output_path = data_config.processed_dir / survey / f"{survey}_processed.parquet"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save processed data
    df_processed.write_parquet(output_path)

    logger.info(f"Preprocessed {len(df_processed)} objects -> {output_path}")
    return output_path


def _apply_survey_preprocessing(df: pl.DataFrame, survey_type: str) -> pl.DataFrame:
    """Apply survey-specific preprocessing pipeline."""
    if survey_type == "gaia":
        return _preprocess_gaia_data(df)
    elif survey_type == "sdss":
        return _preprocess_sdss_data(df)
    elif survey_type == "nsa":
        return _preprocess_nsa_data(df)
    elif survey_type == "linear":
        return _preprocess_linear_data(df)
    else:
        return _preprocess_generic_data(df)


def _preprocess_gaia_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess GAIA data."""
    logger.info("Applying GAIA-specific preprocessing")

    # Essential GAIA preprocessing
    df = df.with_columns(
        [
            # Clean coordinates
            pl.col("ra").alias("ra_deg"),
            pl.col("dec").alias("dec_deg"),
            # Clean magnitudes and add colors
            pl.col("phot_g_mean_mag").alias("g_mag"),
            pl.col("phot_bp_mean_mag").alias("bp_mag"),
            pl.col("phot_rp_mean_mag").alias("rp_mag"),
            # Calculate colors
            (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias("bp_rp"),
            (pl.col("phot_g_mean_mag") - pl.col("phot_rp_mean_mag")).alias("g_rp"),
            # Distance and proper motion
            pl.col("parallax"),
            pl.col("pmra"),
            pl.col("pmdec"),
        ]
    )

    # Filter valid data
    df = df.filter(
        pl.col("ra_deg").is_not_null()
        & pl.col("dec_deg").is_not_null()
        & pl.col("g_mag").is_not_null()
        & (pl.col("g_mag") < 20)  # Reasonable magnitude limit
        & (pl.col("parallax") > 0)  # Positive parallax
    )

    return df


def _preprocess_sdss_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess SDSS data."""
    logger.info("Applying SDSS-specific preprocessing")

    df = df.with_columns(
        [
            # Clean coordinates
            pl.col("ra").alias("ra_deg"),
            pl.col("dec").alias("dec_deg"),
            # SDSS magnitudes
            pl.col("modelMag_u").alias("u_mag"),
            pl.col("modelMag_g").alias("g_mag"),
            pl.col("modelMag_r").alias("r_mag"),
            pl.col("modelMag_i").alias("i_mag"),
            pl.col("modelMag_z").alias("z_mag"),
            # Colors
            (pl.col("modelMag_g") - pl.col("modelMag_r")).alias("g_r"),
            (pl.col("modelMag_r") - pl.col("modelMag_i")).alias("r_i"),
            # Redshift
            pl.col("z").alias("redshift"),
        ]
    )

    # Filter valid data
    df = df.filter(
        pl.col("ra_deg").is_not_null()
        & pl.col("dec_deg").is_not_null()
        & pl.col("g_mag").is_not_null()
        & (pl.col("g_mag") < 22)
    )

    return df


def _preprocess_nsa_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess NSA data."""
    logger.info("Applying NSA-specific preprocessing")

    df = df.with_columns(
        [
            pl.col("RA").alias("ra_deg"),
            pl.col("DEC").alias("dec_deg"),
            pl.col("SERSIC_N").alias("sersic_n"),
            pl.col("SERSIC_TH50").alias("half_light_radius"),
        ]
    )

    return df.filter(pl.col("ra_deg").is_not_null() & pl.col("dec_deg").is_not_null())


def _preprocess_linear_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess LINEAR data."""
    logger.info("Applying LINEAR-specific preprocessing")

    df = df.with_columns(
        [
            pl.col("ra").alias("ra_deg"),
            pl.col("dec").alias("dec_deg"),
            pl.col("magnitude").alias("mag"),
            pl.col("period").alias("period_days"),
            pl.col("amplitude").alias("amplitude_mag"),
        ]
    )

    return df.filter(
        pl.col("ra_deg").is_not_null()
        & pl.col("dec_deg").is_not_null()
        & pl.col("mag").is_not_null()
    )


def _preprocess_generic_data(df: pl.DataFrame) -> pl.DataFrame:
    """Generic preprocessing for unknown surveys."""
    logger.info("Applying generic preprocessing")

    # Try to standardize common columns
    column_mapping = {
        "RA": "ra_deg",
        "Ra": "ra_deg",
        "DEC": "dec_deg",
        "Dec": "dec_deg",
        "MAG": "mag",
        "Mag": "mag",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df = df.rename({old_col: new_col})

    return df


def create_survey_tensordict(df: pl.DataFrame, survey: str) -> SurveyTensorDict:
    """Create SurveyTensorDict from processed DataFrame."""
    logger.info(f"Creating TensorDict for {survey} with {len(df)} objects")

    # Extract spatial coordinates
    if "ra_deg" in df.columns and "dec_deg" in df.columns:
        # Convert to 3D coordinates for spatial consistency
        ra_rad = np.radians(df["ra_deg"].to_numpy())
        dec_rad = np.radians(df["dec_deg"].to_numpy())

        # Unit sphere coordinates
        x = np.cos(dec_rad) * np.cos(ra_rad)
        y = np.cos(dec_rad) * np.sin(ra_rad)
        z = np.sin(dec_rad)

        spatial_data = SpatialTensorDict(
            {
                "x": torch.tensor(x, dtype=torch.float32),
                "y": torch.tensor(y, dtype=torch.float32),
                "z": torch.tensor(z, dtype=torch.float32),
            }
        )
    else:
        # Default spatial data
        n_objects = len(df)
        spatial_data = SpatialTensorDict(
            {
                "x": torch.zeros(n_objects, dtype=torch.float32),
                "y": torch.zeros(n_objects, dtype=torch.float32),
                "z": torch.zeros(n_objects, dtype=torch.float32),
            }
        )

    # Extract photometric data
    mag_columns = [col for col in df.columns if "mag" in col.lower() and "_" not in col]
    color_columns = [
        col
        for col in df.columns
        if "_" in col and any(c in col for c in ["g", "r", "i", "bp", "rp"])
    ]

    photometric_features = {}
    for col in mag_columns:
        photometric_features[col] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float32
        )

    for col in color_columns:
        photometric_features[col] = torch.tensor(
            df[col].to_numpy(), dtype=torch.float32
        )

    photometric_data = (
        PhotometricTensorDict(photometric_features) if photometric_features else None
    )

    # Create SurveyTensorDict
    survey_dict = {
        "spatial": spatial_data,
        "survey_name": survey,
        "num_objects": len(df),
    }

    if photometric_data:
        survey_dict["photometric"] = photometric_data

    return SurveyTensorDict(survey_dict)


def create_training_splits(
    df: pl.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Create train/validation/test splits."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Ratios must sum to 1.0"
    )

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Shuffle data
    df_shuffled = df.sample(fraction=1.0, seed=seed)

    # Create splits
    train_df = df_shuffled.slice(0, n_train)
    val_df = df_shuffled.slice(n_train, n_val)
    test_df = df_shuffled.slice(n_train + n_val)

    logger.info(
        f"Created splits: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df
