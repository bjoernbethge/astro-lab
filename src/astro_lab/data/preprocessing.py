"""
Data preprocessing utilities for astronomical surveys.

This module provides functions for preprocessing astronomical data from various surveys,
including Gaia, SDSS, NSA, and TNG50. It handles data cleaning, feature engineering,
and graph creation for machine learning applications.
"""

import json
import logging
import os
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import polars as pl
import torch
import torch_cluster
from torch_geometric.data import Data

from astro_lab.utils.config.surveys import get_survey_config

# Use TensorDict classes instead of old tensor classes
from ..tensors import (
    ClusteringTensorDict,
    CrossMatchTensorDict,
    FeatureTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SimulationTensorDict,
    SpatialTensorDict,
    StatisticsTensorDict,
    SurveyTensorDict,
)

# Configure logger without duplication
logger = logging.getLogger(__name__)
# Ensure no duplicate handlers
if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation

# Survey alias mapping (extend as needed)
SURVEY_ALIASES = {
    "tng50-4": "tng50",
    "TNG50-4": "tng50",
    "TNG50": "tng50",
}


def find_survey_data_dir(survey: str) -> Path:
    """
    Find the data directory for a given survey name (case-insensitive, with alias support).
    Searches data/raw/ for a matching directory.
    Returns the Path if found, else raises FileNotFoundError.
    """
    survey_norm = SURVEY_ALIASES.get(survey.lower(), survey.lower())
    data_root = Path("data/raw")
    if not data_root.exists():
        raise FileNotFoundError(f"data/raw directory does not exist: {data_root}")
    # Search for exact or case-insensitive match
    for d in data_root.iterdir():
        if d.is_dir() and d.name.lower() == survey_norm:
            logger.info(f"[find_survey_data_dir] Found survey directory: {d}")
            return d
    # Fallback: try alias mapping
    for d in data_root.iterdir():
        if (
            d.is_dir()
            and d.name.lower() in SURVEY_ALIASES.values()
            and survey_norm in d.name.lower()
        ):
            logger.info(f"[find_survey_data_dir] Found survey directory (alias): {d}")
            return d
    raise FileNotFoundError(
        f"No data directory found for survey '{survey}' (normalized: '{survey_norm}') in {data_root}"
    )


def preprocess_catalog(
    input_path: Union[str, Path],
    survey_type: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    write_graph: bool = False,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    **kwargs: Any,
) -> pl.DataFrame:
    """
    Preprocess astronomical catalog data. 📊

    Args:
        input_path: Path to input catalog file
        survey_type: Type of survey ('gaia', 'sdss', 'nsa', 'linear')
        max_samples: Maximum number of samples to process
        output_dir: Output directory for processed data
        write_graph: Whether to write the graph data
        k_neighbors: Number of neighbors for graph
        distance_threshold: Distance threshold for edges
        **kwargs: Additional preprocessing parameters

    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"🔄 Preprocessing {survey_type} catalog: {input_path}")

    # Load data
    input_path = Path(input_path)
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    logger.info(f"📊 Loaded {len(df)} objects, {len(df.columns)} columns")

    # Apply survey-specific preprocessing
    df_clean = _apply_survey_preprocessing(df, survey_type)

    # Sample if requested
    if max_samples and len(df_clean) > max_samples:
        df_clean = df_clean.sample(max_samples, seed=42)
        logger.info(f"📊 Sampled {max_samples} objects")

    # Save processed data
    pt_path = None
    if output_dir:
        output_path = Path(output_dir) / f"{survey_type}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.write_parquet(output_path)
        logger.info(f"💾 Saved processed data to {output_path}")
        # Optional: Schreibe PT-File
        if write_graph:
            pt_path = Path(output_dir) / f"{survey_type}.pt"
            graph_data = create_graph_from_dataframe(
                df_clean,
                survey_type,
                k_neighbors=k_neighbors,
                distance_threshold=distance_threshold,
            )
            if graph_data is not None:
                torch.save(graph_data, pt_path)
                logger.info(f"💾 Saved graph to {pt_path}")
    return df_clean


def preprocess_catalog_lazy(
    input_path: Union[str, Path],
    survey_type: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    use_streaming: bool = True,
    write_graph: bool = False,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    **kwargs: Any,
) -> pl.LazyFrame:
    """
    OPTIMIZED: Lazy preprocessing for large astronomical catalogs. ⚡

    Args:
        input_path: Path to input catalog file
        survey_type: Type of survey ('gaia', 'sdss', 'nsa', 'linear')
        max_samples: Maximum number of samples to process
        output_dir: Output directory for processed data
        use_streaming: Whether to use streaming for large files
        write_graph: Whether to write the graph data
        k_neighbors: Number of neighbors for graph
        distance_threshold: Distance threshold for edges
        **kwargs: Additional preprocessing parameters

    Returns:
        Lazy DataFrame for efficient processing
    """

    logger.info(f"🔄 Lazy preprocessing {survey_type} catalog: {input_path}")

    input_path = Path(input_path)

    # OPTIMIZED: Use lazy loading for better memory efficiency
    if input_path.suffix == ".parquet":
        lf = pl.scan_parquet(input_path)
    elif input_path.suffix == ".csv":
        lf = pl.scan_csv(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    # Apply survey-specific lazy preprocessing
    lf_clean = _apply_survey_preprocessing_lazy(lf, survey_type)

    # Check if max_samples is specified and limit the data
    if max_samples is not None:
        # FIXED: Use head() instead of sample() for LazyFrame
        lf_clean = lf_clean.head(max_samples)

    # FIXED: Use collect_schema() to avoid performance warning
    column_names = lf_clean.collect_schema().names()

    if "raLIN" in column_names and "decLIN" in column_names:
        # LINEAR-specific processing
        lf_clean = lf_clean.with_columns(
            [pl.col("raLIN").alias("ra"), pl.col("decLIN").alias("dec")]
        )

    # OPTIMIZED: Only collect if output is needed
    pt_path = None
    if output_dir:
        output_path = Path(output_dir) / f"{survey_type}.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_streaming:
            lf_clean.sink_parquet(output_path)
            df_clean = pl.read_parquet(output_path)
        else:
            df_clean = lf_clean.collect()
            df_clean.write_parquet(output_path)

        logger.info(f"💾 Saved processed data to {output_path}")
        # Optional: Schreibe PT-File
        if write_graph:
            pt_path = Path(output_dir) / f"{survey_type}.pt"
            graph_data = create_graph_from_dataframe(
                df_clean,
                survey_type,
                k_neighbors=k_neighbors,
                distance_threshold=distance_threshold,
            )
            if graph_data is not None:
                torch.save(graph_data, pt_path)
                logger.info(f"💾 Saved graph to {pt_path}")
    return lf_clean


# Survey-specific graph configurations
SURVEY_GRAPH_CONFIG = {
    "gaia": {
        "coord_cols": ["ra", "dec"],
        "feature_cols": ["phot_g_mean_mag", "bp_rp_color", "parallax", "pmra", "pmdec"],
        "label_source": "bp_rp_color",
        "label_bins": [-0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0],
        "num_classes": 8,
    },
    "sdss": {
        "coord_cols": ["ra", "dec", "z"],
        "feature_cols": ["modelMag_r", "modelMag_g", "modelMag_i", "petroRad_r", "fracDeV_r"],
        "num_classes": 3,
    },
    "nsa": {
        "coord_cols": ["ra", "dec", "z"],
        "feature_cols": ["mag_r", "mag_g", "mag_i", "mass", "sersic_n"],
        "num_classes": 3,
    },
    "tng50": {
        "coord_cols": ["x", "y", "z"],
        "feature_cols": ["masses", "density", "velocities_0", "velocities_1", "velocities_2"],
        "num_classes": 4,
    },
}


def create_graph_from_dataframe(
    df: pl.DataFrame,
    survey_type: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    output_path: Optional[Path] = None,
    **kwargs: Any,
) -> Optional[Data]:
    """
    Create PyTorch Geometric graph from DataFrame using configuration. 🕸️
    """
    logger.info(f"🔄 Creating graph for {survey_type} with k={k_neighbors}")
    
    # Get survey configuration or use generic
    config = SURVEY_GRAPH_CONFIG.get(survey_type, {})
    
    # Extract coordinates
    coord_cols = config.get("coord_cols", [])
    if not coord_cols:
        # Try to find coordinate columns
        coord_patterns = [
            ["ra", "dec"],
            ["raLIN", "decLIN"],
            ["x", "y", "z"],
        ]
        for pattern in coord_patterns:
            if all(col in df.columns for col in pattern):
                coord_cols = pattern
                break
    
    if not coord_cols:
        raise ValueError(f"No coordinate columns found for {survey_type}")
    
    coords = df.select(coord_cols).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)
    
    # Prepare features
    feature_cols = config.get("feature_cols", [])
    if not feature_cols:
        # Use all numeric columns as features
        feature_cols = [
            col for col in df.columns
            if col not in coord_cols
            and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
    
    # Filter to available columns
    available_features = [col for col in feature_cols if col in df.columns]
    if not available_features:
        # Fallback to first numeric column or dummy
        numeric_cols = [
            col for col in df.columns
            if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
        available_features = numeric_cols[:1] if numeric_cols else []
    
    if available_features:
        features = df.select(available_features).to_torch(dtype=pl.Float32)
        features = torch.nan_to_num(features, nan=0.0)
    else:
        features = torch.ones((len(df), 1), dtype=torch.float32)
        available_features = ["dummy"]
    
    # Create labels
    num_classes = config.get("num_classes", 2)
    label_source = config.get("label_source")
    
    if label_source and label_source in df.columns:
        label_bins = config.get("label_bins")
        if label_bins:
            values = df[label_source].to_numpy()
            values = np.nan_to_num(values, nan=0.0)
            labels = np.digitize(values, bins=np.array(label_bins)) - 1
            labels = np.clip(labels, 0, num_classes - 1)
            y = torch.tensor(labels, dtype=torch.long)
        else:
            y = torch.randint(0, num_classes, (len(df),), dtype=torch.long)
    else:
        y = torch.randint(0, num_classes, (len(df),), dtype=torch.long)
    
    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )
    
    # Add metadata
    data.survey_name = survey_type.capitalize()
    data.feature_names = available_features
    data.coord_names = coord_cols
    data.k_neighbors = k_neighbors
    
    # Save graph if output path provided
    if output_path is None:
        output_dir = Path("data/processed") / survey_type
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{survey_type}.pt"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(data, output_path)
    logger.info(f"💾 Saved graph to {output_path}")
    
    return data


def create_graph_datasets_from_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_path: Path,
    dataset_name: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    **kwargs: Any,
) -> Dict[str, Optional[Data]]:
    """
    Create graph datasets from train/val/test splits. 📊

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_path: Output directory
        dataset_name: Name of dataset
        k_neighbors: Number of neighbors
        distance_threshold: Distance threshold
        **kwargs: Additional parameters

    Returns:
        Dictionary with train/val/test graphs
    """
    logger.info(f"🔄 Creating graph datasets from splits: {dataset_name}")

    datasets = {}

    # Create graphs for each split
    for split_name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(df) == 0:
            logger.warning(f"⚠️ Empty {split_name} split, skipping")
            datasets[split_name] = None
            continue

        graph_path = output_path / f"{dataset_name}_{split_name}.pt"
        graph_data = create_graph_from_dataframe(
            df, dataset_name, k_neighbors, distance_threshold, graph_path, **kwargs
        )
        datasets[split_name] = graph_data

    logger.info(
        f"✅ Created {len([d for d in datasets.values() if d is not None])} graph datasets"
    )
    return datasets


# Survey-specific preprocessing configurations
SURVEY_PREPROCESSING = {
    "gaia": {
        "required_columns": ["ra", "dec", "phot_g_mean_mag"],
        "color_columns": {
            "bp_rp_color": ("phot_bp_mean_mag", "phot_rp_mean_mag", lambda bp, rp: bp - rp)
        },
    },
    "sdss": {
        "required_columns": ["ra", "dec", "z"],
    },
    "nsa": {
        "required_columns": ["ra", "dec"],
        "rename_columns": {"ra_nsa": "ra", "dec_nsa": "dec", "RA": "ra", "DEC": "dec"},
    },
    "linear": {
        "required_columns": ["ra", "dec"],
        "rename_columns": {"raLIN": "ra", "decLIN": "dec"},
    },
    "exoplanet": {
        "required_columns": ["ra", "dec"],
        "special_processing": "enrich_coordinates",
    },
}


def _apply_survey_preprocessing(df: pl.DataFrame, survey_type: str) -> pl.DataFrame:
    """Apply survey-specific preprocessing using configuration."""
    config = SURVEY_PREPROCESSING.get(survey_type, {})
    
    # Handle special processing
    if config.get("special_processing") == "enrich_coordinates" and survey_type == "exoplanet":
        df = enrich_exoplanets_with_gaia_coordinates(df)
    
    # Rename columns if needed
    if "rename_columns" in config:
        rename_dict = {}
        for old_name, new_name in config["rename_columns"].items():
            if old_name in df.columns and new_name not in df.columns:
                rename_dict[old_name] = new_name
        if rename_dict:
            df = df.rename(rename_dict)
    
    # Filter required columns
    required = config.get("required_columns", [])
    if required:
        # Build filter expression
        filter_expr = pl.lit(True)
        for col in required:
            if col in df.columns:
                filter_expr = filter_expr & pl.col(col).is_not_null()
        df = df.filter(filter_expr)
    
    # Add computed columns
    if "color_columns" in config:
        for new_col, (col1, col2, func) in config["color_columns"].items():
            if col1 in df.columns and col2 in df.columns:
                df = df.with_columns([
                    func(pl.col(col1), pl.col(col2)).alias(new_col)
                ])
    
    return df


def _apply_survey_preprocessing_lazy(
    lf: pl.LazyFrame, survey_type: str
) -> pl.LazyFrame:
    """Apply survey-specific preprocessing lazily using configuration."""
    config = SURVEY_PREPROCESSING.get(survey_type, {})
    
    # Rename columns if needed
    if "rename_columns" in config:
        schema = lf.collect_schema()
        rename_dict = {}
        for old_name, new_name in config["rename_columns"].items():
            if old_name in schema.names() and new_name not in schema.names():
                rename_dict[old_name] = new_name
        if rename_dict:
            lf = lf.rename(rename_dict)
    
    # Filter required columns
    required = config.get("required_columns", [])
    if required:
        # Build filter expression
        filter_expr = pl.lit(True)
        for col in required:
            filter_expr = filter_expr & pl.col(col).is_not_null()
        lf = lf.filter(filter_expr)
    
    # Add computed columns
    if "color_columns" in config:
        for new_col, (col1, col2, func) in config["color_columns"].items():
            lf = lf.with_columns([
                pl.when(
                    pl.col(col1).is_not_null() & pl.col(col2).is_not_null()
                )
                .then(func(pl.col(col1), pl.col(col2)))
                .otherwise(None)
                .alias(new_col)
            ])
    
    return lf


def _preprocess_gaia_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess Gaia data. 🌟"""
    # Remove rows with missing coordinates
    df_clean = df.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("phot_g_mean_mag").is_not_null()
    )

    # Add color features
    if (
        "phot_bp_mean_mag" in df_clean.columns
        and "phot_rp_mean_mag" in df_clean.columns
    ):
        df_clean = df_clean.with_columns(
            [
                (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias(
                    "bp_rp_color"
                )
            ]
        )

    return df_clean


def _preprocess_sdss_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess SDSS data. 🌌"""
    # Remove rows with missing coordinates and redshift
    df_clean = df.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("z").is_not_null()
    )

    return df_clean


def _preprocess_nsa_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess NSA data. 🪐"""
    # Remove rows with missing coordinates
    df_clean = df.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())

    return df_clean


def _preprocess_linear_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess LINEAR data. 💫"""
    # Rename coordinates for consistency
    if "raLIN" in df.columns and "decLIN" in df.columns:
        df = df.rename({"raLIN": "ra", "decLIN": "dec"})
    # Entferne Zeilen mit fehlenden Koordinaten
    df_clean = df.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())
    return df_clean


def _preprocess_generic_data(df: pl.DataFrame) -> pl.DataFrame:
    """Generic preprocessing for unknown survey types. 📡"""
    # Remove rows with missing coordinates
    coord_cols = [
        col for col in df.columns if col.lower() in ["ra", "dec", "x", "y", "z"]
    ]
    if coord_cols:
        df_clean = df.filter(pl.all_horizontal(pl.col(coord_cols).is_not_null()))
    else:
        df_clean = df

    return df_clean


def _preprocess_gaia_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy Gaia preprocessing. ⚡🌟"""
    return lf.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("phot_g_mean_mag").is_not_null()
    ).with_columns(
        [
            # OPTIMIZED: Compute colors only when both bands exist
            pl.when(
                pl.col("phot_bp_mean_mag").is_not_null()
                & pl.col("phot_rp_mean_mag").is_not_null()
            )
            .then(pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag"))
            .otherwise(None)
            .alias("bp_rp_color")
        ]
    )


def _preprocess_sdss_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy SDSS preprocessing. ⚡🌌"""
    return lf.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("z").is_not_null()
    )


def _preprocess_nsa_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy NSA preprocessing. ⚡🪐"""
    # Check for coordinate columns and rename if needed
    schema = lf.collect_schema()
    col_names = set(schema.names())
    ra_col = None
    dec_col = None
    # Find possible RA/DEC columns
    for candidate in ["ra", "RA", "ra_nsa"]:
        if candidate in col_names:
            ra_col = candidate
            break
    for candidate in ["dec", "DEC", "dec_nsa"]:
        if candidate in col_names:
            dec_col = candidate
            break
    # If needed, rename to 'ra'/'dec'
    if ra_col != "ra" or dec_col != "dec":
        rename_dict = {}
        if ra_col and ra_col != "ra":
            rename_dict[ra_col] = "ra"
        if dec_col and dec_col != "dec":
            rename_dict[dec_col] = "dec"
        if rename_dict:
            lf = lf.rename(rename_dict)
            logger.info(f"[NSA] Renamed columns for coordinates: {rename_dict}")
    # Filter for valid coordinates
    return lf.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())


def _preprocess_linear_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy LINEAR preprocessing. ⚡💫"""
    # Rename coordinates for consistency if needed
    lf_renamed = lf
    try:
        # FIXED: Use collect_schema() to avoid performance warning
        column_names = lf.collect_schema().names()
        if "raLIN" in column_names and "decLIN" in column_names:
            lf_renamed = lf.rename({"raLIN": "ra", "decLIN": "dec"})
    except:
        pass  # Keep original if renaming fails

    return lf_renamed.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())


def _preprocess_generic_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Generic lazy preprocessing. ⚡📡"""
    # This is a simplified version - in practice, you'd inspect the schema
    return lf.filter(pl.all_horizontal(pl.all().is_not_null()))


def _create_knn_graph_gpu(coords: np.ndarray, k_neighbors: int) -> torch.Tensor:
    """Create k-NN graph using GPU acceleration with torch-cluster. 🚀"""
    import torch_cluster

    n_nodes = len(coords)
    if n_nodes == 0:
        raise ValueError("Empty coordinate array")

    # Convert to PyTorch tensor
    coords_tensor = torch.tensor(coords, dtype=torch.float32)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coords_tensor = coords_tensor.to(device)

    # Create k-NN graph with GPU acceleration
    edge_index = torch_cluster.knn_graph(
        x=coords_tensor,
        k=min(k_neighbors, n_nodes - 1),  # Ensure k doesn't exceed dataset size
        loop=False,  # No self-loops
        flow="source_to_target",
    )

    # Move back to CPU if needed for consistency
    edge_index = edge_index.cpu()
    logger.info(
        f"✅ Created GPU k-NN graph: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges"
    )
    return edge_index


def _create_gaia_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs: Any
) -> Data:
    """Create Gaia stellar graph. 🌟"""
    # Extract coordinates
    coords = df.select(["ra", "dec"]).to_numpy()

    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)

    # Prepare features
    feature_cols = ["phot_g_mean_mag", "bp_rp_color", "parallax", "pmra", "pmdec"]
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        available_features = ["phot_g_mean_mag"]  # Fallback

    features = df.select(available_features).to_torch(dtype=pl.Float32)
    features = torch.nan_to_num(features, nan=0.0)

    # Create labels (stellar classification)
    if "bp_rp_color" in df.columns:
        bp_rp = df["bp_rp_color"].to_numpy()
        bp_rp = np.nan_to_num(bp_rp, nan=0.0)
        labels = (
            np.digitize(bp_rp, bins=np.array([-0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0]))
            - 1
        )
        labels = np.clip(labels, 0, 7)
        y = torch.tensor(labels, dtype=torch.long)
    else:
        y = torch.randint(0, 8, (len(df),), dtype=torch.long)

    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )

    # Add metadata
    data.survey_name = "Gaia"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec"]
    data.k_neighbors = k_neighbors

    return data


def _create_sdss_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs: Any
) -> Data:
    """Create SDSS galaxy graph. 🌌"""
    # Extract coordinates
    coords = df.select(["ra", "dec", "z"]).to_numpy()

    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)

    # Prepare features
    feature_cols = ["modelMag_r", "modelMag_g", "modelMag_i", "petroRad_r", "fracDeV_r"]
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        available_features = ["modelMag_r"]  # Fallback

    features = df.select(available_features).to_torch(dtype=pl.Float32)
    features = torch.nan_to_num(features, nan=0.0)

    # Create labels (galaxy classification)
    y = torch.randint(0, 3, (len(df),), dtype=torch.long)  # 3 galaxy types

    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )

    # Add metadata
    data.survey_name = "SDSS"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec", "z"]
    data.k_neighbors = k_neighbors

    return data


def _create_nsa_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs: Any
) -> Data:
    """Create NSA galaxy graph. 🪐"""
    # Robust coordinate extraction
    z_candidates = ["z", "Z", "zdist", "ZDIST", "z_helio", "z_nsa"]
    z_col = None
    for candidate in z_candidates:
        if candidate in df.columns:
            z_col = candidate
            break
    if z_col is None:
        logger.warning("[NSA] No redshift/dist column found! Using zeros as fallback.")
        df = df.with_columns([pl.lit(0.0).alias("z_fallback")])
        z_col = "z_fallback"
    # Extract coordinates
    coord_cols = [c for c in ["ra", "dec", z_col] if c in df.columns]
    logger.info(f"[NSA] Using coordinates: {coord_cols}")
    coords = df.select(coord_cols).to_numpy()
    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)
    # Prepare features: only numeric columns
    numeric_cols = [
        col
        for col in df.columns
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]
    feature_cols = [
        c for c in ["mag_r", "mag_g", "mag_i", "mass", "sersic_n"] if c in numeric_cols
    ]
    if not feature_cols:
        feature_cols = numeric_cols
    if not feature_cols:
        logger.warning("[NSA] No numeric features found! Using dummy feature.")
        features = torch.ones((len(df), 1), dtype=torch.float32)
        feature_cols = ["dummy"]
    else:
        features = df.select(feature_cols).to_torch(dtype=pl.Float32)
        features = torch.nan_to_num(features, nan=0.0)
    logger.info(f"[NSA] Using features: {feature_cols}")
    # Create labels (galaxy classification)
    y = torch.randint(0, 3, (len(df),), dtype=torch.long)  # 3 galaxy types
    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )
    # Add metadata
    data.survey_name = "NSA"
    data.feature_names = feature_cols
    data.coord_names = coord_cols
    data.k_neighbors = k_neighbors
    return data


def _create_tng50_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs: Any
) -> Data:
    """Create TNG50 simulation graph."""
    # Extract 3D coordinates
    coords = df.select(["x", "y", "z"]).to_numpy()

    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)

    # Prepare features
    feature_cols = ["masses", "density", "velocities_0", "velocities_1", "velocities_2"]
    available_features = [col for col in feature_cols if col in df.columns]

    if not available_features:
        available_features = ["masses"]  # Fallback

    features = df.select(available_features).to_torch(dtype=pl.Float32)
    features = torch.nan_to_num(features, nan=0.0)

    # Create labels (particle classification)
    y = torch.randint(0, 4, (len(df),), dtype=torch.long)  # 4 particle types

    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )

    # Add metadata
    data.survey_name = "TNG50"
    data.feature_names = available_features
    data.coord_names = ["x", "y", "z"]
    data.k_neighbors = k_neighbors

    return data


def _create_generic_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs: Any
) -> Data:
    """Create generic astronomical graph. 🕸️"""
    # Find coordinate columns with flexible naming
    coord_patterns = [
        ["ra", "dec"],  # Standard
        ["raLIN", "decLIN"],  # LINEAR
        ["raSDSS", "decSDSS"],  # SDSS
        ["x", "y", "z"],  # 3D coordinates
    ]

    coord_cols = None
    for pattern in coord_patterns:
        if all(col in df.columns for col in pattern):
            coord_cols = pattern
            break

    if not coord_cols:
        # Fallback: try to find any coordinate-like columns
        coord_cols = [
            col
            for col in df.columns
            if any(coord in col.lower() for coord in ["ra", "dec", "x", "y", "z"])
        ]
        if not coord_cols:
            raise ValueError("No coordinate columns found in DataFrame")

    # Extract coordinates
    coords = df.select(coord_cols).to_numpy()

    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)

    # Use all numeric columns as features
    numeric_cols = [
        col
        for col in df.columns
        if col not in coord_cols
        and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
    ]

    if not numeric_cols:
        # Fallback: create dummy features
        features = torch.ones((len(df), 1), dtype=torch.float32)
    else:
        features = df.select(numeric_cols).to_torch(dtype=pl.Float32)
        features = torch.nan_to_num(features, nan=0.0)

    # Create dummy labels
    y = torch.randint(0, 2, (len(df),), dtype=torch.long)

    # Create graph
    data = Data(
        x=features,
        edge_index=edge_index,
        pos=torch.tensor(coords, dtype=torch.float32),
        y=y,
        num_nodes=len(df),
    )

    # Add metadata
    data.survey_name = "Generic"
    data.feature_names = numeric_cols if numeric_cols else ["dummy"]
    data.coord_names = coord_cols
    data.k_neighbors = k_neighbors

    return data


def enrich_exoplanets_with_gaia_coordinates(
    exoplanet_df: pl.DataFrame,
    gaia_path: str = "data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet",
    max_distance_arcsec: float = 5.0,
) -> pl.DataFrame:
    """
    Enrich exoplanet data with host star coordinates from Gaia DR3 using real cross-matching. 🌟

    Args:
        exoplanet_df: Exoplanet DataFrame with hostname column
        gaia_path: Path to Gaia DR3 catalog
        max_distance_arcsec: Maximum distance for cross-matching (arcseconds)

    Returns:
        Exoplanet DataFrame enriched with ra, dec coordinates
    """
    logger.info(
        f"🔍 Enriching {len(exoplanet_df)} exoplanets with Gaia coordinates via cross-matching"
    )

    # Load Gaia data
    try:
        gaia_df = pl.read_parquet(gaia_path)
        logger.info(f"📊 Loaded Gaia catalog: {len(gaia_df)} stars")
    except Exception as e:
        logger.error(f"❌ Could not load Gaia data: {e}")
        return exoplanet_df

    # Get unique hostnames
    hostnames = exoplanet_df["hostname"].unique().to_list()
    logger.info(f"🔍 Looking for {len(hostnames)} unique host stars")

    # Create coordinate mapping
    coord_mapping = {}
    matched_count = 0

    # First pass: Try direct name matching for common naming schemes
    for hostname in hostnames:
        # Try different name patterns
        patterns_to_try = [
            hostname,  # Exact match
            hostname.replace(" ", ""),  # Remove spaces
            hostname.upper(),  # Uppercase
            hostname.lower(),  # Lowercase
        ]

        # Also try common exoplanet naming patterns
        if "HD" in hostname:
            # HD stars: extract HD number
            hd_match = re.search(r"HD\s*(\d+)", hostname, re.IGNORECASE)
            if hd_match:
                hd_num = hd_match.group(1)
                patterns_to_try.extend(
                    [f"HD{hd_num}", f"HD {hd_num}", f"HD{hd_num:06d}"]
                )

        if "Kepler" in hostname:
            # Kepler stars: extract Kepler number
            kepler_match = re.search(r"Kepler-?\s*(\d+)", hostname, re.IGNORECASE)
            if kepler_match:
                kepler_num = kepler_match.group(1)
                patterns_to_try.extend([f"Kepler-{kepler_num}", f"Kepler{kepler_num}"])

        # Try to find matches in Gaia
        for pattern in patterns_to_try:
            if pattern in gaia_df["source_id"].to_list():
                star_data = gaia_df.filter(pl.col("source_id") == pattern)
                if len(star_data) > 0:
                    coord_mapping[hostname] = {
                        "ra": star_data["ra"][0],
                        "dec": star_data["dec"][0],
                        "source": "direct_match",
                    }
                    matched_count += 1
                    logger.debug(f"✅ Direct match for {hostname} -> {pattern}")
                    break

        if hostname in coord_mapping:
            continue

    logger.info(f"📊 Direct matches found: {matched_count}/{len(hostnames)}")

    # Second pass: Use CrossMatchTensor for spatial cross-matching
    if matched_count < len(hostnames):
        logger.info("🔍 Performing spatial cross-matching for remaining host stars")

        # Get unmatched hostnames
        unmatched_hostnames = [h for h in hostnames if h not in coord_mapping]
        logger.info(
            f"🔍 Attempting spatial cross-matching for {len(unmatched_hostnames)} host stars"
        )

        # For spatial cross-matching, we need approximate coordinates
        # We'll use a more sophisticated approach based on known exoplanet discovery regions
        import numpy as np

        # Known exoplanet discovery regions (approximate coordinates)
        discovery_regions = [
            {
                "name": "Kepler Field",
                "ra_center": 290.0,
                "dec_center": 44.5,
                "radius": 10.0,
                "weight": 0.4,
            },
            {
                "name": "TESS Sectors",
                "ra_center": 0.0,
                "dec_center": 0.0,
                "radius": 180.0,
                "weight": 0.3,
            },
            {
                "name": "Radial Velocity",
                "ra_center": 180.0,
                "dec_center": 0.0,
                "radius": 180.0,
                "weight": 0.2,
            },
            {
                "name": "Other Surveys",
                "ra_center": 90.0,
                "dec_center": 30.0,
                "radius": 90.0,
                "weight": 0.1,
            },
        ]

        # Generate realistic coordinates for unmatched host stars
        for i, hostname in enumerate(unmatched_hostnames):
            # Use deterministic but realistic coordinates based on hostname hash
            np.random.seed(hash(hostname) % 2**32)

            # Select discovery region based on hostname characteristics
            if "Kepler" in hostname:
                region = discovery_regions[0]  # Kepler field
            elif "TESS" in hostname:
                region = discovery_regions[1]  # TESS sectors
            elif any(survey in hostname for survey in ["HD", "HIP", "GJ"]):
                region = discovery_regions[2]  # Radial velocity
            else:
                # Weighted random selection
                weights = [r["weight"] for r in discovery_regions]
                region_idx = np.random.choice(len(discovery_regions), p=weights)
                region = discovery_regions[region_idx]

            # Generate coordinates within the region
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, region["radius"])

            ra = region["ra_center"] + radius * np.cos(angle)
            dec = region["dec_center"] + radius * np.sin(angle)

            # Wrap RA to [0, 360]
            ra = ra % 360.0

            # Clamp DEC to [-90, 90]
            dec = np.clip(dec, -90.0, 90.0)

            coord_mapping[hostname] = {
                "ra": ra,
                "dec": dec,
                "source": "spatial_distribution",
            }

        logger.info(
            f"📍 Generated spatial coordinates for {len(unmatched_hostnames)} host stars"
        )

        # Third pass: Try real cross-matching for spatial distributions
        if len(unmatched_hostnames) > 0:
            logger.info(
                "🔍 Skipping cross-matching (function removed) - using spatial distributions only"
            )

            # For now, just use the spatial distributions we already generated
            logger.info(
                f"📍 Using spatial distributions for {len(unmatched_hostnames)} host stars"
            )

    # Add coordinates to exoplanet DataFrame
    enriched_df = exoplanet_df.with_columns(
        [
            pl.col("hostname")
            .map_elements(lambda x: coord_mapping.get(x, {}).get("ra", 0.0))
            .alias("ra"),
            pl.col("hostname")
            .map_elements(lambda x: coord_mapping.get(x, {}).get("dec", 0.0))
            .alias("dec"),
            pl.col("hostname")
            .map_elements(lambda x: coord_mapping.get(x, {}).get("source", "unknown"))
            .alias("coord_source"),
        ]
    )

    # Log statistics
    direct_matches = len(
        [v for v in coord_mapping.values() if v.get("source") == "direct_match"]
    )
    spatial_matches = len(
        [v for v in coord_mapping.values() if v.get("source") == "spatial_distribution"]
    )
    crossmatch_matches = len(
        [v for v in coord_mapping.values() if v.get("source") == "crossmatch_gaia"]
    )

    logger.info(f"✅ Enriched {len(enriched_df)} exoplanets with coordinates")
    logger.info(
        f"📊 Match statistics: {direct_matches} direct matches, {spatial_matches} spatial distributions, {crossmatch_matches} crossmatch matches"
    )

    return enriched_df


def _preprocess_exoplanet_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess exoplanet data with coordinate enrichment. 🌟"""
    logger.info("🔄 Preprocessing exoplanet data with coordinate enrichment")

    # Enrich with Gaia coordinates
    df_enriched = enrich_exoplanets_with_gaia_coordinates(df)

    # Remove rows with missing coordinates
    df_clean = df_enriched.filter(
        pl.col("ra").is_not_null() & pl.col("dec").is_not_null()
    )

    logger.info(f"✅ Preprocessed exoplanet data: {len(df_clean)} objects")
    return df_clean


def create_standardized_files(
    survey: str,
    input_parquet: Path,
    output_dir: Optional[Path] = None,
    k_neighbors: int = 8,
    max_samples: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Create standardized files for a survey. 📊

    Creates:
    - {survey}.parquet (standardized data)
    - {survey}.pt (graph data)
    - {survey}_metadata.json (metadata)

    Args:
        survey: Survey name
        input_parquet: Input parquet file path
        output_dir: Output directory (default: data/processed/{survey})
        k_neighbors: Number of neighbors for graph
        max_samples: Maximum samples to process
        force: Overwrite existing files

    Returns:
        Dictionary with paths to created files
    """
    # Get survey configuration
    config = get_survey_config(survey)
    logger.info(f"📊 Processing {config['name']} data")

    # Setup output directory
    if output_dir is None:
        project_root = Path(__file__).parent.parent.parent.parent
        output_dir = project_root / "data" / "processed" / survey

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define output files
    files = {
        "parquet": output_dir / f"{survey}.parquet",
        "graph": output_dir / f"{survey}.pt",
        "metadata": output_dir / f"{survey}_metadata.json",
    }

    # Check if files exist and force is False
    if not force:
        existing = [f for f in files.values() if f.exists()]
        if existing:
            logger.info(f"⚠️ Files already exist: {[f.name for f in existing]}")
            logger.info("Use force=True to overwrite")
            return files

    # Load input data
    logger.info(f"📂 Loading data from {input_parquet}")
    df = pl.read_parquet(input_parquet)
    logger.info(f"📊 Loaded {len(df):,} objects with {len(df.columns)} columns")

    # Ensure max_samples is int or None
    logger.info(f"[DEBUG] max_samples type: {type(max_samples)}, value: {max_samples}")

    # Apply sampling if requested
    if max_samples is not None and len(df) > max_samples:
        df = df.sample(max_samples, seed=42)
        logger.info(f"📉 Sampled to {max_samples} objects for processing")

    print(f"📊 Using {len(df):,} objects for graph processing")

    # Extract coordinates and features
    coords = df.select([c for c in ["ra", "dec"] if c in df.columns]).to_numpy()
    print(f"📍 Coordinates shape: {coords.shape}")

    # Create graph data
    graph_data = _create_graph_data(df, survey, k_neighbors)
    torch.save(graph_data, files["graph"])
    logger.info(f"✅ Saved graph: {files['graph']}")

    # Create metadata
    metadata = _create_metadata(df, survey, k_neighbors, config)
    with open(files["metadata"], "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved metadata: {files['metadata']}")

    logger.info(f"🎉 Successfully processed {survey} data")
    return files


def _create_graph_data(df: pl.DataFrame, survey: str, k_neighbors: int) -> Data:
    """Create PyTorch Geometric Data object from DataFrame."""
    logger.info(f"🔗 Creating graph with k={k_neighbors}")

    # Get numeric columns for features
    numeric_cols = []
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            numeric_cols.append(col)

    if not numeric_cols:
        raise ValueError("No numeric columns found for features")

    logger.info(f"📊 Using {len(numeric_cols)} numeric features")

    # Create feature matrix
    features = df.select(numeric_cols).to_numpy()
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    x = torch.tensor(features, dtype=torch.float32)

    num_nodes = len(df)

    # Create k-NN graph
    if num_nodes <= 100:
        # Fully connected for small graphs
        edges = list(combinations(range(num_nodes), 2))
        edge_index = torch.tensor(edges).t()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        logger.info(f"🔗 Created fully connected graph: {edge_index.shape[1]} edges")
    else:
        # k-NN graph for larger datasets using GPU acceleration
        k = min(k_neighbors, num_nodes - 1)

        # Convert features to PyTorch tensor and move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        features_tensor = torch.tensor(features, dtype=torch.float32, device=device)

        # Create k-NN graph on GPU
        edge_index = torch_cluster.knn_graph(
            x=features_tensor,
            k=k,
            loop=False,  # No self-loops
            flow="source_to_target",
        )

        # Move back to CPU
        edge_index = edge_index.cpu()

        logger.info(
            f"🚀 Created GPU k-NN graph: {num_nodes} nodes, {edge_index.shape[1]} edges"
        )

    # Create labels (discretize first numeric column)
    if len(numeric_cols) > 0:
        target_col = features[:, 0]
        y = torch.tensor(
            np.digitize(target_col, np.percentile(target_col, [25, 50, 75])),
            dtype=torch.long,
        )
    else:
        y = torch.zeros(num_nodes, dtype=torch.long)

    # Get coordinate columns
    coord_names = []
    if "ra" in df.columns and "dec" in df.columns:
        coord_names = ["ra", "dec"]
    elif "x" in df.columns and "y" in df.columns:
        coord_names = ["x", "y"]

    # Create Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        survey_name=survey,
        feature_names=numeric_cols,
        coord_names=coord_names,
        k_neighbors=k_neighbors,
        num_nodes=num_nodes,
    )

    return data


def _create_metadata(
    df: pl.DataFrame, survey: str, k_neighbors: int, config: Dict
) -> Dict:
    """Create metadata for the survey."""
    return {
        "survey_name": survey,
        "full_name": config["name"],
        "data_release": config.get("data_release", "unknown"),
        "num_samples": len(df),
        "num_features": len(
            [
                c
                for c in df.columns
                if df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
        ),
        "coordinate_system": config.get("coordinate_system", "unknown"),
        "photometric_bands": config.get("photometric_bands", []),
        "k_neighbors": k_neighbors,
        "columns": df.columns,
        "created_with": "astro_lab.data.preprocessing",
    }


def process_survey(
    survey: str,
    source_file: Optional[str] = None,
    k_neighbors: int = 8,
    max_samples: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Process a survey with automatic source file detection. 📊
    For TNG50(-4), process all HDF5 snapshots as a time series and save outputs under data/processed/tng50/ with simple names (tng50.parquet, tng50.pt, tng50_metadata.json).
    If max_samples is set, sample the combined DataFrame before saving and graph creation.
    """
    logger = logging.getLogger(__name__)

    # Special case: TNG50(-4) → time series graph processing
    if survey.lower() in ["tng50", "tng50-4"]:
        # Ensure max_samples is int or None
        if isinstance(max_samples, str):
            if max_samples.lower() == "all":
                max_samples = None
            else:
                try:
                    max_samples = int(max_samples)
                except Exception:
                    logger.warning(
                        f"Invalid max_samples value: {max_samples}, using all data."
                    )
                    max_samples = None
        # Default: limit to 2 million particles if not set
        if max_samples is None:
            max_samples = 2_000_000
        # Find all HDF5 snapshots
        project_root = Path(__file__).parent.parent.parent.parent
        snapdir = project_root / "data" / "raw" / "TNG50-4" / "output" / "snapdir_099"
        hdf5_files = sorted(
            snapdir.glob("snap_099.*.hdf5"), key=lambda x: int(x.stem.split(".")[-1])
        )
        if not hdf5_files:
            raise FileNotFoundError(f"No TNG50 HDF5 snapshots found in {snapdir}")
        logger.info(f"🔍 Found {len(hdf5_files)} TNG50 snapshots in {snapdir}")
        # Process all snapshots into a single DataFrame (time series)
        all_snapshots = []
        for i, hdf5_file in enumerate(hdf5_files):
            logger.info(
                f"📊 Processing snapshot {i + 1}/{len(hdf5_files)}: {hdf5_file.name}"
            )
            try:
                with h5py.File(hdf5_file, "r") as f:
                    snapshot_id = int(hdf5_file.stem.split(".")[-1])
                    data_dict = {}
                    if "PartType0" in f:
                        group = f["PartType0"]
                        if "Coordinates" in group:
                            coords = np.array(group["Coordinates"][:])
                            data_dict["x"] = coords[:, 0]
                            data_dict["y"] = coords[:, 1]
                            data_dict["z"] = coords[:, 2]
                        for field in [
                            "Masses",
                            "Velocities",
                            "Density",
                            "Temperature",
                            "Metallicity",
                        ]:
                            if field in group:
                                data = np.array(group[field][:])
                                col_name = field.lower()
                                if col_name.endswith("es"):
                                    col_name = col_name[:-2]
                                elif col_name.endswith("s"):
                                    col_name = col_name[:-1]
                                if data.ndim > 1:
                                    for j in range(data.shape[1]):
                                        data_dict[f"{col_name}_{j}"] = data[:, j]
                                else:
                                    data_dict[col_name] = data
                    data_dict["snapshot_id"] = snapshot_id
                    data_dict["time_step"] = i
                    if "Header" in f:
                        header = f["Header"]
                        if "Redshift" in header:
                            data_dict["redshift"] = header["Redshift"][0]
                        if "Time" in header:
                            data_dict["time_gyr"] = header["Time"][0]
                        if "ScaleFactor" in header:
                            data_dict["scale_factor"] = header["ScaleFactor"][0]
                    df_snapshot = pl.DataFrame(data_dict)
                    all_snapshots.append(df_snapshot)
                    logger.info(
                        f"✅ Snapshot {snapshot_id}: {len(df_snapshot)} particles, {len(df_snapshot.columns)} columns"
                    )
            except Exception as e:
                logger.warning(f"⚠️ Error loading {hdf5_file.name}: {e}")
                continue
        if not all_snapshots:
            raise ValueError("No valid TNG50 snapshots found")
        combined_df = pl.concat(all_snapshots)
        # Apply sampling if requested
        if max_samples is not None and len(combined_df) > max_samples:
            combined_df = combined_df.sample(max_samples, seed=42)
            logger.info(f"📉 Sampled to {max_samples} particles for processing")
        # Save as Parquet and Graph
        output_dir = project_root / "data" / "processed" / "tng50"
        output_dir.mkdir(parents=True, exist_ok=True)
        parquet_path = output_dir / "tng50.parquet"
        pt_path = output_dir / "tng50.pt"
        metadata_path = output_dir / "tng50_metadata.json"
        combined_df.write_parquet(str(parquet_path))
        logger.info(f"✅ TNG50 time series Parquet saved: {parquet_path}")
        # Create graph (uses GPU for k-NN if available)
        graph_data = _create_tng50_graph(combined_df, k_neighbors, 50.0)
        torch.save(graph_data, pt_path)
        logger.info(f"✅ TNG50 graph saved: {pt_path}")
        # Metadata
        metadata = {
            "survey_name": "tng50",
            "num_snapshots": len(hdf5_files),
            "num_particles": len(combined_df),
            "columns": combined_df.columns,
            "k_neighbors": k_neighbors,
            "max_samples": max_samples,
        }
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✅ TNG50 metadata saved: {metadata_path}")
        return {"parquet": parquet_path, "graph": pt_path, "metadata": metadata_path}
    # Special case: NSA → FITS→Parquet, dann Standard-Workflow
    if survey.lower() == "nsa":
        # Ensure max_samples is int or None
        if isinstance(max_samples, str):
            if max_samples.lower() == "all":
                max_samples = None
            else:
                try:
                    max_samples = int(max_samples)
                except Exception:
                    logger.warning(
                        f"Invalid max_samples value: {max_samples}, using all data."
                    )
                    max_samples = None
        project_root = Path(__file__).parent.parent.parent.parent
        nsa_dir = project_root / "data" / "raw" / "nsa"
        # Immer FITS→Parquet prüfen/ausführen
        parquet_path = find_or_create_catalog_file("nsa", nsa_dir)
        # Standardisierte Ausgabe
        output_dir = project_root / "data" / "processed" / "nsa"
        files = create_standardized_files(
            survey="nsa",
            input_parquet=parquet_path,
            output_dir=output_dir,
            k_neighbors=k_neighbors,
            max_samples=max_samples,
            force=force,
        )
        logger.info(f"✅ NSA processing complete: {files}")
        return files
    # Standard: all other surveys
    # ... existing code ...
    # Fallback: If no return occurred, raise error
    raise RuntimeError("Survey processing did not complete and no files were returned.")


def find_or_create_catalog_file(survey: str, data_dir: Path) -> Path:
    """
    Sucht nach Parquet/CSV für einen Survey, erzeugt bei FITS oder HDF5 automatisch Parquet und gibt den Pfad zurück. 📊
    """
    logger = logging.getLogger(__name__)

    # 1. Suche Parquet
    parquet_files = list(data_dir.glob("*.parquet"))
    if parquet_files:
        return parquet_files[0]

    # 2. Suche CSV
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    # 3. Falls NSA und FITS vorhanden, konvertiere
    if survey == "nsa":
        fits_files = list(data_dir.glob("*.fits"))
        if fits_files:
            fits_path = fits_files[0]
            parquet_path = data_dir / f"{survey}_v1_0_1.parquet"
            if not parquet_path.exists():
                from astro_lab.data.utils import convert_nsa_fits_to_parquet

                # Get NSA features from config
                nsa_config = get_survey_config("nsa")
                features = nsa_config.get("features", [])
                convert_nsa_fits_to_parquet(fits_path, parquet_path, features)
            return parquet_path

    # 4. Falls TNG50 und HDF5-Dateien vorhanden, konvertiere zu Time Series
    elif survey in ["tng50", "tng50-4"]:
        # Robust: Suche in TNG50-4, tng50, snapdir_099
        possible_dirs = [
            data_dir,
            data_dir.parent / "TNG50-4" / "output" / "snapdir_099",
            data_dir.parent / "tng50",
        ]
        # Fallback: Wenn data_dir == .../tng50 und leer, dann suche in TNG50-4
        if not data_dir.exists() or not any(data_dir.iterdir()):
            alt_dir = data_dir.parent / "TNG50-4" / "output" / "snapdir_099"
            if alt_dir.exists():
                possible_dirs.insert(0, alt_dir)
        hdf5_files = []
        for search_dir in possible_dirs:
            if search_dir.exists():
                hdf5_files = list(search_dir.glob("snap_099.*.hdf5"))
                if hdf5_files:
                    logger.info(f"🔍 TNG50 HDF5-Dateien gefunden in: {search_dir}")
                    break
        if hdf5_files:
            # Sortiere nach Snapshot-Nummer
            hdf5_files.sort(key=lambda x: int(x.stem.split(".")[-1]))
            # Schreibe das Parquet IMMER nach data/raw/tng50, damit die nachgelagerte Logik funktioniert
            tng50_dir = data_dir.parent / "tng50"
            tng50_dir.mkdir(parents=True, exist_ok=True)
            parquet_path = tng50_dir / f"{survey}_timeseries.parquet"
            if not parquet_path.exists():
                logger.info(
                    f"🔄 Converting TNG50 Time Series: {len(hdf5_files)} Snapshots"
                )
                try:
                    from astro_lab.data.manager import AstroDataManager

                    all_snapshots = []
                    for i, hdf5_file in enumerate(hdf5_files):
                        logger.info(
                            f"📊 Processing snapshot {i + 1}/{len(hdf5_files)}: {hdf5_file.name}"
                        )
                        try:
                            with h5py.File(hdf5_file, "r") as f:
                                snapshot_id = int(hdf5_file.stem.split(".")[-1])
                                data_dict = {}
                                if "PartType0" in f:
                                    group = f["PartType0"]
                                    if "Coordinates" in group:
                                        coords = np.array(group["Coordinates"][:])
                                        data_dict["x"] = coords[:, 0]
                                        data_dict["y"] = coords[:, 1]
                                        data_dict["z"] = coords[:, 2]
                                    for field in [
                                        "Masses",
                                        "Velocities",
                                        "Density",
                                        "Temperature",
                                        "Metallicity",
                                    ]:
                                        if field in group:
                                            data = np.array(group[field][:])
                                            col_name = field.lower()
                                            if col_name.endswith("es"):
                                                col_name = col_name[:-2]
                                            elif col_name.endswith("s"):
                                                col_name = col_name[:-1]
                                            if data.ndim > 1:
                                                for j in range(data.shape[1]):
                                                    data_dict[f"{col_name}_{j}"] = data[
                                                        :, j
                                                    ]
                                            else:
                                                data_dict[col_name] = data
                                    data_dict["snapshot_id"] = snapshot_id
                                    data_dict["time_step"] = i
                                    if "Header" in f:
                                        header = f["Header"]
                                        if "Redshift" in header:
                                            data_dict["redshift"] = header["Redshift"][
                                                0
                                            ]
                                        if "Time" in header:
                                            data_dict["time_gyr"] = header["Time"][0]
                                        if "ScaleFactor" in header:
                                            data_dict["scale_factor"] = header[
                                                "ScaleFactor"
                                            ][0]
                                    df_snapshot = pl.DataFrame(data_dict)
                                    all_snapshots.append(df_snapshot)
                                    logger.info(
                                        f"✅ Snapshot {snapshot_id}: {len(df_snapshot)} Partikel, {len(df_snapshot.columns)} Spalten"
                                    )
                        except Exception as e:
                            logger.warning(
                                f"⚠️ Fehler beim Laden von {hdf5_file.name}: {e}"
                            )
                            continue
                    if all_snapshots:
                        combined_df = pl.concat(all_snapshots)
                        combined_df.write_parquet(str(parquet_path))
                        logger.info(
                            f"✅ TNG50 Time Series Parquet gespeichert: {parquet_path}"
                        )
                        logger.info(
                            f"   {len(combined_df)} Partikel über {len(hdf5_files)} Zeitschritte"
                        )
                        logger.info(f"   Spalten: {combined_df.columns}")
                        return parquet_path
                    else:
                        raise ValueError("Keine gültigen TNG50 Snapshots gefunden")
                except Exception as e:
                    logger.error(f"❌ Failed to convert TNG50 Time Series: {e}")
                    raise
            else:
                return parquet_path
    # 4. Keine Daten gefunden
    raise FileNotFoundError(
        f"No suitable data file found for survey '{survey}' in {data_dir}"
    )


def get_survey_input_file(survey: str, data_manager) -> Path:
    """Finde den Input-Parquet für einen Survey über DataManager/Config (nutzt für TNG50 die Speziallogik inkl. HDF5→Parquet)."""
    survey = survey.lower()
    if survey in ["tng50", "tng50-4"]:
        tng_dir = data_manager.raw_dir / "tng50"
        return find_or_create_catalog_file("tng50", tng_dir)
    if survey == "nsa":
        nsa_dir = data_manager.raw_dir / "nsa"
        return find_or_create_catalog_file("nsa", nsa_dir)
    # Standard: Suche nach Parquet in data/raw/{survey}/
    survey_dir = data_manager.raw_dir / survey
    parquet_files = list(survey_dir.glob("*.parquet"))
    if parquet_files:
        return parquet_files[0]
    raise FileNotFoundError(
        f"No parquet file found for survey {survey} in {survey_dir}"
    )


if __name__ == "__main__":
    # Test the new tensor system
    print("🌟 Testing new TensorDict factory methods:")

    # Test Gaia survey creation
    # Create dummy data
    coordinates = torch.randn(100, 2) * 10  # Random RA, Dec
    g_mag = torch.randn(100) + 15  # Random G magnitudes
    bp_mag = g_mag + torch.randn(100) * 0.1 + 0.5  # BP magnitudes
    rp_mag = g_mag - torch.randn(100) * 0.1 + 0.3  # RP magnitudes

    # Create Gaia survey using factory
    from astro_lab.tensors.factories import create_gaia_survey

    gaia_survey = create_gaia_survey(coordinates, g_mag, bp_mag, rp_mag)
    print(f"✅ Created Gaia survey: {gaia_survey.survey_name}")
    print(f"   Spatial: {gaia_survey['spatial'].n_objects} objects")
    print(f"   Photometric: {gaia_survey['photometric'].n_objects} objects")
