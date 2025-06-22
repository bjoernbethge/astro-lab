"""
AstroLab Data Preprocessing Module
=================================

Handles data preprocessing and graph creation for astronomical surveys.
Moved from CLI to data module for better organization.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from .config import data_config
from .core import SURVEY_CONFIGS

logger = logging.getLogger(__name__)


def preprocess_catalog(
    input_path: Union[str, Path],
    survey_type: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs
) -> pl.DataFrame:
    """
    Preprocess astronomical catalog data.
    
    Args:
        input_path: Path to input catalog file
        survey_type: Type of survey ('gaia', 'sdss', 'nsa', 'linear')
        max_samples: Maximum number of samples to process
        output_dir: Output directory for processed data
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
    if output_dir:
        output_path = Path(output_dir) / f"{survey_type}_processed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_clean.write_parquet(output_path)
        logger.info(f"💾 Saved processed data to {output_path}")
    
    return df_clean


def create_graph_from_dataframe(
    df: pl.DataFrame,
    survey_type: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    output_path: Optional[Path] = None,
    **kwargs
) -> Optional[Data]:
    """
    Create PyTorch Geometric graph from Polars DataFrame.
    
    Args:
        df: Input DataFrame
        survey_type: Type of survey
        k_neighbors: Number of nearest neighbors
        distance_threshold: Distance threshold for edges
        output_path: Path to save graph
        **kwargs: Additional parameters
        
    Returns:
        PyTorch Geometric Data object
    """
    logger.info(f"🔄 Creating graph for {survey_type} with k={k_neighbors}")
    
    # Apply survey-specific graph creation
    if survey_type == "nsa":
        graph_data = _create_nsa_graph(df, k_neighbors, distance_threshold, **kwargs)
    elif survey_type == "gaia":
        graph_data = _create_gaia_graph(df, k_neighbors, distance_threshold, **kwargs)
    elif survey_type == "sdss":
        graph_data = _create_sdss_graph(df, k_neighbors, distance_threshold, **kwargs)
    elif survey_type == "tng50":
        graph_data = _create_tng50_graph(df, k_neighbors, distance_threshold, **kwargs)
    else:
        graph_data = _create_generic_graph(df, k_neighbors, distance_threshold, **kwargs)
    
    # Save graph if output path provided
    if output_path and graph_data:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_path)
        logger.info(f"💾 Saved graph to {output_path}")
    
    return graph_data


def create_graph_datasets_from_splits(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    output_path: Path,
    dataset_name: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    **kwargs
) -> Dict[str, Optional[Data]]:
    """
    Create graph datasets from train/val/test splits.
    
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
    
    logger.info(f"✅ Created {len([d for d in datasets.values() if d is not None])} graph datasets")
    return datasets


def _apply_survey_preprocessing(df: pl.DataFrame, survey_type: str) -> pl.DataFrame:
    """Apply survey-specific preprocessing."""
    if survey_type == "gaia":
        return _preprocess_gaia_data(df)
    elif survey_type == "sdss":
        return _preprocess_sdss_data(df)
    elif survey_type == "nsa":
        return _preprocess_nsa_data(df)
    elif survey_type == "linear":
        return _preprocess_linear_data(df)
    else:
        logger.warning(f"⚠️ No specific preprocessing for {survey_type}, using generic")
        return _preprocess_generic_data(df)


def _preprocess_gaia_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess Gaia data."""
    # Remove rows with missing coordinates
    df_clean = df.filter(
        pl.col("ra").is_not_null() & 
        pl.col("dec").is_not_null() &
        pl.col("phot_g_mean_mag").is_not_null()
    )
    
    # Add color features
    if "phot_bp_mean_mag" in df_clean.columns and "phot_rp_mean_mag" in df_clean.columns:
        df_clean = df_clean.with_columns([
            (pl.col("phot_bp_mean_mag") - pl.col("phot_rp_mean_mag")).alias("bp_rp_color")
        ])
    
    return df_clean


def _preprocess_sdss_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess SDSS data."""
    # Remove rows with missing coordinates and redshift
    df_clean = df.filter(
        pl.col("ra").is_not_null() & 
        pl.col("dec").is_not_null() &
        pl.col("z").is_not_null()
    )
    
    return df_clean


def _preprocess_nsa_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess NSA data."""
    # Remove rows with missing coordinates
    df_clean = df.filter(
        pl.col("ra").is_not_null() & 
        pl.col("dec").is_not_null()
    )
    
    return df_clean


def _preprocess_linear_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess LINEAR data."""
    # Remove rows with missing coordinates
    df_clean = df.filter(
        pl.col("ra").is_not_null() & 
        pl.col("dec").is_not_null()
    )
    
    return df_clean


def _preprocess_generic_data(df: pl.DataFrame) -> pl.DataFrame:
    """Generic preprocessing for unknown survey types."""
    # Remove rows with missing coordinates
    coord_cols = [col for col in df.columns if col.lower() in ["ra", "dec", "x", "y", "z"]]
    if coord_cols:
        df_clean = df.filter(pl.all_horizontal(pl.col(coord_cols).is_not_null()))
    else:
        df_clean = df
    
    return df_clean


def _create_knn_graph(coords: np.ndarray, k_neighbors: int) -> torch.Tensor:
    """Create k-NN graph from coordinates."""
    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Use haversine distance for 2D sky coordinates
    if coords.shape[1] == 2:
        coords_rad = np.radians(coords)
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)), metric="haversine")
    else:
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(coords)))
    
    nbrs.fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Build edge index
    edges = []
    for i, neighbors in enumerate(indices):
        for j in neighbors[1:]:  # Skip self
            edges.append([i, j])
    
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _create_gaia_graph(df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs) -> Data:
    """Create Gaia stellar graph."""
    # Extract coordinates
    coords = df.select(["ra", "dec"]).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph(coords, k_neighbors)
    
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
        labels = np.digitize(bp_rp, bins=np.array([-0.5, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0])) - 1
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
        num_nodes=len(df)
    )
    
    # Add metadata
    data.survey_name = "Gaia"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec"]
    data.k_neighbors = k_neighbors
    
    return data


def _create_sdss_graph(df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs) -> Data:
    """Create SDSS galaxy graph."""
    # Extract coordinates
    coords = df.select(["ra", "dec", "z"]).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph(coords, k_neighbors)
    
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
        num_nodes=len(df)
    )
    
    # Add metadata
    data.survey_name = "SDSS"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec", "z"]
    data.k_neighbors = k_neighbors
    
    return data


def _create_nsa_graph(df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs) -> Data:
    """Create NSA galaxy graph."""
    # Extract coordinates
    coords = df.select(["ra", "dec", "z"]).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph(coords, k_neighbors)
    
    # Prepare features
    feature_cols = ["mag_r", "mag_g", "mag_i", "mass", "sersic_n"]
    available_features = [col for col in feature_cols if col in df.columns]
    
    if not available_features:
        available_features = ["mag_r"]  # Fallback
    
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
        num_nodes=len(df)
    )
    
    # Add metadata
    data.survey_name = "NSA"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec", "z"]
    data.k_neighbors = k_neighbors
    
    return data


def _create_tng50_graph(df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs) -> Data:
    """Create TNG50 simulation graph."""
    # Extract 3D coordinates
    coords = df.select(["x", "y", "z"]).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph(coords, k_neighbors)
    
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
        num_nodes=len(df)
    )
    
    # Add metadata
    data.survey_name = "TNG50"
    data.feature_names = available_features
    data.coord_names = ["x", "y", "z"]
    data.k_neighbors = k_neighbors
    
    return data


def _create_generic_graph(df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs) -> Data:
    """Create generic astronomical graph."""
    # Find coordinate columns
    coord_cols = [col for col in df.columns if col.lower() in ["ra", "dec", "x", "y", "z"]]
    if not coord_cols:
        raise ValueError("No coordinate columns found in DataFrame")
    
    # Extract coordinates
    coords = df.select(coord_cols).to_numpy()
    
    # Create k-NN graph
    edge_index = _create_knn_graph(coords, k_neighbors)
    
    # Use all numeric columns as features
    numeric_cols = [col for col in df.columns if col not in coord_cols and df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]]
    
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
        num_nodes=len(df)
    )
    
    # Add metadata
    data.survey_name = "Generic"
    data.feature_names = numeric_cols if numeric_cols else ["dummy"]
    data.coord_names = coord_cols
    data.k_neighbors = k_neighbors
    
    return data 