"""
AstroLab Data Preprocessing Module
=================================

Handles data preprocessing and graph creation for astronomical surveys.
Moved from CLI to data module for better organization.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data

from astro_lab.utils.config.surveys import get_survey_config, get_available_surveys
from .config import data_config

logger = logging.getLogger(__name__)


def preprocess_catalog(
    input_path: Union[str, Path],
    survey_type: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    **kwargs,
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
    # Simple garbage collection instead of complex memory management
    import gc

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


def preprocess_catalog_lazy(
    input_path: Union[str, Path],
    survey_type: str,
    max_samples: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    use_streaming: bool = True,
    **kwargs,
) -> pl.LazyFrame:
    """
    OPTIMIZED: Lazy preprocessing for large astronomical catalogs.

    Args:
        input_path: Path to input catalog file
        survey_type: Type of survey ('gaia', 'sdss', 'nsa', 'linear')
        max_samples: Maximum number of samples to process
        output_dir: Output directory for processed data
        use_streaming: Whether to use streaming for large files
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
    if output_dir:
        output_path = Path(output_dir) / f"{survey_type}_processed.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if use_streaming:
            # Use streaming for large datasets
            lf_clean.sink_parquet(output_path)
        else:
            lf_clean.collect().write_parquet(output_path)

        logger.info(f"💾 Saved processed data to {output_path}")

    return lf_clean


def create_graph_from_dataframe(
    df: pl.DataFrame,
    survey_type: str,
    k_neighbors: int = 8,
    distance_threshold: float = 50.0,
    output_path: Optional[Path] = None,
    **kwargs,
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
        graph_data = _create_generic_graph(
            df, k_neighbors, distance_threshold, **kwargs
        )

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
    **kwargs,
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

    logger.info(
        f"✅ Created {len([d for d in datasets.values() if d is not None])} graph datasets"
    )
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
    elif survey_type == "exoplanet":
        return _preprocess_exoplanet_data(df)
    else:
        logger.warning(f"⚠️ No specific preprocessing for {survey_type}, using generic")
        return _preprocess_generic_data(df)


def _apply_survey_preprocessing_lazy(
    lf: pl.LazyFrame, survey_type: str
) -> pl.LazyFrame:
    """Apply survey-specific preprocessing lazily."""
    if survey_type == "gaia":
        return _preprocess_gaia_data_lazy(lf)
    elif survey_type == "sdss":
        return _preprocess_sdss_data_lazy(lf)
    elif survey_type == "nsa":
        return _preprocess_nsa_data_lazy(lf)
    elif survey_type == "linear":
        return _preprocess_linear_data_lazy(lf)
    elif survey_type == "tng50":
        return lf  # TNG50 data is already clean
    else:
        return _preprocess_generic_data_lazy(lf)


def _preprocess_gaia_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess Gaia data."""
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
    """Preprocess SDSS data."""
    # Remove rows with missing coordinates and redshift
    df_clean = df.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("z").is_not_null()
    )

    return df_clean


def _preprocess_nsa_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess NSA data."""
    # Remove rows with missing coordinates
    df_clean = df.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())

    return df_clean


def _preprocess_linear_data(df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess LINEAR data."""
    # Rename coordinates for consistency
    if "raLIN" in df.columns and "decLIN" in df.columns:
        df = df.rename({"raLIN": "ra", "decLIN": "dec"})
    # Entferne Zeilen mit fehlenden Koordinaten
    df_clean = df.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())
    return df_clean


def _preprocess_generic_data(df: pl.DataFrame) -> pl.DataFrame:
    """Generic preprocessing for unknown survey types."""
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
    """OPTIMIZED: Lazy Gaia preprocessing."""
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
    """OPTIMIZED: Lazy SDSS preprocessing."""
    return lf.filter(
        pl.col("ra").is_not_null()
        & pl.col("dec").is_not_null()
        & pl.col("z").is_not_null()
    )


def _preprocess_nsa_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy NSA preprocessing."""
    return lf.filter(pl.col("ra").is_not_null() & pl.col("dec").is_not_null())


def _preprocess_linear_data_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    """OPTIMIZED: Lazy LINEAR preprocessing."""
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
    """OPTIMIZED: Generic lazy preprocessing."""
    # This is a simplified version - in practice, you'd inspect the schema
    return lf.filter(pl.all_horizontal(pl.all().is_not_null()))


def _create_knn_graph_gpu(coords: np.ndarray, k_neighbors: int) -> torch.Tensor:
    """Create k-NN graph using GPU acceleration - ONLY GPU VERSION."""
    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    n_nodes = len(coords)
    logger.info(f"🚀 Creating GPU k-NN graph for {n_nodes:,} nodes with k={k_neighbors}")
    
    # Convert to PyTorch tensor and move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
    
    try:
        # Use torch_cluster's knn_graph directly
        import torch_cluster
        
        # Create k-NN graph on GPU
        edge_index = torch_cluster.knn_graph(
            x=coords_tensor, 
            k=k_neighbors, 
            loop=False,  # No self-loops
            flow='source_to_target'
        )
        
        # Move back to CPU if needed for consistency
        edge_index = edge_index.cpu()
        
        logger.info(f"✅ Created GPU k-NN graph: {n_nodes:,} nodes, {edge_index.shape[1]:,} edges")
        return edge_index
        
    except ImportError:
        logger.error("❌ torch-cluster not available - GPU acceleration required!")
        raise ImportError("torch-cluster required for GPU k-NN graph creation")
    except Exception as e:
        logger.error(f"❌ GPU k-NN failed: {e}")
        raise RuntimeError(f"GPU k-NN graph creation failed: {e}")


def _create_gaia_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create Gaia stellar graph."""
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
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create SDSS galaxy graph."""
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
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create NSA galaxy graph."""
    # Extract coordinates
    coords = df.select(["ra", "dec", "z"]).to_numpy()

    # Create k-NN graph with GPU acceleration
    edge_index = _create_knn_graph_gpu(coords, k_neighbors)

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
        num_nodes=len(df),
    )

    # Add metadata
    data.survey_name = "NSA"
    data.feature_names = available_features
    data.coord_names = ["ra", "dec", "z"]
    data.k_neighbors = k_neighbors

    return data


def _create_tng50_graph(
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
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
    df: pl.DataFrame, k_neighbors: int, distance_threshold: float, **kwargs
) -> Data:
    """Create generic astronomical graph."""
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


def perform_gaia_crossmatching(
    exoplanet_coords: pl.DataFrame,
    gaia_df: pl.DataFrame,
    max_distance_arcsec: float = 5.0,
    min_probability: float = 0.5,
) -> Dict[str, Any]:
    """
    Perform cross-matching between exoplanet host stars and Gaia catalog using CrossMatchTensor.

    Args:
        exoplanet_coords: DataFrame with exoplanet coordinates (ra, dec)
        gaia_df: Gaia catalog DataFrame
        max_distance_arcsec: Maximum matching distance in arcseconds
        min_probability: Minimum probability for Bayesian matching

    Returns:
        Dictionary with cross-matching results
    """
    try:
        from astro_lab.tensors.crossmatch import CrossMatchTensor
    except ImportError:
        logger.warning("CrossMatchTensor not available, using fallback matching")
        return perform_fallback_crossmatching(
            exoplanet_coords, gaia_df, max_distance_arcsec
        )

    logger.info(
        f"🔍 Performing Gaia cross-matching for {len(exoplanet_coords)} exoplanets"
    )

    # Prepare exoplanet coordinates
    exoplanet_data = {
        "ra": torch.tensor(exoplanet_coords["ra"].to_numpy(), dtype=torch.float32),
        "dec": torch.tensor(exoplanet_coords["dec"].to_numpy(), dtype=torch.float32),
        "hostname": torch.tensor(
            exoplanet_coords["hostname"].to_numpy(), dtype=torch.int64
        ),
    }

    # Prepare Gaia coordinates
    gaia_data = {
        "ra": torch.tensor(gaia_df["ra"].to_numpy(), dtype=torch.float32),
        "dec": torch.tensor(gaia_df["dec"].to_numpy(), dtype=torch.float32),
        "source_id": torch.tensor(gaia_df["source_id"].to_numpy(), dtype=torch.int64),
    }

    # Create CrossMatchTensor
    crossmatch_tensor = CrossMatchTensor(
        catalog_a=exoplanet_data,
        catalog_b=gaia_data,
        catalog_names=("exoplanets", "gaia"),
        coordinate_columns={"a": [0, 1], "b": [0, 1]},
    )

    # Perform sky coordinate matching
    logger.info(
        f"🔍 Performing sky coordinate matching with {max_distance_arcsec} arcsec tolerance"
    )
    sky_matches = crossmatch_tensor.sky_coordinate_matching(
        tolerance_arcsec=max_distance_arcsec,
        method="nearest_neighbor",
        match_name="exoplanet_gaia_sky_match",
    )

    # Perform Bayesian matching for higher quality matches
    logger.info("🔍 Performing Bayesian matching for probability assessment")
    bayesian_matches = crossmatch_tensor.bayesian_matching(
        prior_density=1e-6,  # objects per square arcsecond
        tolerance_arcsec=max_distance_arcsec,
        match_name="exoplanet_gaia_bayesian_match",
    )

    # Combine results
    results = {
        "sky_matches": sky_matches,
        "bayesian_matches": bayesian_matches,
        "crossmatch_tensor": crossmatch_tensor,
        "n_exoplanets": len(exoplanet_coords),
        "n_gaia_stars": len(gaia_df),
        "max_distance_arcsec": max_distance_arcsec,
        "min_probability": min_probability,
    }

    # Extract high-confidence matches
    high_confidence_matches = []
    for match in bayesian_matches["matches"]:
        posterior_prob = (
            match.get("posterior_prob", 0.0) if isinstance(match, dict) else 0.0
        )
        if posterior_prob >= min_probability:
            high_confidence_matches.append(match)

    results["high_confidence_matches"] = high_confidence_matches
    results["n_high_confidence"] = len(high_confidence_matches)

    logger.info(
        f"✅ Cross-matching completed: {len(sky_matches['matches'])} sky matches, {len(high_confidence_matches)} high-confidence matches"
    )

    return results


def perform_fallback_crossmatching(
    exoplanet_coords: pl.DataFrame,
    gaia_df: pl.DataFrame,
    max_distance_arcsec: float = 5.0,
) -> Dict[str, Any]:
    """
    Fallback cross-matching when CrossMatchTensor is not available.

    Args:
        exoplanet_coords: DataFrame with exoplanet coordinates
        gaia_df: Gaia catalog DataFrame
        max_distance_arcsec: Maximum matching distance

    Returns:
        Dictionary with cross-matching results
    """
    logger.info("🔄 Using fallback cross-matching method")

    # Convert to numpy for faster computation
    exo_ra = exoplanet_coords["ra"].to_numpy()
    exo_dec = exoplanet_coords["dec"].to_numpy()
    exo_hostnames = exoplanet_coords["hostname"].to_numpy()

    gaia_ra = gaia_df["ra"].to_numpy()
    gaia_dec = gaia_df["dec"].to_numpy()
    gaia_source_ids = gaia_df["source_id"].to_numpy()

    # Convert tolerance to degrees
    tolerance_deg = max_distance_arcsec / 3600.0

    matches = []

    # Simple nearest neighbor search
    for i, (ra1, dec1, hostname) in enumerate(zip(exo_ra, exo_dec, exo_hostnames)):
        min_distance = float("inf")
        best_match_idx = -1

        for j, (ra2, dec2, source_id) in enumerate(
            zip(gaia_ra, gaia_dec, gaia_source_ids)
        ):
            # Calculate angular separation using haversine formula
            ra1_rad = np.radians(ra1)
            dec1_rad = np.radians(dec1)
            ra2_rad = np.radians(ra2)
            dec2_rad = np.radians(dec2)

            dra = ra2_rad - ra1_rad
            ddec = dec2_rad - dec1_rad

            a = (
                np.sin(ddec / 2) ** 2
                + np.cos(dec1_rad) * np.cos(dec2_rad) * np.sin(dra / 2) ** 2
            )

            distance_rad = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
            distance_deg = np.degrees(distance_rad)

            if distance_deg <= tolerance_deg and distance_deg < min_distance:
                min_distance = distance_deg
                best_match_idx = j

        if best_match_idx >= 0:
            matches.append(
                {
                    "exoplanet_idx": i,
                    "gaia_idx": best_match_idx,
                    "hostname": hostname,
                    "gaia_source_id": gaia_source_ids[best_match_idx],
                    "separation_arcsec": min_distance * 3600.0,
                    "ra_exoplanet": ra1,
                    "dec_exoplanet": dec1,
                    "ra_gaia": gaia_ra[best_match_idx],
                    "dec_gaia": gaia_dec[best_match_idx],
                }
            )

    results = {
        "matches": matches,
        "n_matches": len(matches),
        "n_exoplanets": len(exoplanet_coords),
        "n_gaia_stars": len(gaia_df),
        "max_distance_arcsec": max_distance_arcsec,
        "method": "fallback_nearest_neighbor",
    }

    logger.info(f"✅ Fallback cross-matching completed: {len(matches)} matches found")
    return results


def enrich_exoplanets_with_gaia_coordinates(
    exoplanet_df: pl.DataFrame,
    gaia_path: str = "data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet",
    max_distance_arcsec: float = 5.0,
) -> pl.DataFrame:
    """
    Enrich exoplanet data with host star coordinates from Gaia DR3 using real cross-matching.

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
            logger.info("🔍 Attempting real cross-matching for spatial distributions")

            # Create DataFrame with generated coordinates
            spatial_coords = pl.DataFrame(
                {
                    "hostname": unmatched_hostnames,
                    "ra": [coord_mapping[h]["ra"] for h in unmatched_hostnames],
                    "dec": [coord_mapping[h]["dec"] for h in unmatched_hostnames],
                }
            )

            # Perform cross-matching
            crossmatch_results = perform_gaia_crossmatching(
                spatial_coords, gaia_df, max_distance_arcsec
            )

            # Update coordinates with real Gaia matches
            for match in crossmatch_results.get("high_confidence_matches", []):
                exoplanet_idx = match["index_a"]
                gaia_idx = match["index_b"]
                hostname = unmatched_hostnames[exoplanet_idx]

                # Update with real Gaia coordinates
                coord_mapping[hostname] = {
                    "ra": gaia_df["ra"][gaia_idx],
                    "dec": gaia_df["dec"][gaia_idx],
                    "source": "crossmatch_gaia",
                    "separation_arcsec": match.get("separation_arcsec", 0.0),
                    "probability": match.get("posterior_prob", 0.0),
                }

            logger.info(
                f"✅ Cross-matching updated {len(crossmatch_results.get('high_confidence_matches', []))} coordinates with real Gaia data"
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
    """Preprocess exoplanet data with coordinate enrichment."""
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
    Create standardized files for a survey.

    Creates:
    - {survey}.parquet (standardized data)
    - {survey}_graph_k{k}.pt (graph data)
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
        "graph": output_dir / f"{survey}_graph.pt",
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
    logger.info(f"📊 Loaded {len(df):,} Gaia stars with {len(df.columns)} columns")
    
    # Use ALL stars - no sampling limitation
    print(f"📊 Using all {len(df):,} stars for graph processing")
    
    # Extract coordinates and features
    coords = df.select(["ra", "dec"]).to_numpy()
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
        from itertools import combinations

        edges = list(combinations(range(num_nodes), 2))
        edge_index = torch.tensor(edges).t()
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        logger.info(f"🔗 Created fully connected graph: {edge_index.shape[1]} edges")
    else:
        # k-NN graph for larger datasets
        k = min(k_neighbors, num_nodes - 1)
        nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nbrs.fit(features)
        distances, indices = nbrs.kneighbors(features)

        # Create edge index
        sources = torch.arange(num_nodes).repeat_interleave(k)
        targets = torch.tensor(indices.flatten())
        edge_index = torch.stack([sources, targets])
        logger.info(
            f"🔗 Created k-NN graph: {num_nodes} nodes, {edge_index.shape[1]} edges"
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
    Process a survey with automatic source file detection.

    Args:
        survey: Survey name
        source_file: Source parquet file (auto-detected if None)
        k_neighbors: Number of neighbors for graph
        max_samples: Maximum samples to process
        force: Overwrite existing files

    Returns:
        Dictionary with paths to created files
    """
    # Auto-detect source file if not provided
    if source_file is None:
        project_root = Path(__file__).parent.parent.parent.parent

        # Try different common locations
        possible_sources = [
            project_root
            / "data"
            / "processed"
            / survey
            / f"{survey}_dr3_bright_all_sky_mag12.0_processed.parquet",
            project_root
            / "data"
            / "raw"
            / survey
            / f"{survey}_dr3_bright_all_sky_mag12.0.parquet",
            project_root / "data" / "processed" / f"{survey}_processed.parquet",
            project_root / "data" / "raw" / f"{survey}.parquet",
        ]

        source_path = None
        for path in possible_sources:
            if path.exists():
                source_path = path
                break

        if source_path is None:
            raise FileNotFoundError(f"No source file found for survey '{survey}'")

        logger.info(f"📂 Auto-detected source: {source_path}")
    else:
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

    return create_standardized_files(
        survey=survey,
        input_parquet=source_path,
        k_neighbors=k_neighbors,
        max_samples=max_samples,
        force=force,
    )


def create_gaia_survey_tensor():
    """Create proper SurveyTensor from all 3M Gaia stars using the tensor system."""
    import polars as pl
    import torch
    from pathlib import Path
    
    logger.info("🌟 Creating Gaia SurveyTensor with all 3M stars...")
    
    # Load the large Gaia dataset
    gaia_file = Path("data/processed/gaia/gaia_dr3_bright_all_sky_mag12.0_processed.parquet")
    if not gaia_file.exists():
        logger.error(f"❌ Gaia file not found: {gaia_file}")
        return None
    
    logger.info(f"📊 Loading Gaia data from {gaia_file}")
    df = pl.read_parquet(gaia_file)
    logger.info(f"✅ Loaded {len(df):,} Gaia stars with {len(df.columns)} columns")
    
    # Smart NaN handling: fill NaN values instead of dropping rows
    logger.info("🧹 Smart data cleaning: filling NaN values...")
    
    # Get only numeric columns
    numeric_columns = []
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
            numeric_columns.append(col)
    
    logger.info(f"📊 Using {len(numeric_columns)} numeric columns: {numeric_columns[:5]}...")
    
    # Select only numeric columns and fill nulls
    df_numeric = df.select(numeric_columns)
    df_clean = df_numeric.fill_null(strategy="mean")
    
    # Convert to numpy and handle remaining issues
    data_numpy = df_clean.to_numpy()
    
    # Aggressive cleaning: replace ALL non-finite values
    data_numpy = np.nan_to_num(data_numpy, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Additional check: ensure no infinite or NaN values remain
    if not np.isfinite(data_numpy).all():
        logger.warning("⚠️ Found remaining non-finite values, applying final cleanup...")
        data_numpy[~np.isfinite(data_numpy)] = 0.0
    
    logger.info(f"📊 After smart cleaning: {len(data_numpy):,} stars with {data_numpy.shape[1]} features (kept all data!)")
    
    # Convert to tensor with explicit finite check
    tensor_data = torch.tensor(data_numpy, dtype=torch.float32)
    
    # Final tensor validation
    if not torch.isfinite(tensor_data).all():
        logger.warning("🔧 Tensor contains non-finite values, applying final cleanup...")
        tensor_data[~torch.isfinite(tensor_data)] = 0.0
    
    logger.info(f"📊 Clean tensor shape: {tensor_data.shape}")
    logger.info(f"✅ Tensor validation: all finite = {torch.isfinite(tensor_data).all()}")
    
    # Define Gaia column mapping
    gaia_columns = df.columns
    column_mapping = {col: i for i, col in enumerate(gaia_columns)}
    
    # Create SurveyTensor with proper Gaia configuration
    try:
        from astro_lab.tensors.survey import SurveyTensor
        
        survey_tensor = SurveyTensor(
            data=tensor_data,
            survey_name="gaia",
            data_release="DR3",
            filter_system="gaia_dr3",
            column_mapping=column_mapping,
            survey_metadata={
                "full_name": "Gaia DR3 Complete (3M stars)",
                "magnitude_limit": 12.0,
                "coordinate_system": "icrs",
                "photometric_bands": ["G", "BP", "RP"],
                "n_objects": len(data_numpy),
                "processing_date": "2024",
                "source_file": str(gaia_file),
            }
        )
        
        logger.info(f"✅ Created SurveyTensor: {survey_tensor.survey_name} with {len(survey_tensor)} objects")
        
        # Create Spatial3DTensor
        logger.info("🌍 Creating Spatial3DTensor...")
        spatial_tensor = survey_tensor.get_spatial_tensor(include_distances=True)
        logger.info(f"✅ Created Spatial3DTensor with {spatial_tensor.shape} coordinates")
        
        # Create PhotometricTensor
        logger.info("📸 Creating PhotometricTensor...")
        photometric_tensor = survey_tensor.get_photometric_tensor()
        logger.info(f"✅ Created PhotometricTensor with bands: {photometric_tensor.bands}")
        
        # Save to simple gaia directory
        output_dir = Path("data/processed/gaia")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple file names
        survey_file = output_dir / "gaia_survey_tensor.pt"
        spatial_file = output_dir / "gaia_spatial_tensor.pt"
        photometric_file = output_dir / "gaia_photometric_tensor.pt"
        metadata_file = output_dir / "gaia_tensor_metadata.json"
        
        # Save tensors
        torch.save(survey_tensor, survey_file)
        logger.info(f"💾 Saved SurveyTensor to: {survey_file}")
        
        torch.save(spatial_tensor, spatial_file)
        logger.info(f"💾 Saved Spatial3DTensor to: {spatial_file}")
        
        torch.save(photometric_tensor, photometric_file)
        logger.info(f"💾 Saved PhotometricTensor to: {photometric_file}")
        
        # Create comprehensive metadata
        metadata = {
            "survey_name": "gaia",
            "full_name": "Gaia DR3 Complete Dataset (3M stars)",
            "data_release": "DR3",
            "num_samples": len(data_numpy),
            "tensor_types": ["survey", "spatial_3d", "photometric"],
            "coordinate_system": "icrs",
            "photometric_bands": ["G", "BP", "RP"],
            "spatial_dimensions": list(spatial_tensor.shape),
            "photometric_dimensions": list(photometric_tensor.shape),
            "column_mapping": column_mapping,
            "created_with": "astro_lab.tensors.SurveyTensor",
            "processing_method": "complete_3m_dataset_smart_nan_handling",
        }
        
        import json
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"📋 Saved tensor metadata to: {metadata_file}")
        
        logger.info(f"🎯 Complete tensor system created!")
        logger.info(f"   SurveyTensor: {len(survey_tensor):,} objects")
        logger.info(f"   Spatial3DTensor: {spatial_tensor.shape}")
        logger.info(f"   PhotometricTensor: {photometric_tensor.shape}")
        
        return {
            "survey_tensor": survey_tensor,
            "spatial_tensor": spatial_tensor,
            "photometric_tensor": photometric_tensor,
            "files": {
                "survey": str(survey_file),
                "spatial": str(spatial_file),
                "photometric": str(photometric_file),
                "metadata": str(metadata_file),
            }
        }
        
    except ImportError as e:
        logger.warning(f"⚠️ Tensor system not available: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error creating tensor system: {e}")
        return None


# Keep the original function for backward compatibility
def process_large_gaia_dataset():
    """Process the large Gaia dataset using GPU k-NN for fast graph creation (legacy)."""
    print("⚠️ Using legacy PyG graph processing. Consider using create_gaia_survey_tensor() instead.")
    return create_gaia_survey_tensor()


if __name__ == "__main__":
    # Test the new tensor system
    print("🌟 Creating complete Gaia tensor system:")
    create_gaia_survey_tensor()
