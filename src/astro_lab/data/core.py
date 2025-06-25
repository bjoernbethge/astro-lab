"""
AstroLab Data Core Module
=========================

Core data handling and processing for astronomical datasets.
Clean PyTorch Geometric dataset implementation with Polars support.
Updated for TensorDict architecture.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import astropy.units as u
import lightning as L
import numpy as np
import polars as pl
import torch
import torch_cluster
import torch_geometric
import torch_geometric.transforms as T
from astropy.coordinates import SkyCoord
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr
from torch_geometric.loader import DataLoader

# Import TensorDict classes directly - no fallbacks needed
from ..tensors import (
    ClusteringTensorDict,
    CrossMatchTensorDict,
    EarthSatelliteTensorDict,
    FeatureTensorDict,
    LightcurveTensorDict,
    PhotometricTensorDict,
    SimulationTensorDict,
    SpatialTensorDict,
    StatisticsTensorDict,
    SurveyTensorDict,
)

# Import survey configurations from centralized location
from ..utils.config.surveys import get_survey_config
from .config import data_config

logger = logging.getLogger(__name__)

# Set environment variable for NumPy 2.x compatibility with bpy and other modules
os.environ["NUMPY_EXPERIMENTAL_ARRAY_API"] = "1"

# Configure logging
logger = logging.getLogger(__name__)

# Add PyG classes to safe globals for torch.load
torch.serialization.add_safe_globals(
    [
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.Data,
        torch_geometric.data.batch.Batch,
        torch_geometric.data.data.DataTensorAttr,
        torch_geometric.data.data.DataEdgeAttr,
        torch_geometric.data.data.Data,
        torch_geometric.data.batch.Batch,
        torch_geometric.data.hetero_data.HeteroData,
        torch_geometric.data.storage.BaseStorage,
        torch_geometric.data.storage.NodeStorage,
        torch_geometric.data.storage.EdgeStorage,
        torch_geometric.data.storage.GlobalStorage,
    ]
)

# =========================================================================
# 🚀 PERFORMANCE OPTIMIZATION - CUDA, Polars, PyTorch 2025 Best Practices
# =========================================================================


def get_optimal_device() -> torch.device:
    """Get optimal device with CUDA optimization."""
    if torch.cuda.is_available():
        # Set CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Use first available GPU
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def get_optimal_batch_size(
    dataset_size: int, model_complexity: int = 64, memory_safety_factor: float = 0.8
) -> int:
    """
    Calculate optimal batch size based on available memory and dataset size.

    Args:
        dataset_size: Number of samples in dataset
        model_complexity: Hidden dimension or complexity factor
        memory_safety_factor: Safety factor for memory usage (0.8 = 80% of available)

    Returns:
        Optimal batch size
    """
    if torch.cuda.is_available():
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        gpu_memory_gb = gpu_memory / (1024**3)

        # Estimate memory per sample (rough heuristic)
        memory_per_sample = model_complexity * 4  # bytes per sample
        max_samples = int(
            (gpu_memory_gb * 1024**3 * memory_safety_factor) / memory_per_sample
        )

        # Conservative batch size
        optimal_batch_size = min(max_samples, dataset_size, 128)

        # Ensure reasonable minimum
        return max(optimal_batch_size, 8)
    else:
        # CPU: use smaller batches
        return min(dataset_size, 32)


def get_optimal_num_workers() -> int:
    """Get optimal number of workers for DataLoader."""
    cpu_count = os.cpu_count() or 4

    if torch.cuda.is_available():
        # GPU: fewer workers since GPU is bottleneck
        return min(cpu_count // 2, 4)
    else:
        # CPU: more workers for parallel processing
        return min(cpu_count - 1, 8)


# Initialize optimizations
# optimize_polars_settings()  # CAUSES MEMORY LEAKS - removed
# optimize_torch_settings()   # Keep this separate

# =========================================================================
# 🌟 COSMIC WEB ANALYSIS FUNCTIONS - Density-based for all surveys
# =========================================================================


def calculate_local_density(
    positions: Union[torch.Tensor, np.ndarray],
    radius_pc: float = 1000.0,
    max_neighbors: int = 100,
) -> torch.Tensor:
    """
    Calculate local density for each object using GPU acceleration.

    Args:
        positions: 3D positions (N, 3) in Mpc
        radius_pc: Radius for local density calculation in pc
        max_neighbors: Maximum neighbors to consider

    Returns:
        Local density for each object in obj/pc³
    """
    # Convert to pc for local density calculation
    positions_pc = positions * 1e6  # Mpc to pc

    # Use GPU-accelerated radius search
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions_tensor = torch.tensor(positions_pc, dtype=torch.float32, device=device)

    # Create radius graph
    edge_index = torch_cluster.radius_graph(
        x=positions_tensor, r=radius_pc, loop=False, flow="source_to_target"
    )

    # Count neighbors for each point
    neighbor_counts = []
    for i in range(len(positions_pc)):
        # Count edges where this point is the source
        n_neighbors = (edge_index[0] == i).sum().item()
        neighbor_counts.append(n_neighbors)

    # Vectorized density calculation
    volume = (4 / 3) * np.pi * (radius_pc**3)
    densities = np.array(neighbor_counts) / volume

    return torch.tensor(densities, dtype=torch.float32)


def adaptive_cosmic_web_clustering(
    spatial_tensor: Any,  # Spatial3DTensor or fallback
    coords_3d: np.ndarray,
    scale_mpc: float,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Adaptive Cosmic Web Clustering based on local density.

    Args:
        spatial_tensor: Spatial3DTensor with clustering methods
        coords_3d: 3D coordinates in Mpc
        scale_mpc: Scale for clustering in Mpc
        verbose: Verbose output

    Returns:
        Tuple of (clustering results, local density)
    """
    # Calculate local density
    radius_pc = scale_mpc * 1_000_000  # Scale in pc
    local_density = calculate_local_density(coords_3d, radius_pc)

    # Adaptive parameters based on density
    mean_density = np.mean(local_density)
    std_density = np.std(local_density)

    # eps based on density variation
    eps_pc = radius_pc * (1 + std_density / max(mean_density, 1e-30))

    # min_samples based on local density
    min_samples = max(2, int(mean_density * 0.1))

    if verbose:
        logger.info(
            f"  📊 Local density: {mean_density:.2e} ± {std_density:.2e} obj/pc³"
        )
        logger.info(
            f"  🎯 Adaptive eps: {eps_pc / 1_000_000:.1f} Mpc, min_samples: {min_samples}"
        )

    # Density-based clustering
    results = spatial_tensor.cosmic_web_clustering(
        eps_pc=eps_pc,
        min_samples=min_samples,
        algorithm="dbscan",
    )

    # Ensure local_density is a numpy array
    if isinstance(local_density, torch.Tensor):
        local_density_np = local_density.detach().cpu().numpy()
    else:
        local_density_np = np.asarray(local_density)

    return results, local_density_np


def analyze_cosmic_web(
    survey_tensor: Any,  # SurveyTensor or fallback
    scales_mpc: List[float] = [5.0, 10.0, 20.0, 50.0],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Complete Cosmic Web analysis for a SurveyTensor.

    Args:
        survey_tensor: SurveyTensor with spatial data
        scales_mpc: List of scales for multi-scale analysis
        verbose: Verbose output

    Returns:
        Dictionary with all analysis results
    """
    if verbose:
        logger.info(f"🌌 COSMIC WEB ANALYSIS: {survey_tensor.survey_name}")
        logger.info("=" * 50)

    # Extract spatial tensor
    spatial_tensor = survey_tensor.get_spatial_tensor()
    coords_3d = spatial_tensor.cartesian.detach().cpu().numpy()

    # OPTIMIZED: Early exit for small datasets
    if len(coords_3d) < 10:
        logger.warning("⚠️ Dataset too small for cosmic web analysis")
        return {
            "survey_name": survey_tensor.survey_name,
            "n_objects": len(coords_3d),
            "error": "Dataset too small",
            "results_by_scale": {},
        }

    # OPTIMIZED: Cached global metrics computation
    coord_min = coords_3d.min(axis=0)
    coord_max = coords_3d.max(axis=0)
    coord_range = coord_max - coord_min
    total_volume = np.prod(coord_range)
    global_density = len(coords_3d) / total_volume

    if verbose:
        logger.info(f"📊 Dataset: {len(coords_3d):,} objects")
        logger.info(f"📏 Volume: {total_volume:.0f} Mpc³")
        logger.info(f"🌍 Global density: {global_density:.2e} obj/Mpc³")

    # Multi-scale clustering
    results_summary = {}

    for scale in scales_mpc:
        if verbose:
            logger.info(f"\n🕸️ Scale {scale} Mpc:")

        start_time = time.time()

        try:
            results, local_density = adaptive_cosmic_web_clustering(
                spatial_tensor, coords_3d, scale, verbose=verbose
            )

            cluster_time = time.time() - start_time

            n_groups = results["n_clusters"]
            n_noise = results["n_noise"]

            # OPTIMIZED: Vectorized statistics computation
            if isinstance(local_density, torch.Tensor):
                local_density_np = local_density.detach().cpu().numpy()
            else:
                local_density_np = np.asarray(local_density)

            results_summary[scale] = {
                "n_clusters": n_groups,
                "n_noise": n_noise,
                "grouped_fraction": (len(coords_3d) - n_noise) / len(coords_3d),
                "time_s": cluster_time,
                "mean_local_density": float(np.mean(local_density_np)),
                "density_variation": float(np.std(local_density_np)),
                "local_density_stats": {
                    "min": float(np.min(local_density_np)),
                    "max": float(np.max(local_density_np)),
                    "median": float(np.median(local_density_np)),
                },
            }

            if verbose:
                logger.info(f"  ⏱️ Completed in {cluster_time:.1f}s")
                logger.info(
                    f"  Groups: {n_groups}, Grouped: {(len(coords_3d) - n_noise) / len(coords_3d) * 100:.1f}%"
                )

        except Exception as e:
            if verbose:
                logger.error(f"  ❌ Error: {e}")
            continue

    return {
        "survey_name": survey_tensor.survey_name,
        "n_objects": len(coords_3d),
        "coordinates": coords_3d,
        "total_volume": total_volume,
        "global_density": global_density,
        "results_by_scale": results_summary,
    }


def create_cosmic_web_loader(
    survey: str,
    max_samples: Optional[int] = None,
    scales_mpc: List[float] = [5.0, 10.0, 20.0, 50.0],
    device: Optional[torch.device] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Convenience function: Load survey and perform cosmic web analysis.

    Args:
        survey: Survey name ('gaia', 'sdss', 'nsa', 'linear', etc.)
        max_samples: Maximum number of objects
        scales_mpc: Scales for multi-scale analysis
        device: Device for processing (auto-detect if None)
        **kwargs: Additional parameters for survey loading

    Returns:
        Cosmic web analysis results
    """
    # Auto-detect optimal device
    device = device or get_optimal_device()

    logger.info(f"🚀 Using device: {device}")
    if device.type == "cuda":
        logger.info(
            f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )

    # Use survey-specific loaders
    if survey == "gaia":
        survey_tensor = load_survey_data(
            survey="gaia", max_samples=max_samples, return_tensor=True, **kwargs
        )
    elif survey == "sdss":
        survey_tensor = load_survey_data(
            survey="sdss", max_samples=max_samples, return_tensor=True, **kwargs
        )
    elif survey == "nsa":
        survey_tensor = load_survey_data(
            survey="nsa", max_samples=max_samples, return_tensor=True, **kwargs
        )
    elif survey == "linear":
        # Create SurveyTensor directly for LINEAR
        data_path = Path("data/raw/linear/linear_raw.parquet")
        if not data_path.exists():
            raise FileNotFoundError(f"LINEAR data not found: {data_path}")

        df = pl.read_parquet(data_path)
        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, seed=42)

        survey_tensor = _polars_to_survey_tensor(df, "linear")
    elif survey == "tng":
        survey_tensor = load_tng50_data(max_samples=max_samples)
        # Convert to SurveyTensor if needed
        if not hasattr(survey_tensor, "get_spatial_tensor"):
            # Create SurveyTensor from TNG data
            coords = survey_tensor[0].pos.detach().cpu().numpy()
            survey_tensor = SpatialTensorDict(coordinates=coords, unit="Mpc")
    elif survey == "exoplanet":
        # Load exoplanet data
        data_path = Path(
            "data/processed/exoplanet_graphs/raw/confirmed_exoplanets.parquet"
        )
        if not data_path.exists():
            raise FileNotFoundError(f"Exoplanet data not found: {data_path}")

        df = pl.read_parquet(data_path)
        df = df.filter(pl.col("sy_dist").is_not_null() & (pl.col("sy_dist") > 0))

        if max_samples and len(df) > max_samples:
            df = df.sample(max_samples, seed=42)

        survey_tensor = _polars_to_survey_tensor(df, "exoplanet")
    else:
        # Use generic loader for any other survey
        survey_tensor = load_survey_data(
            survey=survey, max_samples=max_samples, return_tensor=True, **kwargs
        )

    # Move to optimal device if possible
    if hasattr(survey_tensor, "to") and device.type == "cuda":
        try:
            survey_tensor = survey_tensor.to(device)
        except Exception as e:
            logger.warning(f"⚠️ Could not move to GPU: {e}")

    # Perform cosmic web analysis
    results = analyze_cosmic_web(survey_tensor, scales_mpc=scales_mpc, verbose=True)

    return results


# Survey configurations - DRY principle with TENSOR METADATA
def _polars_to_survey_tensor(
    df: pl.DataFrame, survey: str, survey_metadata: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Convert Polars DataFrame to SurveyTensor with survey-specific metadata.

    Args:
        df: Polars DataFrame with astronomical data
        survey: Survey name ('gaia', 'sdss', etc.)
        survey_metadata: Additional metadata

    Returns:
        SurveyTensor with properly configured metadata
    """

    config = get_survey_config(survey)

    # Convert to PyTorch tensor
    tensor_data = torch.tensor(df.to_numpy(), dtype=torch.float32)

    # Prepare metadata
    metadata = {
        "survey_name": survey,
        "data_release": config["data_release"],
        "filter_system": config["filter_system"],
        "column_mapping": config["tensor_column_mapping"],
        "coordinate_system": config["coordinate_system"],
        "photometric_bands": config["photometric_bands"],
        "n_objects": len(df),
        "survey_metadata": {
            "original_columns": df.columns,
            "data_shape": df.shape,
            "coord_cols": config["coord_cols"],
            "mag_cols": config["mag_cols"],
            "extra_cols": config["extra_cols"],
        },
    }

    if survey_metadata:
        metadata["survey_metadata"].update(survey_metadata)

    return SurveyTensorDict(data=tensor_data, **metadata)


class AstroDataset(InMemoryDataset):
    """
    Minimal PyTorch Geometric dataset für astronomische Daten.
    Lädt Daten einmal beim Init und hält sie im Speicher.
    """

    def __init__(
        self,
        root: str,
        survey: str,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        use_streaming: bool = True,
    ):
        self.survey = survey
        self.k_neighbors = k_neighbors
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.use_streaming = use_streaming
        self.survey_config = get_survey_config(survey)
        super().__init__(root, transform, pre_transform, pre_filter)
        self._processed_dir = Path(self.root) / "processed" / self.survey
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._pt_path = self._processed_dir / f"{self.survey}.pt"
        self._parquet_path = self._processed_dir / f"{self.survey}.parquet"
        self._load_data()

    def _load_data(self):
        """Lädt Daten einmal beim Initialisieren."""
        if self._pt_path.exists():
            print(f"🔄 Lade Graph aus: {self._pt_path}")
            try:
                graph_data = torch.load(self._pt_path, weights_only=False)
                self.data, self.slices = self.collate([graph_data])
                return
            except Exception:
                print("⚠️ Fehler beim Laden von .pt, nutze Parquet...")
        if not self._parquet_path.exists():
            raise FileNotFoundError(f"Parquet nicht gefunden: {self._parquet_path}")
        print(f"🔄 Lade Parquet: {self._parquet_path}")
        df = pl.read_parquet(self._parquet_path)
        graph_data = self._create_graph_from_dataframe(df)
        self.data, self.slices = self.collate([graph_data])
        torch.save(graph_data, self._pt_path)
        print(f"💾 Graph gespeichert: {self._pt_path}")

    def _create_graph_from_dataframe(self, df: pl.DataFrame) -> Data:
        coord_cols = self.survey_config.get("coordinates", ["ra", "dec"])
        feature_cols = self.survey_config.get("features", coord_cols)
        coords = df.select([c for c in coord_cols if c in df.columns]).to_numpy()
        features = (
            df.select([c for c in feature_cols if c in df.columns]).to_numpy()
            if feature_cols
            else coords
        )
        if features.shape[1] == 0:
            features = coords
        pos = torch.tensor(coords, dtype=torch.float32)
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.zeros(len(df), dtype=torch.long)
        device = get_optimal_device()
        pos_device = pos.to(device)
        edge_index = torch_cluster.knn_graph(pos_device, k=self.k_neighbors, batch=None)
        edge_index = edge_index.cpu()
        data = Data(x=x, pos=pos, edge_index=edge_index, y=y, num_nodes=len(pos))
        data.survey_name = self.survey
        data.survey = self.survey
        data.k_neighbors = self.k_neighbors
        return data

    def len(self) -> int:
        return 1 if hasattr(self.data, "num_nodes") else 0

    def get(self, idx: int) -> Data:
        if idx == 0:
            return self.data
        else:
            raise IndexError(f"Index {idx} out of range for single graph dataset")

    def get_info(self) -> Dict[str, Any]:
        if len(self) == 0:
            return {"error": "Dataset empty", "columns": []}

        sample = self[0]

        # Get standard survey columns based on survey type
        standard_columns = self._get_standard_columns_for_survey()

        return {
            "survey": self.survey,
            "num_samples": len(self),
            "num_nodes": sample.num_nodes if hasattr(sample, "num_nodes") else 0,
            "num_edges": sample.edge_index.shape[1]
            if hasattr(sample, "edge_index")
            else 0,
            "num_features": sample.x.shape[1] if hasattr(sample, "x") else 0,
            "k_neighbors": self.k_neighbors,
            "columns": standard_columns,  # Add columns info
        }

    def _get_standard_columns_for_survey(self) -> List[str]:
        """Get standard column names for the survey type."""
        try:
            from astro_lab.utils.config.surveys import get_survey_config

            config = get_survey_config(self.survey)
            # Combine coordinate, magnitude, and extra columns
            columns = []
            columns.extend(config.get("coord_cols", ["ra", "dec"]))
            columns.extend(config.get("mag_cols", []))
            columns.extend(config.get("extra_cols", []))
            return columns
        except Exception:
            # Fallback for unknown surveys
            return (
                ["ra", "dec", "mag"]
                if self.survey in ["gaia", "nsa", "exoplanet"]
                else []
            )


def load_survey_data(
    survey: str,
    max_samples: Optional[int] = None,
    return_tensor: bool = True,
    **kwargs: Any,
) -> Union[AstroDataset, "SurveyTensorDict"]:
    """
    Load survey data as an AstroDataset or SurveyTensor.
    Generic function for all surveys.
    """
    from .manager import data_manager

    # Ensure root is passed to AstroDataset
    if "root" not in kwargs:
        kwargs["root"] = str(data_config.base_dir)

    dataset = AstroDataset(survey=survey, max_samples=max_samples, **kwargs)

    if return_tensor:
        # Assumes process() creates a tensor accessible via dataset
        # This part needs to be aligned with AstroDataset implementation
        if hasattr(dataset, "to_tensor"):
            return dataset.to_tensor()
        else:
            # Fallback or error
            raise NotImplementedError(
                "AstroDataset must have a method to convert to SurveyTensor"
            )
    return dataset


def load_tng50_data(
    max_samples: Optional[int] = None,
    particle_type: str = "PartType0",
    return_tensor: bool = False,
) -> Union[AstroDataset, Any]:
    """Load TNG50 simulation data as a survey-like dataset.

    Args:
        max_samples: Maximum number of particles to load (None = all)
        particle_type: Particle type to load (PartType0, PartType1, PartType4, PartType5)
        return_tensor: Whether to return as tensor instead of AstroDataset

    Returns:
        AstroDataset or tensor with TNG50 simulation data
    """
    config = get_survey_config("tng50")

    # Load TNG50 data from processed files
    data_path = Path("data/processed/tng50_combined.parquet")
    if not data_path.exists():
        print(f"⚠️ TNG50 data not found at {data_path}. Generating demo data...")
        # Create AstroDataset with survey parameter and generate demo data
        dataset = AstroDataset(
            survey="tng50", max_samples=max_samples, return_tensor=False
        )
        dataset.download()  # Generate demo data
        # Don't set dataset.data directly - let PyG handle it properly
        return dataset

    # Load with Polars
    df = pl.read_parquet(data_path)

    # Filter by particle type if specified
    if particle_type and "particle_type" in df.columns:
        df = df.filter(pl.col("particle_type") == particle_type)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=42)

    # Create AstroDataset
    dataset = AstroDataset(
        name=config["name"],
        data=df,
        coord_cols=config["coord_cols"],
        mag_cols=config["mag_cols"],
        extra_cols=config["extra_cols"],
        color_pairs=config["color_pairs"],
        tensor_metadata=config,
    )

    if return_tensor:
        # Return as tensor if requested
        return dataset.get_spatial_tensor()

    return dataset


def load_tng50_temporal_data(
    max_samples: Optional[int] = None, snapshot_id: Optional[int] = None
) -> AstroDataset:
    """
    Load TNG50 temporal simulation data.

    Returns:
        AstroDataset with TNG50 temporal simulation data
    """

    # Load TNG50 temporal data from processed files
    data_path = Path(
        "data/processed/tng50_temporal_100mb/processed/tng50_temporal_graphs_r1.0.pt"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"TNG50 temporal data not found at {data_path}. Run preprocessing first."
        )

    # Load PyTorch tensors
    temporal_data = torch.load(data_path, weights_only=False)

    # Extract data from temporal structure
    if snapshot_id is not None:
        # Load specific snapshot
        if snapshot_id >= len(temporal_data):
            raise ValueError(
                f"Snapshot {snapshot_id} not available. Max: {len(temporal_data) - 1}"
            )

        graph_data = temporal_data[snapshot_id]

        # Handle dict format (actual format of the data)
        if isinstance(graph_data, dict):
            positions = graph_data["x"][:, :3].numpy()  # x, y, z
            masses = graph_data["x"][:, 3].numpy()  # mass
            velocities = graph_data["x"][:, 4:7].numpy()  # vx, vy, vz
            particle_types = graph_data["x"][:, 7].numpy()  # particle_type

            # Get cosmological parameters
            redshifts = graph_data.get("redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = graph_data.get("time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = graph_data.get("scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        elif hasattr(graph_data, "x"):
            # Standard PyTorch Geometric format
            positions = graph_data.x[:, :3].numpy()  # x, y, z
            masses = graph_data.x[:, 3].numpy()  # mass
            velocities = graph_data.x[:, 4:7].numpy()  # vx, vy, vz
            particle_types = graph_data.x[:, 7].numpy()  # particle_type

            # Get cosmological parameters
            redshifts = getattr(graph_data, "redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = getattr(graph_data, "time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = getattr(graph_data, "scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        elif hasattr(graph_data, "pos"):
            # Alternative format with pos attribute
            positions = graph_data.pos[:, :3].numpy()
            masses = (
                graph_data.mass.numpy()
                if hasattr(graph_data, "mass")
                else np.ones(len(positions))
            )
            velocities = (
                graph_data.vel.numpy()
                if hasattr(graph_data, "vel")
                else np.zeros((len(positions), 3))
            )
            particle_types = (
                graph_data.particle_type.numpy()
                if hasattr(graph_data, "particle_type")
                else np.zeros(len(positions))
            )

            # Get cosmological parameters
            redshifts = getattr(graph_data, "redshift", 0.0)
            if hasattr(redshifts, "item"):
                redshifts = redshifts.item()
            elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                redshifts = (
                    redshifts[0].item()
                    if hasattr(redshifts[0], "item")
                    else float(redshifts[0])
                )
            else:
                redshifts = 0.0

            time_gyr = getattr(graph_data, "time_gyr", 0.0)
            if hasattr(time_gyr, "item"):
                time_gyr = time_gyr.item()
            elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                time_gyr = (
                    time_gyr[0].item()
                    if hasattr(time_gyr[0], "item")
                    else float(time_gyr[0])
                )
            else:
                time_gyr = 0.0

            scale_factor = getattr(graph_data, "scale_factor", 1.0)
            if hasattr(scale_factor, "item"):
                scale_factor = scale_factor.item()
            elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                scale_factor = (
                    scale_factor[0].item()
                    if hasattr(scale_factor[0], "item")
                    else float(scale_factor[0])
                )
            else:
                scale_factor = 1.0

        else:
            # Fallback: assume it's a dictionary or other format
            raise ValueError(
                f"Unknown TNG50 temporal data format for snapshot {snapshot_id}"
            )

        # Create DataFrame for single snapshot
        df_data = {
            "x": positions[:, 0],
            "y": positions[:, 1],
            "z": positions[:, 2],
            "mass": masses,
            "velocity_0": velocities[:, 0],
            "velocity_1": velocities[:, 1],
            "velocity_2": velocities[:, 2],
            "particle_type": particle_types,
            "snapshot_id": snapshot_id,
            "redshift": redshifts,
            "time_gyr": time_gyr,
            "scale_factor": scale_factor,
        }
        df = pl.DataFrame(df_data)

    else:
        # Load all snapshots combined
        all_data = []
        for i, graph_data in enumerate(temporal_data):
            # Handle dict format (actual format of the data)
            if isinstance(graph_data, dict):
                positions = graph_data["x"][:, :3].numpy()
                masses = graph_data["x"][:, 3].numpy()
                velocities = graph_data["x"][:, 4:7].numpy()
                particle_types = graph_data["x"][:, 7].numpy()

                # Get cosmological parameters
                redshifts = graph_data.get("redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = graph_data.get("time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = graph_data.get("scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            elif hasattr(graph_data, "x"):
                positions = graph_data.x[:, :3].numpy()
                masses = graph_data.x[:, 3].numpy()
                velocities = graph_data.x[:, 4:7].numpy()
                particle_types = graph_data.x[:, 7].numpy()

                # Get cosmological parameters
                redshifts = getattr(graph_data, "redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = getattr(graph_data, "time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = getattr(graph_data, "scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            elif hasattr(graph_data, "pos"):
                positions = graph_data.pos[:, :3].numpy()
                masses = (
                    graph_data.mass.numpy()
                    if hasattr(graph_data, "mass")
                    else np.ones(len(positions))
                )
                velocities = (
                    graph_data.vel.numpy()
                    if hasattr(graph_data, "vel")
                    else np.zeros((len(positions), 3))
                )
                particle_types = (
                    graph_data.particle_type.numpy()
                    if hasattr(graph_data, "particle_type")
                    else np.zeros(len(positions))
                )

                # Get cosmological parameters
                redshifts = getattr(graph_data, "redshift", 0.0)
                if hasattr(redshifts, "item"):
                    redshifts = redshifts.item()
                elif hasattr(redshifts, "__len__") and len(redshifts) > 0:
                    redshifts = (
                        redshifts[0].item()
                        if hasattr(redshifts[0], "item")
                        else float(redshifts[0])
                    )
                else:
                    redshifts = 0.0

                time_gyr = getattr(graph_data, "time_gyr", 0.0)
                if hasattr(time_gyr, "item"):
                    time_gyr = time_gyr.item()
                elif hasattr(time_gyr, "__len__") and len(time_gyr) > 0:
                    time_gyr = (
                        time_gyr[0].item()
                        if hasattr(time_gyr[0], "item")
                        else float(time_gyr[0])
                    )
                else:
                    time_gyr = 0.0

                scale_factor = getattr(graph_data, "scale_factor", 1.0)
                if hasattr(scale_factor, "item"):
                    scale_factor = scale_factor.item()
                elif hasattr(scale_factor, "__len__") and len(scale_factor) > 0:
                    scale_factor = (
                        scale_factor[0].item()
                        if hasattr(scale_factor[0], "item")
                        else float(scale_factor[0])
                    )
                else:
                    scale_factor = 1.0

            else:
                continue  # Skip unknown format

            snapshot_data = {
                "x": positions[:, 0],
                "y": positions[:, 1],
                "z": positions[:, 2],
                "mass": masses,
                "velocity_0": velocities[:, 0],
                "velocity_1": velocities[:, 1],
                "velocity_2": velocities[:, 2],
                "particle_type": particle_types,
                "snapshot_id": i,
                "redshift": redshifts,
                "time_gyr": time_gyr,
                "scale_factor": scale_factor,
            }
            all_data.append(pl.DataFrame(snapshot_data))

        df = pl.concat(all_data)

    # Apply sampling if requested
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, seed=42)

    # Create AstroDataset
    dataset = AstroDataset(
        survey="tng50_temporal",
        data_path=None,  # We already have the data
        k_neighbors=8,
        max_samples=max_samples,
        force_reload=False,
        return_tensor=True,  # Return as SurveyTensor
    )

    # Set the processed data directly
    dataset._df_cache = df
    dataset._processed_data = [df]  # Ensure it's treated as processed

    return dataset


def detect_survey_type(dataset_name: str, df: Optional[pl.DataFrame]) -> str:
    """Detect survey type from dataset name or data structure."""
    name_lower = dataset_name.lower()
    if "gaia" in name_lower:
        return "gaia"
    elif "sdss" in name_lower:
        return "sdss"
    elif "nsa" in name_lower:
        return "nsa"
    elif "linear" in name_lower:
        return "linear"
    elif "exoplanet" in name_lower:
        return "exoplanet"
    elif "tng" in name_lower:  # More general for TNG50, TNG100 etc.
        return "tng"
    elif "rrlyrae" in name_lower:
        return "rrlyrae"
    # Fallback based on columns if name is not indicative
    if df is not None:
        cols = {c.lower() for c in df.columns}
        if "phot_g_mean_mag" in cols and "parallax" in cols:
            return "gaia"
        if "specobjid" in cols and "z" in cols:
            return "sdss"
        if "iauname" in cols and "nsa_version" in cols:
            return "nsa"

    return "generic"


def benchmark_performance(
    survey: str = "linear", max_samples: int = 1000, verbose: bool = True
) -> Dict[str, Any]:
    """
    Benchmark performance of cosmic web analysis with different optimizations.

    Args:
        survey: Survey to benchmark
        max_samples: Number of samples to test
        verbose: Print detailed results

    Returns:
        Performance benchmark results
    """
    if verbose:
        print("🚀 PERFORMANCE BENCHMARK")
        print("=" * 50)

    # Test device detection
    device = get_optimal_device()
    if verbose:
        print(f"📱 Device: {device}")
        if device.type == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"📊 GPU Memory: {gpu_memory:.1f} GB")

    # Test batch size optimization
    optimal_batch_size = get_optimal_batch_size(max_samples)
    optimal_workers = get_optimal_num_workers()

    if verbose:
        print(f"⚡ Optimal batch size: {optimal_batch_size}")
        print(f"🔧 Optimal workers: {optimal_workers}")

    # Benchmark cosmic web analysis
    start_time = time.time()

    try:
        results = create_cosmic_web_loader(
            survey=survey,
            max_samples=max_samples,
            scales_mpc=[5.0, 10.0],  # Fewer scales for benchmark
            device=device,
        )

        total_time = time.time() - start_time

        if verbose:
            print(f"⏱️ Total time: {total_time:.2f}s")
            print(f"📊 Objects processed: {results['n_objects']:,}")
            print(f"🚀 Throughput: {results['n_objects'] / total_time:.0f} objects/s")

        return {
            "device": str(device),
            "optimal_batch_size": optimal_batch_size,
            "optimal_workers": optimal_workers,
            "total_time": total_time,
            "objects_processed": results["n_objects"],
            "throughput": results["n_objects"] / total_time,
            "success": True,
        }

    except Exception as e:
        if verbose:
            print(f"❌ Benchmark failed: {e}")

        return {
            "device": str(device),
            "optimal_batch_size": optimal_batch_size,
            "optimal_workers": optimal_workers,
            "total_time": time.time() - start_time,
            "error": str(e),
            "success": False,
        }


def print_performance_info():
    """Print current performance configuration."""
    print("🚀 PERFORMANCE CONFIGURATION")
    print("=" * 40)

    # Device info
    device = get_optimal_device()
    print(f"📱 Device: {device}")

    if device.type == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"📊 GPU Memory: {gpu_memory:.1f} GB")
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"⚡ cuDNN: {torch.backends.cudnn.version()}")

    # CPU info
    cpu_count = os.cpu_count()
    print(f"🖥️ CPU Cores: {cpu_count}")

    # Optimal settings
    optimal_workers = get_optimal_num_workers()
    print(f"🔧 Optimal Workers: {optimal_workers}")

    # PyTorch optimizations
    print(f"⚡ cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"🔒 cuDNN Deterministic: {torch.backends.cudnn.deterministic}")
    print(f"🚀 TF32 Enabled: {torch.backends.cuda.matmul.allow_tf32}")

    # Polars optimizations
    try:
        string_cache = pl.Config.get_global_string_cache()
        print(f"📊 Polars String Cache: {string_cache}")
    except AttributeError:
        try:
            string_cache = pl.Config.get_string_cache()
            print(f"📊 Polars String Cache: {string_cache}")
        except AttributeError:
            print("📊 Polars String Cache: Auto-managed")
