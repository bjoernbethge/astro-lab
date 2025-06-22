"""
AstroLab Data Core Module
=========================

Core data handling and processing for astronomical datasets.
Clean PyTorch Geometric dataset implementation with Polars support.
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
import torch_geometric
import torch_geometric.transforms as T
from astropy.coordinates import SkyCoord
from sklearn.neighbors import BallTree, NearestNeighbors
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from .config import data_config

logger = logging.getLogger(__name__)

# Set environment variable for NumPy 2.x compatibility with bpy and other modules
os.environ["NUMPY_EXPERIMENTAL_ARRAY_API"] = "1"

# Configure logging
logger = logging.getLogger(__name__)

# 🌟 TENSOR INTEGRATION - Import all tensor types
# Core dependencies - should always be available
from astro_lab.tensors import (
    LightcurveTensor,
    PhotometricTensor,
    SimulationTensor,
    Spatial3DTensor,
    SpectralTensor,
    SurveyTensor,
)


# Function defined here to avoid circular import
def create_graph_datasets_from_splits(
    train_indices: List[int],
    val_indices: List[int],
    test_indices: List[int],
    data_list: List[Data],
    **kwargs,
) -> Tuple[List[Data], List[Data], List[Data]]:
    """Create graph datasets from train/val/test splits."""
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    return train_data, val_data, test_data


# PyTorch Geometric integration - core dependency

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
    Calculate local density for each object.

    Args:
        positions: 3D positions (N, 3) in Mpc
        radius_pc: Radius for local density calculation in pc
        max_neighbors: Maximum neighbors to consider

    Returns:
        Local density for each object in obj/pc³
    """
    # Convert to pc for local density calculation
    positions_pc = positions * 1e6  # Mpc to pc

    # BallTree for efficient radius searches
    tree = BallTree(positions_pc.numpy())

    # OPTIMIZED: Use query_radius with count_only for better performance
    neighbor_counts = tree.query_radius(
        positions_pc.numpy(), r=radius_pc, count_only=True
    )

    # OPTIMIZED: Vectorized density calculation
    volume = (4 / 3) * np.pi * (radius_pc**3)
    densities = neighbor_counts / volume

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

    return results, local_density


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
    **kwargs,
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
        survey_tensor = load_gaia_data(
            max_samples=max_samples, return_tensor=True, **kwargs
        )
    elif survey == "sdss":
        survey_tensor = load_sdss_data(
            max_samples=max_samples, return_tensor=True, **kwargs
        )
    elif survey == "nsa":
        survey_tensor = load_nsa_data(
            max_samples=max_samples, return_tensor=True, **kwargs
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
            survey_tensor = Spatial3DTensor(coords, unit="Mpc")
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
        raise ValueError(f"Unknown survey: {survey}")

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
SURVEY_CONFIGS = {
    "gaia": {
        "name": "Gaia DR3",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
        "extra_cols": ["parallax", "pmra", "pmdec"],
        "color_pairs": [
            ("phot_g_mean_mag", "phot_bp_mean_mag"),
            ("phot_bp_mean_mag", "phot_rp_mean_mag"),
        ],
        "default_limit": 12.0,
        "url": "gaia",
        # 🌟 TENSOR METADATA
        "filter_system": "gaia",
        "data_release": "DR3",
        "coordinate_system": "icrs",
        "photometric_bands": ["G", "BP", "RP"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "parallax": 2,
            "pmra": 3,
            "pmdec": 4,
            "phot_g_mean_mag": 5,
            "phot_bp_mean_mag": 6,
            "phot_rp_mean_mag": 7,
        },
    },
    "sdss": {
        "name": "SDSS DR17",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": [
            "modelMag_u",
            "modelMag_g",
            "modelMag_r",
            "modelMag_i",
            "modelMag_z",
        ],
        "extra_cols": ["petroRad_r", "fracDeV_r"],
        "color_pairs": [("modelMag_g", "modelMag_r"), ("modelMag_r", "modelMag_i")],
        "default_limit": 20.0,
        "url": "sdss",
        # 🌟 TENSOR METADATA
        "filter_system": "sdss",
        "data_release": "DR17",
        "coordinate_system": "icrs",
        "photometric_bands": ["u", "g", "r", "i", "z"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "z": 2,
            "modelMag_u": 3,
            "modelMag_g": 4,
            "modelMag_r": 5,
            "modelMag_i": 6,
            "modelMag_z": 7,
            "petroRad_r": 8,
            "fracDeV_r": 9,
        },
    },
    "nsa": {
        "name": "NASA Sloan Atlas",
        "coord_cols": ["ra", "dec", "z"],
        "mag_cols": ["mag_g", "mag_r", "mag_i"],
        "extra_cols": ["sersic_n", "sersic_ba", "mass"],
        "color_pairs": [("mag_g", "mag_r"), ("mag_r", "mag_i")],
        "default_limit": 18.0,
        "url": "nsa",
        # 🌟 TENSOR METADATA
        "filter_system": "sdss",
        "data_release": "v1_0_1",
        "coordinate_system": "icrs",
        "photometric_bands": ["g", "r", "i"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "z": 2,
            "mag_g": 3,
            "mag_r": 4,
            "mag_i": 5,
            "sersic_n": 6,
            "sersic_ba": 7,
            "mass": 8,
        },
    },
    "linear": {
        "name": "LINEAR Lightcurves",
        "coord_cols": ["ra", "dec"],
        "mag_cols": ["mag_mean", "mag_amp"],
        "extra_cols": ["period", "period_error"],
        "color_pairs": [],
        "default_limit": 16.0,
        "url": "linear",
        # 🌟 TENSOR METADATA
        "filter_system": "johnson",
        "data_release": "final",
        "coordinate_system": "icrs",
        "photometric_bands": ["V"],
        "tensor_column_mapping": {
            "ra": 0,
            "dec": 1,
            "mag_mean": 2,
            "mag_amp": 3,
            "period": 4,
            "period_error": 5,
        },
    },
    # 🌟 NEW: TNG50 als vollwertiger Survey
    "tng50": {
        "name": "TNG50 Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": [],  # No magnitudes for simulation
        "extra_cols": ["masses", "velocities_0", "velocities_1", "velocities_2"],
        "color_pairs": [],
        "default_limit": None,  # Keine Magnitude-Limits
        "url": "tng50",
        # 🌟 TENSOR METADATA for Simulation
        "filter_system": "none",
        "data_release": "TNG50-4",
        "coordinate_system": "cartesian",
        "photometric_bands": [],
        "particle_types": ["PartType0", "PartType1", "PartType4", "PartType5"],
        "tensor_column_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "masses": 3,
            "velocities_0": 4,
            "velocities_1": 5,
            "velocities_2": 6,
        },
        # Simulation-specific metadata
        "simulation_metadata": {
            "box_size": 35.0,  # Mpc/h
            "redshift": 0.0,
            "snapshot": 99,
            "cosmology": "Planck2018",
        },
    },
    # 🌟 NEW: TNG50-Temporal als vollwertiger Survey
    "tng50_temporal": {
        "name": "TNG50 Temporal Simulation",
        "coord_cols": ["x", "y", "z"],
        "mag_cols": [],  # No magnitudes for simulation
        "extra_cols": [
            "mass",
            "velocity_0",
            "velocity_1",
            "velocity_2",
            "particle_type",
            "snapshot_id",
            "redshift",
            "time_gyr",
            "scale_factor",
        ],
        "color_pairs": [],
        "default_limit": None,  # Keine Magnitude-Limits
        "url": "tng50_temporal",
        # 🌟 TENSOR METADATA for temporal simulation
        "filter_system": "none",
        "data_release": "TNG50-4-Temporal",
        "coordinate_system": "cartesian",
        "photometric_bands": [],
        "particle_types": ["PartType0", "PartType1", "PartType4", "PartType5"],
        "tensor_column_mapping": {
            "x": 0,
            "y": 1,
            "z": 2,
            "mass": 3,
            "velocity_0": 4,
            "velocity_1": 5,
            "velocity_2": 6,
            "particle_type": 7,
            "snapshot_id": 8,
            "redshift": 9,
            "time_gyr": 10,
            "scale_factor": 11,
        },
        # Temporal simulation-specific metadata
        "temporal_evolution": True,
    },
}


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

    if survey not in SURVEY_CONFIGS:
        raise ValueError(f"Survey {survey} not supported")

    config = SURVEY_CONFIGS[survey]

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

    return SurveyTensor(data=tensor_data, **metadata)


class AstroDataset(InMemoryDataset):
    """
    Clean PyTorch Geometric dataset for astronomical data.

    Loads pre-processed .pt files from data/processed/<survey>/processed/.
    No data processing or generation - only loading existing files.
    """

    def __init__(
        self,
        root: Optional[Union[str, Path]] = None,
        survey: Optional[str] = None,
        max_samples: Optional[int] = None,
        k_neighbors: int = 8,
        distance_threshold: float = 50.0,
        return_tensor: bool = True,
        transform: Optional[Any] = None,
        force_reload: bool = False,
        **kwargs,
    ):
        """Initialize AstroDataset with survey-specific configuration."""
        # Set root directory
        if root is None:
            root = data_config.processed_dir / (survey or "generic")

        # Store configuration
        self.survey = survey
        self.max_samples = max_samples
        self.k_neighbors = k_neighbors
        self.distance_threshold = distance_threshold
        self.return_tensor = return_tensor

        # Initialize data attribute before calling parent
        self._data = None

        # Call parent constructor
        super().__init__(root, transform, force_reload=force_reload)

        # Set up survey-specific configuration
        if survey and survey in SURVEY_CONFIGS:
            config = SURVEY_CONFIGS[survey]
            self.coord_cols = config.get("coord_cols", ["ra", "dec"])
            self.mag_cols = config.get("mag_cols", [])
            self.extra_cols = config.get("extra_cols", [])
            self.color_pairs = config.get("color_pairs", [])
            self.tensor_metadata = config
        else:
            # Default configuration
            self.coord_cols = ["ra", "dec"]
            self.mag_cols = ["mag"]
            self.extra_cols = []
            self.color_pairs = []
            self.tensor_metadata = {}

    @property
    def data(self):
        """Get the data attribute."""
        return self._data

    @data.setter
    def data(self, value):
        """Set the data attribute."""
        self._data = value

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names - not used for .pt loading."""
        return []

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names for .pt file selection."""
        suffix = f"_n{self.max_samples}" if self.max_samples else ""
        tensor_suffix = "_tensor" if self.return_tensor else ""
        return [f"{self.survey}_k{self.k_neighbors}{suffix}{tensor_suffix}.pt"]

    def download(self) -> None:
        """Load dataset from .pt files - no fake data generation."""
        if self.data is not None:
            return
        self.load(None)  # Load from .pt file

    def _load_polars_dataframe(self) -> pl.DataFrame:
        """Load data as Polars DataFrame - only from real .pt files."""
        if self.data is None:
            self.load(None)  # Load from .pt file

        # Convert PyG Data to Polars DataFrame if needed
        if hasattr(self.data, "x") and hasattr(self.data, "feature_names"):
            # Convert PyG Data to DataFrame
            feature_data = {}
            for i, name in enumerate(self.data.feature_names):
                feature_data[name] = self.data.x[:, i].numpy()

            # Add coordinate columns if available
            if hasattr(self.data, "coord_names") and hasattr(self.data, "pos"):
                for i, name in enumerate(self.data.coord_names):
                    feature_data[name] = self.data.pos[:, i].numpy()

            return pl.DataFrame(feature_data)
        else:
            raise ValueError("Data is not in expected PyG format")

    def process(self):
        """No processing - data should already be processed as .pt files."""
        raise NotImplementedError(
            "AstroDataset does not process data. Use pre-processed .pt files from data/processed/<survey>/processed/."
        )

    def _find_best_pt_file(self) -> Optional[Path]:
        """Find the best matching .pt file based on max_samples and k_neighbors."""
        # Use project root for data path - go up 4 levels from core.py to get to project root
        project_root = Path(__file__).parent.parent.parent.parent
        processed_dir = project_root / "data" / "processed" / self.survey / "processed"

        # Add test data directory for testing
        test_data_dir = (
            project_root
            / "test"
            / "tensors"
            / "data"
            / "processed"
            / self.survey
            / "processed"
        )
        possible_dirs = [processed_dir]
        if test_data_dir.exists():
            possible_dirs.insert(0, test_data_dir)  # Prioritize test data

        all_files = []
        for processed_dir in possible_dirs:
            if not processed_dir.exists():
                continue
            patterns = [
                f"{self.survey}_graph_k{self.k_neighbors}_n*.pt",  # New format: gaia_graph_k8_n100.pt
                f"{self.survey}_k{self.k_neighbors}_n*.pt",  # Old format: gaia_k8_n100.pt
                f"{self.survey}_graph_k{self.k_neighbors}.pt",  # New format: gaia_graph_k8.pt
                f"{self.survey}_k{self.k_neighbors}.pt",  # Old format: gaia_k8.pt
                f"{self.survey}_mag*.pt",  # Magnitude-based files
            ]
            for pattern in patterns:
                files = list(processed_dir.glob(pattern))
                all_files.extend(files)
        if not all_files:
            return None

        def extract_n(f):
            m = re.search(r"_n(\d+)", f.name)
            if m:
                return int(m.group(1))
            return f.stat().st_size

        all_files = sorted(all_files, key=extract_n)
        best = None
        for f in all_files:
            n = extract_n(f)
            if self.max_samples is None or n <= self.max_samples:
                best = f
            else:
                break
        return best or all_files[-1]  # fallback: largest

    def load(self, path: Union[str, Path]):
        """Load dataset from .pt file."""
        pt_file = self._find_best_pt_file()
        if pt_file and pt_file.exists():
            logger.info(f"📦 Loading graph data from {pt_file}")
            data = torch.load(pt_file)

            if isinstance(data, list):
                self.data, self.slices = self.collate(data)
            elif isinstance(data, dict) and "data" in data and "slices" in data:
                self.data, self.slices = data["data"], data["slices"]
            else:
                # fallback: treat as single Data object
                self.data, self.slices = self.collate([data])
            return

        # No .pt file found
        raise FileNotFoundError(
            f"No suitable .pt file found for {self.survey} (k={self.k_neighbors}, n={self.max_samples}). "
            f"Please ensure data is pre-processed and available in data/processed/{self.survey}/processed/."
        )

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if len(self) == 0:
            return {"error": "Dataset not loaded"}

        data = self[0]
        return {
            "survey": self.survey,
            "survey_name": data.survey_name,
            "num_nodes": data.num_nodes,
            "num_edges": data.edge_index.shape[1],
            "num_features": data.x.shape[1],
            "feature_names": data.feature_names,
            "coordinate_names": data.coord_names,
            "k_neighbors": data.k_neighbors,
            "avg_degree": data.edge_index.shape[1] / data.num_nodes,
        }

    def _download(self):
        """Override _download to load .pt files directly."""
        self.load(None)  # Load from .pt file - no fallback to fake data

    def _process(self):
        """Override _process to load .pt files directly."""
        self.load(None)  # Load from .pt file - no fallback to fake data


class AstroDataModule(L.LightningDataModule):
    """
    Clean Lightning DataModule for astronomical data.

    Eliminates complex setup logic with simple, direct approach.
    """

    def __init__(
        self,
        survey: str,
        data_path: Optional[str] = None,
        batch_size: Optional[int] = None,  # Auto-detect if None
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: Optional[int] = None,  # Auto-detect if None
        force_reload: bool = False,
        device: Optional[torch.device] = None,  # Auto-detect if None
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Auto-detect optimal settings
        self.device = device or get_optimal_device()
        self.num_workers = num_workers or get_optimal_num_workers()

        # Create dataset first to get size for batch size optimization
        self.dataset = AstroDataset(
            survey=survey,
            data_path=data_path,
            k_neighbors=k_neighbors,
            max_samples=max_samples,
            force_reload=force_reload,
            **kwargs,
        )

        # Auto-detect optimal batch size
        if batch_size is None:
            dataset_size = len(self.dataset) if len(self.dataset) > 0 else 1000
            self.batch_size = get_optimal_batch_size(dataset_size)
        else:
            self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        """Setup with automatic train/val/test splits."""
        if len(self.dataset) == 0:
            return

        data = self.dataset[0]
        num_nodes = data.num_nodes

        # Create random splits
        indices = torch.randperm(num_nodes)

        train_size = int(self.hparams.train_ratio * num_nodes)
        val_size = int(self.hparams.val_ratio * num_nodes)

        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size : train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size :]] = True

        print(
            f"📊 Split {data.survey_name}: "
            f"Train={data.train_mask.sum()}, "
            f"Val={data.val_mask.sum()}, "
            f"Test={data.test_mask.sum()}"
        )

        # Move data to optimal device
        if self.device.type == "cuda":
            data = data.to(self.device)
            print(f"🚀 Moved data to {self.device}")

    def train_dataloader(self):
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,  # Graph datasets use batch_size=1
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )

    def val_dataloader(self):
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )

    def test_dataloader(self):
        return DataLoader(
            [self.dataset[0]],
            batch_size=1,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=self.device.type == "cuda",
        )


# Factory function - replaces all the individual create_* functions
def create_astro_dataloader(survey: str, batch_size: int = 1, **kwargs) -> DataLoader:
    """
    Universal factory for astronomical data loaders.

    Replaces create_gaia_dataloader, create_sdss_dataloader, etc.
    """
    dataset = AstroDataset(survey=survey, **kwargs)
    return DataLoader(dataset, batch_size=batch_size)


def create_astro_datamodule(survey: str, **kwargs) -> AstroDataModule:
    """
    Universal factory for astronomical data modules.

    Replaces GaiaDataModule, ExoplanetDataModule, etc.
    """
    return AstroDataModule(survey=survey, **kwargs)


# 🌟 ENHANCED LOAD FUNCTIONS with automatic tensor conversion


def load_gaia_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load Gaia DR3 stellar catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with Gaia data or AstroDataset
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        # Create temporary dataset to get DataFrame
        temp_dataset = AstroDataset(
            survey="gaia", max_samples=max_samples, return_tensor=False, **kwargs
        )
        temp_dataset.load(None)  # Load from .pt file
        df = temp_dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "gaia", {"max_samples": max_samples})

    # Create AstroDataset with survey parameter
    dataset = AstroDataset(
        survey="gaia", max_samples=max_samples, return_tensor=False, **kwargs
    )
    dataset.load(None)  # Load from .pt file

    return dataset


def load_sdss_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load SDSS DR17 galaxy catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with SDSS data or AstroDataset
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        # Create temporary dataset to get DataFrame
        temp_dataset = AstroDataset(
            survey="sdss", max_samples=max_samples, return_tensor=False, **kwargs
        )
        temp_dataset.load(None)  # Load from .pt file
        df = temp_dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "sdss", {"max_samples": max_samples})

    # Create AstroDataset with survey parameter
    dataset = AstroDataset(
        survey="sdss", max_samples=max_samples, return_tensor=False, **kwargs
    )
    dataset.load(None)  # Load from .pt file

    return dataset


def load_nsa_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "SurveyTensor"]:
    """
    Load NSA galaxy catalog.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return SurveyTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        SurveyTensor with NSA data or AstroDataset
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        # Create temporary dataset to get DataFrame
        temp_dataset = AstroDataset(
            survey="nsa", max_samples=max_samples, return_tensor=False, **kwargs
        )
        temp_dataset.load(None)  # Load from .pt file
        df = temp_dataset._load_polars_dataframe()
        return _polars_to_survey_tensor(df, "nsa", {"max_samples": max_samples})

    # Create AstroDataset with survey parameter
    dataset = AstroDataset(
        survey="nsa", max_samples=max_samples, return_tensor=False, **kwargs
    )
    dataset.load(None)  # Load from .pt file

    return dataset


def load_lightcurve_data(
    max_samples: int = 5000,
    return_tensor: bool = True,  # 🌟 Default to tensor!
    **kwargs,
) -> Union[AstroDataset, "LightcurveTensor"]:
    """
    Load LINEAR lightcurve data.

    Args:
        max_samples: Maximum number of samples
        return_tensor: Return LightcurveTensor instead of dataset (recommended!)
        **kwargs: Additional arguments

    Returns:
        LightcurveTensor with lightcurve data or AstroDataset
    """
    if return_tensor and TENSOR_INTEGRATION_AVAILABLE:
        dataset = AstroDataset(
            survey="linear", max_samples=max_samples, return_tensor=False, **kwargs
        )
        dataset.download()
        df = dataset._load_polars_dataframe()

        # Convert to LightcurveTensor for time series data
        n_samples = len(df)
        times = torch.arange(n_samples, dtype=torch.float32)  # Demo time array
        magnitudes = torch.tensor(df["mag_mean"].to_numpy(), dtype=torch.float32)

        return LightcurveTensor(
            data=torch.stack([times, magnitudes], dim=1),
            bands=["V"],
            time_format="sequential",
            coordinate_system="icrs",
            survey_name="linear",
        )

    return AstroDataset(
        survey="linear",
        k_neighbors=8,
        max_samples=max_samples,
        return_tensor=return_tensor,
        **kwargs,
    )


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
    config = SURVEY_CONFIGS["tng50"]

    # Load TNG50 data from processed files
    data_path = Path("data/processed/tng50_combined.parquet")
    if not data_path.exists():
        print(f"⚠️ TNG50 data not found at {data_path}. Generating demo data...")
        # Create AstroDataset with survey parameter and generate demo data
        dataset = AstroDataset(
            survey="tng50", max_samples=max_samples, return_tensor=False
        )
        dataset.download()  # Generate demo data
        dataset.data = dataset._load_polars_dataframe()
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
    """Load TNG50 temporal simulation data as a survey-like dataset.

    Args:
        max_samples: Maximum number of particles to load (None = all)
        snapshot_id: Specific snapshot to load (None = all snapshots)

    Returns:
        AstroDataset with TNG50 temporal simulation data
    """
    config = SURVEY_CONFIGS["tng50_temporal"]

    # Load TNG50 temporal data from processed files
    data_path = Path(
        "data/processed/tng50_temporal_100mb/processed/tng50_temporal_graphs_r1.0.pt"
    )
    if not data_path.exists():
        raise FileNotFoundError(
            f"TNG50 temporal data not found at {data_path}. Run preprocessing first."
        )

    # Load PyTorch tensors
    import torch

    temporal_data = torch.load(data_path, map_location="cpu", weights_only=False)

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
    """
    Detect survey type from filename and columns.

    Args:
        dataset_name: Name of the dataset file
        df: Polars DataFrame with astronomical data (can be None)

    Returns:
        Survey type string
    """
    name_lower = dataset_name.lower()

    # Handle None DataFrame case
    if df is None:
        # Try to detect from filename only
        if "tng50" in name_lower or "parttype" in name_lower:
            return "tng50"
        elif "nsa" in name_lower:
            return "nsa"
        elif "gaia" in name_lower:
            return "gaia"
        elif "sdss" in name_lower:
            return "sdss"
        elif "linear" in name_lower:
            return "linear"
        else:
            return "generic"

    columns = [col.lower() for col in df.columns]

    # TNG50 simulation data
    if "tng50" in name_lower or "parttype" in name_lower:
        return "tng50"
    elif (
        "x" in columns
        and "y" in columns
        and "z" in columns
        and any("velocities" in col for col in columns)
    ):
        return "tng50"
    # Observational surveys
    elif "nsa" in name_lower or any("petromag" in col for col in columns):
        return "nsa"
    elif "gaia" in name_lower or "phot_g_mean_mag" in columns:
        return "gaia"
    elif "sdss" in name_lower or any("modelmag" in col for col in columns):
        return "sdss"
    else:
        return "generic"


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
        survey_type: Type of survey ('nsa', 'gaia', 'sdss', 'generic')
        k_neighbors: Number of neighbors for graph construction
        distance_threshold: Distance threshold for edges
        output_path: Optional path to save graph
        **kwargs: Additional arguments

    Returns:
        PyTorch Geometric Data object or None if PyG not available
    """

    coords = df.select(coord_cols).to_numpy()
    all_features = df.to_numpy()

    print(
        f"   📊 Coordinate range: X[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}]"
    )
    print(
        f"   📊 Coordinate range: Y[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]"
    )
    print(
        f"   📊 Coordinate range: Z[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]"
    )

    # For TNG50, use 3D distance in comoving coordinates
    # Distance threshold is in simulation units (Mpc/h)
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(df)), metric="euclidean")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Create edge list (exclude self-connections)
    edge_list = []
    edge_distances = []
    edge_velocities = []  # Store relative velocities as edge features

    # Extract velocity columns if available
    vel_cols = ["velocities_0", "velocities_1", "velocities_2"]
    has_velocities = all(col in df.columns for col in vel_cols)

    if has_velocities:
        velocities = df.select(vel_cols).to_numpy()

    for i, (dists, neighs) in enumerate(zip(distances, indices)):
        for dist, neigh in zip(dists[1:], neighs[1:]):  # Skip self
            if dist <= distance_threshold:
                edge_list.append([i, neigh])
                edge_distances.append(dist)

                # Add relative velocity as edge feature if available
                if has_velocities:
                    rel_velocity = np.linalg.norm(velocities[i] - velocities[neigh])
                    edge_velocities.append(rel_velocity)

    if not edge_list:
        print(f"   ⚠️ No edges found with distance threshold {distance_threshold}")
        # Create a minimal graph with self-loops
        edge_list = [[i, i] for i in range(min(10, len(df)))]
        edge_distances = [0.0] * len(edge_list)
        edge_velocities = [0.0] * len(edge_list) if has_velocities else []

    edge_index = torch.tensor(np.array(edge_list).T, dtype=torch.long)

    # Create edge attributes (distance + relative velocity if available)
    if has_velocities and edge_velocities:
        edge_attr = torch.tensor(
            np.column_stack([edge_distances, edge_velocities]), dtype=torch.float
        )
    else:
        edge_attr = torch.tensor(edge_distances, dtype=torch.float).unsqueeze(1)

    node_features = torch.tensor(all_features, dtype=torch.float)

    print(f"   ✅ TNG50 Graph: {len(all_features):,} nodes, {len(edge_list):,} edges")
    if has_velocities:
        print("   📊 Edge features: distance + relative velocity")
    else:
        print("   📊 Edge features: distance only")

    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=torch.tensor(coords, dtype=torch.float),  # 3D positions
        num_nodes=len(all_features),
    )


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
