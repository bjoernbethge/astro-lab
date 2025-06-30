"""
Lightning DataModule for AstroLab
================================

Lightning DataModule with enhanced integration of astronomical features,
reduced redundancy, and better cooperation between components.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import lightning as L
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from astro_lab.config import (
    get_data_config,
    get_survey_config,
    get_survey_optimization,
)
from astro_lab.data.datasets import SurveyGraphDataset
from astro_lab.tensors import (
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SpectralTensorDict,
    SurveyTensorDict,
)

logger = logging.getLogger(__name__)


class AstroDataModule(L.LightningDataModule):
    """
    Enhanced Lightning DataModule with deep integration of astronomical features.

    Major improvements:
    - Direct integration with specialized TensorDict classes
    - Survey-specific feature extractors and transformations
    - Unified configuration through central config system
    - Memory-efficient caching and lazy loading
    - Astronomical domain-aware splitting strategies
    - Multi-scale graph construction support
    """

    def __init__(
        self,
        survey: str,
        # Core parameters
        batch_size: int = 32,
        num_workers: int = 4,
        # Graph construction parameters
        graph_config: Optional[Dict[str, Any]] = None,
        # Data selection
        max_samples: Optional[int] = None,
        selection_criteria: Optional[Dict[str, Any]] = None,
        # Split configuration
        split_config: Optional[Dict[str, Any]] = None,
        # Transform pipeline
        transform_pipeline: Optional[List[Union[str, Callable]]] = None,
        # Cache configuration
        cache_config: Optional[Dict[str, Any]] = None,
        # Optional overrides
        data_root: Optional[str] = None,
        force_reload: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Core configuration
        self.survey = survey
        self.data_config = get_data_config()
        self.survey_config = get_survey_config(survey)
        self.optimization_config = get_survey_optimization(survey)

        # Use optimized batch size if available
        self.batch_size = batch_size
        if self.optimization_config and batch_size == 32:  # Default not overridden
            self.batch_size = self.optimization_config.batch_size

        self.num_workers = num_workers
        self.max_samples = max_samples
        self.force_reload = force_reload

        # Data root with central config
        self.data_root = Path(data_root) if data_root else self.data_config.base_dir

        # Graph construction configuration with defaults
        default_graph_config = {
            "method": "knn",
            "k_neighbors": 8,
            "use_3d": True,
            "cosmic_web_scales": [5.0, 10.0, 25.0],  # For multi-scale analysis
            "include_edge_features": True,
        }
        self.graph_config = {**default_graph_config, **(graph_config or {})}

        # Selection criteria for data filtering
        self.selection_criteria = selection_criteria or {}

        # Split configuration with astronomical awareness
        default_split_config = {
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "strategy": "spatial",  # spatial, temporal, random, stratified
            "stratify_by": None,  # Feature to stratify by
            "spatial_method": "clustering",  # clustering, grid, random
            "temporal_bins": 10,  # For temporal splitting
        }
        self.split_config = {**default_split_config, **(split_config or {})}

        # Transform pipeline setup
        self.transform_pipeline = self._build_transform_pipeline(transform_pipeline)

        # Cache configuration
        default_cache_config = {
            "enable": True,
            "cache_processed": True,
            "cache_graphs": True,
            "cache_dir": self.data_root / "cache" / survey,
        }
        self.cache_config = {**default_cache_config, **(cache_config or {})}

        # Initialize cache directory if enabled
        if self.cache_config["enable"]:
            self.cache_config["cache_dir"].mkdir(parents=True, exist_ok=True)

        # Dataset components (initialized in setup)
        self.dataset: Optional[SurveyGraphDataset] = None
        self.tensor_dict: Optional[SurveyTensorDict] = None
        self._splits: Dict[str, Any] = {"train": None, "val": None, "test": None}
        self._cached_properties: Dict[str, Any] = {}

    def _build_transform_pipeline(
        self, pipeline: Optional[List[Union[str, Callable]]]
    ) -> Optional[Compose]:
        """Build transformation pipeline from config or callables."""
        if not pipeline:
            # Default survey-specific transforms
            pipeline = self._get_default_transforms()

        transforms = []
        for transform in pipeline:
            if isinstance(transform, str):
                # Load predefined transforms by name
                transforms.append(self._get_transform_by_name(transform))
            elif callable(transform):
                transforms.append(transform)
            else:
                logger.warning(f"Invalid transform: {transform}")

        return Compose(transforms) if transforms else None

    def _get_default_transforms(self) -> List[str]:
        """Get default transforms based on survey type."""
        survey_type = self.survey_config.get("type", "catalog")

        if survey_type == "stellar":
            return ["normalize_magnitudes", "compute_colors", "add_kinematics"]
        elif survey_type == "galaxy":
            return [
                "normalize_magnitudes",
                "compute_morphology",
                "add_redshift_features",
            ]
        elif survey_type == "timeseries":
            return ["normalize_lightcurves", "extract_variability_features"]
        else:
            return ["normalize_features"]

    def _get_transform_by_name(self, name: str) -> Callable:
        """Get predefined transform by name."""
        transform_registry = {
            "normalize_magnitudes": self._normalize_magnitudes,
            "compute_colors": self._compute_colors,
            "add_kinematics": self._add_kinematics,
            "compute_morphology": self._compute_morphology,
            "add_redshift_features": self._add_redshift_features,
            "normalize_lightcurves": self._normalize_lightcurves,
            "extract_variability_features": self._extract_variability_features,
            "normalize_features": self._normalize_features,
        }

        if name not in transform_registry:
            raise ValueError(f"Unknown transform: {name}")

        return transform_registry[name]

    def prepare_data(self):
        """Download and preprocess data if needed. Called only on rank 0."""
        # Check cache first
        cache_path = self._get_cache_path("dataset")
        if cache_path.exists() and not self.force_reload:
            logger.info(f"Dataset cache exists: {cache_path}")
            return

        # Create dataset to trigger download/processing
        _ = SurveyGraphDataset(
            root=str(self.data_root),
            survey=self.survey,
            k_neighbors=self.graph_config["k_neighbors"],
            max_samples=self.max_samples,
            force_reload=self.force_reload,
        )

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with enhanced astronomical features."""
        if self.dataset is None:
            self._load_or_create_dataset()
            self._create_tensor_dict()
            self._apply_selection_criteria()
            self._extract_features()
            self._create_splits()
            self._log_dataset_info()

    def _load_or_create_dataset(self):
        """Load dataset with caching support."""
        cache_path = self._get_cache_path("dataset")

        if cache_path.exists() and not self.force_reload:
            logger.info(f"Loading cached dataset from {cache_path}")
            self.dataset = torch.load(cache_path, map_location="cpu")
        else:
            # Create new dataset
            self.dataset = SurveyGraphDataset(
                root=str(self.data_root),
                survey=self.survey,
                k_neighbors=self.graph_config["k_neighbors"],
                use_3d_coordinates=self.graph_config["use_3d"],
                max_samples=self.max_samples,
                transform=self.transform_pipeline,
                force_reload=self.force_reload,
            )

            # Cache if enabled
            if self.cache_config["enable"] and self.cache_config["cache_processed"]:
                torch.save(self.dataset, cache_path)
                logger.info(f"Cached dataset to {cache_path}")

    def _create_tensor_dict(self):
        """Create specialized TensorDict based on survey type."""
        if not self.dataset or len(self.dataset) == 0:
            return

        survey_type = self.survey_config.get("type", "catalog")
        graph = self.dataset[0]

        # Create appropriate TensorDict subclass
        if survey_type == "stellar":
            self.tensor_dict = self._create_spatial_tensor_dict(graph)
        elif survey_type == "galaxy":
            self.tensor_dict = self._create_cosmology_tensor_dict(graph)
        elif survey_type == "photometric":
            self.tensor_dict = self._create_photometric_tensor_dict(graph)
        elif survey_type == "spectroscopic":
            self.tensor_dict = self._create_spectral_tensor_dict(graph)
        elif survey_type == "timeseries":
            self.tensor_dict = self._create_lightcurve_tensor_dict(graph)
        else:
            # Generic survey tensor dict
            self.tensor_dict = SurveyTensorDict.from_graph(graph)

    def _create_spatial_tensor_dict(self, graph: Data) -> SpatialTensorDict:
        """Create spatial tensor dict for stellar surveys."""
        coords = graph.pos if hasattr(graph, "pos") else graph.x[:, :3]

        spatial = SpatialTensorDict(
            coordinates=coords,
            coordinate_system=self.survey_config.get("coordinate_system", "icrs"),
            unit=self.survey_config.get("distance_unit", "pc"),
        )

        # Add cosmic web analysis if requested
        if "cosmic_web" in self.graph_config:
            for scale in self.graph_config["cosmic_web_scales"]:
                labels = spatial.cosmic_web_clustering(eps_pc=scale)
                graph[f"cluster_labels_{scale}pc"] = labels

        return spatial

    def _create_photometric_tensor_dict(self, graph: Data) -> PhotometricTensorDict:
        """Create photometric tensor dict for multi-band surveys."""
        # Extract magnitude columns
        mag_columns = self.survey_config.get("magnitude_columns", [])
        if mag_columns and hasattr(graph, "x"):
            mags = graph.x[:, : len(mag_columns)]

            photom = PhotometricTensorDict(
                magnitudes=mags,
                bands=mag_columns,
                filter_system=self.survey_config.get("magnitude_system", "AB"),
            )

            # Add colors as features
            colors = photom.compute_colors()
            graph.color_features = colors

            return photom

        return None

    def _apply_selection_criteria(self):
        """Apply selection criteria to filter data."""
        if not self.selection_criteria or not self.dataset:
            return

        # Example criteria: magnitude limits, quality flags, etc.
        for criterion, value in self.selection_criteria.items():
            if criterion == "magnitude_limit" and hasattr(
                self.dataset[0], "magnitudes"
            ):
                mask = self.dataset[0].magnitudes[:, 0] < value
                self._apply_mask(mask)
            elif criterion == "quality_flag" and hasattr(self.dataset[0], "quality"):
                mask = self.dataset[0].quality >= value
                self._apply_mask(mask)

    def _extract_features(self):
        """Extract astronomical features using tensor dicts."""
        if not self.tensor_dict or not self.dataset:
            return

        graph = self.dataset[0]

        # Extract features based on tensor dict type
        if isinstance(self.tensor_dict, SpatialTensorDict):
            # Add kinematic features
            if hasattr(graph, "proper_motion"):
                graph.kinematic_features = (
                    self.tensor_dict.compute_tangential_velocity()
                )

        elif isinstance(self.tensor_dict, PhotometricTensorDict):
            # Color indices already added
            pass

        elif isinstance(self.tensor_dict, SpectralTensorDict):
            # Extract spectral features
            graph.spectral_features = self.tensor_dict.extract_spectral_features()

        elif isinstance(self.tensor_dict, LightcurveTensorDict):
            # Extract variability features
            graph.variability_features = self.tensor_dict.extract_variability_features()

    def _create_splits(self):
        """Create train/val/test splits using configured strategy."""
        if not self.dataset:
            return

        strategy = self.split_config["strategy"]

        if strategy == "spatial":
            self._create_spatial_splits()
        elif strategy == "temporal":
            self._create_temporal_splits()
        elif strategy == "stratified":
            self._create_stratified_splits()
        else:
            self._create_random_splits()

    def _create_spatial_splits(self):
        """Create spatially-aware splits using astronomical clustering."""
        if len(self.dataset) == 1:
            # Single graph - use spatial clustering
            graph = self.dataset[0]

            if self.tensor_dict and isinstance(self.tensor_dict, SpatialTensorDict):
                # Use cosmic web clustering for splits
                method = self.split_config.get("spatial_method", "clustering")

                if method == "clustering":
                    # Use existing clustering or compute new one
                    scale = self.graph_config["cosmic_web_scales"][0]
                    cluster_key = f"cluster_labels_{scale}pc"

                    if hasattr(graph, cluster_key):
                        cluster_labels = getattr(graph, cluster_key)
                    else:
                        cluster_labels = self.tensor_dict.cosmic_web_clustering(
                            eps_pc=scale
                        )

                    # Assign clusters to splits
                    self._assign_clusters_to_splits(graph, cluster_labels)

                elif method == "grid":
                    # Grid-based splitting
                    self._create_grid_splits(graph)

            else:
                # Fallback to coordinate-based splitting
                self._create_coordinate_splits(graph)

            # All splits point to same dataset
            for split in ["train", "val", "test"]:
                self._splits[split] = self.dataset
        else:
            # Multiple graphs - use spatial distribution
            self._create_multi_graph_spatial_splits()

    def _assign_clusters_to_splits(self, graph: Data, cluster_labels: torch.Tensor):
        """Assign clusters to train/val/test splits."""
        unique_clusters = torch.unique(cluster_labels)
        n_clusters = len(unique_clusters)

        # Shuffle clusters
        perm = torch.randperm(n_clusters)
        shuffled_clusters = unique_clusters[perm]

        # Calculate split sizes
        train_size = int(self.split_config["train_ratio"] * n_clusters)
        val_size = int(self.split_config["val_ratio"] * n_clusters)

        # Assign clusters
        train_clusters = shuffled_clusters[:train_size]
        val_clusters = shuffled_clusters[train_size : train_size + val_size]
        test_clusters = shuffled_clusters[train_size + val_size :]

        # Create masks
        graph.train_mask = torch.isin(cluster_labels, train_clusters)
        graph.val_mask = torch.isin(cluster_labels, val_clusters)
        graph.test_mask = torch.isin(cluster_labels, test_clusters)

    def _get_cache_path(self, cache_type: str) -> Path:
        """Get cache path for given type."""
        filename = f"{self.survey}_{cache_type}"
        if cache_type == "dataset":
            filename += f"_k{self.graph_config['k_neighbors']}"
        if self.max_samples:
            filename += f"_n{self.max_samples}"
        filename += ".pt"

        return self.cache_config["cache_dir"] / filename

    def _log_dataset_info(self):
        """Log comprehensive dataset information."""
        if not self.dataset:
            return

        info = {
            "survey": self.survey,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "graph_method": self.graph_config["method"],
            "split_strategy": self.split_config["strategy"],
        }

        # Add tensor dict info
        if self.tensor_dict:
            info["tensor_dict_type"] = type(self.tensor_dict).__name__
            if hasattr(self.tensor_dict, "coordinate_system"):
                info["coordinate_system"] = self.tensor_dict.coordinate_system

        # Add feature info
        if len(self.dataset) > 0:
            graph = self.dataset[0]
            info["num_nodes"] = (
                graph.num_nodes if hasattr(graph, "num_nodes") else graph.x.size(0)
            )
            info["num_features"] = graph.x.size(1) if hasattr(graph, "x") else 0

            # List available features
            features = []
            for attr in [
                "color_features",
                "kinematic_features",
                "spectral_features",
                "variability_features",
            ]:
                if hasattr(graph, attr):
                    features.append(attr)
            info["extracted_features"] = features

        logger.info(f"Dataset configuration: {info}")

    # Transform methods
    def _normalize_magnitudes(self, data: Data) -> Data:
        """Normalize magnitude features."""
        if (
            hasattr(data, "x")
            and self.tensor_dict
            and isinstance(self.tensor_dict, PhotometricTensorDict)
        ):
            # Use photometric tensor dict normalization
            data.x = self.tensor_dict.normalize_magnitudes(data.x)
        return data

    def _compute_colors(self, data: Data) -> Data:
        """Compute color indices."""
        if self.tensor_dict and isinstance(self.tensor_dict, PhotometricTensorDict):
            colors = self.tensor_dict.compute_colors()
            data.color_features = colors
        return data

    def _add_kinematics(self, data: Data) -> Data:
        """Add kinematic features."""
        if self.tensor_dict and isinstance(self.tensor_dict, SpatialTensorDict):
            if hasattr(data, "proper_motion"):
                data.kinematic_features = self.tensor_dict.compute_tangential_velocity()
        return data

    def _normalize_features(self, data: Data) -> Data:
        """Generic feature normalization."""
        if hasattr(data, "x"):
            # Simple standardization
            mean = data.x.mean(dim=0, keepdim=True)
            std = data.x.std(dim=0, keepdim=True)
            data.x = (data.x - mean) / (std + 1e-8)
        return data

    # DataLoader creation methods remain similar but use optimized settings
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader with optimized settings."""
        if not self.dataset:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        # Apply optimization config if available
        batch_size = self.batch_size
        if self.optimization_config:
            batch_size = self.optimization_config.batch_size

        # Use neighbor sampling for large graphs
        if self._should_use_neighbor_sampling():
            return self._create_neighbor_loader("train", shuffle=True)

        return self._create_standard_loader(
            self._splits["train"], batch_size, shuffle=True
        )

    def _should_use_neighbor_sampling(self) -> bool:
        """Determine if neighbor sampling should be used."""
        if len(self.dataset) == 1 and hasattr(self.dataset[0], "num_nodes"):
            # Use neighbor sampling for large single graphs
            return self.dataset[0].num_nodes > 10000
        return False

    def _create_standard_loader(
        self, dataset: Any, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        """Create standard DataLoader with all optimizations."""
        # Build loader arguments
        loader_args = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }

        # Add prefetch factor for multi-worker loading
        if self.num_workers > 0:
            loader_args["prefetch_factor"] = 2

        return DataLoader(dataset, **loader_args)

    @property
    def num_features(self) -> int:
        """Get number of features including extracted ones."""
        if "num_features" not in self._cached_properties:
            total_features = 0

            if self.dataset and len(self.dataset) > 0:
                graph = self.dataset[0]

                # Base features
                if hasattr(graph, "x"):
                    total_features = graph.x.size(1)

                # Add extracted features
                for attr in [
                    "color_features",
                    "kinematic_features",
                    "spectral_features",
                    "variability_features",
                ]:
                    if hasattr(graph, attr):
                        feat = getattr(graph, attr)
                        if isinstance(feat, torch.Tensor):
                            total_features += feat.size(1) if feat.dim() > 1 else 1

            self._cached_properties["num_features"] = total_features

        return self._cached_properties["num_features"]

    @property
    def num_classes(self) -> int:
        """Get number of classes from survey config or data."""
        if "num_classes" not in self._cached_properties:
            # Try survey config first
            num_classes = self.survey_config.get("num_classes", None)

            if num_classes is None and self.dataset and len(self.dataset) > 0:
                # Infer from data
                graph = self.dataset[0]
                if hasattr(graph, "y") and graph.y is not None:
                    if graph.y.dim() == 1:
                        num_classes = len(torch.unique(graph.y))
                    else:
                        num_classes = graph.y.size(1)
                else:
                    num_classes = 2  # Default binary

            self._cached_properties["num_classes"] = num_classes or 2

        return self._cached_properties["num_classes"]


def create_datamodule(survey: str, **kwargs) -> AstroDataModule:
    """
    Create an optimized DataModule for the given survey.

    This factory function automatically applies survey-specific optimizations
    and configurations from the central config system.
    """
    # Get survey-specific configurations
    survey_config = get_survey_config(survey)
    optimization = get_survey_optimization(survey)

    # Build default graph config based on survey type
    survey_type = survey_config.get("type", "catalog")
    default_graph_config = {
        "method": "knn",
        "k_neighbors": 8,
        "use_3d": True,
    }

    # Add survey-specific graph settings
    if survey_type == "stellar":
        default_graph_config["cosmic_web_scales"] = [5.0, 10.0, 25.0, 50.0]
    elif survey_type == "galaxy":
        default_graph_config["cosmic_web_scales"] = [1.0, 5.0, 10.0, 20.0]  # Mpc

    # Merge with user config
    if "graph_config" in kwargs:
        kwargs["graph_config"] = {**default_graph_config, **kwargs["graph_config"]}
    else:
        kwargs["graph_config"] = default_graph_config

    # Apply optimization settings if not overridden
    if optimization:
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = optimization.batch_size
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = optimization.num_workers

    # Set default transform pipeline if not provided
    if "transform_pipeline" not in kwargs:
        if survey_type == "stellar":
            kwargs["transform_pipeline"] = [
                "normalize_magnitudes",
                "compute_colors",
                "add_kinematics",
            ]
        elif survey_type == "photometric":
            kwargs["transform_pipeline"] = ["normalize_magnitudes", "compute_colors"]

    return AstroDataModule(survey=survey, **kwargs)
