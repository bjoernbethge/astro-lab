"""
AstroLab DataModule - Optimized Lightning DataModule (2025)
==========================================================

Lightning DataModule for astronomical datasets with modern optimizations.
Implements best practices for GPU training and memory efficiency.
"""

import logging
import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import numpy as np
import polars as pl
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.loader import DataLoader

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
from .config import data_config
from .datasets import SurveyGraphDataset
from .utils import (
    check_graph_consistency,
    create_optimized_dataloader,
    get_graph_statistics,
    sample_subgraph_random,
)

logger = logging.getLogger(__name__)


class AstroDataModule(L.LightningDataModule):
    """
    Optimized Lightning DataModule for astronomical survey data.

    Features:
    - Automatic DataLoader optimization for GPUs
    - Smart memory pinning and prefetching
    - Support for all model types (node, graph, temporal, point)
    - Memory-efficient batch processing
    """

    # Model type normalization mapping
    MODEL_TYPE_MAPPING = {
        # Canonical types
        "node": "node",
        "graph": "graph",
        "temporal": "temporal",
        "point": "point",
        # Alternative names
        "astro_node_gnn": "node",
        "astro_graph_gnn": "graph",
        "astro_temporal_gnn": "temporal",
        "astro_pointnet": "point",
        "AstroNodeGNN": "node",
        "AstroGraphGNN": "graph",
        "AstroTemporalGNN": "temporal",
        "AstroPointNet": "point",
    }

    def __init__(
        self,
        survey: str,
        data_root: Optional[str] = None,
        k_neighbors: int = 8,
        max_samples: Optional[int] = None,
        batch_size: int = 32,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        num_workers: int = 0,
        model_type: Optional[str] = None,
        subgraph_size: int = 100,
        # DataLoader optimization parameters
        pin_memory: bool = False,
        persistent_workers: bool = False,
        prefetch_factor: Optional[int] = None,
        drop_last: bool = True,
        # Advanced memory optimization
        use_smart_pinning: bool = True,
        non_blocking_transfer: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Core parameters
        self.survey = survey
        self.data_root = data_root or tempfile.gettempdir()
        self.k_neighbors = k_neighbors
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.subgraph_size = subgraph_size
        self.drop_last = drop_last

        # Normalize model type
        if model_type:
            self.model_type = self.MODEL_TYPE_MAPPING.get(model_type, model_type)
            if model_type != self.model_type:
                logger.info(f"Normalized model_type: {model_type} -> {self.model_type}")
        else:
            self.model_type = "node"  # Default

        # GPU detection for optimization
        self.use_gpu = torch.cuda.is_available()
        self.target_device = torch.device("cuda" if self.use_gpu else "cpu")

        # DataLoader optimization based on 2025 best practices
        if self.use_gpu and num_workers > 0:
            self.pin_memory = pin_memory
            self.persistent_workers = persistent_workers
            self.prefetch_factor = prefetch_factor if prefetch_factor else 2
        else:
            self.pin_memory = False
            self.persistent_workers = False
            self.prefetch_factor = None

        # Advanced memory optimization flags
        self.use_smart_pinning = use_smart_pinning and self.use_gpu
        self.non_blocking_transfer = non_blocking_transfer and self.use_gpu

        # Will be created in setup()
        self.lightning_data = None
        self.dataset = None
        self._main_data = None
        self._num_features = None
        self._num_classes = None

        logger.info(
            f"🔧 AstroDataModule initialized:\n"
            f"   Dataset: {survey}, Model type: {self.model_type}\n"
            f"   Batch size: {batch_size}, Workers: {num_workers}\n"
            f"   GPU: {self.use_gpu}, Pin memory: {self.pin_memory}\n"
            f"   Smart pinning: {self.use_smart_pinning}"
        )

    def prepare_data(self):
        """Download or prepare data. Called only on rank 0."""
        # This is called before setup() and only on the main process
        # Use for downloading or one-time preparation
        logger.info(f"📥 Preparing data for {self.survey}")

        # Ensure data directories exist
        Path(self.data_root).mkdir(parents=True, exist_ok=True)

    def setup(self, stage: Optional[str] = None):
        """Setup datasets with optimized loading."""
        if self.lightning_data is None and self._main_data is None:
            logger.info(f"📊 Setting up {self.survey} dataset")

            # Create the survey dataset
            self.dataset = SurveyGraphDataset(
                root=self.data_root,
                survey=self.survey,
                k_neighbors=self.k_neighbors,
                max_samples=self.max_samples,
            )

            if len(self.dataset) == 0:
                raise RuntimeError(f"Dataset {self.survey} is empty")

            # Get the main data object
            self._main_data = self.dataset._data
            if not isinstance(self._main_data, Data):
                raise TypeError(f"Expected Data object, got {type(self._main_data)}")

            # Extract features and classes info
            self._extract_data_info()

            # Create synthetic labels if needed
            self._ensure_labels()

            # Setup based on model type
            if self.model_type in ["graph", "point"]:
                # For graph-level and point cloud tasks
                self._setup_graph_level()
            elif self.model_type == "temporal":
                # For temporal tasks
                self._setup_temporal()
            else:
                # Default: node-level tasks
                self._setup_node_level()

            logger.info(f"✅ Setup complete for {self.model_type} model")
            logger.info(
                f"   Features: {self.num_features}, Classes: {self.num_classes}"
            )

    def _extract_data_info(self):
        """Extract feature and class information from data."""
        if hasattr(self._main_data, "x") and self._main_data.x is not None:
            if self._main_data.x.dim() >= 2:
                self._num_features = self._main_data.x.size(1)
            else:
                self._num_features = 1
        else:
            self._num_features = 1
            logger.warning("No features found in data, using num_features=1")

        if hasattr(self._main_data, "y") and self._main_data.y is not None:
            self._num_classes = len(torch.unique(self._main_data.y))
        else:
            self._num_classes = 2  # Default binary

    def _ensure_labels(self):
        """Ensure the data has labels for training."""
        data = self._main_data

        if not hasattr(data, "y") or data.y is None:
            logger.warning("No labels found. Creating synthetic binary labels.")
            if hasattr(data, "x") and data.x is not None:
                # Use feature-based synthetic labels
                if data.x.dim() >= 2:
                    feature_sums = data.x.sum(dim=1)
                else:
                    feature_sums = data.x
                median_val = feature_sums.median()
                data.y = (feature_sums > median_val).long()
            else:
                # Random binary labels as fallback
                num_nodes = data.num_nodes if hasattr(data, "num_nodes") else 100
                data.y = torch.randint(0, 2, (num_nodes,))

            self._num_classes = 2

    def _setup_node_level(self):
        """Setup for node-level tasks with optimized sampling."""
        # Create train/val/test masks
        self._create_node_splits()

        # Configure neighborhood sampling for efficiency
        if self.batch_size > 1:
            # Use smaller neighborhoods for larger batches
            num_neighbors = [15, 10] if self.batch_size <= 64 else [10, 5]
        else:
            num_neighbors = [25, 15]  # Larger neighborhoods for full-batch

        # Use LightningNodeData for efficient node classification
        self.lightning_data = LightningNodeData(
            self._main_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            num_neighbors=num_neighbors,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

        logger.info(
            f"✅ Node-level setup: {self._main_data.num_nodes} nodes, "
            f"neighbors={num_neighbors}"
        )

    def _setup_graph_level(self):
        """Setup for graph-level tasks with efficient subgraph creation."""
        import torch
        from torch_geometric.utils import subgraph

        # Ensure we have valid data
        if self._main_data is None:
            raise ValueError("No main data available for graph-level setup")

        if (
            not hasattr(self._main_data, "num_nodes")
            or self._main_data.num_nodes is None
        ):
            raise ValueError("Main data has no valid num_nodes")

        # Calculate optimal number of subgraphs
        num_nodes = self._main_data.num_nodes
        subgraph_size = self.subgraph_size or 100  # Default if None
        num_subgraphs = max(
            self.batch_size * 3,  # At least 3x batch size for variety
            min(num_nodes // subgraph_size, 1000),  # Cap at 1000
        )

        device = (
            self._main_data.x.device
            if hasattr(self._main_data, "x") and self._main_data.x is not None
            else "cpu"
        )

        # Pre-allocate lists for efficiency
        subgraphs = []

        # Use efficient batch processing for subgraph creation
        logger.info(f"Creating {num_subgraphs} subgraphs...")

        for i in range(num_subgraphs):
            # Efficient node sampling
            if num_nodes > subgraph_size:
                # Use more efficient sampling for large graphs
                start_idx = torch.randint(0, num_nodes - subgraph_size, (1,)).item()
                node_idx = torch.arange(
                    start_idx, start_idx + subgraph_size, device=device
                )
                # Shuffle for randomness
                node_idx = node_idx[torch.randperm(subgraph_size, device=device)]
            else:
                node_idx = torch.arange(num_nodes, device=device)

            # Extract subgraph efficiently
            edge_index = self._main_data.edge_index.to(device)
            edge_index_sub, _ = subgraph(
                node_idx, edge_index, relabel_nodes=True, num_nodes=num_nodes
            )

            # Create subgraph data - keep on CPU for DataLoader
            x_data = None
            if hasattr(self._main_data, "x") and self._main_data.x is not None:
                x_data = self._main_data.x[node_idx].cpu()

            data = Data(
                x=x_data,
                edge_index=edge_index_sub.cpu(),
                y=torch.tensor(
                    [i % (self._num_classes or 2)], dtype=torch.long
                ),  # Balanced labels with default
            )

            # Add additional attributes if present
            for attr in ["edge_attr", "pos"]:
                if hasattr(self._main_data, attr):
                    attr_value = getattr(self._main_data, attr)
                    if attr_value is not None:
                        setattr(data, attr, attr_value[node_idx].cpu())

            subgraphs.append(data)

        # Split into train/val/test with stratification
        num_train = int(len(subgraphs) * self.train_ratio)
        num_val = int(len(subgraphs) * self.val_ratio)

        # Shuffle for better distribution
        indices = torch.randperm(len(subgraphs))
        subgraphs = [subgraphs[i] for i in indices]

        self.train_graphs = subgraphs[:num_train]
        self.val_graphs = subgraphs[num_train : num_train + num_val]
        self.test_graphs = subgraphs[num_train + num_val :]

        logger.info(
            f"✅ Graph-level setup - Train: {len(self.train_graphs)}, "
            f"Val: {len(self.val_graphs)}, Test: {len(self.test_graphs)}"
        )

    def _setup_temporal(self):
        """Setup for temporal tasks (placeholder for future implementation)."""
        # For now, use node-level setup
        # TODO: Implement proper temporal data handling
        logger.warning("Temporal setup not fully implemented, using node-level setup")
        self._setup_node_level()

    def _create_node_splits(self):
        """Create train/val/test masks with stratification."""
        data = self._main_data
        num_nodes = data.num_nodes

        # Initialize masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        # Stratified split if labels exist
        if hasattr(data, "y") and data.y is not None:
            # Get indices for each class
            unique_classes = torch.unique(data.y)

            for c in unique_classes:
                class_indices = torch.where(data.y == c)[0]
                n_class = len(class_indices)

                # Shuffle indices
                perm = torch.randperm(n_class)
                class_indices = class_indices[perm]

                # Calculate splits
                train_size = int(n_class * self.train_ratio)
                val_size = int(n_class * self.val_ratio)

                # Assign masks
                data.train_mask[class_indices[:train_size]] = True
                data.val_mask[class_indices[train_size : train_size + val_size]] = True
                data.test_mask[class_indices[train_size + val_size :]] = True
        else:
            # Random split
            indices = torch.randperm(num_nodes)
            train_size = int(num_nodes * self.train_ratio)
            val_size = int(num_nodes * self.val_ratio)

            data.train_mask[indices[:train_size]] = True
            data.val_mask[indices[train_size : train_size + val_size]] = True
            data.test_mask[indices[train_size + val_size :]] = True

        logger.info(
            f"Node splits - Train: {data.train_mask.sum().item()}, "
            f"Val: {data.val_mask.sum().item()}, Test: {data.test_mask.sum().item()}"
        )

    def _get_dataloader_kwargs(self) -> dict:
        """Get optimized DataLoader kwargs based on 2025 best practices."""
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "drop_last": self.drop_last,
        }

        # Add prefetch_factor only if using workers
        if self.num_workers > 0 and self.prefetch_factor:
            kwargs["prefetch_factor"] = self.prefetch_factor

        # Add custom collate function for memory efficiency
        if self.model_type in ["graph", "point"]:
            kwargs["collate_fn"] = self._efficient_collate

        return kwargs

    def _efficient_collate(self, batch):
        """Memory-efficient collate function for graph data."""
        # Use PyG's Batch for efficient batching
        return Batch.from_data_list(batch)

    def train_dataloader(self):
        """Get optimized training dataloader."""
        if self.lightning_data is None and not hasattr(self, "train_graphs"):
            self.setup()

        if self.model_type in ["graph", "point"]:
            # Graph-level dataloader
            kwargs = self._get_dataloader_kwargs()
            kwargs["shuffle"] = True
            return DataLoader(self.train_graphs, **kwargs)
        else:
            # Node-level dataloader
            return self.lightning_data.train_dataloader()

    def val_dataloader(self):
        """Get optimized validation dataloader."""
        if self.lightning_data is None and not hasattr(self, "val_graphs"):
            self.setup()

        if self.model_type in ["graph", "point"]:
            # Graph-level dataloader
            kwargs = self._get_dataloader_kwargs()
            kwargs["shuffle"] = False
            kwargs["drop_last"] = False  # Keep all validation samples
            return DataLoader(self.val_graphs, **kwargs)
        else:
            # Node-level dataloader
            return self.lightning_data.val_dataloader()

    def test_dataloader(self):
        """Get optimized test dataloader."""
        if self.lightning_data is None and not hasattr(self, "test_graphs"):
            self.setup()

        if self.model_type in ["graph", "point"]:
            # Graph-level dataloader
            kwargs = self._get_dataloader_kwargs()
            kwargs["shuffle"] = False
            kwargs["drop_last"] = False  # Keep all test samples
            return DataLoader(self.test_graphs, **kwargs)
        else:
            # Node-level dataloader
            return self.lightning_data.test_dataloader()

    def predict_dataloader(self):
        """Get prediction dataloader."""
        return self.test_dataloader()

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        if self._num_classes is not None:
            return self._num_classes
        return 2  # Default binary

    @property
    def num_features(self) -> int:
        """Number of node features."""
        if self._num_features is not None:
            return self._num_features
        return 1  # Default

    def teardown(self, stage: Optional[str] = None):
        """Clean up after training/testing."""
        # Clear references for garbage collection
        self._main_data = None
        self.lightning_data = None

        # Clear GPU cache if used
        if self.use_gpu:
            torch.cuda.empty_cache()

        # Use central memory management
        from astro_lab.memory import force_comprehensive_cleanup

        force_comprehensive_cleanup()

        logger.info("🧹 Cleaned up DataModule resources")

    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        if self.dataset is None:
            return {"error": "Dataset not initialized"}

        info = self.dataset.get_info()

        # Add DataModule-specific info
        info.update(
            {
                # Basic info
                "batch_size": self.batch_size,
                "num_workers": self.num_workers,
                "model_type": self.model_type,
                "train_ratio": self.train_ratio,
                "val_ratio": self.val_ratio,
                "num_classes": self.num_classes,
                "num_features": self.num_features,
                # Optimization info
                "pin_memory": self.pin_memory,
                "persistent_workers": self.persistent_workers,
                "prefetch_factor": self.prefetch_factor,
                "drop_last": self.drop_last,
                "use_smart_pinning": self.use_smart_pinning,
                "non_blocking_transfer": self.non_blocking_transfer,
                "target_device": str(self.target_device),
                "use_gpu": self.use_gpu,
            }
        )

        # Add split info if available
        if hasattr(self, "_main_data") and self._main_data is not None:
            if hasattr(self._main_data, "train_mask"):
                info["train_samples"] = self._main_data.train_mask.sum().item()
                info["val_samples"] = self._main_data.val_mask.sum().item()
                info["test_samples"] = self._main_data.test_mask.sum().item()
            elif hasattr(self, "train_graphs"):
                info["train_graphs"] = len(self.train_graphs)
                info["val_graphs"] = len(self.val_graphs)
                info["test_graphs"] = len(self.test_graphs)

        return info

    def state_dict(self) -> Dict[str, Any]:
        """Save DataModule state."""
        return {
            "survey": self.survey,
            "k_neighbors": self.k_neighbors,
            "max_samples": self.max_samples,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "model_type": self.model_type,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load DataModule state."""
        for key, value in state_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)


# Factory function for convenience
def create_optimized_datamodule(
    survey: str, model_type: str, batch_size: int = 32, **kwargs
) -> AstroDataModule:
    """
    Create an optimized DataModule with automatic GPU detection.

    Args:
        survey: Dataset name (e.g., 'gaia', 'sdss')
        model_type: Model type ('node', 'graph', 'temporal', 'point')
        batch_size: Batch size
        **kwargs: Additional parameters

    Returns:
        Configured AstroDataModule
    """
    # Auto-detect GPU and set optimal defaults
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        # GPU-optimized settings
        defaults = {
            "num_workers": min(8, multiprocessing.cpu_count()),
            "pin_memory": True,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "drop_last": True,
        }
    else:
        # CPU settings
        defaults = {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "drop_last": False,
        }

    # Update with user kwargs
    defaults.update(kwargs)

    return AstroDataModule(
        survey=survey, model_type=model_type, batch_size=batch_size, **defaults
    )
