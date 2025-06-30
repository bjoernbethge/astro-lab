"""
Lightning DataModule using PyG Lightning Wrappers
================================================

Enhanced with large-scale training support:
- NeighborLoader for mini-batch sampling
- ClusterGCN and GraphSAINT samplers
- Memory-efficient data loading
- Dynamic batching support
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import lightning as L
import torch
import torch_cluster
from torch.utils.data import Subset
from torch_geometric.data import Data, Dataset
from torch_geometric.data.lightning import LightningDataset, LightningNodeData
from torch_geometric.loader import (
    DataLoader,
    NeighborLoader,
    ClusterData,
    ClusterLoader,
    GraphSAINTSampler,
    GraphSAINTNodeSampler,
    GraphSAINTEdgeSampler,
    GraphSAINTRandomWalkSampler,
)
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import fps  # Farthest point sampling
from torch_sparse import coalesce

from astro_lab.config import get_data_config, get_survey_config
from astro_lab.data.datasets import AstroPointCloudDataset, SurveyGraphDataset
from astro_lab.data.preprocessors import get_preprocessor

logger = logging.getLogger(__name__)


class AstroLightningDataset(L.LightningDataModule):
    """
    Lightning DataModule for AstroLab datasets with large-scale support.

    Features:
    - Multiple sampling strategies (NeighborLoader, ClusterGCN, GraphSAINT)
    - Memory-efficient data loading
    - Dynamic batching for variable-size graphs
    - Adaptive batch size based on GPU memory
    """

    def __init__(
        self,
        survey: str,
        task: str = "node_classification",
        dataset_type: str = "graph",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        k_neighbors: int = 20,
        # Large-scale sampling parameters
        sampling_strategy: str = "none",  # "none", "neighbor", "cluster", "saint"
        neighbor_sizes: List[int] = [25, 10],
        num_clusters: int = 1500,
        saint_sample_coverage: int = 50,
        saint_walk_length: int = 2,
        # Dynamic batching
        enable_dynamic_batching: bool = False,
        min_batch_size: int = 1,
        max_batch_size: int = 512,
        # Dataset parameters
        max_samples: Optional[int] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_subgraphs: int = 1000,
        points_per_subgraph: int = 500,
        # Other parameters
        data_root: Optional[str] = None,
        force_reload: bool = False,
        # Partitioning parameters
        partition_method: Optional[str] = None,  # "metis", "random"
        num_partitions: int = 4,
        **kwargs,
    ):
        """
        Initialize enhanced AstroLab Lightning DataModule.

        Args:
            survey: Survey name (gaia, sdss, etc.)
            task: Task type (node_classification, graph_classification, etc.)
            dataset_type: Dataset type (graph, point_cloud, etc.)
            batch_size: Batch size for training
            num_workers: Number of DataLoader workers
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
            prefetch_factor: Number of batches to prefetch
            k_neighbors: Number of k-nearest neighbors for graph construction
            sampling_strategy: Sampling strategy for large graphs
            neighbor_sizes: List of neighbor sizes for NeighborLoader
            num_clusters: Number of clusters for ClusterGCN
            saint_sample_coverage: Sample coverage for GraphSAINT
            saint_walk_length: Walk length for GraphSAINT random walk
            enable_dynamic_batching: Enable dynamic batch size adjustment
            min_batch_size: Minimum batch size for dynamic batching
            max_batch_size: Maximum batch size for dynamic batching
            max_samples: Maximum number of samples for development/testing
            train_ratio: Ratio of training samples
            val_ratio: Ratio of validation samples
            num_subgraphs: Number of subgraphs for point cloud datasets
            points_per_subgraph: Number of points per subgraph
            data_root: Data root directory
            force_reload: Whether to force reload data
            partition_method: Graph partitioning method for distributed training
            num_partitions: Number of partitions for distributed training
            **kwargs: Additional arguments
        """
        super().__init__()

        self.survey = survey
        self.task = task
        self.dataset_type = dataset_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.k_neighbors = k_neighbors
        
        # Large-scale parameters
        self.sampling_strategy = sampling_strategy
        self.neighbor_sizes = neighbor_sizes
        self.num_clusters = num_clusters
        self.saint_sample_coverage = saint_sample_coverage
        self.saint_walk_length = saint_walk_length
        
        # Dynamic batching
        self.enable_dynamic_batching = enable_dynamic_batching
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = batch_size
        
        # Dataset parameters
        self.max_samples = max_samples
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_subgraphs = num_subgraphs
        self.points_per_subgraph = points_per_subgraph
        self.force_reload = force_reload
        
        # Partitioning parameters
        self.partition_method = partition_method
        self.num_partitions = num_partitions
        
        self.kwargs = kwargs

        # Get configurations
        self.data_config = get_data_config()
        self.survey_config = get_survey_config(survey)
        self.data_root = Path(data_root) if data_root else self.data_config.base_dir

        # Dataset components
        self.full_dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Large-scale components
        self.cluster_data = None
        self.saint_sampler = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        logger.info(f"Initialized {dataset_type} Lightning DataModule for {survey}")
        logger.info(f"Sampling strategy: {sampling_strategy}")
        if max_samples:
            logger.info(f"Development mode: limiting to {max_samples} samples")

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for the given stage with large-scale support."""
        if self.full_dataset is None:
            # Create the appropriate dataset
            if self.dataset_type == "point_cloud":
                self.full_dataset = AstroPointCloudDataset(
                    root=str(self.data_root),
                    survey=self.survey,
                    k_neighbors=self.k_neighbors,
                    num_subgraphs=self.num_subgraphs,
                    points_per_subgraph=self.points_per_subgraph,
                    force_reload=self.force_reload,
                    **self.kwargs,
                )
            else:
                self.full_dataset = SurveyGraphDataset(
                    root=str(self.data_root),
                    survey=self.survey,
                    k_neighbors=self.k_neighbors,
                    force_reload=self.force_reload,
                    **self.kwargs,
                )

            # Setup splits based on dataset type
            self._setup_splits()
            
            # Setup large-scale components if needed
            if self.sampling_strategy != "none":
                self._setup_large_scale_sampling()
                
            # Setup dynamic batching if enabled
            if self.enable_dynamic_batching:
                self._setup_dynamic_batching()

    def _setup_splits(self):
        """Setup train/val/test splits."""
        n_samples = len(self.full_dataset)

        # For single-graph datasets, use node-level splits
        if n_samples == 1 and self.dataset_type == "graph":
            logger.info("Single graph detected, using node-level splits")
            self._setup_node_level_splits()
        else:
            # Standard graph-level splits
            self._setup_graph_level_splits(n_samples)

    def _setup_node_level_splits(self):
        """Setup node-level splits for single graph."""
        graph = self.full_dataset[0]
        if hasattr(graph, "num_nodes") and graph.num_nodes is not None:
            num_nodes = int(graph.num_nodes)

            # Create node-level splits
            n_train = int(num_nodes * self.train_ratio)
            n_val = int(num_nodes * self.val_ratio)

            # Create node indices
            node_indices = torch.randperm(num_nodes)
            train_nodes = node_indices[:n_train]
            val_nodes = node_indices[n_train : n_train + n_val]
            test_nodes = node_indices[n_train + n_val :]

            # Create node masks
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)

            train_mask[train_nodes] = True
            val_mask[val_nodes] = True
            test_mask[test_nodes] = True

            # Add masks to graph
            graph.train_mask = train_mask
            graph.val_mask = val_mask
            graph.test_mask = test_mask

            # Use the same graph for all splits (with different masks)
            self.train_dataset = self.full_dataset
            self.val_dataset = self.full_dataset
            self.test_dataset = self.full_dataset

            logger.info(
                f"Node-level splits: train={train_mask.sum()}, "
                f"val={val_mask.sum()}, test={test_mask.sum()}"
            )
        else:
            # Fallback to standard splits
            logger.warning("Graph has no num_nodes attribute, using standard splits")
            self.train_dataset = self.full_dataset
            self.val_dataset = None
            self.test_dataset = None

    def _setup_graph_level_splits(self, n_samples: int):
        """Setup graph-level splits."""
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)

        # Create indices
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train : n_train + n_val].tolist()
        test_indices = indices[n_train + n_val :].tolist()

        # Create proper dataset subsets
        self.train_dataset = (
            Subset(self.full_dataset, train_indices)
            if train_indices
            else self.full_dataset
        )
        self.val_dataset = (
            Subset(self.full_dataset, val_indices) if val_indices else None
        )
        self.test_dataset = (
            Subset(self.full_dataset, test_indices) if test_indices else None
        )

        logger.info(f"Created {self.dataset_type} dataset with {n_samples} samples")
        logger.info(
            f"Splits: train={len(train_indices)}, val={len(val_indices)}, "
            f"test={len(test_indices)}"
        )

    def _setup_large_scale_sampling(self):
        """Setup large-scale sampling strategies."""
        if self.sampling_strategy == "cluster":
            self._setup_cluster_sampling()
        elif self.sampling_strategy == "saint":
            self._setup_saint_sampling()
        # NeighborLoader is handled in dataloader creation

    def _setup_cluster_sampling(self):
        """Setup ClusterGCN sampling."""
        if len(self.full_dataset) == 1:  # Single large graph
            graph = self.full_dataset[0]
            self.cluster_data = ClusterData(
                graph, 
                num_parts=self.num_clusters,
                recursive=False,
                save_dir=self.data_root / "cluster_data"
            )
            logger.info(f"Created {self.num_clusters} clusters for ClusterGCN")

    def _setup_saint_sampling(self):
        """Setup GraphSAINT sampling."""
        if len(self.full_dataset) == 1:  # Single large graph
            graph = self.full_dataset[0]
            # Choose sampler based on graph properties
            if graph.num_edges < 1000000:
                self.saint_sampler = GraphSAINTNodeSampler
            elif graph.num_edges < 10000000:
                self.saint_sampler = GraphSAINTEdgeSampler
            else:
                self.saint_sampler = GraphSAINTRandomWalkSampler
            logger.info(f"Using {self.saint_sampler.__name__} for GraphSAINT")

    def _setup_dynamic_batching(self):
        """Setup dynamic batching based on GPU memory."""
        if torch.cuda.is_available():
            # Get GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            # Adjust batch size based on memory
            if gpu_memory < 8:
                self.current_batch_size = min(self.batch_size, 16)
            elif gpu_memory < 16:
                self.current_batch_size = min(self.batch_size, 32)
            else:
                self.current_batch_size = self.batch_size
                
            logger.info(
                f"Dynamic batching: GPU memory {gpu_memory:.1f}GB, "
                f"batch size {self.current_batch_size}"
            )

    def train_dataloader(self):
        """Create training dataloader with large-scale support."""
        if self.train_loader is not None:
            return self.train_loader
            
        # NeighborLoader for node-level tasks
        if self.sampling_strategy == "neighbor" and self.task.startswith("node"):
            if len(self.train_dataset) == 1:  # Single graph
                graph = self.train_dataset[0]
                self.train_loader = NeighborLoader(
                    graph,
                    num_neighbors=self.neighbor_sizes,
                    batch_size=self.current_batch_size,
                    input_nodes=graph.train_mask,
                    shuffle=True,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers and self.num_workers > 0,
                    pin_memory=self.pin_memory and torch.cuda.is_available(),
                )
                return self.train_loader
        
        # ClusterLoader for ClusterGCN
        elif self.sampling_strategy == "cluster" and self.cluster_data is not None:
            self.train_loader = ClusterLoader(
                self.cluster_data,
                batch_size=max(1, self.current_batch_size // 10),  # Fewer clusters per batch
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers and self.num_workers > 0,
            )
            return self.train_loader
        
        # GraphSAINT sampler
        elif self.sampling_strategy == "saint" and self.saint_sampler is not None:
            graph = self.train_dataset[0]
            self.train_loader = self.saint_sampler(
                graph,
                batch_size=self.current_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                sample_coverage=self.saint_sample_coverage,
                walk_length=self.saint_walk_length if hasattr(self, 'walk_length') else 2,
                num_steps=30,  # Number of iterations per epoch
            )
            return self.train_loader
        
        # Default DataLoader
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.current_batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory and torch.cuda.is_available(),
                persistent_workers=self.persistent_workers and self.num_workers > 0,
                prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                multiprocessing_context="spawn" if self.num_workers > 0 else None,
            )

    def val_dataloader(self):
        """Create validation dataloader with large-scale support."""
        if self.val_dataset is None:
            return None
            
        if self.val_loader is not None:
            return self.val_loader
            
        # Similar logic to train_dataloader but without shuffle
        if self.sampling_strategy == "neighbor" and self.task.startswith("node"):
            if len(self.val_dataset) == 1:
                graph = self.val_dataset[0]
                self.val_loader = NeighborLoader(
                    graph,
                    num_neighbors=self.neighbor_sizes,
                    batch_size=self.current_batch_size,
                    input_nodes=graph.val_mask,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers and self.num_workers > 0,
                    pin_memory=self.pin_memory and torch.cuda.is_available(),
                )
                return self.val_loader
        
        # Default validation loader
        return DataLoader(
            self.val_dataset,
            batch_size=self.current_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        """Create test dataloader with large-scale support."""
        if self.test_dataset is None:
            return None
            
        if self.test_loader is not None:
            return self.test_loader
            
        # Similar logic to val_dataloader
        if self.sampling_strategy == "neighbor" and self.task.startswith("node"):
            if len(self.test_dataset) == 1:
                graph = self.test_dataset[0]
                self.test_loader = NeighborLoader(
                    graph,
                    num_neighbors=self.neighbor_sizes,
                    batch_size=self.current_batch_size,
                    input_nodes=graph.test_mask,
                    shuffle=False,
                    num_workers=self.num_workers,
                    persistent_workers=self.persistent_workers and self.num_workers > 0,
                    pin_memory=self.pin_memory and torch.cuda.is_available(),
                )
                return self.test_loader
        
        # Default test loader
        return DataLoader(
            self.test_dataset,
            batch_size=self.current_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def adjust_batch_size(self, factor: float = 1.0):
        """Dynamically adjust batch size."""
        if self.enable_dynamic_batching:
            new_size = int(self.current_batch_size * factor)
            self.current_batch_size = max(
                self.min_batch_size, 
                min(self.max_batch_size, new_size)
            )
            logger.info(f"Adjusted batch size to {self.current_batch_size}")
            
            # Reset loaders to use new batch size
            self.train_loader = None
            self.val_loader = None
            self.test_loader = None

    @property
    def num_features(self) -> int:
        """Get number of features from the dataset."""
        if self.full_dataset is None:
            return 0
        if hasattr(self.full_dataset, "num_features"):
            return self.full_dataset.num_features
        # Get from first sample
        sample = self.full_dataset[0]
        return sample.x.shape[1] if hasattr(sample, "x") else 0

    @property
    def num_classes(self) -> int:
        """Get number of classes from the dataset."""
        if self.full_dataset is None:
            return 1
        if hasattr(self.full_dataset, "num_classes"):
            return self.full_dataset.num_classes
        # Get from first sample
        sample = self.full_dataset[0]
        if hasattr(sample, "y"):
            if sample.y.dim() == 0:
                return int(sample.y.max().item()) + 1
            else:
                return sample.y.shape[-1]
        return 1  # Default for regression


class AstroLightningNodeData(LightningNodeData):
    """
    Lightning wrapper for node-level tasks with enhanced sampling.

    Handles single large graphs with node-level predictions using
    efficient neighbor sampling and spatial splits.
    """

    def __init__(
        self,
        data: Data,
        # DataLoader parameters
        batch_size: int = 128,
        num_workers: int = 4,
        # Neighbor sampling
        num_neighbors: List[int] = [25, 10],
        sampling_strategy: str = "neighbor",  # "neighbor", "cluster", "saint"
        # Split parameters
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        split_method: str = "spatial",  # "random" or "spatial"
        **kwargs,
    ):
        """Initialize enhanced node-level Lightning wrapper."""
        # Create train/val/test masks
        num_nodes = data.num_nodes

        if split_method == "spatial":
            # Use spatial clustering for splits
            masks = self._create_spatial_splits(data, train_ratio, val_ratio)
        else:
            # Random splits
            masks = self._create_random_splits(num_nodes, train_ratio, val_ratio)

        data.train_mask = masks["train"]
        data.val_mask = masks["val"]
        data.test_mask = masks["test"]

        # Store sampling strategy
        self.sampling_strategy = sampling_strategy

        # Initialize parent class
        super().__init__(
            data=data,
            batch_size=batch_size,
            num_workers=num_workers,
            num_neighbors=num_neighbors,
            **kwargs,
        )

        logger.info(f"Created node-level data with {num_nodes} nodes")
        logger.info(
            f"Train: {masks['train'].sum()}, Val: {masks['val'].sum()}, "
            f"Test: {masks['test'].sum()}"
        )
        logger.info(f"Sampling strategy: {sampling_strategy}")

    def _create_random_splits(
        self, num_nodes: int, train_ratio: float, val_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Create random train/val/test splits."""
        indices = torch.randperm(num_nodes)

        n_train = int(num_nodes * train_ratio)
        n_val = int(num_nodes * val_ratio)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train : n_train + n_val]] = True
        test_mask[indices[n_train + n_val :]] = True

        return {"train": train_mask, "val": val_mask, "test": test_mask}

    def _create_spatial_splits(
        self, data: Data, train_ratio: float, val_ratio: float
    ) -> Dict[str, torch.Tensor]:
        """Create spatially-aware splits using PyG clustering."""
        # Get positions
        if hasattr(data, "pos"):
            pos = data.pos
        elif hasattr(data, "x") and data.x.shape[1] >= 3:
            pos = data.x[:, :3]
        else:
            # Fallback to random if no positions
            return self._create_random_splits(data.num_nodes, train_ratio, val_ratio)

        # Use farthest point sampling for diverse cluster centers
        n_clusters = min(100, data.num_nodes // 50)  # Adaptive cluster count
        cluster_centers = fps(pos, ratio=n_clusters / data.num_nodes)

        # Build k-NN graph for clustering
        edge_index = knn_graph(pos, k=5, flow="target_to_source")

        # Simple spatial clustering using nearest cluster center
        distances = torch.cdist(pos, pos[cluster_centers])
        cluster_labels = distances.argmin(dim=1)

        # Assign clusters to train/val/test
        unique_clusters = torch.unique(cluster_labels)
        n_train_clusters = int(len(unique_clusters) * train_ratio)
        n_val_clusters = int(len(unique_clusters) * val_ratio)

        # Shuffle clusters
        perm = torch.randperm(len(unique_clusters))
        train_clusters = unique_clusters[perm[:n_train_clusters]]
        val_clusters = unique_clusters[
            perm[n_train_clusters : n_train_clusters + n_val_clusters]
        ]
        test_clusters = unique_clusters[perm[n_train_clusters + n_val_clusters :]]

        # Create masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

        for c in train_clusters:
            train_mask[cluster_labels == c] = True
        for c in val_clusters:
            val_mask[cluster_labels == c] = True
        for c in test_clusters:
            test_mask[cluster_labels == c] = True

        return {"train": train_mask, "val": val_mask, "test": test_mask}


def create_pyg_kmeans_clustering(
    positions: torch.Tensor, n_clusters: int
) -> torch.Tensor:
    """
    K-means clustering using PyTorch operations only.

    Args:
        positions: Node positions [N, D]
        n_clusters: Number of clusters

    Returns:
        Cluster labels [N]
    """
    device = positions.device
    n_points = positions.shape[0]

    # Initialize cluster centers using FPS
    center_indices = fps(positions, ratio=n_clusters / n_points)
    centers = positions[center_indices]

    # K-means iterations
    max_iters = 100
    for _ in range(max_iters):
        # Assign points to nearest center
        distances = torch.cdist(positions, centers)
        labels = distances.argmin(dim=1)

        # Update centers
        new_centers = torch.zeros_like(centers)
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                new_centers[k] = positions[mask].mean(dim=0)
            else:
                # Keep old center if no points assigned
                new_centers[k] = centers[k]

        # Check convergence
        if torch.allclose(centers, new_centers, rtol=1e-5):
            break

        centers = new_centers

    return labels


def create_lightning_datamodule(
    survey: str,
    task: str = "graph",  # "graph", "node", or "link"
    dataset_type: str = "auto",  # "graph", "point_cloud", or "auto"
    **kwargs,
) -> Union[AstroLightningDataset, AstroLightningNodeData]:
    """
    Factory function to create appropriate Lightning data module.

    Args:
        survey: Survey name (e.g., 'gaia', 'sdss')
        task: Task type ("graph", "node", or "link")
        dataset_type: Dataset type or "auto" for automatic selection
        **kwargs: Additional parameters for the data module

    Returns:
        Lightning data module instance
    """
    # Auto-detect dataset type based on task
    if dataset_type == "auto":
        if task == "node":
            # For node tasks, use single graph
            dataset_type = "graph"
        else:
            # For graph tasks, use point cloud for better batching
            dataset_type = "point_cloud"

    # Create appropriate data module
    if task == "node":
        # Load single graph dataset
        dataset = SurveyGraphDataset(
            root=kwargs.get("data_root", str(get_data_config().base_dir)),
            survey=survey,
            k_neighbors=kwargs.get("k_neighbors", 20),
            force_reload=kwargs.get("force_reload", False),
        )

        # Get the single graph
        data = dataset[0]

        # Create node-level Lightning wrapper
        return AstroLightningNodeData(
            data=data,
            batch_size=kwargs.get("batch_size", 128),
            num_workers=kwargs.get("num_workers", 4),
            num_neighbors=kwargs.get("neighbor_sizes", [25, 10]),
            sampling_strategy=kwargs.get("sampling_strategy", "neighbor"),
            train_ratio=kwargs.get("train_ratio", 0.6),
            val_ratio=kwargs.get("val_ratio", 0.2),
            split_method=kwargs.get("split_method", "spatial"),
        )
    else:
        # Create dataset-level Lightning wrapper
        return AstroLightningDataset(
            survey=survey,
            task=task,
            dataset_type=dataset_type,
            batch_size=kwargs.get("batch_size", 32),
            num_workers=kwargs.get("num_workers", 4),
            pin_memory=kwargs.get("pin_memory", True),
            persistent_workers=kwargs.get("persistent_workers", True),
            prefetch_factor=kwargs.get("prefetch_factor", 2),
            k_neighbors=kwargs.get("k_neighbors", 20),
            sampling_strategy=kwargs.get("sampling_strategy", "none"),
            neighbor_sizes=kwargs.get("neighbor_sizes", [25, 10]),
            num_clusters=kwargs.get("num_clusters", 1500),
            saint_sample_coverage=kwargs.get("saint_sample_coverage", 50),
            saint_walk_length=kwargs.get("saint_walk_length", 2),
            enable_dynamic_batching=kwargs.get("enable_dynamic_batching", False),
            min_batch_size=kwargs.get("min_batch_size", 1),
            max_batch_size=kwargs.get("max_batch_size", 512),
            max_samples=kwargs.get("max_samples"),
            train_ratio=kwargs.get("train_ratio", 0.8),
            val_ratio=kwargs.get("val_ratio", 0.1),
            num_subgraphs=kwargs.get("num_subgraphs", 1000),
            points_per_subgraph=kwargs.get("points_per_subgraph", 500),
            data_root=kwargs.get("data_root"),
            force_reload=kwargs.get("force_reload", False),
            partition_method=kwargs.get("partition_method"),
            num_partitions=kwargs.get("num_partitions", 4),
        )
