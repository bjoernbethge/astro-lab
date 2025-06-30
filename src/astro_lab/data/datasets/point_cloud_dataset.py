"""
Point Cloud Dataset for Astronomical Data
========================================

Optimized InMemoryDataset that creates multiple point cloud subgraphs
for efficient batching and training with point cloud models.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch
import torch_cluster
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn.pool import fps  # Farthest point sampling
from torch_geometric.transforms import KNNGraph

from astro_lab.config import get_data_config, get_survey_config
from astro_lab.data.preprocessors import get_preprocessor
from astro_lab.memory import clear_cuda_cache

logger = logging.getLogger(__name__)


class AstroPointCloudDataset(InMemoryDataset):
    """
    PyTorch Geometric InMemoryDataset that creates multiple point cloud subgraphs
    for astronomical survey data.

    All data is loaded into memory for maximum performance.
    """

    def __init__(
        self,
        root: str,
        survey: str,
        k_neighbors: Optional[int] = None,
        num_subgraphs: int = 1000,
        points_per_subgraph: int = 500,
        overlap_ratio: float = 0.1,
        use_3d_coordinates: bool = True,
        transform: Optional[Any] = None,
        pre_transform: Optional[Any] = None,
        pre_filter: Optional[Any] = None,
        force_reload: bool = False,
        **kwargs,
    ):
        """
        Initialize the in-memory point cloud dataset.

        Args:
            root: Root directory for the dataset
            survey: Survey name (e.g., 'gaia', 'sdss')
            k_neighbors: Number of neighbors for kNN graph
            num_subgraphs: Number of subgraphs to create
            points_per_subgraph: Approximate number of points per subgraph
            overlap_ratio: Ratio of overlapping points between subgraphs
            use_3d_coordinates: Whether to use 3D coordinates
            transform: Optional transform to apply
            pre_transform: Optional pre-transform
            pre_filter: Optional pre-filter
            force_reload: Force reprocessing of data
        """
        self.use_3d_coordinates = use_3d_coordinates
        self.force_reload = force_reload
        self.survey = survey
        self.num_subgraphs = num_subgraphs
        self.points_per_subgraph = points_per_subgraph
        self.overlap_ratio = overlap_ratio

        # Get configurations
        self.survey_config = get_survey_config(survey)
        self.data_config = get_data_config()

        # Set k_neighbors with defaults
        self.k_neighbors = k_neighbors or self.survey_config.get("k_neighbors", 20)

        # Initialize base class - InMemoryDataset loads data into self.data
        super().__init__(root, transform, pre_transform, pre_filter)

        # Load data into memory
        self._load_data_list()

    @property
    def raw_file_names(self) -> List[str]:
        """Raw file names."""
        return [f"{self.survey}_raw.parquet"]

    @property
    def processed_file_names(self) -> List[str]:
        """Processed file names."""
        base = f"{self.survey}_pc_{self.num_subgraphs}_{self.points_per_subgraph}"
        if self.use_3d_coordinates:
            base += "_3d"
        return [f"{base}_data.pt"]  # Single file for InMemoryDataset

    def download(self):
        """Download raw data if needed."""
        raw_path = Path(self.raw_dir) / self.raw_file_names[0]
        if not raw_path.exists():
            logger.info(f"Downloading {self.survey} data...")
            preprocessor = get_preprocessor(self.survey)
            df = preprocessor.load_data()
            df.write_parquet(raw_path)
            logger.info(f"Saved to {raw_path}")

    def process(self):
        """Process raw data into multiple point cloud subgraphs and load into memory."""
        # Check if already processed
        processed_path = Path(self.processed_dir) / self.processed_file_names[0]
        if not self.force_reload and processed_path.exists():
            logger.info(f"Loading cached processed data from {processed_path}")
            self.data = torch.load(processed_path)
            return

        logger.info(
            f"Processing {self.survey} data into {self.num_subgraphs} point clouds..."
        )

        # Load data via preprocessor
        preprocessor = get_preprocessor(self.survey)

        # Try to load pre-processed parquet if available
        processed_parquet = (
            Path(self.processed_dir).parent / f"{self.survey}_processed.parquet"
        )
        if processed_parquet.exists() and not self.force_reload:
            logger.info(f"Loading pre-processed data from {processed_parquet}")
            df = pl.read_parquet(processed_parquet)
        else:
            df = preprocessor.load_data()

            # Limit data size for initial processing
            max_sources = 1_000_000  # Process at most 1M sources
            if len(df) > max_sources:
                logger.warning(
                    f"Large dataset detected ({len(df):,} sources). Sampling {max_sources:,} for processing."
                )
                df = df.sample(n=max_sources, seed=42)

            # Save processed data for future use
            df.write_parquet(processed_parquet)
            logger.info(f"Saved processed data to {processed_parquet}")

        # Convert to TensorDict with GPU support
        tensor_dict = preprocessor.create_tensordict(
            df, use_gpu=torch.cuda.is_available()
        )

        # Extract all data
        all_features, all_positions, all_labels = self._extract_data(tensor_dict)

        # Move to CPU for partitioning if on GPU
        if all_positions.is_cuda:
            all_positions_cpu = all_positions.cpu()
            all_features_cpu = (
                all_features.cpu() if all_features.is_cuda else all_features
            )
            all_labels_cpu = (
                all_labels.cpu()
                if all_labels is not None and all_labels.is_cuda
                else all_labels
            )
        else:
            all_positions_cpu = all_positions
            all_features_cpu = all_features
            all_labels_cpu = all_labels

        # Create spatial partitions using PyG clustering
        partitions = self._create_spatial_partitions_pyg(all_positions_cpu)

        # Create all point cloud graphs in memory
        data_list = []
        for i in range(min(len(partitions), self.num_subgraphs)):
            partition_indices = partitions[i]

            # Extract partition data
            features = all_features_cpu[partition_indices]
            positions = all_positions_cpu[partition_indices]
            labels = (
                all_labels_cpu[partition_indices]
                if all_labels_cpu is not None
                else None
            )

            # Create point cloud graph
            graph = self._create_point_cloud_graph(features, positions, labels)
            data_list.append(graph)

            if (i + 1) % 100 == 0:
                logger.info(
                    f"Created {i + 1}/{min(len(partitions), self.num_subgraphs)} point clouds"
                )

        # Load into memory (this is what InMemoryDataset does)
        self.data = data_list

        # Save to disk for future use
        torch.save(self.data, processed_path)
        logger.info(f"Saved {len(self.data)} point cloud subgraphs to memory and disk")

        # Cleanup
        clear_cuda_cache()

    def _load_data_list(self):
        """Load data list into memory."""
        # InMemoryDataset handles this automatically in process()
        pass

    def _extract_data(
        self, tensor_dict
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Extract features, positions, and labels from tensor dict."""
        # Extract features
        if "features" in tensor_dict:
            features = tensor_dict["features"]
        elif "x" in tensor_dict:
            features = tensor_dict["x"]
        else:
            features = None

        # Extract coordinates
        if "spatial" in tensor_dict:
            positions = tensor_dict["spatial"].get(
                "coordinates", tensor_dict["spatial"].get("pos", None)
            )
        elif features is not None and features.shape[1] >= 3:
            positions = features[:, :3]
        else:
            raise ValueError("No coordinate data found")

        if positions is None:
            raise ValueError("No positions found")

        # If no features, use positions as features
        if features is None:
            features = positions.clone()

        # Extract labels
        if "labels" in tensor_dict:
            labels = tensor_dict["labels"]
        elif "y" in tensor_dict:
            labels = tensor_dict["y"]
        else:
            # Create synthetic labels using PyG k-means
            logger.info("Creating synthetic labels via PyG clustering...")
            labels = self._pyg_kmeans(positions, n_clusters=5)

        return features, positions, labels

    def _pyg_kmeans(
        self, positions: torch.Tensor, n_clusters: int, max_iters: int = 10
    ) -> torch.Tensor:
        """K-means clustering using PyTorch operations with GPU support - optimized version."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        positions = positions.to(device)
        n_points = positions.shape[0]

        # For very large datasets, reduce number of clusters
        if n_points > 100000:
            actual_clusters = min(n_clusters, 100)
            logger.info(
                f"Large dataset: using {actual_clusters} clusters instead of {n_clusters}"
            )
            n_clusters = actual_clusters

        # Initialize cluster centers using FPS (on GPU) - this is fast
        ratio = min(n_clusters / n_points, 1.0)
        batch = torch.zeros(n_points, dtype=torch.long, device=device)
        center_indices = fps(positions, batch, ratio=ratio)

        # If we got fewer centers than requested, sample randomly for the rest
        if len(center_indices) < n_clusters:
            remaining = n_clusters - len(center_indices)
            other_indices = torch.randperm(n_points, device=device)[:remaining]
            center_indices = torch.cat([center_indices, other_indices])

        centers = positions[center_indices[:n_clusters]]

        # K-means iterations (reduced for speed)
        for i in range(max_iters):
            # Batch process distances for memory efficiency
            batch_size = min(50000, n_points)
            labels = torch.zeros(n_points, dtype=torch.long, device=device)

            for start_idx in range(0, n_points, batch_size):
                end_idx = min(start_idx + batch_size, n_points)
                batch_positions = positions[start_idx:end_idx]

                # Compute distances for this batch
                distances = torch.cdist(batch_positions, centers)
                labels[start_idx:end_idx] = distances.argmin(dim=1)

            # Update centers
            new_centers = torch.zeros_like(centers)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = positions[mask].mean(dim=0)
                else:
                    # Keep old center if no points assigned
                    new_centers[k] = centers[k]

            # Check convergence (early stopping after 3 iterations)
            if i >= 3 and torch.allclose(centers, new_centers, rtol=1e-3):
                break

            centers = new_centers

        return labels.cpu()

    def _create_spatial_partitions_pyg(
        self, positions: torch.Tensor
    ) -> List[torch.Tensor]:
        """Create spatial partitions using PyG clustering with GPU support."""
        n_points = positions.shape[0]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move positions to GPU
        positions = positions.to(device)

        # Calculate number of clusters needed
        n_clusters = max(self.num_subgraphs, n_points // self.points_per_subgraph)
        n_clusters = min(n_clusters, n_points // 10)  # Ensure enough points per cluster

        # For large datasets, use a more reasonable number of clusters
        if n_clusters > 1000:
            logger.warning(
                f"Reducing clusters from {n_clusters} to 1000 for performance"
            )
            n_clusters = 1000

        logger.info(f"Creating {n_clusters} spatial clusters using PyG on {device}...")

        # Use grid-based partitioning for very large datasets
        if n_points > 500000:
            logger.info("Using grid-based partitioning for large dataset")
            return self._grid_based_partitions(positions, n_clusters)

        # Use PyG k-means clustering
        cluster_labels = self._pyg_kmeans(positions, n_clusters)

        partitions = []

        for i in range(n_clusters):
            # Get points in this cluster
            cluster_mask = cluster_labels == i
            cluster_indices = torch.where(cluster_mask)[0]

            if len(cluster_indices) == 0:
                continue

            # Add overlap with nearby clusters if requested
            if self.overlap_ratio > 0 and len(cluster_indices) > 10:
                cluster_center = positions[cluster_indices].mean(dim=0)

                # Find distances to all points (batch process for memory efficiency)
                batch_size = min(100000, n_points)
                distances = torch.zeros(n_points, device=device)

                for start_idx in range(0, n_points, batch_size):
                    end_idx = min(start_idx + batch_size, n_points)
                    batch_positions = positions[start_idx:end_idx]
                    distances[start_idx:end_idx] = torch.norm(
                        batch_positions - cluster_center, dim=1
                    )

                # Add nearest points as overlap
                n_overlap = int(len(cluster_indices) * self.overlap_ratio)
                k = min(len(cluster_indices) + n_overlap, n_points)
                _, nearest_indices = torch.topk(distances, k=k, largest=False)

                # Combine cluster and overlap indices
                all_indices = torch.unique(
                    torch.cat([cluster_indices, nearest_indices[:k]])
                )
                partitions.append(all_indices.cpu())
            else:
                partitions.append(cluster_indices.cpu())

        # Ensure minimum points per partition
        min_points = max(50, self.k_neighbors * 2)
        partitions = [p for p in partitions if len(p) >= min_points]

        # Shuffle partitions
        perm = torch.randperm(len(partitions))
        partitions = [partitions[i] for i in perm]

        logger.info(f"Created {len(partitions)} partitions")

        return partitions[: self.num_subgraphs]

    def _grid_based_partitions(
        self, positions: torch.Tensor, n_partitions: int
    ) -> List[torch.Tensor]:
        """Create grid-based partitions for very large datasets."""
        device = positions.device
        n_points = positions.shape[0]

        # Compute bounding box
        min_coords = positions.min(dim=0)[0]
        max_coords = positions.max(dim=0)[0]

        # Calculate grid dimensions - use more cells to ensure we get enough partitions
        grid_dim = int(np.ceil(n_partitions ** (1 / 3))) * 2  # Double for more cells

        # Create grid cells
        x_edges = torch.linspace(
            min_coords[0], max_coords[0], grid_dim + 1, device=device
        )
        y_edges = torch.linspace(
            min_coords[1], max_coords[1], grid_dim + 1, device=device
        )
        z_edges = torch.linspace(
            min_coords[2], max_coords[2], grid_dim + 1, device=device
        )

        partitions = []
        min_points_per_cell = max(50, self.k_neighbors * 2)

        # Collect all non-empty cells first
        all_cells = []
        for i in range(grid_dim):
            for j in range(grid_dim):
                for k in range(grid_dim):
                    # Find points in this grid cell
                    mask = (
                        (positions[:, 0] >= x_edges[i])
                        & (positions[:, 0] < x_edges[i + 1])
                        & (positions[:, 1] >= y_edges[j])
                        & (positions[:, 1] < y_edges[j + 1])
                        & (positions[:, 2] >= z_edges[k])
                        & (positions[:, 2] < z_edges[k + 1])
                    )

                    indices = torch.where(mask)[0]

                    if len(indices) >= min_points_per_cell:
                        all_cells.append(indices.cpu())

        # If we have too few cells, use random sampling fallback
        if len(all_cells) < n_partitions:
            logger.info(
                f"Grid produced only {len(all_cells)} cells, using random sampling for remaining"
            )

            # Use the grid cells we have
            partitions = all_cells

            # Add random samples for the rest
            all_indices = torch.arange(n_points, device=device)
            remaining_needed = n_partitions - len(partitions)

            for _ in range(remaining_needed):
                # Random center point
                center_idx = torch.randint(0, n_points, (1,), device=device).item()
                center = positions[center_idx]

                # Find nearest points
                distances = torch.norm(positions - center, dim=1)
                _, nearest = torch.topk(
                    distances, k=min(self.points_per_subgraph, n_points), largest=False
                )

                partitions.append(nearest.cpu())
        else:
            # Randomly select required number of cells
            selected_indices = torch.randperm(len(all_cells))[:n_partitions]
            partitions = [all_cells[i] for i in selected_indices]

        return partitions

    def _create_point_cloud_graph(
        self,
        features: torch.Tensor,
        positions: torch.Tensor,
        labels: Optional[torch.Tensor],
    ) -> Data:
        """Create a point cloud graph from features and positions."""
        # Create PyG Data object
        graph = Data(x=features, pos=positions)

        if labels is not None:
            graph.y = labels

        # Add k-NN edges for graph structure
        knn_transform = KNNGraph(k=self.k_neighbors)
        graph = knn_transform(graph)

        # Add metadata
        graph.num_nodes = features.size(0)
        graph.num_edges = graph.edge_index.size(1)

        return graph

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        # Sample first subgraph for statistics
        if len(self.data) > 0:
            sample_data = self.data[0]
            avg_nodes = sample_data.num_nodes
            avg_edges = sample_data.num_edges
            num_features = sample_data.x.size(1)
            num_classes = (
                int(sample_data.y.max().item() + 1) if hasattr(sample_data, "y") else 0
            )
        else:
            avg_nodes = avg_edges = num_features = num_classes = 0

        return {
            "survey": self.survey,
            "num_subgraphs": len(self.data),
            "k_neighbors": self.k_neighbors,
            "points_per_subgraph": self.points_per_subgraph,
            "overlap_ratio": self.overlap_ratio,
            "use_3d_coordinates": self.use_3d_coordinates,
            "avg_nodes_per_subgraph": avg_nodes,
            "avg_edges_per_subgraph": avg_edges,
            "num_features": num_features,
            "num_classes": num_classes,
            "survey_config": {
                "name": self.survey_config.get("name", self.survey),
                "type": self.survey_config.get("type", "catalog"),
                "coordinate_system": self.survey_config.get(
                    "coordinate_system", "icrs"
                ),
            },
        }
