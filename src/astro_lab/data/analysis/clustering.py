"""
Spatial Clustering for Astronomical Data Analysis
================================================

Efficient clustering algorithms using PyTorch Geometric and TensorDict integration.
"""

import logging
from typing import Dict, List, Optional, Union

import torch
from torch_geometric.nn import radius_graph
from torch_geometric.utils import to_undirected

from astro_lab.models.autoencoders.base import BaseAutoencoder
from astro_lab.models.autoencoders.pointcloud_autoencoder import PointCloudAutoencoder
from astro_lab.tensors import SpatialTensorDict

logger = logging.getLogger(__name__)


class SpatialClustering:
    """
    Efficient spatial clustering using PyTorch Geometric operations.

    Features:
    - Multi-scale clustering analysis
    - Friends-of-Friends algorithm
    - DBSCAN implementation
    - TensorDict integration
    - GPU acceleration
    """

    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        random_state: int = 42,
    ):
        """Initialize spatial clustering."""
        self.device = device
        self.random_state = random_state

        if random_state is not None:
            torch.manual_seed(random_state)

        logger.info(f"🔗 SpatialClustering initialized on {self.device}")

    def multi_scale_analysis(
        self,
        coordinates: Union[torch.Tensor, SpatialTensorDict],
        scales: List[float],
        min_samples: int = 5,
    ) -> Dict[str, Dict]:
        """
        Perform multi-scale clustering analysis.

        Args:
            coordinates: Coordinates tensor [N, 3] or SpatialTensorDict
            scales: Clustering scales in appropriate units
            min_samples: Minimum cluster size

        Returns:
            Dictionary with clustering results per scale
        """
        # Handle TensorDict input
        if isinstance(coordinates, SpatialTensorDict):
            coords = coordinates.coordinates
        else:
            coords = coordinates

        coords = coords.to(self.device)

        results = {}
        n_total = coords.size(0)

        logger.debug(f"🔗 Multi-scale clustering: {len(scales)} scales")

        for scale in scales:
            # Use Friends-of-Friends clustering
            labels = self.friends_of_friends(
                coords, linking_length=scale, min_group_size=min_samples
            )

            # Analyze results
            analysis = self._analyze_clustering(labels, n_total)

            results[f"{scale:.1f}"] = {"scale": scale, "labels": labels, **analysis}

            logger.debug(f"  {scale:.1f}: {analysis['n_clusters']} clusters")

        return results

    def friends_of_friends(
        self, coordinates: torch.Tensor, linking_length: float, min_group_size: int = 2
    ) -> torch.Tensor:
        """
        Friends-of-Friends clustering using PyG radius_graph.

        Args:
            coordinates: Object coordinates [N, 3]
            linking_length: Maximum linking distance
            min_group_size: Minimum group size

        Returns:
            Cluster labels (noise points labeled as -1)
        """
        device = coordinates.device
        n_points = coordinates.size(0)

        # Handle edge case
        if n_points == 0:
            return torch.empty(0, dtype=torch.long, device=device)

        # Use PyG radius_graph
        edge_index = radius_graph(
            coordinates,
            r=linking_length,
            batch=None,
            loop=False,
            max_num_neighbors=n_points,
        )

        # Make undirected
        edge_index = to_undirected(edge_index, num_nodes=n_points)

        # Handle no edges case
        if edge_index.size(1) == 0:
            return torch.full((n_points,), -1, dtype=torch.long, device=device)

        # Find connected components
        labels = self._connected_components_torch(edge_index, n_points)

        # Filter by minimum group size
        unique_labels, counts = torch.unique(labels, return_counts=True)
        small_clusters = unique_labels[counts < min_group_size]

        # Mark small clusters as noise
        for cluster_id in small_clusters:
            labels[labels == cluster_id] = -1

        # Relabel remaining clusters consecutively
        valid_clusters = torch.unique(labels[labels >= 0])

        # Handle case where no valid clusters remain
        if len(valid_clusters) == 0:
            return torch.full((n_points,), -1, dtype=torch.long, device=device)

        # Create label mapping only for valid clusters
        max_original_label = labels.max().item()
        if max_original_label >= 0:
            label_map = torch.full(
                (max_original_label + 1,), -1, dtype=torch.long, device=device
            )
            for i, cluster_id in enumerate(valid_clusters):
                label_map[cluster_id] = i

            # Apply relabeling only to non-noise points
            mask = labels >= 0
            labels[mask] = label_map[labels[mask]]

        return labels

    def dbscan_torch(
        self, coordinates: torch.Tensor, eps: float, min_samples: int = 5
    ) -> torch.Tensor:
        """
        DBSCAN clustering using PyTorch operations.

        Args:
            coordinates: Points to cluster [N, 3]
            eps: Maximum distance between points
            min_samples: Minimum points for core point

        Returns:
            Cluster labels (-1 for noise)
        """
        n_points = coordinates.size(0)
        device = coordinates.device

        # Build radius graph
        edge_index = radius_graph(
            coordinates, r=eps, loop=True, max_num_neighbors=n_points
        )

        # Count neighbors for each point
        neighbor_counts = torch.zeros(n_points, device=device)
        src = edge_index[0]
        neighbor_counts.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))

        # Identify core points
        core_points = neighbor_counts >= min_samples

        # Initialize labels
        labels = torch.full((n_points,), -1, dtype=torch.long, device=device)
        cluster_id = 0

        # Process each core point
        for i in range(n_points):
            if core_points[i] and labels[i] == -1:
                # Start new cluster
                labels[i] = cluster_id

                # Get neighbors to process
                mask = edge_index[0] == i
                to_process = edge_index[1, mask].unique()

                # Expand cluster
                processed = torch.zeros(n_points, dtype=torch.bool, device=device)
                processed[i] = True

                while len(to_process) > 0:
                    # Process next point
                    current = to_process[0]
                    to_process = to_process[1:]

                    if labels[current] == -1:
                        labels[current] = cluster_id

                        if core_points[current] and not processed[current]:
                            # Add neighbors of core point
                            mask = edge_index[0] == current
                            neighbors = edge_index[1, mask]
                            new_points = neighbors[labels[neighbors] == -1]
                            to_process = torch.cat([to_process, new_points]).unique()

                    processed[current] = True

                cluster_id += 1

        return labels

    def _connected_components_torch(
        self, edge_index: torch.Tensor, n_nodes: int
    ) -> torch.Tensor:
        """Find connected components using pure PyTorch operations."""
        device = edge_index.device

        # Initialize each node as its own component
        labels = torch.arange(n_nodes, device=device)

        # Iteratively merge components
        changed = True
        while changed:
            changed = False

            # Get minimum label among neighbors
            src, dst = edge_index
            neighbor_labels = labels[dst]

            # Scatter min to update labels
            new_labels = labels.clone()
            new_labels.scatter_reduce_(
                0, src, neighbor_labels, reduce="amin", include_self=True
            )

            # Check if any labels changed
            if not torch.equal(labels, new_labels):
                changed = True
                labels = new_labels

        # Relabel consecutively
        unique_labels = torch.unique(labels)
        label_map = torch.zeros(n_nodes, dtype=torch.long, device=device)
        for i, label in enumerate(unique_labels):
            label_map[labels == label] = i

        return label_map

    def _analyze_clustering(self, labels: torch.Tensor, n_total: int) -> Dict:
        """Analyze clustering results."""
        # Count clusters and noise
        unique_labels, counts = torch.unique(labels, return_counts=True)

        cluster_mask = unique_labels >= 0
        n_clusters = cluster_mask.sum().item()
        n_noise = (labels == -1).sum().item()
        n_grouped = n_total - n_noise

        # Cluster statistics
        if n_clusters > 0:
            cluster_sizes = counts[cluster_mask]
            largest_cluster = cluster_sizes.max().item()
            mean_cluster_size = cluster_sizes.float().mean().item()
        else:
            largest_cluster = 0
            mean_cluster_size = 0.0

        return {
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "n_grouped": n_grouped,
            "grouped_fraction": n_grouped / n_total if n_total > 0 else 0.0,
            "largest_cluster_size": largest_cluster,
            "mean_cluster_size": mean_cluster_size,
            "cluster_sizes": counts[cluster_mask].tolist() if n_clusters > 0 else [],
        }


def analyze_with_autoencoder(
    coordinates: Union[torch.Tensor, SpatialTensorDict],
    autoencoder: Optional[BaseAutoencoder] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Analyze spatial data using autoencoder for dimensionality reduction.

    Args:
        coordinates: Input coordinates
        autoencoder: Pre-trained autoencoder (optional)
        device: Computation device

    Returns:
        Analysis results with latent representations
    """
    if autoencoder is None:
        # Create default autoencoder
        autoencoder = PointCloudAutoencoder(
            input_dim=3, latent_dim=16, hidden_dim=64, use_geometric=True
        ).to(device)

    # Get latent representations
    latent_repr = autoencoder.encode(coordinates)

    # Perform clustering on latent space
    clustering = SpatialClustering(device=device)
    clustering_results = clustering.multi_scale_analysis(
        latent_repr, scales=[0.5, 1.0, 2.0], min_samples=3
    )

    return {
        "latent_representations": latent_repr,
        "clustering_results": clustering_results,
        "autoencoder": autoencoder,
    }
