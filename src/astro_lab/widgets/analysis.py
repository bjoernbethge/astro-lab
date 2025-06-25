"""
Analysis Module - GPU-accelerated data analysis
==============================================

Provides GPU-accelerated analysis methods using PyTorch Geometric
and torch_cluster with simple CPU fallbacks.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch_cluster
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from torch_geometric.nn import radius_graph

from ..tensors import SurveyTensorDict

logger = logging.getLogger(__name__)


class AnalysisModule:
    """
    GPU-accelerated data analysis.
    """

    def find_neighbors(
        self,
        survey_tensor: SurveyTensorDict,
        k: int = 10,
        radius: Optional[float] = None,
        use_gpu: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        GPU-accelerated neighbor finding using torch_cluster and torch_geometric.

        Args:
            survey_tensor: SurveyTensor with spatial data
            k: Number of nearest neighbors (if radius is None)
            radius: Radius for neighbor search (if provided, overrides k)
            use_gpu: Whether to use GPU acceleration

        Returns:
            Dictionary with 'edge_index' and 'distances'
        """
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data

        device = torch.device(
            "cuda" if torch.cuda.is_available() and use_gpu else "cpu"
        )
        coords_device = coords.to(device)

        logger.info(f"Finding neighbors on {device} for {len(coords)} points...")

        if radius is not None:
            # Radius-based search
            edge_index = radius_graph(
                x=coords_device, r=radius, loop=False, flow="source_to_target"
            )
        else:
            # k-NN search
            edge_index = torch_cluster.knn_graph(
                x=coords_device, k=k, loop=False, flow="source_to_target"
            )

        # Calculate distances
        distances = torch.norm(
            coords_device[edge_index[0]] - coords_device[edge_index[1]], dim=1
        )

        return {"edge_index": edge_index.cpu(), "distances": distances.cpu()}

    def cluster_data(
        self,
        survey_tensor: SurveyTensorDict,
        eps: float = 10.0,
        min_samples: int = 5,
        algorithm: str = "dbscan",
    ) -> Dict[str, Any]:
        """
        GPU-accelerated clustering analysis.

        Args:
            survey_tensor: SurveyTensor with spatial data
            eps: Clustering radius
            min_samples: Minimum samples for core points
            algorithm: 'dbscan' or 'hierarchical'

        Returns:
            Dictionary with cluster labels, statistics, and analysis results
        """
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data.cpu().numpy()

        logger.info(f"Clustering {len(coords)} points with {algorithm}...")

        if algorithm == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            from sklearn.cluster import AgglomerativeClustering

            clusterer = AgglomerativeClustering(distance_threshold=eps, linkage="ward")

        labels = clusterer.fit_predict(coords)
        labels_tensor = torch.from_numpy(labels)

        # Analyze results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = sum(1 for label in labels if label == -1)

        # Calculate cluster properties
        cluster_stats = {}
        if n_clusters > 0:
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_mask = labels == cluster_id
                cluster_coords = coords[cluster_mask]

                # Cluster center and size
                center = cluster_coords.mean(axis=0)
                distances = np.linalg.norm(cluster_coords - center, axis=1)

                cluster_stats[cluster_id] = {
                    "n_points": int(cluster_mask.sum()),
                    "center": center,
                    "radius": float(distances.max()),
                    "density": float(
                        cluster_mask.sum()
                        / (4 / 3 * np.pi * max(distances.max(), 1e-6) ** 3)
                    ),
                }

        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

        return {
            "cluster_labels": labels_tensor,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_stats": cluster_stats,
            "coords": torch.from_numpy(coords),
        }

    def analyze_density(
        self, survey_tensor: SurveyTensorDict, radius: float = 5.0, use_gpu: bool = True
    ) -> torch.Tensor:
        """
        GPU-accelerated local density analysis.

        Args:
            survey_tensor: SurveyTensor with spatial data
            radius: Radius for density calculation
            use_gpu: Whether to use GPU acceleration

        Returns:
            Local density for each point
        """
        if not use_gpu:
            logger.info("Using CPU density analysis...")
            return self._analyze_density_cpu(survey_tensor, radius)

        logger.info("Using GPU-accelerated density analysis...")
        return self._analyze_density_gpu(survey_tensor, radius)

    def analyze_structure(
        self, survey_tensor: SurveyTensorDict, k: int = 10, use_gpu: bool = True
    ) -> Dict[str, Any]:
        """
        GPU-accelerated structure analysis.

        Args:
            survey_tensor: SurveyTensor with spatial data
            k: Number of neighbors for analysis
            use_gpu: Whether to use GPU acceleration

        Returns:
            Dictionary with structure analysis results
        """
        if not use_gpu:
            logger.info("Using CPU structure analysis...")
            return self._analyze_structure_cpu(survey_tensor, k)

        logger.info("Using GPU-accelerated structure analysis...")
        return self._analyze_structure_gpu(survey_tensor, k)

    def _analyze_density_gpu(
        self, survey_tensor: SurveyTensorDict, radius: float
    ) -> torch.Tensor:
        """GPU-accelerated density analysis using PyTorch Geometric."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords_device = coords.to(device)

        logger.info(f"GPU density analysis on {device}...")

        # Create radius graph using PyTorch Geometric
        edge_index = radius_graph(
            x=coords_device, r=radius, loop=False, flow="source_to_target"
        )

        # Count neighbors for each point
        densities = []
        for i in range(len(coords)):
            n_neighbors = (edge_index[0] == i).sum().item()
            volume = (4 / 3) * np.pi * radius**3
            density = n_neighbors / volume
            densities.append(density)

        return torch.tensor(densities, dtype=torch.float32)

    def _analyze_density_cpu(
        self, survey_tensor: SurveyTensorDict, radius: float
    ) -> torch.Tensor:
        """CPU fallback for density analysis."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data.cpu().numpy()

        tree = BallTree(coords)
        densities = []

        for i in range(len(coords)):
            neighbors = tree.query_radius([coords[i]], r=radius)[0]
            n_neighbors = len(neighbors) - 1  # Exclude self
            volume = (4 / 3) * np.pi * radius**3
            density = n_neighbors / volume
            densities.append(density)

        return torch.tensor(densities, dtype=torch.float32)

    def _analyze_structure_gpu(
        self, survey_tensor: SurveyTensorDict, k: int
    ) -> Dict[str, Any]:
        """GPU-accelerated structure analysis."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords_device = coords.to(device)

        logger.info(f"GPU structure analysis on {device}...")

        # Create k-NN graph using torch_cluster
        edge_index = torch_cluster.knn_graph(
            x=coords_device, k=k, loop=False, flow="source_to_target"
        )

        # Analyze structure
        num_nodes = len(coords)
        num_edges = edge_index.shape[1]

        # Calculate degrees
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)

        # Calculate average degree
        avg_degree = degrees.float().mean().item()

        # Calculate graph density
        max_edges = num_nodes * (num_nodes - 1) / 2
        graph_density = num_edges / max_edges if max_edges > 0 else 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "graph_density": graph_density,
            "degrees": degrees.cpu(),
        }

    def _analyze_structure_cpu(
        self, survey_tensor: SurveyTensorDict, k: int
    ) -> Dict[str, Any]:
        """CPU fallback for structure analysis."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data.cpu().numpy()

        tree = BallTree(coords)
        distances, indices = tree.query(coords, k=k + 1)  # +1 to exclude self

        # Create edge list
        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip first (self)
                edge_list.append([i, j])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        # Analyze structure
        num_nodes = len(coords)
        num_edges = edge_index.shape[1]

        # Calculate degrees
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)

        # Calculate average degree
        avg_degree = degrees.float().mean().item()

        # Calculate graph density
        max_edges = num_nodes * (num_nodes - 1) / 2
        graph_density = num_edges / max_edges if max_edges > 0 else 0

        return {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "avg_degree": avg_degree,
            "graph_density": graph_density,
            "degrees": degrees,
        }
