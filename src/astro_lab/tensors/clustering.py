"""
Clustering Tensor for Astronomical Clustering
=============================================

Specialized tensor for astronomical clustering operations including
galaxy clusters, stellar associations, and large-scale structure.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.spatial.distance as distance
import torch
from scipy import stats
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans

# GPU-accelerated imports
import torch_cluster
from torch_geometric.nn import radius_graph

from .base import AstroTensorBase


class ClusteringTensor(AstroTensorBase):
    """
    Tensor for astronomical clustering operations.

    Provides specialized clustering algorithms for:
    - Galaxy cluster detection
    - Stellar association identification
    - Large-scale structure analysis
    - Friends-of-friends linking
    - Hierarchical clustering with astronomical metrics
    """

    _metadata_fields = [
        "coordinate_system",
        "clustering_algorithms",
        "cluster_labels",
        "cluster_properties",
        "distance_metrics",
        "clustering_parameters",
        "spatial_scales",
        "astronomical_context",
        "n_features",
    ]

    def __init__(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        features: Optional[Union[torch.Tensor, np.ndarray]] = None,
        coordinate_system: str = "cartesian",
        astronomical_context: str = "general",
        **kwargs,
    ):
        """
        Initialize clustering tensor.

        Args:
            positions: Position data [N, D]
            features: Optional feature data [N, F]
            coordinate_system: Coordinate system
            astronomical_context: Astronomical context
            **kwargs: Additional metadata
        """
        # Combine positions and features if available
        if features is not None:
            pos_tensor = torch.as_tensor(positions, dtype=torch.float32)
            feat_tensor = torch.as_tensor(features, dtype=torch.float32)
            
            # Validate that positions and features have the same number of objects
            if pos_tensor.shape[0] != feat_tensor.shape[0]:
                raise ValueError(f"Positions and features must have the same number of objects. Got {pos_tensor.shape[0]} positions and {feat_tensor.shape[0]} features.")
            
            data = torch.cat([pos_tensor, feat_tensor], dim=1)
        else:
            data = torch.as_tensor(positions, dtype=torch.float32)

        # Initialize base class
        super().__init__(data, **kwargs)

        # Set metadata
        n_features = feat_tensor.shape[1] if features is not None else 0
        self._metadata.update({
            "tensor_type": "clustering",
            "coordinate_system": coordinate_system,
            "astronomical_context": astronomical_context,
            "n_spatial_dims": positions.shape[1] if hasattr(positions, 'shape') else 3,
            "n_features": n_features,
            "clustering_algorithms": {},
            "cluster_labels": {},
            "cluster_properties": {},
        })

    @property
    def positions(self) -> torch.Tensor:
        """Position data."""
        if self.features is not None:
            return self._data[:, :3]  # Assume first 3 columns are positions
        return self._data

    @property
    def features(self) -> Optional[torch.Tensor]:
        """Feature data."""
        if self._data.shape[1] > 3:
            return self._data[:, 3:]
        return None

    def dbscan_clustering(
        self,
        eps: float = 1.0,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm_name: str = "dbscan",
    ) -> torch.Tensor:
        """
        Perform DBSCAN clustering with GPU-accelerated eps estimation.

        Args:
            eps: Clustering radius
            min_samples: Minimum samples for core points
            metric: Distance metric
            algorithm_name: Name for storing results

        Returns:
            Cluster labels (-1 for noise)
        """
        positions = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            positions = self._preprocess_spherical_coordinates(positions)

        # Auto-estimate eps if not provided
        if eps is None:
            eps = self._estimate_eps(min_samples)

        # Perform DBSCAN
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = clusterer.fit_predict(positions)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "n_clusters": len(np.unique(labels[labels >= 0])),
                "n_noise": (labels == -1).sum(),
            },
        )

        return labels_tensor

    def hdbscan_clustering(
        self,
        min_cluster_size: int = 5,
        min_samples: Optional[int] = None,
        algorithm_name: str = "hdbscan",
    ) -> torch.Tensor:
        """
        Perform HDBSCAN clustering for hierarchical density-based clustering.

        Args:
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples in neighborhood
            algorithm_name: Name for storing results

        Returns:
            Cluster labels (-1 for noise)
        """
        try:
            import hdbscan
        except ImportError:
            raise ImportError("hdbscan package required for HDBSCAN clustering")

        positions = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            positions = self._preprocess_spherical_coordinates(positions)

        # Perform HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples
        )
        labels = clusterer.fit_predict(positions)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": len(np.unique(labels[labels >= 0])),
                "n_noise": (labels == -1).sum(),
                "cluster_persistence": clusterer.cluster_persistence_
                if hasattr(clusterer, "cluster_persistence_")
                else None,
            },
        )

        return labels_tensor

    def friends_of_friends(
        self,
        linking_length: float = 1.0,
        min_group_size: int = 2,
        algorithm_name: str = "fof",
    ) -> torch.Tensor:
        """
        Friends-of-friends clustering for large-scale structure.

        Args:
            linking_length: Linking length for group formation
            min_group_size: Minimum group size
            algorithm_name: Name for storing results

        Returns:
            Group labels (-1 for isolated objects)
        """
        positions = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            positions = self._preprocess_spherical_coordinates(positions)

        # Use GPU-accelerated radius search for FoF
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
        
        # Create radius graph
        edge_index = radius_graph(
            x=positions_tensor,
            r=linking_length,
            loop=False,
            flow='source_to_target'
        )
        
        # Move back to CPU for processing
        edge_index = edge_index.cpu().numpy()
        
        # Convert to adjacency matrix
        n_points = len(positions)
        adjacency = csr_matrix((np.ones(len(edge_index[0])), edge_index), shape=(n_points, n_points))
        
        # Find connected components
        from scipy.sparse.csgraph import connected_components
        n_components, labels = connected_components(adjacency, directed=False)
        
        # Filter by minimum group size
        for i in range(n_components):
            if np.sum(labels == i) < min_group_size:
                labels[labels == i] = -1
        
        # Relabel to consecutive integers
        unique_labels = np.unique(labels[labels >= 0])
        label_map = {old: new for new, old in enumerate(unique_labels)}
        labels = np.array([label_map.get(label, -1) for label in labels])

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "linking_length": linking_length,
                "min_group_size": min_group_size,
                "n_groups": len(np.unique(labels[labels >= 0])),
                "n_isolated": (labels == -1).sum(),
            },
        )

        return labels_tensor

    def hierarchical_clustering(
        self,
        n_clusters: int,
        linkage: str = "ward",
        distance_threshold: Optional[float] = None,
        algorithm_name: str = "hierarchical",
    ) -> torch.Tensor:
        """
        Hierarchical clustering with astronomical distance metrics.

        Args:
            n_clusters: Number of clusters (ignored if distance_threshold set)
            linkage: Linkage criterion
            distance_threshold: Distance threshold for automatic cluster number
            algorithm_name: Name for storing results

        Returns:
            Cluster labels
        """

        # Use both positions and features if available
        if self.features is not None:
            data = self._data.cpu().numpy()
        else:
            data = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            pos_data = self.positions.cpu().numpy()
            pos_data = self._preprocess_spherical_coordinates(pos_data)
            if self.features is not None:
                feat_data = self.features.cpu().numpy()
                data = np.concatenate([pos_data, feat_data], axis=1)
            else:
                data = pos_data

        # Perform hierarchical clustering
        from sklearn.cluster import AgglomerativeClustering

        if distance_threshold is not None:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage=linkage,
            )
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, linkage=linkage
            )

        labels = clusterer.fit_predict(data)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "n_clusters": n_clusters if distance_threshold is None else len(np.unique(labels)),
                "linkage": linkage,
                "distance_threshold": distance_threshold,
            },
        )

        return labels_tensor

    def kmeans_clustering(
        self,
        n_clusters: int,
        algorithm_name: str = "kmeans",
        **kwargs,
    ) -> torch.Tensor:
        """
        K-means clustering for astronomical data.

        Args:
            n_clusters: Number of clusters
            algorithm_name: Name for storing results
            **kwargs: Additional KMeans parameters

        Returns:
            Cluster labels
        """
        # Use both positions and features if available
        if self.features is not None:
            data = self._data.cpu().numpy()
        else:
            data = self.positions.cpu().numpy()

        # Apply coordinate-specific preprocessing
        if self.get_metadata("coordinate_system") == "spherical":
            pos_data = self.positions.cpu().numpy()
            pos_data = self._preprocess_spherical_coordinates(pos_data)
            if self.features is not None:
                feat_data = self.features.cpu().numpy()
                data = np.concatenate([pos_data, feat_data], axis=1)
        else:
                data = pos_data

        # Perform K-means
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
        labels = clusterer.fit_predict(data)

        # Store results
        labels_tensor = torch.from_numpy(labels)
        self._store_clustering_results(
            algorithm_name,
            labels_tensor,
            {
                "n_clusters": n_clusters,
                "inertia": clusterer.inertia_,
                "cluster_centers": clusterer.cluster_centers_,
            },
        )

        return labels_tensor

    def get_cluster_properties(self, algorithm_name: str) -> Dict[str, Any]:
        """Get properties for a specific clustering algorithm."""
        return self._metadata.get("cluster_properties", {}).get(algorithm_name, {})

    def get_cluster_labels(self, algorithm_name: str) -> Optional[torch.Tensor]:
        """Get cluster labels for a specific algorithm."""
        return self._metadata.get("cluster_labels", {}).get(algorithm_name)

    def list_clustering_algorithms(self) -> List[str]:
        """List available clustering algorithms."""
        return list(self._metadata.get("cluster_labels", {}).keys())

    def _store_clustering_results(
        self, algorithm_name: str, labels: torch.Tensor, properties: Dict[str, Any]
    ) -> None:
        """Store clustering results in metadata."""
        if "cluster_labels" not in self._metadata:
            self._metadata["cluster_labels"] = {}
        if "cluster_properties" not in self._metadata:
            self._metadata["cluster_properties"] = {}

        self._metadata["cluster_labels"][algorithm_name] = labels
        self._metadata["cluster_properties"][algorithm_name] = properties

    # Helper methods
    def _estimate_eps(self, min_samples: int) -> float:
        """Estimate optimal eps parameter for DBSCAN using GPU acceleration."""
        positions = self.positions.cpu().numpy()

        # Use GPU-accelerated k-distance graph method
        try:
            # Convert to PyTorch tensor and move to GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            positions_tensor = torch.tensor(positions, dtype=torch.float32, device=device)
            
            # Create k-NN graph on GPU to get distances
            edge_index = torch_cluster.knn_graph(
                x=positions_tensor, 
                k=min_samples, 
                loop=False,  # No self-loops
                flow='source_to_target'
            )
            
            # Calculate distances for each edge
            edge_distances = torch.norm(
                positions_tensor[edge_index[0]] - positions_tensor[edge_index[1]], 
                dim=1
            )
            
            # Group distances by source node and get k-th distance for each
            k_distances = []
            for i in range(len(positions)):
                node_edges = edge_index[0] == i
                if node_edges.sum() >= min_samples:
                    node_distances = edge_distances[node_edges]
                    k_distances.append(node_distances[min_samples - 1].item())
            
            if k_distances:
                # Find elbow point (simplified)
                eps = np.percentile(k_distances, 90)  # Use 90th percentile as heuristic
                return float(eps)
            else:
                # Fallback to sklearn if not enough edges
                raise RuntimeError("Not enough edges for eps estimation")
                
        except (ImportError, RuntimeError):
            # Fallback to sklearn if torch_cluster not available or failed
            from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min_samples)
        nbrs.fit(positions)
        distances, indices = nbrs.kneighbors(positions)

        # Sort distances to k-th nearest neighbor
        k_distances = np.sort(distances[:, min_samples - 1])

        # Find elbow point (simplified)
        eps = np.percentile(k_distances, 90)  # Use 90th percentile as heuristic

        return float(eps)

    def _preprocess_spherical_coordinates(self, positions: np.ndarray) -> np.ndarray:
        """Preprocess spherical coordinates for clustering."""
        if positions.shape[1] == 2:  # RA, Dec
            ra, dec = positions[:, 0], positions[:, 1]
            # Convert to Cartesian on unit sphere
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)
            return np.column_stack([x, y, z])
        elif positions.shape[1] == 3:  # RA, Dec, Distance
            ra, dec, dist = positions[:, 0], positions[:, 1], positions[:, 2]
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            x = dist * np.cos(dec_rad) * np.cos(ra_rad)
            y = dist * np.cos(dec_rad) * np.sin(ra_rad)
            z = dist * np.sin(dec_rad)
            return np.column_stack([x, y, z])
        else:
            return positions

    def __repr__(self) -> str:
        n_algorithms = len(self.get_metadata("cluster_labels", {}))
        return (
            f"ClusteringTensor(objects={len(self)}, "
            f"coord_system={self.get_metadata('coordinate_system')}, "
            f"algorithms={n_algorithms})"
        )
