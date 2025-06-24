"""
Simplified graph utilities for astronomical data.

Basic graph construction and analysis tools that work with the new tensor architecture.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Forward references for type hints
if TYPE_CHECKING:
    from astro_lab.tensors import Spatial3DTensor

# Removed duplicate TYPE_CHECKING block

# Import PyTorch Geometric and sklearn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

# GPU-accelerated imports
import torch_cluster


def _gpu_knn_graph(coords: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
    """
    Create KNN graph using GPU acceleration with torch_cluster - FAST VERSION.
    
    Args:
        coords: Coordinate tensor [N, 3]
        k: Number of nearest neighbors
        **kwargs: Additional arguments (ignored for GPU implementation)
    
    Returns:
        Edge index tensor [2, num_edges]
    """
    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Convert to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords_gpu = coords.to(device)
    
    # Use torch_cluster's knn_graph for GPU acceleration
    edge_index = torch_cluster.knn_graph(
        x=coords_gpu, 
        k=k, 
        loop=False,  # No self-loops
        flow='source_to_target'
    )
    
    # Move back to CPU for consistency
    edge_index = edge_index.cpu()
    
    return edge_index


def _sklearn_knn_graph(coords: torch.Tensor, k: int, **kwargs) -> torch.Tensor:
    """
    Create KNN graph using sklearn's NearestNeighbors as fallback for torch-cluster.

    Args:
        coords: Coordinate tensor [N, 3]
        k: Number of nearest neighbors
        **kwargs: Additional arguments (ignored for sklearn implementation)

    Returns:
        Edge index tensor [2, num_edges]
    """

    # Convert to numpy for sklearn
    coords_np = coords.detach().cpu().numpy()

    # Use GPU-accelerated radius search as fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords_tensor = torch.tensor(coords_np, dtype=torch.float32, device=device)
    
    # Create k-NN graph on GPU
    edge_index = torch_cluster.knn_graph(
        x=coords_tensor, 
        k=k, 
        loop=False,  # No self-loops
        flow='source_to_target'
    )
    
    # Move back to CPU for consistency
    edge_index = edge_index.cpu()

    return edge_index


def create_knn_graph(coords: torch.Tensor, k: int, use_gpu: bool = True, **kwargs) -> torch.Tensor:
    """
    Create KNN graph with GPU acceleration by default.
    
    Args:
        coords: Coordinate tensor [N, 3]
        k: Number of nearest neighbors
        use_gpu: Whether to use GPU acceleration (default: True)
        **kwargs: Additional arguments
    
    Returns:
        Edge index tensor [2, num_edges]
    """
    if use_gpu:
        return _gpu_knn_graph(coords, k, **kwargs)
    else:
        return _sklearn_knn_graph(coords, k, **kwargs)


def create_radius_graph(coords: torch.Tensor, radius: float, **kwargs) -> torch.Tensor:
    """
    Create radius graph using GPU acceleration.
    
    Args:
        coords: Coordinate tensor [N, 3]
        radius: Radius for neighbor search
        **kwargs: Additional arguments
    
    Returns:
        Edge index tensor [2, num_edges]
    """
    if len(coords) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Convert to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    coords_gpu = coords.to(device)
    
    # Use torch_geometric's radius_graph for GPU acceleration
    edge_index = radius_graph(
        x=coords_gpu,
        r=radius,
        loop=False,  # No self-loops
        flow='source_to_target'
    )
    
    # Move back to CPU for consistency
    edge_index = edge_index.cpu()
    
    return edge_index


def create_astronomical_graph(
    coords: torch.Tensor,
    features: Optional[torch.Tensor] = None,
    k_neighbors: int = 8,
    radius: Optional[float] = None,
    **kwargs,
) -> Data:
    """
    Create PyTorch Geometric Data object for astronomical data.
    
    Args:
        coords: Coordinate tensor [N, 3]
        features: Optional feature tensor [N, F]
        k_neighbors: Number of nearest neighbors for graph construction
        radius: Alternative radius for graph construction
        **kwargs: Additional arguments
    
    Returns:
        PyTorch Geometric Data object
    """
    # Create edges
    if radius is not None:
        edge_index = create_radius_graph(coords, radius, **kwargs)
    else:
        edge_index = create_knn_graph(coords, k_neighbors, **kwargs)
    
    # Create node features
    if features is not None:
        x = torch.cat([coords, features], dim=1)
    else:
        x = coords
    
    return Data(x=x, edge_index=edge_index, pos=coords)


def analyze_graph_structure(edge_index: torch.Tensor, num_nodes: int) -> Dict[str, Any]:
    """
    Analyze graph structure and connectivity.
    
    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph
    
    Returns:
        Dictionary with graph statistics
    """
    num_edges = edge_index.shape[1]
    
    # Calculate degree distribution
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.scatter_add_(0, edge_index[0], torch.ones(num_edges, dtype=torch.long))
    
    # Calculate statistics
    mean_degree = degrees.float().mean().item()
    max_degree = degrees.max().item()
    min_degree = degrees.min().item()
    
    # Calculate connectivity
    isolated_nodes = (degrees == 0).sum().item()
    connected_nodes = num_nodes - isolated_nodes
    
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "mean_degree": mean_degree,
        "max_degree": max_degree,
        "min_degree": min_degree,
        "isolated_nodes": isolated_nodes,
        "connected_nodes": connected_nodes,
        "connectivity_ratio": connected_nodes / num_nodes if num_nodes > 0 else 0.0,
    }


def cluster_graph_nodes(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    algorithm: str = "dbscan",
    **kwargs,
) -> torch.Tensor:
    """
    Cluster nodes in the graph based on spatial proximity.
    
    Args:
        coords: Coordinate tensor [N, 3]
        edge_index: Edge index tensor [2, num_edges]
        algorithm: Clustering algorithm ('dbscan', 'kmeans')
        **kwargs: Additional clustering parameters
    
    Returns:
        Cluster labels tensor [N]
    """
    coords_np = coords.detach().cpu().numpy()
    
    if algorithm == "dbscan":
        clusterer = DBSCAN(**kwargs)
    elif algorithm == "kmeans":
        n_clusters = kwargs.get("n_clusters", 5)
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")
    
    labels = clusterer.fit_predict(coords_np)
    return torch.from_numpy(labels)


def create_spatial_graph(
    spatial_tensor: "Spatial3DTensor",
    method: str = "knn",
    k: int = 5,
    radius: float = 1.0,
    **kwargs: Any,
) -> Data:
    """
    Create a spatial graph from Spatial3DTensor.

    Args:
        spatial_tensor: Input spatial tensor
        method: Graph construction method ('knn', 'radius')
        k: Number of neighbors for KNN
        radius: Radius for radius graph
        **kwargs: Additional arguments

    Returns:
        PyTorch Geometric Data object
    """

    # Get Cartesian coordinates
    coords = spatial_tensor.cartesian

    # Create edges
    if method == "knn":
        # Use GPU-accelerated KNN by default
        edge_index = _gpu_knn_graph(coords, k=k, **kwargs)
    elif method == "radius":
        edge_index = radius_graph(coords, r=radius, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Create data object
    data = Data(
        x=coords,
        edge_index=edge_index,
        pos=coords,
    )

    # Add metadata
    if hasattr(spatial_tensor, "_metadata"):
        for key, value in spatial_tensor._metadata.items():
            if isinstance(value, torch.Tensor):
                setattr(data, key, value)

    return data


def calculate_graph_metrics(data: Data) -> dict:
    """
    Calculate basic graph metrics.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dictionary of graph metrics
    """
    # Use data.x if available, otherwise fall back to data.pos
    if hasattr(data, "x") and data.x is not None:
        num_nodes = data.x.size(0)
    elif hasattr(data, "pos") and data.pos is not None:
        num_nodes = data.pos.size(0)
    else:
        raise ValueError("Data object must have either 'x' or 'pos' attribute")

    num_edges = data.edge_index.size(1)

    # Calculate degree statistics
    degrees = torch.bincount(data.edge_index[0], minlength=num_nodes)

    # Calculate clustering coefficient (simplified version)
    clustering_coeff = _calculate_clustering_coefficient(data.edge_index, num_nodes)

    metrics = {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "avg_degree": float(degrees.float().mean()),
        "max_degree": int(degrees.max()),
        "min_degree": int(degrees.min()),
        "density": 2 * num_edges / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0.0,
        "clustering_coefficient": clustering_coeff,
    }

    return metrics


def _calculate_clustering_coefficient(
    edge_index: torch.Tensor, num_nodes: int
) -> float:
    """
    Calculate average clustering coefficient for the graph.

    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph

    Returns:
        Average clustering coefficient
    """
    # Create adjacency matrix
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    # Make undirected
    adj = adj | adj.t()

    clustering_coeffs = []

    for node in range(num_nodes):
        # Get neighbors
        neighbors = torch.where(adj[node])[0]
        degree = len(neighbors)

        if degree < 2:
            # No triangles possible with less than 2 neighbors
            clustering_coeffs.append(0.0)
            continue

        # Count triangles
        triangles = 0
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                if adj[neighbors[i], neighbors[j]]:
                    triangles += 1

        # Clustering coefficient for this node
        possible_triangles = degree * (degree - 1) // 2
        clustering_coeff = (
            triangles / possible_triangles if possible_triangles > 0 else 0.0
        )
        clustering_coeffs.append(clustering_coeff)

    # Return average clustering coefficient
    return float(np.mean(clustering_coeffs))


def spatial_distance_matrix(
    coords: torch.Tensor, metric: str = "euclidean"
) -> torch.Tensor:
    """
    Compute pairwise distance matrix.

    Args:
        coords: Coordinate tensor [N, 3]
        metric: Distance metric ('euclidean', 'angular')

    Returns:
        Distance matrix [N, N]
    """
    if metric == "euclidean":
        diff = coords.unsqueeze(1) - coords.unsqueeze(0)
        distances = torch.norm(diff, dim=-1)
    elif metric == "angular":
        # Normalize to unit vectors
        unit_coords = coords / torch.norm(coords, dim=-1, keepdim=True)
        # Compute dot product
        dot_products = torch.mm(unit_coords, unit_coords.t())
        # Clamp to avoid numerical issues
        dot_products = torch.clamp(dot_products, -1.0, 1.0)
        # Convert to angular distance
        distances = torch.acos(dot_products)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distances


def cluster_and_analyze(
    coords: torch.Tensor,
    algorithm: str = "dbscan",
    eps: float = 0.5,
    min_samples: int = 5,
    n_clusters: int = None,
    use_gpu: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Performs clustering and statistics calculation on the given coordinates.
    Args:
        coords: Coordinates as tensor [N, D]
        algorithm: 'dbscan', 'agglomerative', 'kmeans', 'fof'
        eps: Radius for DBSCAN/Agglomerative/FoF
        min_samples: Minimum for DBSCAN
        n_clusters: Number of clusters for KMeans/Agglomerative
        use_gpu: GPU for neighborhood search (not for clustering itself)
        **kwargs: Additional parameters
    Returns:
        Dict with labels, statistics, n_clusters, n_noise, etc.
    """
    coords_np = coords.detach().cpu().numpy()
    labels = None
    if algorithm == "dbscan":
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
        labels = clusterer.fit_predict(coords_np)
    elif algorithm == "agglomerative":
        if n_clusters is not None:
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            clusterer = AgglomerativeClustering(distance_threshold=eps, n_clusters=None, **kwargs)
        labels = clusterer.fit_predict(coords_np)
    elif algorithm == "kmeans":
        if n_clusters is None:
            n_clusters = 5
        clusterer = KMeans(n_clusters=n_clusters, **kwargs)
        labels = clusterer.fit_predict(coords_np)
    elif algorithm == "fof":
        # Friends-of-Friends as DBSCAN with only eps
        clusterer = DBSCAN(eps=eps, min_samples=1, **kwargs)
        labels = clusterer.fit_predict(coords_np)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    labels_tensor = torch.from_numpy(labels).to(coords.device)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = sum(1 for label in labels if label == -1)
    # Calculate statistics
    cluster_stats = {}
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        cluster_mask = labels == cluster_id
        cluster_coords = coords_np[cluster_mask]
        center = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center, axis=1)
        cluster_stats[cluster_id] = {
            "n_points": int(cluster_mask.sum()),
            "center": center,
            "radius": float(distances.max()),
            "density": float(
                cluster_mask.sum() / (4/3 * np.pi * max(distances.max(), 1e-6)**3)
            ),
        }
    return {
        "cluster_labels": labels_tensor,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_stats": cluster_stats,
        "coords": coords.detach().cpu(),
    }


__all__ = [
    "create_spatial_graph",
    "calculate_graph_metrics",
    "spatial_distance_matrix",
]
