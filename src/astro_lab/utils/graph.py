"""
Simplified graph utilities for astronomical data.

Basic graph construction and analysis tools that work with the new tensor architecture.
"""

from typing import List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import torch

# Forward references for type hints
if TYPE_CHECKING:
    from astro_lab.tensors import Spatial3DTensor

# Removed duplicate TYPE_CHECKING block

# Check for PyTorch Geometric and sklearn
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import radius_graph
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

try:
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


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
    if not SKLEARN_AVAILABLE:
        raise ImportError("sklearn required for KNN graph construction")
    
    # Convert to numpy for sklearn
    coords_np = coords.detach().cpu().numpy()
    
    # Create NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(coords_np)
    
    # Find k+1 nearest neighbors (including self)
    distances, indices = nbrs.kneighbors(coords_np)
    
    # Create edge list (exclude self-connections)
    edge_list = []
    for i in range(len(coords_np)):
        for j in range(1, k+1):  # Skip first index (self)
            neighbor_idx = indices[i, j]
            edge_list.append([i, neighbor_idx])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    return edge_index


def create_spatial_graph(
    spatial_tensor: "Spatial3DTensor",
    method: str = "knn",
    k: int = 5,
    radius: float = 1.0,
    **kwargs,
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
    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("PyTorch Geometric required for graph operations")

    # Get Cartesian coordinates
    coords = spatial_tensor.cartesian

    # Create edges
    if method == "knn":
        # Use sklearn-based KNN as fallback
        edge_index = _sklearn_knn_graph(coords, k=k, **kwargs)
    elif method == "radius":
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric required for radius graph")
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
    if hasattr(data, 'x') and data.x is not None:
        num_nodes = data.x.size(0)
    elif hasattr(data, 'pos') and data.pos is not None:
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


def _calculate_clustering_coefficient(edge_index: torch.Tensor, num_nodes: int) -> float:
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
        clustering_coeff = triangles / possible_triangles if possible_triangles > 0 else 0.0
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


__all__ = [
    "create_spatial_graph",
    "calculate_graph_metrics",
    "spatial_distance_matrix",
    "TORCH_GEOMETRIC_AVAILABLE",
]
