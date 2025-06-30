"""
PyG-native clustering utilities
===============================

Clustering functions using only PyTorch and PyTorch Geometric.
No sklearn or scipy dependencies.
"""

import torch
from torch_geometric.nn.pool import fps
from torch_geometric.nn import knn_graph


def create_pyg_kmeans(
    positions: torch.Tensor, 
    n_clusters: int, 
    max_iters: int = 100,
    tol: float = 1e-5
) -> torch.Tensor:
    """
    K-means clustering using PyTorch operations only.
    
    Args:
        positions: Node positions [N, D]
        n_clusters: Number of clusters
        max_iters: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Cluster labels [N]
    """
    device = positions.device
    n_points = positions.shape[0]
    
    # Initialize cluster centers using FPS
    ratio = min(n_clusters / n_points, 1.0)
    center_indices = fps(positions, ratio=ratio)
    
    # If we got fewer centers than requested, sample randomly for the rest
    if len(center_indices) < n_clusters:
        remaining = n_clusters - len(center_indices)
        other_indices = torch.randperm(n_points, device=device)[:remaining]
        center_indices = torch.cat([center_indices, other_indices])
    
    centers = positions[center_indices[:n_clusters]]
    
    # K-means iterations
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
        if torch.allclose(centers, new_centers, rtol=tol):
            break
            
        centers = new_centers
    
    return labels


def spatial_clustering_fps(
    positions: torch.Tensor,
    n_clusters: int,
    return_centers: bool = False
) -> torch.Tensor:
    """
    Spatial clustering using Farthest Point Sampling.
    
    A fast alternative to k-means that works well for spatial data.
    
    Args:
        positions: Node positions [N, D]
        n_clusters: Number of clusters
        return_centers: Whether to return cluster centers
        
    Returns:
        Cluster labels [N] and optionally centers [n_clusters, D]
    """
    n_points = positions.shape[0]
    
    # Use FPS to get diverse cluster centers
    ratio = min(n_clusters / n_points, 1.0)
    center_indices = fps(positions, ratio=ratio)
    
    # Ensure we have exactly n_clusters centers
    if len(center_indices) < n_clusters:
        # Add random points if needed
        remaining = n_clusters - len(center_indices)
        other_indices = torch.randperm(n_points)[:remaining]
        center_indices = torch.cat([center_indices, other_indices])
    else:
        center_indices = center_indices[:n_clusters]
    
    centers = positions[center_indices]
    
    # Assign each point to nearest center
    distances = torch.cdist(positions, centers)
    labels = distances.argmin(dim=1)
    
    if return_centers:
        return labels, centers
    return labels


def graph_based_clustering(
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    n_clusters: int,
    method: str = "spectral"
) -> torch.Tensor:
    """
    Graph-based clustering using graph structure.
    
    Args:
        positions: Node positions [N, D]
        edge_index: Graph edges [2, E]
        n_clusters: Number of clusters
        method: Clustering method ("spectral" or "propagation")
        
    Returns:
        Cluster labels [N]
    """
    if method == "propagation":
        # Simple label propagation
        return _label_propagation(edge_index, n_clusters, positions.shape[0])
    else:
        # Fallback to spatial clustering
        return spatial_clustering_fps(positions, n_clusters)


def _label_propagation(
    edge_index: torch.Tensor,
    n_clusters: int,
    num_nodes: int,
    max_iters: int = 100
) -> torch.Tensor:
    """
    Simple label propagation for clustering.
    """
    # Initialize random labels
    labels = torch.randint(0, n_clusters, (num_nodes,))
    
    for _ in range(max_iters):
        new_labels = labels.clone()
        
        # For each node, adopt the most common label among neighbors
        for node in range(num_nodes):
            # Find neighbors
            neighbors = edge_index[1][edge_index[0] == node]
            if len(neighbors) > 0:
                neighbor_labels = labels[neighbors]
                # Find most common label
                unique_labels, counts = torch.unique(neighbor_labels, return_counts=True)
                new_labels[node] = unique_labels[counts.argmax()]
        
        # Check convergence
        if torch.equal(labels, new_labels):
            break
            
        labels = new_labels
    
    return labels


def hierarchical_fps_clustering(
    positions: torch.Tensor,
    n_levels: int = 3,
    reduction_factor: float = 0.5
) -> Dict[str, torch.Tensor]:
    """
    Hierarchical clustering using repeated FPS.
    
    Args:
        positions: Node positions [N, D]
        n_levels: Number of hierarchy levels
        reduction_factor: Points reduction per level
        
    Returns:
        Dictionary with cluster labels at each level
    """
    hierarchies = {}
    current_positions = positions
    current_indices = torch.arange(positions.shape[0])
    
    for level in range(n_levels):
        # Sample points for next level
        ratio = reduction_factor ** (level + 1)
        sampled_indices = fps(current_positions, ratio=ratio)
        
        # Assign all points to nearest sampled point
        sampled_positions = current_positions[sampled_indices]
        distances = torch.cdist(positions, sampled_positions)
        labels = distances.argmin(dim=1)
        
        hierarchies[f"level_{level}"] = {
            "labels": labels,
            "centers": sampled_positions,
            "n_clusters": len(sampled_indices)
        }
        
        # Update for next level
        current_positions = sampled_positions
        current_indices = current_indices[sampled_indices]
    
    return hierarchies
