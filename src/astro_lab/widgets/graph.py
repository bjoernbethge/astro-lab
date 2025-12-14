"""
Simplified graph utilities for astronomical data.

Basic graph construction and analysis tools that work with the new tensor architecture.
Includes cosmic web analysis functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Import PyTorch Geometric and sklearn
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from torch_geometric.data import Data

# Import centralized graph builders
from astro_lab.data.graphs import (
    create_astronomical_graph,
    create_knn_graph,
    create_radius_graph,
)
from astro_lab.tensors import SpatialTensorDict, SurveyTensorDict
from astro_lab.data.cosmic_web import CosmicWebAnalyzer


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
    Cluster nodes in a graph using various algorithms.

    Args:
        coords: Node coordinates [N, D]
        edge_index: Edge index tensor [2, num_edges]
        algorithm: Clustering algorithm ("dbscan", "agglomerative", "kmeans", 
                                       "spectral", "meanshift", "optics", "hdbscan")
        **kwargs: Algorithm-specific parameters

    Returns:
        Cluster labels tensor [N]
    """
    coords_np = coords.detach().cpu().numpy()

    if algorithm == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        clustering = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == "agglomerative":
        n_clusters = kwargs.get("n_clusters", 3)
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
    elif algorithm == "kmeans":
        n_clusters = kwargs.get("n_clusters", 3)
        clustering = KMeans(n_clusters=n_clusters, random_state=42)
    elif algorithm == "spectral":
        n_clusters = kwargs.get("n_clusters", 3)
        affinity = kwargs.get("affinity", "nearest_neighbors")
        clustering = SpectralClustering(
            n_clusters=n_clusters, 
            affinity=affinity,
            random_state=42
        )
    elif algorithm == "meanshift":
        from sklearn.cluster import MeanShift
        bandwidth = kwargs.get("bandwidth", None)
        clustering = MeanShift(bandwidth=bandwidth)
    elif algorithm == "optics":
        from sklearn.cluster import OPTICS
        min_samples = kwargs.get("min_samples", 5)
        eps = kwargs.get("eps", None)
        clustering = OPTICS(min_samples=min_samples, max_eps=eps)
    elif algorithm == "hdbscan":
        try:
            import hdbscan
            min_cluster_size = kwargs.get("min_cluster_size", 5)
            min_samples = kwargs.get("min_samples", None)
            clustering = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples
            )
        except ImportError:
            raise ImportError(
                "HDBSCAN not installed. Install with: pip install hdbscan"
            )
    else:
        raise ValueError(f"Unknown clustering algorithm: {algorithm}")

    labels = clustering.fit_predict(coords_np)
    return torch.tensor(labels, dtype=torch.long)


def create_spatial_graph(
    spatial_tensor: SpatialTensorDict,
    method: str = "knn",
    k: int = 5,
    radius: float = 1.0,
    **kwargs: Any,
) -> Data:
    """
    Create graph from SpatialTensorDict using centralized builders.

    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        method: Graph construction method ("knn", "radius", "astronomical")
        k: Number of neighbors for KNN
        radius: Radius for radius-based graph
        **kwargs: Additional arguments

    Returns:
        PyTorch Geometric Data object
    """
    # Create SurveyTensorDict wrapper
    survey_tensor = SurveyTensorDict({"spatial": spatial_tensor})

    if method == "knn":
        return create_knn_graph(survey_tensor, k_neighbors=k, **kwargs)
    elif method == "radius":
        return create_radius_graph(survey_tensor, radius=radius, **kwargs)
    elif method == "astronomical":
        return create_astronomical_graph(survey_tensor, k_neighbors=k, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_graph_metrics(data: Data) -> dict:
    """
    Calculate comprehensive graph metrics.

    Args:
        data: PyTorch Geometric Data object

    Returns:
        Dictionary with graph metrics
    """
    if not hasattr(data, "edge_index") or data.edge_index is None:
        return {"error": "No edge_index found in data"}

    # Basic structure analysis
    structure = analyze_graph_structure(data.edge_index, data.num_nodes)

    # Clustering coefficient
    clustering_coeff = _calculate_clustering_coefficient(
        data.edge_index, data.num_nodes
    )

    # Feature statistics
    feature_stats = {}
    if hasattr(data, "x") and data.x is not None:
        feature_stats = {
            "num_features": data.x.size(1),
            "feature_mean": data.x.mean().item(),
            "feature_std": data.x.std().item(),
        }

    # Position statistics
    position_stats = {}
    if hasattr(data, "pos") and data.pos is not None:
        pos = data.pos
        position_stats = {
            "num_dimensions": pos.size(1),
            "position_range": {
                "min": pos.min().item(),
                "max": pos.max().item(),
            },
        }

    return {
        **structure,
        "clustering_coefficient": clustering_coeff,
        **feature_stats,
        **position_stats,
    }


def _calculate_clustering_coefficient(
    edge_index: torch.Tensor, num_nodes: int
) -> float:
    """
    Calculate the average clustering coefficient of the graph.

    Args:
        edge_index: Edge index tensor [2, num_edges]
        num_nodes: Number of nodes

    Returns:
        Average clustering coefficient
    """
    if num_nodes == 0:
        return 0.0

    # Convert to adjacency matrix for easier computation
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
    adj[edge_index[0], edge_index[1]] = True

    # Calculate clustering coefficient for each node (vectorized)
    # Use matrix multiplication to count triangles more efficiently
    # adj^2[i,j] counts 2-step paths from i to j
    # adj^3 diagonal gives number of triangles × 2 for each node
    
    clustering_coeffs = torch.zeros(num_nodes, dtype=torch.float32)
    
    # Convert to float for matrix multiplication
    adj_float = adj.float()
    
    # Compute A^2 and A^3 for triangle counting
    adj_squared = torch.mm(adj_float, adj_float)
    
    # For each node, count triangles
    for i in range(num_nodes):
        # Get neighbors of node i
        neighbors = torch.where(adj[i])[0]
        k = len(neighbors)
        
        if k < 2:
            clustering_coeffs[i] = 0.0
            continue
        
        # Count triangles using adjacency matrix: 
        # Number of triangles = (neighbors[i] & neighbors[j]) for all j in neighbors
        # This is equivalent to counting common neighbors
        neighbor_adj = adj[neighbors][:, neighbors]  # Subgraph of neighbors
        triangles = neighbor_adj.sum().item() / 2  # Divide by 2 as edges are counted twice
        
        # Clustering coefficient
        max_triangles = k * (k - 1) / 2
        if max_triangles > 0:
            clustering_coeffs[i] = triangles / max_triangles
        else:
            clustering_coeffs[i] = 0.0
    
    return clustering_coeffs.mean().item()


def spatial_distance_matrix(
    coords: torch.Tensor, metric: str = "euclidean"
) -> torch.Tensor:
    """
    Calculate pairwise spatial distances between coordinates.

    Args:
        coords: Coordinate tensor [N, D]
        metric: Distance metric ("euclidean", "manhattan", "cosine")

    Returns:
        Distance matrix [N, N]
    """
    if metric == "euclidean":
        # Efficient euclidean distance calculation
        dist_sq = torch.cdist(coords, coords, p=2)
        return dist_sq
    elif metric == "manhattan":
        return torch.cdist(coords, coords, p=1)
    elif metric == "cosine":
        # Normalize coordinates for cosine distance
        coords_norm = coords / torch.norm(coords, dim=1, keepdim=True)
        return 1 - torch.mm(coords_norm, coords_norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")


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
    Perform clustering and return comprehensive analysis.

    Args:
        coords: Coordinate tensor [N, D]
        algorithm: Clustering algorithm
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
        n_clusters: Number of clusters for K-means/Agglomerative
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional clustering parameters

    Returns:
        Dictionary with clustering results and analysis
    """
    # Perform clustering
    labels = cluster_graph_nodes(
        coords,
        torch.empty(
            (2, 0), dtype=torch.long
        ),  # Empty edge_index for coordinate-only clustering
        algorithm=algorithm,
        eps=eps,
        min_samples=min_samples,
        n_clusters=n_clusters,
        **kwargs,
    )

    # Calculate cluster statistics
    unique_labels = torch.unique(labels)
    n_clusters_found = len(unique_labels)

    cluster_sizes = []
    cluster_centers = []

    for label in unique_labels:
        mask = labels == label
        cluster_sizes.append(mask.sum().item())
        cluster_centers.append(coords[mask].mean(dim=0))

    cluster_centers = torch.stack(cluster_centers)

    # Calculate silhouette score (simplified)
    if n_clusters_found > 1:
        # Calculate average distance to cluster center
        avg_intra_cluster = 0
        for i, label in enumerate(unique_labels):
            mask = labels == label
            cluster_coords = coords[mask]
            center = cluster_centers[i]
            avg_intra_cluster += torch.norm(cluster_coords - center, dim=1).mean()
        avg_intra_cluster /= n_clusters_found

        # Calculate average distance to nearest cluster center
        avg_inter_cluster = 0
        for i, center in enumerate(cluster_centers):
            other_centers = torch.cat([cluster_centers[:i], cluster_centers[i + 1 :]])
            if len(other_centers) > 0:
                min_dist = torch.norm(other_centers - center, dim=1).min()
                avg_inter_cluster += min_dist
        avg_inter_cluster /= n_clusters_found

        silhouette_score = (avg_inter_cluster - avg_intra_cluster) / max(
            avg_inter_cluster, avg_intra_cluster
        )
    else:
        silhouette_score = 0.0

    return {
        "algorithm": algorithm,
        "n_clusters": n_clusters_found,
        "labels": labels,
        "cluster_sizes": cluster_sizes,
        "cluster_centers": cluster_centers,
        "silhouette_score": silhouette_score,
        "coordinate_range": {
            "min": coords.min().item(),
            "max": coords.max().item(),
        },
    }


def analyze_cosmic_web_structure(
    spatial_tensor: SpatialTensorDict,
    scales: List[float] = [5.0, 10.0, 25.0, 50.0],
    min_samples: int = 5,
    algorithm: str = "dbscan",
    use_existing_analyzer: bool = True,
) -> Dict[str, Any]:
    """
    Analyze cosmic web structure at multiple scales.
    
    Args:
        spatial_tensor: Spatial coordinates tensor
        scales: List of clustering scales (in same units as coordinates)
        min_samples: Minimum samples for clustering
        algorithm: Clustering algorithm to use
        use_existing_analyzer: Use the CosmicWebAnalyzer from data module
        
    Returns:
        Dictionary with multi-scale clustering results
    """
    if use_existing_analyzer:
        # Use the existing cosmic web clustering method
        results = {
            "scales": scales,
            "clustering_results": {},
            "density_analysis": None,
            "structure_analysis": None,
        }
        
        # Multi-scale clustering
        for scale in scales:
            labels = spatial_tensor.cosmic_web_clustering(
                eps_pc=scale,
                min_samples=min_samples,
                algorithm=algorithm,
            )
            
            unique_labels = torch.unique(labels)
            n_clusters = len(unique_labels[unique_labels >= 0])
            n_noise = (labels == -1).sum().item()
            
            results["clustering_results"][f"{scale}_units"] = {
                "labels": labels,
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "n_grouped": len(labels) - n_noise,
                "grouped_fraction": (len(labels) - n_noise) / len(labels),
            }
            
        # Add density analysis
        density_counts = spatial_tensor.analyze_local_density(radius_pc=scales[0])
        results["density_analysis"] = {
            "counts": density_counts,
            "mean_density": density_counts.float().mean().item(),
            "std_density": density_counts.float().std().item(),
        }
        
        # Add structure analysis
        structure = spatial_tensor.cosmic_web_structure(grid_size_pc=scales[-1])
        results["structure_analysis"] = structure
        
    else:
        # Use cluster_and_analyze for each scale
        coords = spatial_tensor["coordinates"]
        results = {
            "scales": scales,
            "clustering_results": {},
        }
        
        for scale in scales:
            cluster_result = cluster_and_analyze(
                coords,
                algorithm=algorithm,
                eps=scale,
                min_samples=min_samples,
            )
            
            results["clustering_results"][f"{scale}_units"] = cluster_result
            
    return results


def cosmic_web_connectivity_analysis(
    spatial_tensor: SpatialTensorDict,
    cluster_labels: torch.Tensor,
    scale: float,
) -> Dict[str, Any]:
    """
    Analyze connectivity patterns in cosmic web clusters.
    
    Args:
        spatial_tensor: Spatial coordinates
        cluster_labels: Cluster assignments
        scale: Scale at which clustering was performed
        
    Returns:
        Dictionary with connectivity metrics
    """
    coords = spatial_tensor["coordinates"]
    unique_labels = torch.unique(cluster_labels)
    clusters = unique_labels[unique_labels >= 0]
    
    connectivity = {
        "inter_cluster_distances": [],
        "cluster_properties": {},
        "filament_candidates": [],
    }
    
    # Analyze each cluster
    for label in clusters:
        mask = cluster_labels == label
        cluster_coords = coords[mask]
        
        # Cluster properties
        center = cluster_coords.mean(dim=0)
        radius = torch.norm(cluster_coords - center, dim=1).max().item()
        
        connectivity["cluster_properties"][label.item()] = {
            "size": mask.sum().item(),
            "center": center.tolist(),
            "radius": radius,
            "density": mask.sum().item() / (4/3 * np.pi * radius**3) if radius > 0 else 0,
        }
    
    # Inter-cluster distances
    cluster_centers = []
    for label in clusters:
        mask = cluster_labels == label
        center = coords[mask].mean(dim=0)
        cluster_centers.append(center)
        
    if len(cluster_centers) > 1:
        cluster_centers = torch.stack(cluster_centers)
        distances = torch.cdist(cluster_centers, cluster_centers)
        
        # Find potential filaments (clusters closer than 2*scale)
        filament_threshold = 2 * scale
        filament_pairs = torch.where(
            (distances < filament_threshold) & (distances > 0)
        )
        
        for i, j in zip(filament_pairs[0].tolist(), filament_pairs[1].tolist()):
            if i < j:  # Avoid duplicates
                connectivity["filament_candidates"].append({
                    "cluster_1": clusters[i].item(),
                    "cluster_2": clusters[j].item(),
                    "distance": distances[i, j].item(),
                })
    
    return connectivity


# The filament detection functions have been moved to data/cosmic_web.py
# as they are data analysis functions, not visualization functions
