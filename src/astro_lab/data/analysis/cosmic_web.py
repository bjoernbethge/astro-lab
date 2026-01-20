"""
Cosmic Web Analysis for Large-Scale Structure Detection - PyTorch Geometric 2025
==============================================================================

Advanced cosmic web analysis with PyTorch Geometric 2025 features and optimizations.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch_geometric.nn import (
    fps,
    knn_graph,
    radius_graph,
    voxel_grid,
)
from torch_geometric.utils import (
    coalesce,
    contains_self_loops,
    degree,
    to_undirected,
)

from astro_lab.tensors import SpatialTensorDict
from astro_lab.utils.device import get_default_device
from astro_lab.utils.tensor import extract_coordinates

logger = logging.getLogger(__name__)


class ScalableCosmicWebAnalyzer:
    """
    Scalable cosmic web analysis using PyTorch Geometric 2025.

    Features:
    - Multi-scale cosmic web detection with native PyG functions
    - Filament and void identification using graph algorithms
    - GPU-accelerated processing with torch.compile support
    - Memory-efficient batch processing
    - TensorDict integration for flexible data handling
    """

    def __init__(
        self,
        device: str = None,
        max_points_per_batch: int = 100000,
    ):
        """
        Initialize cosmic web analyzer.

        Args:
            device: Computation device (default: auto-detect)
            max_points_per_batch: Maximum points to process at once
        """
        if device is None:
            device = get_default_device()
        self.device = torch.device(device)
        self.max_points_per_batch = max_points_per_batch

        logger.info(f"🌌 ScalableCosmicWebAnalyzer initialized on {self.device}")

    def analyze_cosmic_web(
        self,
        coordinates: Union[torch.Tensor, SpatialTensorDict],
        density_field: Optional[torch.Tensor] = None,
        scales: Optional[List[float]] = None,
        density_threshold: float = 0.5,
        use_adaptive_sampling: bool = True,
    ) -> Dict:
        """
        Analyze cosmic web structure at multiple scales.

        Args:
            coordinates: Galaxy/particle coordinates [N, 3]
            density_field: Optional pre-computed density field [N]
            scales: Analysis scales in Mpc (default: [1.0, 2.0, 5.0, 10.0])
            density_threshold: Threshold for high-density regions
            use_adaptive_sampling: Use FPS for large datasets

        Returns:
            Comprehensive cosmic web analysis results
        """
        # Handle different input types using utility function
        coords = extract_coordinates(coordinates)
        coords = coords.to(self.device)
        n_points = coords.shape[0]

        if density_field is not None:
            density_field = density_field.to(self.device)

        if scales is None:
            scales = [1.0, 2.0, 5.0, 10.0]

        logger.info(
            f"🌌 Cosmic web analysis: {n_points:,} points, {len(scales)} scales"
        )

        # Adaptive sampling for large datasets
        if use_adaptive_sampling and n_points > self.max_points_per_batch:
            coords, indices = self._adaptive_sampling(coords)
            if density_field is not None:
                density_field = density_field[indices]
            logger.info(f"   Adaptively sampled to {coords.shape[0]:,} points")

        results = {}

        # Multi-scale analysis
        for scale in scales:
            scale_results = self._analyze_at_scale(
                coords, density_field, scale, density_threshold
            )
            results[f"scale_{scale:.1f}"] = scale_results

        # Combine multi-scale results
        combined = self._combine_multi_scale_results(results)

        return {
            "multi_scale": results,
            "combined": combined,
            "coordinates": coords,
            "scales": scales,
            "device": str(self.device),
        }

    def _adaptive_sampling(
        self, coordinates: torch.Tensor, ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive sampling using FPS."""
        indices = fps(coordinates, batch=None, ratio=ratio, random_start=True)
        return coordinates[indices], indices

    def _analyze_at_scale(
        self,
        coordinates: torch.Tensor,
        density_field: Optional[torch.Tensor],
        scale: float,
        density_threshold: float,
    ) -> Dict:
        """Analyze cosmic web at a specific scale."""
        n_points = coordinates.size(0)

        # Build connectivity graph using PyG 2025 functions
        if scale < 5.0:  # Use k-NN for smaller scales
            k = min(32, n_points - 1)
            edge_index = knn_graph(
                coordinates, k=k, batch=None, loop=False, flow="source_to_target"
            )
        else:  # Use radius graph for larger scales
            edge_index = radius_graph(
                coordinates,
                r=scale,
                batch=None,
                loop=False,
                max_num_neighbors=64,
                flow="source_to_target",
            )

        # Make undirected and remove duplicates
        edge_index = to_undirected(edge_index, num_nodes=n_points)
        edge_index = coalesce(edge_index, num_nodes=n_points)

        # Calculate local density if not provided
        if density_field is None:
            density_field = self._calculate_local_density(coordinates, edge_index)

        # Identify cosmic web components
        filaments = self._identify_filaments(
            coordinates, edge_index, density_field, scale
        )
        voids = self._identify_voids(coordinates, density_field, density_threshold)
        nodes = self._identify_nodes(coordinates, edge_index, density_field, scale)

        # Calculate statistics
        stats = self._calculate_statistics(edge_index, density_field, n_points)

        return {
            "scale": scale,
            "filaments": filaments,
            "voids": voids,
            "nodes": nodes,
            "density_field": density_field,
            "connectivity": edge_index,
            "statistics": stats,
        }

    def _calculate_local_density(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local density using graph connectivity."""
        n_points = coordinates.size(0)

        # Calculate node degrees
        node_degrees = degree(edge_index[0], num_nodes=n_points)

        # Also consider spatial density
        if edge_index.shape[1] > 0:
            # Calculate average edge length per node
            edge_lengths = torch.zeros(n_points, device=coordinates.device)
            edge_counts = torch.zeros(n_points, device=coordinates.device)

            # Compute edge lengths
            src, dst = edge_index
            distances = torch.norm(coordinates[dst] - coordinates[src], dim=1)

            # Accumulate distances
            edge_lengths.scatter_add_(0, src, distances)
            edge_counts.scatter_add_(0, src, torch.ones_like(distances))

            # Average distance to neighbors (inverse for density)
            avg_distances = edge_lengths / edge_counts.clamp(min=1.0)
            spatial_density = 1.0 / (avg_distances + 1e-6)

            # Combine degree and spatial density
            density = node_degrees * spatial_density
            density = density / density.max().clamp(min=1.0)
        else:
            density = torch.zeros(n_points, device=coordinates.device)

        return density

    def _calculate_anisotropy(
        self, coordinates: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate local anisotropy using eigenvalue analysis."""
        n_points = coordinates.size(0)
        device = coordinates.device

        anisotropy = torch.zeros(n_points, device=device)

        # Process in batches for memory efficiency
        batch_size = min(1000, n_points)

        for start_idx in range(0, n_points, batch_size):
            end_idx = min(start_idx + batch_size, n_points)

            for i in range(start_idx, end_idx):
                # Find neighbors
                neighbor_mask = edge_index[0] == i
                if neighbor_mask.sum() > 2:
                    neighbors = edge_index[1, neighbor_mask]

                    # Calculate direction vectors
                    directions = coordinates[neighbors] - coordinates[i]

                    # Compute covariance matrix
                    cov = torch.mm(directions.T, directions) / len(directions)

                    # Compute eigenvalues
                    try:
                        eigenvalues = torch.linalg.eigvalsh(cov)

                        # Anisotropy: ratio of largest to smallest eigenvalue
                        anisotropy[i] = eigenvalues[-1] / (eigenvalues[0] + 1e-6)
                    except Exception:
                        anisotropy[i] = 1.0

        # Normalize anisotropy
        anisotropy = torch.clamp(anisotropy, 0.0, 10.0) / 10.0

        return anisotropy

    def _identify_filaments(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        density_field: torch.Tensor,
        scale: float,
    ) -> Dict:
        """Identify filamentary structures using anisotropy and density."""
        n_points = coordinates.size(0)
        device = coordinates.device

        # Calculate local anisotropy
        anisotropy = self._calculate_anisotropy(coordinates, edge_index)

        # Filament criteria: high anisotropy, medium density
        filament_mask = (
            (anisotropy > 0.6) & (density_field > 0.3) & (density_field < 0.8)
        )

        # Find connected filament components
        filament_indices = torch.where(filament_mask)[0]

        if len(filament_indices) > 0:
            # Create subgraph of filament points
            filament_edges = []
            for i in range(edge_index.shape[1]):
                if filament_mask[edge_index[0, i]] and filament_mask[edge_index[1, i]]:
                    filament_edges.append(i)

            if filament_edges:
                filament_edge_index = edge_index[:, filament_edges]
                filament_labels = self._connected_components_pyg(
                    filament_edge_index, n_points
                )
            else:
                filament_labels = torch.full(
                    (n_points,), -1, dtype=torch.long, device=device
                )
        else:
            filament_labels = torch.full(
                (n_points,), -1, dtype=torch.long, device=device
            )

        # Filter and validate filaments
        valid_filaments = self._filter_filaments(
            coordinates, filament_labels, scale, min_length=2.0 * scale
        )

        return {
            "mask": filament_mask,
            "labels": filament_labels,
            "anisotropy": anisotropy,
            "valid_filaments": valid_filaments,
            "n_filaments": len(valid_filaments),
        }

    def _identify_voids(
        self, coordinates: torch.Tensor, density_field: torch.Tensor, threshold: float
    ) -> Dict:
        """Identify void regions using density field."""
        # Low density regions
        void_mask = density_field < threshold

        # Extract void coordinates
        void_indices = torch.where(void_mask)[0]
        void_coords = coordinates[void_indices]

        if void_coords.size(0) > 0:
            # Use voxel grid for void clustering
            voxel_cluster = voxel_grid(
                void_coords,
                batch=None,
                size=threshold * 10.0,  # Adaptive voxel size
            )

            # Convert voxel clusters to void labels
            unique_voxels = torch.unique(voxel_cluster)
            void_labels = torch.zeros_like(voxel_cluster)

            for i, voxel_id in enumerate(unique_voxels):
                void_labels[voxel_cluster == voxel_id] = i
        else:
            void_labels = torch.empty(0, dtype=torch.long, device=coordinates.device)

        return {
            "mask": void_mask,
            "indices": void_indices,
            "coordinates": void_coords,
            "labels": void_labels,
            "n_voids": len(torch.unique(void_labels)) if len(void_labels) > 0 else 0,
        }

    def _identify_nodes(
        self,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        density_field: torch.Tensor,
        scale: float,
    ) -> Dict:
        """Identify node regions (high-density clusters)."""
        # Calculate local anisotropy
        anisotropy = self._calculate_anisotropy(coordinates, edge_index)

        # Node criteria: high density, low anisotropy (spherical)
        node_mask = (density_field > 0.7) & (anisotropy < 0.4)

        # Find connected node components
        node_indices = torch.where(node_mask)[0]

        if len(node_indices) > 0:
            # Extract node subgraph
            node_edges = []
            for i in range(edge_index.shape[1]):
                if node_mask[edge_index[0, i]] and node_mask[edge_index[1, i]]:
                    node_edges.append(i)

            if node_edges:
                node_edge_index = edge_index[:, node_edges]
                node_labels = self._connected_components_pyg(
                    node_edge_index, coordinates.size(0)
                )
            else:
                node_labels = torch.full(
                    (coordinates.size(0),),
                    -1,
                    dtype=torch.long,
                    device=coordinates.device,
                )
        else:
            node_labels = torch.full(
                (coordinates.size(0),), -1, dtype=torch.long, device=coordinates.device
            )

        # Count valid nodes
        valid_nodes = torch.unique(node_labels[node_labels >= 0])

        return {
            "mask": node_mask,
            "indices": node_indices,
            "labels": node_labels,
            "n_nodes": len(valid_nodes),
            "anisotropy": anisotropy[node_mask]
            if node_mask.any()
            else torch.tensor([]),
        }

    def _connected_components_pyg(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """Find connected components using PyG utilities."""
        device = edge_index.device

        # Union-Find algorithm for connected components
        parent = torch.arange(num_nodes, device=device)

        # Process edges
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]

            # Find roots
            root_src = src
            while parent[root_src] != root_src:
                parent[root_src] = parent[parent[root_src]]  # Path compression
                root_src = parent[root_src]

            root_dst = dst
            while parent[root_dst] != root_dst:
                parent[root_dst] = parent[parent[root_dst]]  # Path compression
                root_dst = parent[root_dst]

            # Union
            if root_src != root_dst:
                parent[root_src] = root_dst

        # Final path compression and labeling
        labels = torch.zeros(num_nodes, dtype=torch.long, device=device)
        component_map = {}
        next_label = 0

        for i in range(num_nodes):
            # Find root with path compression
            root = i
            while parent[root] != root:
                parent[root] = parent[parent[root]]
                root = parent[root]

            # Assign label
            if root not in component_map:
                component_map[root] = next_label
                next_label += 1
            labels[i] = component_map[root]

        return labels

    def _filter_filaments(
        self,
        coordinates: torch.Tensor,
        labels: torch.Tensor,
        scale: float,
        min_length: float,
    ) -> List[Dict]:
        """Filter filaments by physical properties."""
        valid_filaments = []

        unique_labels = torch.unique(labels[labels >= 0])

        for label in unique_labels:
            filament_mask = labels == label
            filament_coords = coordinates[filament_mask]

            if len(filament_coords) < 3:  # Need at least 3 points
                continue

            # Calculate filament properties
            properties = self._calculate_filament_properties(filament_coords)

            # Apply filters
            if properties["length"] > min_length:
                valid_filaments.append(
                    {
                        "label": label.item(),
                        "coordinates": filament_coords,
                        "n_points": len(filament_coords),
                        **properties,
                    }
                )

        return valid_filaments

    def _calculate_filament_properties(self, coordinates: torch.Tensor) -> Dict:
        """Calculate physical properties of a filament."""
        # Filament length (approximate using max span)
        pairwise_distances = torch.cdist(coordinates, coordinates)
        max_distance = pairwise_distances.max().item()

        # Filament thickness (average nearest neighbor distance)
        if len(coordinates) > 1:
            # Set diagonal to inf to exclude self-distances
            pairwise_distances.fill_diagonal_(float("inf"))
            min_distances = pairwise_distances.min(dim=1)[0]
            avg_thickness = min_distances.mean().item()
        else:
            avg_thickness = 0.0

        # Filament straightness (ratio of end-to-end distance to total length)
        if len(coordinates) > 2:
            # Simple approximation using PCA
            centered = coordinates - coordinates.mean(dim=0)
            _, _, v = torch.svd(centered)

            # Project onto principal axis
            projections = torch.mm(centered, v[:, 0:1])
            straightness = (projections.max() - projections.min()) / max_distance
            straightness = straightness.item()
        else:
            straightness = 1.0

        return {
            "length": max_distance,
            "thickness": avg_thickness,
            "straightness": straightness,
        }

    def _calculate_statistics(
        self, edge_index: torch.Tensor, density_field: torch.Tensor, num_nodes: int
    ) -> Dict:
        """Calculate graph and density statistics."""
        stats = {}

        # Graph statistics
        if edge_index.shape[1] > 0:
            degrees = degree(edge_index[0], num_nodes=num_nodes)
            stats["degree"] = {
                "mean": degrees.mean().item(),
                "std": degrees.std().item(),
                "min": degrees.min().item(),
                "max": degrees.max().item(),
            }

            stats["graph"] = {
                "num_nodes": num_nodes,
                "num_edges": edge_index.shape[1],
                "avg_degree": degrees.mean().item(),
                "has_self_loops": contains_self_loops(edge_index),
            }
        else:
            stats["degree"] = {"mean": 0, "std": 0, "min": 0, "max": 0}
            stats["graph"] = {
                "num_nodes": num_nodes,
                "num_edges": 0,
                "avg_degree": 0,
                "has_self_loops": False,
            }

        # Density statistics
        stats["density"] = {
            "mean": density_field.mean().item(),
            "std": density_field.std().item(),
            "min": density_field.min().item(),
            "max": density_field.max().item(),
        }

        return stats

    def _combine_multi_scale_results(self, results: Dict) -> Dict:
        """Combine results from multiple scales into summary statistics."""
        combined = {
            "filaments": [],
            "voids": [],
            "nodes": [],
            "statistics": {},
        }

        # Collect all filaments across scales
        all_filament_lengths = []
        for scale_name, scale_result in results.items():
            scale = scale_result["scale"]

            # Add scale info to each filament
            for filament in scale_result["filaments"]["valid_filaments"]:
                filament_with_scale = filament.copy()
                filament_with_scale["scale"] = scale
                combined["filaments"].append(filament_with_scale)
                all_filament_lengths.append(filament["length"])

        # Aggregate statistics
        if all_filament_lengths:
            combined["statistics"]["filament_lengths"] = {
                "mean": torch.tensor(all_filament_lengths).mean().item(),
                "std": torch.tensor(all_filament_lengths).std().item(),
                "min": min(all_filament_lengths),
                "max": max(all_filament_lengths),
                "total": len(all_filament_lengths),
            }

        # Aggregate void counts
        total_voids = sum(r["voids"]["n_voids"] for r in results.values())
        combined["statistics"]["total_voids"] = total_voids

        # Aggregate node counts
        total_nodes = sum(r["nodes"]["n_nodes"] for r in results.values())
        combined["statistics"]["total_nodes"] = total_nodes

        # Multi-scale density statistics
        all_densities = []
        for scale_result in results.values():
            all_densities.append(scale_result["density_field"])

        if all_densities:
            combined_density = torch.cat(all_densities)
            combined["statistics"]["multi_scale_density"] = {
                "mean": combined_density.mean().item(),
                "std": combined_density.std().item(),
                "min": combined_density.min().item(),
                "max": combined_density.max().item(),
            }

        return combined


def analyze_cosmic_web(
    coordinates: Union[torch.Tensor, SpatialTensorDict],
    density_field: Optional[torch.Tensor] = None,
    scales: Optional[List[float]] = None,
    density_threshold: float = 0.5,
    use_adaptive_sampling: bool = True,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Top-level function for cosmic web analysis, for import convenience.
    Instantiates ScalableCosmicWebAnalyzer and calls its analyze_cosmic_web method.
    """
    analyzer = ScalableCosmicWebAnalyzer(device=device)
    return analyzer.analyze_cosmic_web(
        coordinates=coordinates,
        density_field=density_field,
        scales=scales,
        density_threshold=density_threshold,
        use_adaptive_sampling=use_adaptive_sampling,
    )


def analyze_cosmic_web_50m(
    coordinates: Union[torch.Tensor, SpatialTensorDict],
    density_field: Optional[torch.Tensor] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict:
    """
    Optimized cosmic web analysis for very large datasets (50M+ points).

    Uses adaptive sampling and efficient memory management for massive
    astronomical datasets.

    Args:
        coordinates: Galaxy/particle coordinates
        density_field: Optional pre-computed density field
        device: Computation device

    Returns:
        Cosmic web analysis results with multi-scale structure
    """
    analyzer = ScalableCosmicWebAnalyzer(device=device, max_points_per_batch=50000)

    return analyzer.analyze_cosmic_web(
        coordinates=coordinates,
        density_field=density_field,
        scales=[1.0, 2.0, 5.0, 10.0, 20.0],  # Extended scale range
        density_threshold=0.3,
        use_adaptive_sampling=True,
    )


def quick_cosmic_web_analysis(
    coordinates: Union[torch.Tensor, SpatialTensorDict],
    scale: float = 5.0,
) -> Dict:
    """
    Quick single-scale cosmic web analysis for interactive use.

    Args:
        coordinates: Input coordinates
        scale: Analysis scale in Mpc

    Returns:
        Single-scale analysis results
    """
    analyzer = ScalableCosmicWebAnalyzer()

    return analyzer.analyze_cosmic_web(
        coordinates=coordinates,
        scales=[scale],
        use_adaptive_sampling=False,
    )
