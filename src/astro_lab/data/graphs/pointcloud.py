"""
Point Cloud Graph Building for Astronomical Data - PyTorch Geometric
===================================================================

Simplified point cloud processing using PyTorch Geometric transforms.
"""

from typing import Optional

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from torch_geometric.transforms import GridSampling

from astro_lab.tensors import SpatialTensorDict


class PointCloudGraphBuilder:
    """Simplified point cloud graph builder using PyTorch Geometric."""

    def __init__(
        self,
        k_neighbors: int = 16,
        radius: float = 10.0,
        use_hierarchical: bool = False,
        grid_voxel_size: float = 2.0,
        device: Optional[torch.device] = None,
    ):
        """Initialize point cloud graph builder."""
        self.k_neighbors = k_neighbors
        self.radius = radius
        self.use_hierarchical = use_hierarchical
        self.grid_voxel_size = grid_voxel_size
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Setup transforms
        if self.use_hierarchical:
            self.grid_sampler = GridSampling(size=self.grid_voxel_size)

    def build_from_spatial(self, spatial_tensor: SpatialTensorDict) -> Data:
        """Build point cloud graph from SpatialTensorDict."""
        # Extract positions
        positions = spatial_tensor["coordinates"].to(self.device)

        # Create initial data object
        data = Data(pos=positions, num_nodes=positions.shape[0])

        # Apply hierarchical sampling if requested
        if self.use_hierarchical and positions.shape[0] > 1000:
            data = self.grid_sampler(data)
            positions = data.pos

        # Build k-NN graph
        edge_index = knn_graph(positions, k=self.k_neighbors, batch=None)
        data.edge_index = edge_index

        # Use positions as features
        data.x = positions

        return data


def create_pointcloud_graph(
    spatial_tensor: SpatialTensorDict,
    k_neighbors: int = 16,
    use_hierarchical: bool = False,
    **kwargs,
) -> Data:
    """Create point cloud graph from spatial tensor."""
    builder = PointCloudGraphBuilder(
        k_neighbors=k_neighbors,
        use_hierarchical=use_hierarchical,
        **kwargs,
    )
    return builder.build_from_spatial(spatial_tensor)


def create_multiscale_pointcloud_graph(
    spatial_tensor: SpatialTensorDict,
    scales: list = [5.0, 10.0, 20.0],
    **kwargs,
) -> list:
    """Create multi-scale point cloud graphs."""
    graphs = []
    for scale in scales:
        graph = create_pointcloud_graph(
            spatial_tensor,
            radius=scale,
            **kwargs,
        )
        graphs.append(graph)
    return graphs


def create_point_cloud_graph(
    coordinates: torch.Tensor,
    k_neighbors: int = 8,
    max_radius: Optional[float] = None,
    device: str = "cuda",
) -> Data:
    """
    Create graph from point cloud using k-nearest neighbors.

    Args:
        coordinates: Point coordinates [N, 3]
        k_neighbors: Number of neighbors to connect
        max_radius: Maximum connection radius (optional)
        device: Device for computation

    Returns:
        PyG Data object with edge_index
    """
    # Move to device
    coordinates.to(device)
