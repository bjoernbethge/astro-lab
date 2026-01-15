import torch
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from .RadiusSampler import RadiusSampler


class AdaptiveRadiusSampler(RadiusSampler):
    """Adaptive radius sampler that adjusts radius based on local density."""

    def __init__(
        self,
        base_radius: float = 10.0,
        target_neighbors: int = 20,
        min_radius: float = 1.0,
        max_radius: float = 100.0,
        **kwargs,
    ):
        """Initialize adaptive radius sampler.

        Args:
            base_radius: Starting radius
            target_neighbors: Target number of neighbors
            min_radius: Minimum allowed radius
            max_radius: Maximum allowed radius
            **kwargs: Additional config
        """
        super().__init__(radius=base_radius, **kwargs)
        self.target_neighbors = target_neighbors
        self.min_radius = min_radius
        self.max_radius = max_radius

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create adaptive radius graph.

        Adjusts radius per node to achieve target neighbor count.
        """
        self.validate_inputs(coordinates, features)

        # First pass with base radius

        edge_index = radius_graph(
            coordinates,
            r=self.radius,
            loop=self.loop,
            max_num_neighbors=self.max_num_neighbors,
        )

        # Count neighbors per node
        neighbor_counts = torch.bincount(edge_index[0], minlength=coordinates.size(0))

        # Identify nodes needing adjustment
        too_few = neighbor_counts < self.target_neighbors // 2
        too_many = neighbor_counts > self.target_neighbors * 2

        # Adaptive radius adjustment (simplified)
        if too_few.any() or too_many.any():
            # This is a simplified version - in practice you'd want
            # per-node radius adjustment
            avg_neighbors = neighbor_counts.float().mean()
            if avg_neighbors < self.target_neighbors:
                self.radius = min(self.radius * 1.5, self.max_radius)
            elif avg_neighbors > self.target_neighbors * 1.5:
                self.radius = max(self.radius * 0.7, self.min_radius)

            # Recreate graph with adjusted radius
            edge_index = radius_graph(
                coordinates,
                r=self.radius,
                loop=self.loop,
                max_num_neighbors=self.max_num_neighbors,
            )

        # Continue with standard processing
        return super().create_graph(coordinates, features, **kwargs)
