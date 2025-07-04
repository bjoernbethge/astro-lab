"""GraphSAINT samplers for large-scale astronomical graphs.

Implements GraphSAINT sampling strategies for scalable GNN training.
"""

from typing import Any, Dict, Optional

import torch
from torch_geometric.data import Data
from torch_geometric.loader import (
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
)

from .base import AstroLabSampler, SpatialSamplerMixin


class GraphSAINTSampler(AstroLabSampler, SpatialSamplerMixin):
    """Base class for GraphSAINT sampling strategies."""

    def __init__(
        self,
        batch_size: int = 1000,
        num_steps: int = 5,
        sample_coverage: int = 10,
        walk_length: int = 2,
        num_workers: int = 0,
        **kwargs,
    ):
        """Initialize GraphSAINT sampler.

        Args:
            batch_size: Number of nodes/edges per batch
            num_steps: Number of iterations per epoch
            sample_coverage: How many times to sample each node
            walk_length: Length of random walk (for RW sampler)
            num_workers: Number of data loading workers
            **kwargs: Additional config
        """
        super().__init__(
            {
                "batch_size": batch_size,
                "num_steps": num_steps,
                "sample_coverage": sample_coverage,
                "walk_length": walk_length,
                "num_workers": num_workers,
                **kwargs,
            }
        )
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.sample_coverage = sample_coverage
        self.walk_length = walk_length
        self.num_workers = num_workers

    def create_graph(
        self, coordinates: torch.Tensor, features: torch.Tensor, **kwargs
    ) -> Data:
        """Create graph for GraphSAINT sampling.

        Args:
            coordinates: 3D positions [N, 3]
            features: Node features [N, F]
            **kwargs: Additional data

        Returns:
            PyG Data object
        """
        self.validate_inputs(coordinates, features)

        # Create k-NN graph
        k = min(30, coordinates.size(0) // 20)
        edge_index = self.knn_graph(coordinates, k=k, loop=False)

        # Calculate edge features and weights
        edge_attr = self.calculate_edge_features(coordinates, edge_index)

        # Edge weights based on inverse distance for importance sampling
        distances = edge_attr[:, 0]
        edge_weight = 1.0 / (distances + 1e-6)
        edge_weight = edge_weight / edge_weight.sum()  # Normalize

        # Create Data object
        data = Data(
            x=features,
            pos=coordinates,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_weight=edge_weight,
        )

        # Add additional attributes
        for key, value in kwargs.items():
            setattr(data, key, value)

        # Add node weights for importance sampling
        node_degrees = torch.bincount(
            edge_index[0], minlength=coordinates.size(0)
        ).float()
        node_weight = node_degrees / node_degrees.sum()
        data.node_weight = node_weight

        # Update stats
        self.sampling_stats.update(
            {
                "num_nodes": coordinates.size(0),
                "num_edges": edge_index.size(1),
                "avg_degree": edge_index.size(1) / coordinates.size(0),
            }
        )

        return data


class GraphSAINTNodeSamplerWrapper(GraphSAINTSampler):
    """GraphSAINT node-based sampling for astronomical graphs."""

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> GraphSAINTNodeSampler:
        """Create GraphSAINT node sampler.

        Args:
            data: PyG Data object
            batch_size: Override default batch size
            shuffle: Whether to shuffle
            **kwargs: Additional DataLoader args

        Returns:
            GraphSAINTNodeSampler instance
        """
        batch_size = batch_size or self.batch_size

        return GraphSAINTNodeSampler(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            num_steps=self.num_steps,
            sample_coverage=self.sample_coverage,
            save_dir=None,  # Don't save subgraphs
            log=False,
            **kwargs,
        )


class GraphSAINTEdgeSamplerWrapper(GraphSAINTSampler):
    """GraphSAINT edge-based sampling for astronomical graphs."""

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> GraphSAINTEdgeSampler:
        """Create GraphSAINT edge sampler.

        Args:
            data: PyG Data object
            batch_size: Override default batch size
            shuffle: Whether to shuffle
            **kwargs: Additional DataLoader args

        Returns:
            GraphSAINTEdgeSampler instance
        """
        batch_size = batch_size or self.batch_size

        return GraphSAINTEdgeSampler(
            data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            num_steps=self.num_steps,
            sample_coverage=self.sample_coverage,
            save_dir=None,
            log=False,
            **kwargs,
        )


class GraphSAINTRandomWalkSamplerWrapper(GraphSAINTSampler):
    """GraphSAINT random walk sampling for astronomical graphs."""

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ) -> GraphSAINTRandomWalkSampler:
        """Create GraphSAINT random walk sampler.

        Args:
            data: PyG Data object
            batch_size: Override default batch size
            shuffle: Whether to shuffle
            **kwargs: Additional DataLoader args

        Returns:
            GraphSAINTRandomWalkSampler instance
        """
        batch_size = batch_size or self.batch_size

        return GraphSAINTRandomWalkSampler(
            data,
            batch_size=batch_size,
            walk_length=self.walk_length,
            shuffle=shuffle,
            num_workers=self.num_workers,
            num_steps=self.num_steps,
            sample_coverage=self.sample_coverage,
            save_dir=None,
            log=False,
            **kwargs,
        )


class AdaptiveGraphSAINTSampler(GraphSAINTSampler):
    """Adaptive GraphSAINT sampler that switches strategies based on graph properties."""

    def __init__(
        self,
        batch_size: int = 1000,
        num_steps: int = 5,
        density_threshold: float = 0.01,
        **kwargs,
    ):
        """Initialize adaptive GraphSAINT sampler.

        Args:
            batch_size: Number of nodes per batch
            num_steps: Number of iterations per epoch
            density_threshold: Graph density threshold for strategy selection
            **kwargs: Additional config
        """
        super().__init__(batch_size=batch_size, num_steps=num_steps, **kwargs)
        self.density_threshold = density_threshold
        self.selected_strategy = None

    def create_dataloader(
        self,
        data: Data,
        batch_size: Optional[int] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        """Create appropriate GraphSAINT sampler based on graph properties.

        Selects:
        - Node sampler for dense graphs
        - Edge sampler for sparse graphs
        - Random walk for very sparse graphs
        """
        batch_size = batch_size or self.batch_size

        # Calculate graph density
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        max_edges = num_nodes * (num_nodes - 1)
        density = num_edges / max_edges if max_edges > 0 else 0

        # Select strategy based on density
        if density > self.density_threshold * 2:
            # Dense graph: use node sampling
            self.selected_strategy = "node"
            return GraphSAINTNodeSampler(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                num_steps=self.num_steps,
                sample_coverage=self.sample_coverage,
            )
        elif density > self.density_threshold / 2:
            # Medium density: use edge sampling
            self.selected_strategy = "edge"
            return GraphSAINTEdgeSampler(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.num_workers,
                num_steps=self.num_steps,
                sample_coverage=self.sample_coverage,
            )
        else:
            # Sparse graph: use random walk
            self.selected_strategy = "random_walk"
            return GraphSAINTRandomWalkSampler(
                data,
                batch_size=batch_size,
                walk_length=self.walk_length,
                shuffle=shuffle,
                num_workers=self.num_workers,
                num_steps=self.num_steps,
                sample_coverage=self.sample_coverage,
            )

    def get_sampling_info(self) -> Dict[str, Any]:
        """Get sampling information including selected strategy."""
        info = super().get_sampling_info()
        info["selected_strategy"] = self.selected_strategy
        info["density_threshold"] = self.density_threshold
        return info
