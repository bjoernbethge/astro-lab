"""
Concrete Graph Builders
======================

Implementation of specific graph building strategies.
"""

from typing import Optional, Union

import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, radius_graph

from astro_lab.tensors import SurveyTensorDict

from .base import BaseGraphBuilder, GraphConfig


class KNNGraphBuilder(BaseGraphBuilder):
    """K-Nearest Neighbors graph builder."""

    def __init__(self, k_neighbors: int = 8, **kwargs):
        config = GraphConfig(method="knn", k_neighbors=k_neighbors, **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build KNN graph from SurveyTensorDict."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Create KNN graph
        edge_index = knn_graph(
            coords, k=self.config.k_neighbors, batch=None, loop=self.config.self_loops
        )

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add metadata
        data.graph_type = "knn"
        data.k_neighbors = self.config.k_neighbors

        return data


class RadiusGraphBuilder(BaseGraphBuilder):
    """Radius-based graph builder."""

    def __init__(self, radius: float = 1.0, **kwargs):
        config = GraphConfig(method="radius", radius=radius, **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build radius graph from SurveyTensorDict."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Create radius graph
        edge_index = radius_graph(
            coords,
            r=self.config.radius,
            batch=None,
            loop=self.config.self_loops,
            max_num_neighbors=self.config.k_neighbors,
        )

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add metadata
        data.graph_type = "radius"
        data.radius = self.config.radius

        return data


class AstronomicalGraphBuilder(BaseGraphBuilder):
    """Astronomical graph builder with specialized logic."""

    def __init__(self, **kwargs):
        config = GraphConfig(method="astronomical", **kwargs)
        super().__init__(config)

    def build(self, survey_tensor: SurveyTensorDict) -> Data:
        """Build astronomical graph from SurveyTensorDict."""
        self.validate_input(survey_tensor)

        # Extract coordinates and features
        coords = self.extract_coordinates(survey_tensor)
        features = self.extract_features(survey_tensor)

        # Use astronomical distance metric if spherical coordinates
        if self.config.coordinate_system == "spherical":
            edge_index = self._create_spherical_graph(coords)
        else:
            # Use KNN for cartesian coordinates
            edge_index = knn_graph(
                coords,
                k=self.config.k_neighbors,
                batch=None,
                loop=self.config.self_loops,
            )

        # Create PyG Data object
        data = self.create_data_object(features, edge_index, coords)

        # Add astronomical metadata
        data.graph_type = "astronomical"
        data.coordinate_system = self.config.coordinate_system
        data.use_3d = self.config.use_3d_coordinates

        # Add survey metadata if available
        if hasattr(survey_tensor, "meta") and survey_tensor.meta:
            data.survey_name = survey_tensor.meta.get("survey_name", "unknown")

        return data

    def _create_spherical_graph(self, coords: torch.Tensor) -> torch.Tensor:
        """Create graph using spherical distance metric."""
        # Convert to spherical coordinates if needed
        if coords.size(-1) == 3:
            # Assume x, y, z cartesian - convert to spherical
            ra = torch.atan2(coords[:, 1], coords[:, 0])
            dec = torch.asin(
                torch.clamp(coords[:, 2] / torch.norm(coords, dim=-1), -1, 1)
            )
            spherical_coords = torch.stack([ra, dec], dim=-1)
        else:
            spherical_coords = coords

        # Use angular distance for spherical coordinates
        edge_index = self._angular_knn_graph(spherical_coords, self.config.k_neighbors)

        return edge_index

    def _angular_knn_graph(self, coords: torch.Tensor, k: int) -> torch.Tensor:
        """Create KNN graph using angular distance."""
        # Compute angular distances
        cos_dec = torch.cos(coords[:, 1])
        sin_dec = torch.sin(coords[:, 1])

        # Compute pairwise angular distances
        cos_dist = torch.cos(
            coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0)
        ) * cos_dec.unsqueeze(1) * cos_dec.unsqueeze(0) + sin_dec.unsqueeze(
            1
        ) * sin_dec.unsqueeze(0)

        # Clamp to avoid numerical issues
        cos_dist = torch.clamp(cos_dist, -1.0, 1.0)
        distances = torch.acos(cos_dist)

        # Get k nearest neighbors
        _, indices = torch.topk(distances, k + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # Remove self-connections

        # Create edge index
        num_nodes = coords.size(0)
        source_nodes = torch.arange(num_nodes).unsqueeze(1).expand(-1, k)
        edge_index = torch.stack([source_nodes.flatten(), indices.flatten()])

        return edge_index


# Convenience functions for easy usage
def create_knn_graph(
    survey_tensor: SurveyTensorDict, k_neighbors: int = 8, **kwargs
) -> Data:
    """Create KNN graph from SurveyTensorDict."""
    builder = KNNGraphBuilder(k_neighbors=k_neighbors, **kwargs)
    return builder.build(survey_tensor)


def create_radius_graph(
    survey_tensor: SurveyTensorDict, radius: float = 1.0, **kwargs
) -> Data:
    """Create radius graph from SurveyTensorDict."""
    builder = RadiusGraphBuilder(radius=radius, **kwargs)
    return builder.build(survey_tensor)


def create_astronomical_graph(survey_tensor: SurveyTensorDict, **kwargs) -> Data:
    """Create astronomical graph from SurveyTensorDict."""
    builder = AstronomicalGraphBuilder(**kwargs)
    return builder.build(survey_tensor)
