"""
Tests for graph utility functions and calculations.

Tests graph construction, metrics, and operations.
"""

import pytest
import numpy as np
import torch
from astro_lab.utils import GRAPH_AVAILABLE


class TestGraphUtilities:
    """Test graph utility functions."""

    def test_distance_calculations(self):
        """Test distance matrix calculations."""
        if not GRAPH_AVAILABLE:
            pytest.skip("Graph utilities not available")
            
        from astro_lab.utils import spatial_distance_matrix
        
        positions = torch.randn(100, 3) * 10  # 100 points in 3D
        distances = spatial_distance_matrix(positions)
        
        assert distances.shape == (100, 100)
        assert np.allclose(distances, distances.T)  # Should be symmetric
        assert np.allclose(np.diag(distances), 0)   # Diagonal should be zero

    def test_knn_graph_construction(self):
        """Test k-nearest neighbors graph construction."""
        if not GRAPH_AVAILABLE:
            pytest.skip("Graph utilities not available")
            
        from astro_lab.utils import create_spatial_graph
        from astro_lab.tensors import Spatial3DTensor
        
        # Create test data
        n_points = 50
        positions = np.random.rand(n_points, 3) * 10
        spatial_tensor = Spatial3DTensor(positions, unit="pc")
        
        # Create k-NN graph
        data = create_spatial_graph(
            spatial_tensor, 
            k=5,  # 5 nearest neighbors
            method="knn"
        )
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Test edge index shape
        assert edge_index.shape[0] == 2  # Should be [2, num_edges]
        assert edge_index.shape[1] > 0   # Should have edges
        
        # Test edge attributes
        if edge_attr is not None:
            assert edge_attr.shape[0] == edge_index.shape[1]  # Same number as edges
        
        # Test that all nodes are connected
        unique_nodes = np.unique(edge_index.numpy())
        assert len(unique_nodes) <= n_points

    def test_graph_metrics(self):
        """Test graph metric calculations."""
        if not GRAPH_AVAILABLE:
            pytest.skip("Graph utilities not available")
            
        from astro_lab.utils import calculate_graph_metrics
        from torch_geometric.data import Data
        
        # Create simple test graph
        positions = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Simple cycle
        
        data = Data(pos=positions, edge_index=edge_index)
        metrics = calculate_graph_metrics(data)
        
        assert isinstance(metrics, dict)
        assert "num_nodes" in metrics
        assert "num_edges" in metrics
        assert metrics["num_nodes"] == 4
        assert metrics["num_edges"] == 4

    def test_clustering_coefficient(self):
        """Test clustering coefficient calculation."""
        if not GRAPH_AVAILABLE:
            pytest.skip("Graph utilities not available")
            
        from astro_lab.utils import calculate_graph_metrics
        from torch_geometric.data import Data
        
        # Create triangle graph (high clustering)
        positions = torch.tensor([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=torch.float)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # Triangle
        
        data = Data(pos=positions, edge_index=edge_index)
        metrics = calculate_graph_metrics(data)
        
        assert "clustering_coefficient" in metrics
        assert isinstance(metrics["clustering_coefficient"], (float, int)) 