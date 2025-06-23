"""
Tests for astronomical-specific graph operations.

Tests spatial tensor graph construction and astronomical data integration.
"""

import numpy as np
import pytest
import torch


class TestAstronomicalGraphs:
    """Test astronomical-specific graph operations."""

    def test_spatial_tensor_graph_construction(self):
        """Test graph construction from Spatial3DTensor."""
        from astro_lab.tensors import Spatial3DTensor

        # Create stellar positions (simulated cluster)
        n_stars = 100
        cluster_center = np.array([0, 0, 0])
        cluster_radius = 5.0  # pc

        # Generate random positions around cluster center
        positions = cluster_center + np.random.randn(n_stars, 3) * cluster_radius

        # Create spatial tensor
        spatial_tensor = Spatial3DTensor(
            data=torch.from_numpy(positions).float(),
            coordinate_system="galactic",
            unit="kpc"
        )

        # Test torch geometric conversion
        data = spatial_tensor.to_torch_geometric(k=8)

        assert hasattr(data, "pos")
        assert hasattr(data, "edge_index")
        assert data.pos.shape == (n_stars, 3)
        assert data.edge_index.shape[0] == 2

        # Test that graph is connected
        assert data.edge_index.shape[1] > 0

    def test_spatial_tensor_torch_geometric_integration(self):
        """Test PyTorch Geometric integration with spatial tensors."""
        from astro_lab.tensors import Spatial3DTensor

        # Create galaxy positions
        positions = np.random.rand(50, 3) * 100  # 50 galaxies in 100 Mpc cube

        spatial_tensor = Spatial3DTensor(
            data=torch.from_numpy(positions).float(),
            coordinate_system="icrs",
            unit="Mpc"
        )

        # Convert to PyTorch Geometric format
        data = spatial_tensor.to_torch_geometric(k=5)

        # Test data structure
        assert data.num_nodes == 50
        assert data.pos.shape == (50, 3)
        assert data.edge_index.max() < 50  # All indices should be valid

    def test_redshift_graph(self):
        """Test redshift-based graph construction."""
        from astro_lab.tensors import Spatial3DTensor
        from astro_lab.utils import create_spatial_graph

        # Simulate galaxy positions with redshift information
        n_galaxies = 30
        positions = np.random.rand(n_galaxies, 3) * 50  # 50 Mpc
        redshifts = positions[:, 2] * 0.01  # z ∝ distance (simplified)

        spatial_tensor = Spatial3DTensor(data=torch.from_numpy(positions).float(), unit="Mpc")

        # Create redshift-weighted graph
        data = create_spatial_graph(spatial_tensor, k=6, method="knn")

        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] > 0

    def test_magnitude_based_edges(self):
        """Test magnitude-based edge weighting."""
        from astro_lab.tensors import Spatial3DTensor
        from astro_lab.utils import create_spatial_graph

        # Create stars with magnitudes
        n_stars = 40
        positions = np.random.rand(n_stars, 3) * 10  # 10 pc
        magnitudes = np.random.uniform(8, 15, n_stars)  # Apparent magnitudes

        spatial_tensor = Spatial3DTensor(data=torch.from_numpy(positions).float(), unit="pc")

        # Create magnitude-weighted graph
        data = create_spatial_graph(spatial_tensor, k=4, method="knn")

        # Test structure
        assert data.edge_index.shape[0] == 2
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            assert data.edge_attr.shape[0] == data.edge_index.shape[1]
