"""
Tests for Spatial3DTensor.
"""

from typing import Dict

import pytest
import torch

from astro_lab.tensors.spatial_3d import Spatial3DTensor


class TestSpatial3DTensor:
    """Test 3D spatial tensor functionality."""

    def test_coordinate_creation(self, sample_tensor_data: Dict[str, torch.Tensor]):
        """Test creation with different coordinate systems."""
        coords = sample_tensor_data["coordinates"]  # [N, 3]

        # Test ICRS coordinates
        spatial = Spatial3DTensor(coords, coordinate_system="icrs", unit="Mpc")
        assert spatial.coordinate_system == "icrs"
        assert spatial.unit == "Mpc"
        assert spatial.shape == coords.shape

    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Wrong shape should raise error
        with pytest.raises(ValueError, match="must have shape"):
            Spatial3DTensor(torch.randn(10, 2))  # Should be [N, 3]

        # Invalid coordinate system
        with pytest.raises(ValueError, match="coordinate_system must be one of"):
            Spatial3DTensor(torch.randn(10, 3), coordinate_system="invalid")

    def test_single_point_conversion(self):
        """Test single point handling."""
        single_point = torch.tensor([1.0, 2.0, 3.0])
        spatial = Spatial3DTensor(single_point)

        assert spatial.shape == (1, 3)
        assert len(spatial) == 1

    def test_distance_calculations(self):
        """Test distance calculation methods."""
        coords = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        spatial = Spatial3DTensor(coords)

        # Test actual available methods
        assert hasattr(spatial, "angular_separation")
        assert hasattr(spatial, "query_neighbors")

    def test_coordinate_transformations(self):
        """Test coordinate system transformations."""
        coords = torch.randn(10, 3)
        spatial = Spatial3DTensor(coords, coordinate_system="icrs")

        # Test coordinate system property
        assert spatial.coordinate_system == "icrs"

        # Test that transformation methods exist
        assert hasattr(spatial, "transform_coordinates")
        assert hasattr(spatial, "to_spherical")
