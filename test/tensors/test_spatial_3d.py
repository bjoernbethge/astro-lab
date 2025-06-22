"""
Tests for spatial 3D tensor operations.
"""

import pytest
import torch

from astro_lab.tensors.spatial_3d import Spatial3DTensor


class TestSpatial3DTensor:
    """Test spatial 3D tensor functionality."""

    def test_coordinate_creation(self):
        """Test coordinate tensor creation."""
        # Create test coordinates directly
        coords = torch.randn(100, 3)  # [N, 3]
        
        tensor = Spatial3DTensor(coords)
        
        assert tensor.shape == (100, 3)
        assert tensor.cartesian.shape == (100, 3)
        assert torch.equal(tensor.cartesian, coords)

    def test_distance_calculation(self):
        """Test distance calculations between points."""
        # Create test coordinates
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        
        tensor = Spatial3DTensor(coords)
        
        # Test distance to origin
        distances = tensor.distance_to_origin()
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        
        torch.testing.assert_close(distances, expected, rtol=1e-6, atol=1e-6)

    def test_query_neighbors(self):
        """Test neighbor querying."""
        # Create test coordinates
        coords = torch.randn(50, 3)
        tensor = Spatial3DTensor(coords)
        
        # Query neighbors
        query_point = torch.randn(3)
        neighbors, distances = tensor.query_neighbors(query_point, radius=2.0)
        
        assert isinstance(neighbors, torch.Tensor)
        assert isinstance(distances, torch.Tensor)
        assert len(neighbors) == len(distances)

    def test_spherical_conversion(self):
        """Test spherical coordinate conversion."""
        # Create test coordinates
        coords = torch.randn(20, 3)
        tensor = Spatial3DTensor(coords)
        
        # Test spherical transformation
        ra, dec, distance = tensor.to_spherical()
        
        assert isinstance(ra, torch.Tensor)
        assert isinstance(dec, torch.Tensor)
        assert isinstance(distance, torch.Tensor)
        assert ra.shape == (20,)
        assert dec.shape == (20,)
        assert distance.shape == (20,)
        assert torch.all(distance >= 0)  # distance should be positive

    def test_coordinate_system_properties(self):
        """Test coordinate system properties."""
        coords = torch.randn(10, 3)
        tensor = Spatial3DTensor(coords, coordinate_system="galactic", unit="kpc")
        
        assert tensor.coordinate_system == "galactic"
        assert tensor.unit == "kpc"
        assert tensor.epoch == 2000.0

    def test_from_spherical_creation(self):
        """Test creation from spherical coordinates."""
        ra = torch.tensor([0.0, 90.0, 180.0])  # degrees
        dec = torch.tensor([0.0, 45.0, -45.0])  # degrees
        distance = torch.tensor([1.0, 2.0, 3.0])  # Mpc
        
        tensor = Spatial3DTensor.from_spherical(ra, dec, distance)
        
        assert tensor.shape == (3, 3)
        assert tensor.coordinate_system == "icrs"
        assert tensor.unit == "Mpc"

    def test_angular_separation(self):
        """Test angular separation calculation."""
        # Create two sets of coordinates
        coords1 = torch.randn(10, 3)
        coords2 = torch.randn(10, 3)
        
        tensor1 = Spatial3DTensor(coords1)
        tensor2 = Spatial3DTensor(coords2)
        
        # Calculate angular separation
        separation = tensor1.angular_separation(tensor2)
        
        assert isinstance(separation, torch.Tensor)
        assert separation.shape == (10,)
        assert torch.all(separation >= 0)  # separation should be positive

    def test_cone_search(self):
        """Test cone search functionality."""
        coords = torch.randn(100, 3)
        tensor = Spatial3DTensor(coords)
        
        # Perform cone search
        center = torch.tensor([0.0, 0.0, 1.0])
        radius_deg = 10.0
        
        result = tensor.cone_search(center, radius_deg)
        
        assert isinstance(result, torch.Tensor)
        assert len(result) <= 100  # Should not return more points than input
