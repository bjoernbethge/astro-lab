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
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data)
        
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
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data)
        
        # Test distance to origin
        distances = tensor.distance_to_origin()
        expected = torch.tensor([0.0, 1.0, 1.0, 1.0])
        
        torch.testing.assert_close(distances, expected, rtol=1e-6, atol=1e-6)

    def test_query_neighbors(self):
        """Test neighbor querying."""
        # Create test coordinates
        coords = torch.randn(50, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data)
        
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
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data)
        
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
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data, coordinate_system="galactic", unit="kpc")
        
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
        assert tensor.unit == "kpc"  # Default unit

    def test_angular_separation(self):
        """Test angular separation calculation."""
        # Create two sets of coordinates
        coords1 = torch.randn(10, 3)
        coords2 = torch.randn(10, 3)
        
        x1, y1, z1 = coords1[:, 0], coords1[:, 1], coords1[:, 2]
        x2, y2, z2 = coords2[:, 0], coords2[:, 1], coords2[:, 2]
        
        data1 = torch.stack([x1, y1, z1], dim=-1)
        tensor1 = Spatial3DTensor(data1)
        data2 = torch.stack([x2, y2, z2], dim=-1)
        tensor2 = Spatial3DTensor(data2)
        
        # Calculate angular separation
        separation = tensor1.angular_separation(tensor2)
        
        assert isinstance(separation, torch.Tensor)
        assert separation.shape == (10,)
        assert torch.all(separation >= 0)  # separation should be positive

    def test_cone_search(self):
        """Test cone search functionality."""
        coords = torch.randn(100, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        tensor = Spatial3DTensor(data)
        
        # Perform cone search
        center = torch.tensor([0.0, 0.0, 1.0])
        radius_deg = 10.0
        
        result = tensor.cone_search(center, radius_deg)
        
        assert isinstance(result, torch.Tensor)
        assert len(result) <= 100  # Should not return more points than input
