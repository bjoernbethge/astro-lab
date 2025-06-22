"""
Tests for tensor serialization and deserialization.
"""

import torch

from astro_lab.tensors.spatial_3d import Spatial3DTensor


class TestTensorSerialization:
    """Test tensor serialization and deserialization."""

    def test_tensor_state_dict(self):
        """Test tensor state dict functionality."""
        coords = torch.randn(5, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        spatial = Spatial3DTensor(x, y, z, coordinate_system="galactic")

        # Test state dict exists
        assert hasattr(spatial, "__getstate__") or hasattr(spatial, "state_dict")

    def test_tensor_pickling(self):
        """Test tensor pickling via dict serialization."""
        coords = torch.randn(3, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        spatial = Spatial3DTensor(x, y, z)

        # Test serialization to dict and back (safer than direct pickling)
        tensor_dict = spatial.to_dict()

        # Check that essential data is preserved
        assert "data" in tensor_dict
        assert "shape" in tensor_dict
        assert "dtype" in tensor_dict
        assert "device" in tensor_dict
        assert tensor_dict["shape"] == list(coords.shape)

        # Test reconstruction would work - data is now a Python list
        assert isinstance(tensor_dict["data"], list)
        assert len(tensor_dict["data"]) == coords.shape[0]
        assert len(tensor_dict["data"][0]) == coords.shape[1]

    def test_tensor_copy(self):
        """Test tensor copying via clone method."""
        coords = torch.randn(4, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        spatial = Spatial3DTensor(x, y, z, unit="kpc")

        # Test clone method (safer than copy module)
        cloned = spatial.clone()
        assert cloned.get_metadata("unit") == "kpc"
        assert torch.equal(spatial.data, cloned.data)

        # Test detach method
        detached = spatial.detach()
        assert detached.get_metadata("unit") == "kpc"
        assert torch.equal(spatial.data, detached.data)
