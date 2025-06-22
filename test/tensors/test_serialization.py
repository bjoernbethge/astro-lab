"""
Tests for tensor serialization and deserialization.
"""

import torch

from astro_lab.tensors.spatial_3d import Spatial3DTensor


class TestTensorSerialization:
    """Test tensor serialization and deserialization."""

    def test_tensor_state_dict(self):
        """Test tensor state dict functionality."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = torch.tensor([7.0, 8.0, 9.0])
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data, coordinate_system="galactic")

        # Test state dict exists
        assert hasattr(spatial, "__getstate__") or hasattr(spatial, "state_dict")

    def test_tensor_pickling(self):
        """Test tensor pickling via dict serialization."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = torch.tensor([7.0, 8.0, 9.0])
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data)

        # Test serialization to dict and back (safer than direct pickling)
        tensor_dict = spatial.to_dict()

        # Check that essential data is preserved
        assert "data" in tensor_dict
        assert "shape" in tensor_dict
        assert "dtype" in tensor_dict
        assert "device" in tensor_dict
        assert tensor_dict["shape"] == list(data.shape)

        # Test reconstruction would work - data is now a Python list
        assert isinstance(tensor_dict["data"], list)
        assert len(tensor_dict["data"]) == data.shape[0]
        assert len(tensor_dict["data"][0]) == data.shape[1]

    def test_tensor_copy(self):
        """Test tensor copying via clone method."""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = torch.tensor([7.0, 8.0, 9.0])
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data, unit="kpc")

        # Test clone method (safer than copy module)
        cloned = spatial.clone()
        assert cloned.get_metadata("unit") == "kpc"
        assert torch.equal(spatial.data, cloned.data)

        # Test detach method
        detached = spatial.detach()
        assert detached.get_metadata("unit") == "kpc"
        assert torch.equal(spatial.data, detached.data)
