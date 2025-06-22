"""
Tests for tensor interoperability.
"""

import pytest
import torch

from astro_lab.tensors.photometric import PhotometricTensor
from astro_lab.tensors.spatial_3d import Spatial3DTensor


class TestTensorInteroperability:
    """Test interoperability between different tensor types."""

    def test_tensor_combination(self):
        """Test combining different tensor types."""
        # Create spatial coordinates
        coords = torch.randn(10, 3)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data)

        # Create photometry for same objects
        mags = torch.randn(10, 5)
        phot = PhotometricTensor(mags, bands=["u", "g", "r", "i", "z"])

        # Both should have same number of objects
        assert len(spatial) == len(phot)

    def test_device_consistency(self, device: torch.device):
        """Test device consistency across tensor types."""
        # Skip if CUDA not available
        if device.type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        coords = torch.randn(5, 3, device=device)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data)

        # Device comparison should check type separately if CUDA
        if device.type == "cuda":
            assert spatial.device.type == device.type
            assert spatial.data.device.type == device.type
        else:
            assert spatial.device == device
            assert spatial.data.device == device

    def test_batch_operations(self):
        """Test batch operations on tensors."""
        # Create batch of spatial coordinates
        batch_coords = torch.randn(32, 10, 3)  # 32 samples, 10 objects each

        # Process each sample
        for i in range(batch_coords.shape[0]):
            x, y, z = batch_coords[i, :, 0], batch_coords[i, :, 1], batch_coords[i, :, 2]
            data = torch.stack([x, y, z], dim=-1)
            spatial = Spatial3DTensor(data)
            assert spatial.shape == (10, 3)

    @pytest.mark.cuda
    def test_cuda_operations(self, skip_if_no_cuda):
        """Test CUDA operations if available."""
        device = torch.device("cuda")

        # Create large tensor on GPU
        coords = torch.randn(1000, 3, device=device)
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        data = torch.stack([x, y, z], dim=-1)
        spatial = Spatial3DTensor(data)

        assert spatial.device.type == device.type

        # Test operations work on GPU
        masked = spatial.apply_mask(torch.rand(1000, device=device) > 0.5)
        assert masked.device.type == device.type
