"""
Tests for performance-related utilities.

Tests memory management, batch processing, and GPU operations.
"""

import pytest
import torch
from astro_lab.data.core import get_optimal_device, get_optimal_batch_size


class TestPerformanceUtilities:
    """Test performance-related utilities (device-agnostic)."""

    def test_memory_efficient_operations(self):
        """Test memory-efficient tensor operations (device-agnostic)."""
        from astro_lab.tensors import Spatial3DTensor
        device = get_optimal_device()
        # Create large dataset on device
        n_objects = 10000
        positions = torch.randn(n_objects, 3, device=device) * 100
        spatial_tensor = Spatial3DTensor(positions, unit="Mpc")
        # Test memory info
        memory_info = spatial_tensor.memory_info()
        assert isinstance(memory_info, dict)
        assert "numel" in memory_info
        assert "storage_size" in memory_info
        assert memory_info["numel"] == n_objects * 3
        assert spatial_tensor.device.type == device.type

    def test_batch_processing(self):
        """Test batch processing capabilities (device-agnostic)."""
        from astro_lab.tensors import Spatial3DTensor
        device = get_optimal_device()
        # Create batch of spatial data on device
        batch_size = get_optimal_batch_size(5)
        n_objects = 100
        batch_data = torch.randn(batch_size, n_objects, 3, device=device)
        # Process each batch element
        results = []
        for i in range(batch_size):
            spatial_tensor = Spatial3DTensor(batch_data[i], unit="kpc")
            results.append(spatial_tensor.shape[0])
            assert spatial_tensor.device.type == device.type
        assert len(results) == batch_size
        assert all(r == n_objects for r in results)

    @pytest.mark.cuda
    def test_gpu_memory_management(self):
        """Test GPU memory management for large datasets (device-agnostic)."""
        from astro_lab.tensors import Spatial3DTensor
        device = get_optimal_device()
        if device.type != "cuda":
            pytest.skip("CUDA not available")
        n_objects = 50000
        # Create data on GPU
        positions = torch.randn(n_objects, 3, device=device)
        spatial_tensor = Spatial3DTensor(positions, unit="Mpc")
        assert spatial_tensor.device.type == "cuda"
        # Test memory transfer
        cpu_tensor = spatial_tensor.cpu()
        assert cpu_tensor.device.type == "cpu"
        # Test memory info
        memory_info = spatial_tensor.memory_info()
        assert "cuda" in memory_info["device"] 