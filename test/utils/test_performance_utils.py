"""
Tests for performance-related utilities.

Tests memory management, batch processing, and GPU operations.
"""

import pytest
import torch


class TestPerformanceUtilities:
    """Test performance-related utilities."""

    def test_memory_efficient_operations(self):
        """Test memory-efficient tensor operations."""
        from astro_lab.tensors import Spatial3DTensor
        
        # Create large dataset
        n_objects = 10000
        positions = torch.randn(n_objects, 3) * 100
        
        spatial_tensor = Spatial3DTensor(positions, unit="Mpc")
        
        # Test memory info
        memory_info = spatial_tensor.memory_info()
        
        assert isinstance(memory_info, dict)
        assert "numel" in memory_info
        assert "storage_size" in memory_info
        assert memory_info["numel"] == n_objects * 3

    def test_batch_processing(self):
        """Test batch processing capabilities."""
        from astro_lab.tensors import Spatial3DTensor
        
        # Create batch of spatial data
        batch_size = 5
        n_objects = 100
        batch_data = torch.randn(batch_size, n_objects, 3)
        
        # Process each batch element
        results = []
        for i in range(batch_size):
            spatial_tensor = Spatial3DTensor(batch_data[i], unit="kpc")
            results.append(spatial_tensor.shape[0])
        
        assert len(results) == batch_size
        assert all(r == n_objects for r in results)

    @pytest.mark.cuda
    def test_gpu_memory_management(self, skip_if_no_cuda):
        """Test GPU memory management for large datasets."""
        from astro_lab.tensors import Spatial3DTensor
        
        device = torch.device("cuda")
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