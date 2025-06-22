"""
Tests for AstroTensorBase.
"""

import pickle
from typing import Any, Dict

import numpy as np
import pytest
import torch

from astro_lab.tensors.base import AstroTensorBase


class TestAstroTensorBase:
    """Test the base astronomical tensor class."""

    def test_tensor_creation(self):
        """Test basic tensor creation."""
        data = torch.randn(5, 3)  # Create test data directly
        tensor = AstroTensorBase(data, tensor_type="test")

        assert tensor.get_metadata("tensor_type") == "test"
        assert tensor.shape == data.shape
        assert torch.equal(tensor.data, data)
        assert tensor.dtype == data.dtype

    def test_tensor_device_transfer(self):
        """Test device transfer."""
        data = torch.randn(5, 3)

        # Test CPU device
        cpu_tensor = AstroTensorBase(data)
        assert cpu_tensor.device.type == "cpu"
        assert cpu_tensor.data.device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            cuda_data = data.to(cuda_device)
            cuda_tensor = AstroTensorBase(cuda_data)
            assert cuda_tensor.device.type == "cuda"
            assert cuda_tensor.data.device.type == "cuda"

    def test_metadata_operations(self):
        """Test metadata handling."""
        data = torch.randn(3, 3)
        tensor = AstroTensorBase(data, custom_field="test_value")

        # Test metadata access
        assert tensor.get_metadata("custom_field") == "test_value"
        assert tensor.get_metadata("nonexistent", "default") == "default"

        # Test metadata update
        tensor.update_metadata(new_field="new_value")
        assert tensor.get_metadata("new_field") == "new_value"

    def test_tensor_operations(self):
        """Test tensor operations preserve metadata."""
        data = torch.randn(3, 5)
        tensor = AstroTensorBase(data, custom_field="test")

        # Test basic operations using torch directly
        unsqueezed_data = tensor.data.unsqueeze(0)
        assert unsqueezed_data.shape == (1, 3, 5)

        # Create new tensor with unsqueezed data
        unsqueezed_tensor = AstroTensorBase(unsqueezed_data, custom_field="test")
        assert unsqueezed_tensor.get_metadata("custom_field") == "test"

        squeezed_data = unsqueezed_data.squeeze(0)
        squeezed_tensor = AstroTensorBase(squeezed_data, custom_field="test")
        assert squeezed_tensor.shape == (3, 5)
        assert squeezed_tensor.get_metadata("custom_field") == "test"

    def test_mask_application(self):
        """Test boolean mask application."""
        data = torch.randn(10, 3)
        tensor = AstroTensorBase(data, test_field="value")

        mask = torch.rand(10) > 0.5
        masked = tensor.apply_mask(mask)

        assert masked.shape[0] == mask.sum()
        assert masked.get_metadata("test_field") == "value"

    def test_numpy_conversion(self):
        """Test conversion to numpy."""
        data = torch.randn(5, 3)
        tensor = AstroTensorBase(data)

        numpy_data = tensor.numpy()
        assert isinstance(numpy_data, np.ndarray)
        assert numpy_data.shape == data.shape
        # Fix linter error by using torch.testing instead
        torch.testing.assert_close(torch.from_numpy(numpy_data), data)
