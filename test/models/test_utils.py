"""
Tests for AstroLab model utilities.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.utils import (
    AttentionPooling,
    count_parameters,
    filter_kwargs,
    freeze_layers,
    get_activation,
    get_device,
    get_model_summary,
    get_pooling,
    initialize_weights,
    move_batch_to_device,
    set_dropout_rate,
    unfreeze_layers,
)


class TestFilterKwargs:
    """Test the filter_kwargs utility function."""

    def test_filter_kwargs_valid_params(self):
        """Test filtering with valid parameters."""

        class TestClass:
            def __init__(self, param1: int, param2: str, param3: float = 1.0):
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        kwargs = {
            "param1": 42,
            "param2": "test",
            "param3": 3.14,
            "invalid_param": "should_be_filtered",
        }

        filtered = filter_kwargs(TestClass, **kwargs)

        assert "param1" in filtered
        assert "param2" in filtered
        assert "param3" in filtered
        assert "invalid_param" not in filtered
        assert filtered["param1"] == 42
        assert filtered["param2"] == "test"
        assert filtered["param3"] == 3.14

    def test_filter_kwargs_no_invalid_params(self):
        """Test filtering when no invalid parameters are provided."""

        class TestClass:
            def __init__(self, param1: int, param2: str):
                self.param1 = param1
                self.param2 = param2

        kwargs = {"param1": 42, "param2": "test"}
        filtered = filter_kwargs(TestClass, **kwargs)

        assert filtered == kwargs

    def test_filter_kwargs_empty_kwargs(self):
        """Test filtering with empty kwargs."""

        class TestClass:
            def __init__(self, param1: int = 1):
                self.param1 = param1

        filtered = filter_kwargs(TestClass)
        assert filtered == {}


class TestGetActivation:
    """Test the get_activation utility function."""

    def test_get_activation_valid_names(self):
        """Test getting valid activation functions."""
        activations = ["relu", "gelu", "swish", "tanh", "leaky_relu", "elu", "mish"]

        for name in activations:
            activation = get_activation(name)
            assert isinstance(activation, nn.Module)

    def test_get_activation_case_insensitive(self):
        """Test that activation names are case insensitive."""
        activation1 = get_activation("ReLU")
        activation2 = get_activation("relu")
        assert isinstance(activation1, nn.ReLU)
        assert isinstance(activation2, nn.ReLU)

    def test_get_activation_invalid_name(self):
        """Test getting invalid activation function."""
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("invalid_activation")


class TestGetPooling:
    """Test the get_pooling utility function."""

    def test_get_pooling_valid_types(self):
        """Test getting valid pooling types."""
        pooling_types = ["mean", "max", "add", "attention"]

        for pooling_type in pooling_types:
            result = get_pooling(pooling_type)
            assert result == pooling_type

    def test_get_pooling_case_insensitive(self):
        """Test that pooling types are case insensitive."""
        result1 = get_pooling("MEAN")
        result2 = get_pooling("mean")
        assert result1 == "mean"
        assert result2 == "mean"

    def test_get_pooling_invalid_type(self):
        """Test getting invalid pooling type."""
        with pytest.raises(ValueError, match="Unknown pooling"):
            get_pooling("invalid_pooling")


class TestGetDevice:
    """Test the get_device utility function."""

    def test_get_device_none(self):
        """Test getting device when None is provided."""
        device = get_device(None)
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]

    def test_get_device_string(self):
        """Test getting device from string."""
        device = get_device("cpu")
        assert device == torch.device("cpu")

    def test_get_device_torch_device(self):
        """Test getting device from torch.device."""
        device = get_device(torch.device("cpu"))
        assert device == torch.device("cpu")


class TestInitializeWeights:
    """Test the initialize_weights utility function."""

    def test_initialize_weights_simple_model(self):
        """Test weight initialization for a simple model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        # Store original weights
        original_weights = []
        for name, param in model.named_parameters():
            if "weight" in name:
                original_weights.append(param.clone())

        # Initialize weights
        initialize_weights(model)

        # Check that weights have changed
        weight_idx = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                assert not torch.allclose(param, original_weights[weight_idx])
                weight_idx += 1


class TestCountParameters:
    """Test the count_parameters utility function."""

    def test_count_parameters_simple_model(self):
        """Test parameter counting for a simple model."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        count = count_parameters(model)
        expected = 10 * 20 + 20 + 20 * 5 + 5  # weights + biases
        assert count == expected

    def test_count_parameters_with_frozen_layers(self):
        """Test parameter counting with frozen layers."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        count = count_parameters(model)
        expected = 20 * 5 + 5  # only second layer parameters
        assert count == expected


class TestMoveBatchToDevice:
    """Test the move_batch_to_device utility function."""

    def test_move_batch_to_device_tensors(self):
        """Test moving tensor batch to device."""
        batch = {
            "x": torch.randn(10, 5),
            "y": torch.randn(10),
            "metadata": {"key": "value"},
        }

        device = torch.device("cpu")
        moved_batch = move_batch_to_device(batch, device)

        assert moved_batch["x"].device == device
        assert moved_batch["y"].device == device
        assert moved_batch["metadata"] == batch["metadata"]

    def test_move_batch_to_device_tensor_lists(self):
        """Test moving batch with tensor lists to device."""
        batch = {"x": [torch.randn(5, 3), torch.randn(5, 3)], "y": torch.randn(10)}

        device = torch.device("cpu")
        moved_batch = move_batch_to_device(batch, device)

        for tensor in moved_batch["x"]:
            assert tensor.device == device
        assert moved_batch["y"].device == device


class TestSetDropoutRate:
    """Test the set_dropout_rate utility function."""

    def test_set_dropout_rate(self):
        """Test setting dropout rate for all dropout layers."""
        model = nn.Sequential(
            nn.Linear(10, 20), nn.Dropout(0.1), nn.Linear(20, 5), nn.Dropout(0.2)
        )

        set_dropout_rate(model, 0.5)

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                assert module.p == 0.5


class TestFreezeUnfreezeLayers:
    """Test the freeze_layers and unfreeze_layers utility functions."""

    def test_freeze_all_layers(self):
        """Test freezing all layers."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        freeze_layers(model)

        for param in model.parameters():
            assert not param.requires_grad

    def test_freeze_specific_layers(self):
        """Test freezing specific layers."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        freeze_layers(model, ["0"])  # Freeze first layer

        for name, param in model.named_parameters():
            if "0" in name:
                assert not param.requires_grad
            else:
                assert param.requires_grad

    def test_unfreeze_all_layers(self):
        """Test unfreezing all layers."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        # First freeze all
        freeze_layers(model)
        # Then unfreeze all
        unfreeze_layers(model)

        for param in model.parameters():
            assert param.requires_grad


class TestGetModelSummary:
    """Test the get_model_summary utility function."""

    def test_get_model_summary_basic(self):
        """Test getting basic model summary."""
        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))

        summary = get_model_summary(model)

        assert "Sequential" in summary
        assert "Total trainable parameters" in summary
        assert "Linear" in summary

    def test_get_model_summary_with_input_shape(self):
        """Test getting model summary with input shape."""
        model = nn.Sequential(nn.Linear(10, 20), nn.Linear(20, 5))

        summary = get_model_summary(model, input_shape=(32, 10))

        assert "Expected input shape" in summary
        assert "(32, 10)" in summary


class TestAttentionPooling:
    """Test the AttentionPooling module."""

    def test_attention_pooling_basic(self):
        """Test basic attention pooling functionality."""
        pooling = AttentionPooling(hidden_dim=64)
        x = torch.randn(10, 64)  # 10 nodes, 64 features

        output = pooling(x)

        assert output.shape == (1, 64)
        assert torch.allclose(output.sum(), output.sum())  # Check for NaN

    def test_attention_pooling_with_batch(self):
        """Test attention pooling with batch information."""
        pooling = AttentionPooling(hidden_dim=64)
        x = torch.randn(10, 64)
        batch = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])  # 3 graphs

        output = pooling(x, batch)

        assert output.shape == (1, 64)
