"""
Tests for Output Heads
=====================

Tests for output head registry and implementations.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.output_heads import (
    ClassificationHead,
    CosmologicalHead,
    MultiTaskHead,
    OutputHeadRegistry,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)


class TestOutputHeadRegistry:
    """Test the output head registry system."""

    def test_registry_initialization(self):
        """Test registry initializes with default heads."""
        registry = OutputHeadRegistry()

        # Check that default heads are registered
        available_heads = registry.list_available()
        expected_heads = [
            "regression",
            "classification",
            "period_detection",
            "shape_modeling",
            "multi_task",
            "cosmological",
        ]

        for head_name in expected_heads:
            assert head_name in available_heads

    def test_create_head_with_params(self):
        """Test creating heads with different parameters."""
        registry = OutputHeadRegistry()

        # Test regression head
        reg_head = registry.create("regression", hidden_dim=128, output_dim=1)
        assert isinstance(reg_head, RegressionHead)

        # Test classification head
        cls_head = registry.create("classification", hidden_dim=64, output_dim=5)
        assert isinstance(cls_head, ClassificationHead)


class TestRegressionHead:
    """Test RegressionHead implementation."""

    def test_initialization(self):
        """Test RegressionHead initializes correctly."""
        head = RegressionHead(hidden_dim=64, output_dim=3, dropout=0.2)

        assert isinstance(head, nn.Module)
        assert hasattr(head, "head")
        assert isinstance(head.head, nn.Sequential)

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        head = RegressionHead(hidden_dim=32, output_dim=2)

        x = torch.randn(10, 32)
        output = head(x)

        assert output.shape == (10, 2)
        assert not torch.isnan(output).any()

    def test_different_output_dims(self):
        """Test with different output dimensions."""
        # Single output
        head_1d = RegressionHead(hidden_dim=16, output_dim=1)
        x = torch.randn(5, 16)
        output_1d = head_1d(x)
        assert output_1d.shape == (5, 1)

        # Multiple outputs
        head_5d = RegressionHead(hidden_dim=32, output_dim=5)
        x = torch.randn(8, 32)
        output_5d = head_5d(x)
        assert output_5d.shape == (8, 5)


class TestClassificationHead:
    """Test ClassificationHead implementation."""

    def test_initialization(self):
        """Test ClassificationHead initializes correctly."""
        head = ClassificationHead(hidden_dim=128, num_classes=10, dropout=0.3)

        assert isinstance(head, nn.Module)
        assert hasattr(head, "head")
        assert isinstance(head.head, nn.Sequential)
        assert head.num_classes == 10

    def test_forward_pass(self):
        """Test forward pass produces correct output shape."""
        head = ClassificationHead(hidden_dim=64, num_classes=5)

        x = torch.randn(12, 64)
        output = head(x)

        assert output.shape == (12, 5)
        assert not torch.isnan(output).any()

    def test_binary_vs_multiclass(self):
        """Test binary vs multiclass classification."""
        # Binary classification (uses sigmoid)
        head_binary = ClassificationHead(hidden_dim=32, num_classes=1)
        x = torch.randn(6, 32)
        output_binary = head_binary(x)
        assert output_binary.shape == (6, 1)
        # Binary output should be between 0 and 1
        assert (output_binary >= 0).all() and (output_binary <= 1).all()

        # Multiclass (uses log_softmax)
        head_multi = ClassificationHead(hidden_dim=32, num_classes=4)
        output_multi = head_multi(x)
        assert output_multi.shape == (6, 4)
        # Log probabilities should be <= 0
        assert (output_multi <= 0).all()


class TestPeriodDetectionHead:
    """Test PeriodDetectionHead for astronomical period detection."""

    def test_initialization(self):
        """Test PeriodDetectionHead initializes correctly."""
        head = PeriodDetectionHead(
            hidden_dim=256,
            output_dim=1,
            period_range=(0.5, 50.0),
            confidence_output=True,
        )

        assert isinstance(head, nn.Module)
        assert hasattr(head, "period_head")
        assert hasattr(head, "confidence_head")
        assert head.period_range == (0.5, 50.0)
        assert head.confidence_output == True

    def test_forward_pass(self):
        """Test forward pass produces period and confidence."""
        head = PeriodDetectionHead(
            hidden_dim=128,
            output_dim=1,
            period_range=(0.1, 100.0),
            confidence_output=True,
        )

        x = torch.randn(8, 128)
        output = head(x)

        assert isinstance(output, dict)
        assert "period" in output
        assert "confidence" in output
        assert output["period"].shape == (8, 1)
        assert output["confidence"].shape == (8, 1)
        assert not torch.isnan(output["period"]).any()
        assert not torch.isnan(output["confidence"]).any()

        # Check period bounds
        assert (output["period"] >= 0.1).all()
        assert (output["period"] <= 100.0).all()

        # Check confidence bounds
        assert (output["confidence"] >= 0).all()
        assert (output["confidence"] <= 1).all()

    def test_without_confidence(self):
        """Test period detection without confidence output."""
        head = PeriodDetectionHead(hidden_dim=64, output_dim=1, confidence_output=False)

        x = torch.randn(4, 64)
        output = head(x)

        assert isinstance(output, dict)
        assert "period" in output
        assert "confidence" not in output


class TestShapeModelingHead:
    """Test ShapeModelingHead for astronomical shape modeling."""

    def test_initialization(self):
        """Test ShapeModelingHead initializes correctly."""
        head = ShapeModelingHead(hidden_dim=512, output_dim=6, shape_params=6)

        assert isinstance(head, nn.Module)
        assert hasattr(head, "shape_head")
        assert head.shape_params == 6

    def test_forward_pass(self):
        """Test forward pass produces shape parameters."""
        head = ShapeModelingHead(hidden_dim=256, output_dim=4, shape_params=4)

        x = torch.randn(6, 256)
        output = head(x)

        assert isinstance(output, dict)
        # Check that output contains expected keys based on implementation
        assert len(output) > 0
        for key, value in output.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == 6  # batch size


class TestMultiTaskHead:
    """Test MultiTaskHead for multiple simultaneous tasks."""

    def test_initialization(self):
        """Test MultiTaskHead initializes correctly."""
        task_configs = {
            "classification": {"type": "classification", "output_dim": 5},
            "regression": {"type": "regression", "output_dim": 2},
        }

        head = MultiTaskHead(hidden_dim=128, task_configs=task_configs)

        assert isinstance(head, nn.Module)
        assert hasattr(head, "task_heads")

    def test_forward_pass(self):
        """Test forward pass produces outputs for all tasks."""
        task_configs = {
            "cls": {"type": "classification", "output_dim": 3},
            "reg": {"type": "regression", "output_dim": 1},
        }

        head = MultiTaskHead(hidden_dim=64, task_configs=task_configs)

        x = torch.randn(10, 64)
        outputs = head(x)

        assert isinstance(outputs, dict)
        assert len(outputs) >= 1  # Should have at least one task output


class TestCosmologicalHead:
    """Test CosmologicalHead for cosmological parameter estimation."""

    def test_initialization(self):
        """Test CosmologicalHead initializes correctly."""
        head = CosmologicalHead(hidden_dim=512, output_dim=6)

        assert isinstance(head, nn.Module)
        assert hasattr(head, "param_heads")  # Correct attribute name

    def test_forward_pass(self):
        """Test forward pass produces cosmological parameters."""
        head = CosmologicalHead(hidden_dim=256, output_dim=3)

        x = torch.randn(4, 256)
        output = head(x)

        assert isinstance(output, dict)
        # Check that output contains expected keys based on implementation
        assert len(output) > 0
        for key, value in output.items():
            assert isinstance(value, torch.Tensor)
            assert value.shape[0] == 4  # batch size


class TestConvenienceFunctions:
    """Test convenience functions for output heads."""

    def test_create_output_head_function(self):
        """Test create_output_head convenience function."""
        # Test regression head
        reg_head = create_output_head("regression", hidden_dim=64, output_dim=3)
        assert isinstance(reg_head, RegressionHead)

        # Test classification head
        cls_head = create_output_head("classification", hidden_dim=128, output_dim=10)
        assert isinstance(cls_head, ClassificationHead)

    def test_invalid_head_type(self):
        """Test error handling for invalid head types."""
        with pytest.raises(ValueError, match="Unknown output head"):
            create_output_head("nonexistent_head", hidden_dim=64, output_dim=1)


class TestOutputHeadIntegration:
    """Test integration between different output heads."""

    def test_heads_with_same_hidden_dim(self):
        """Test that different heads work with the same hidden dimension."""
        hidden_dim = 128
        batch_size = 6

        # Create different heads with same hidden dimension
        reg_head = RegressionHead(hidden_dim=hidden_dim, output_dim=2)
        cls_head = ClassificationHead(hidden_dim=hidden_dim, num_classes=4)
        period_head = PeriodDetectionHead(hidden_dim=hidden_dim, output_dim=1)

        # Test with same input
        x = torch.randn(batch_size, hidden_dim)

        reg_output = reg_head(x)
        cls_output = cls_head(x)
        period_output = period_head(x)

        assert reg_output.shape == (batch_size, 2)
        assert cls_output.shape == (batch_size, 4)
        assert isinstance(period_output, dict)

    def test_parameter_sharing_compatibility(self):
        """Test that heads are compatible for parameter sharing scenarios."""
        # All heads should accept hidden_dim as first parameter
        hidden_dim = 256

        heads = [
            RegressionHead(hidden_dim, output_dim=1),
            ClassificationHead(hidden_dim, num_classes=5),
            PeriodDetectionHead(hidden_dim, output_dim=1),
            ShapeModelingHead(hidden_dim, output_dim=4, shape_params=4),
            CosmologicalHead(hidden_dim, output_dim=6),
        ]

        # All should be nn.Module instances
        for head in heads:
            assert isinstance(head, nn.Module)

        # All should handle the same input
        x = torch.randn(8, hidden_dim)
        for head in heads:
            output = head(x)
            # Each head returns different output format, but should not error
            assert output is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
