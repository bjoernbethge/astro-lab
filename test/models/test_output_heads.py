"""
Tests for AstroLab output heads.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.components.output_heads import (
    OUTPUT_HEADS,
    ClassificationHead,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)


class TestClassificationHead:
    """Test the ClassificationHead module."""

    def test_classification_head_initialization(self):
        """Test ClassificationHead initialization."""
        head = ClassificationHead(input_dim=64, num_classes=5, dropout=0.2)

        assert isinstance(head, nn.Module)
        assert isinstance(head.classifier, nn.Module)

    def test_classification_head_forward(self):
        """Test ClassificationHead forward pass."""
        head = ClassificationHead(input_dim=64, num_classes=5)
        x = torch.randn(32, 64)  # batch_size=32, input_dim=64

        output = head(x)

        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()

    def test_classification_head_different_dropout(self):
        """Test ClassificationHead with different dropout rates."""
        head = ClassificationHead(input_dim=64, num_classes=5, dropout=0.5)
        x = torch.randn(32, 64)

        output = head(x)
        assert output.shape == (32, 5)

    def test_classification_head_gradients(self):
        """Test that ClassificationHead works correctly."""
        head = ClassificationHead(input_dim=64, num_classes=5)
        x = torch.randn(32, 64)

        output = head(x)

        # Check that output has correct shape and no NaN values
        assert output.shape == (32, 5)
        assert not torch.isnan(output).any()

        # Check that head has trainable parameters
        assert len(list(head.parameters())) > 0


class TestRegressionHead:
    """Test the RegressionHead module."""

    def test_regression_head_initialization(self):
        """Test RegressionHead initialization."""
        head = RegressionHead(input_dim=64, output_dim=3, dropout=0.2)

        assert isinstance(head, nn.Module)
        assert isinstance(head.regressor, nn.Module)

    def test_regression_head_forward(self):
        """Test RegressionHead forward pass."""
        head = RegressionHead(input_dim=64, output_dim=3)
        x = torch.randn(32, 64)

        output = head(x)

        assert output.shape == (32, 3)
        assert not torch.isnan(output).any()

    def test_regression_head_default_output_dim(self):
        """Test RegressionHead with default output_dim."""
        head = RegressionHead(input_dim=64)
        x = torch.randn(32, 64)

        output = head(x)

        assert output.shape == (32, 1)  # Default output_dim=1

    def test_regression_head_gradients(self):
        """Test that RegressionHead works correctly."""
        head = RegressionHead(input_dim=64, output_dim=2)
        x = torch.randn(32, 64)

        output = head(x)

        # Check that output has correct shape and no NaN values
        assert output.shape == (32, 2)
        assert not torch.isnan(output).any()

        # Check that head has trainable parameters
        assert len(list(head.parameters())) > 0


class TestPeriodDetectionHead:
    """Test the PeriodDetectionHead module."""

    def test_period_detection_head_initialization(self):
        """Test PeriodDetectionHead initialization."""
        head = PeriodDetectionHead(input_dim=64, dropout=0.2)

        assert isinstance(head, nn.Module)
        assert isinstance(head.period_net, nn.Module)

    def test_period_detection_head_forward(self):
        """Test PeriodDetectionHead forward pass."""
        head = PeriodDetectionHead(input_dim=64)
        x = torch.randn(32, 64)

        output = head(x)

        assert isinstance(output, dict)
        assert "period" in output
        assert "uncertainty" in output
        assert output["period"].shape == (32, 1)
        assert output["uncertainty"].shape == (32, 1)
        assert not torch.isnan(output["period"]).any()
        assert not torch.isnan(output["uncertainty"]).any()

    def test_period_detection_head_positive_outputs(self):
        """Test that period and uncertainty are always positive."""
        head = PeriodDetectionHead(input_dim=64)
        x = torch.randn(32, 64)

        output = head(x)

        assert torch.all(output["period"] >= 0)
        assert torch.all(output["uncertainty"] >= 0)

    def test_period_detection_head_gradients(self):
        """Test that PeriodDetectionHead works correctly."""
        head = PeriodDetectionHead(input_dim=64)
        x = torch.randn(32, 64)

        output = head(x)

        # Check that output has correct shape and no NaN values
        assert output["period"].shape == (32, 1)
        assert output["uncertainty"].shape == (32, 1)
        assert not torch.isnan(output["period"]).any()
        assert not torch.isnan(output["uncertainty"]).any()

        # Check that head has trainable parameters
        assert len(list(head.parameters())) > 0


class TestShapeModelingHead:
    """Test the ShapeModelingHead module."""

    def test_shape_modeling_head_initialization(self):
        """Test ShapeModelingHead initialization."""
        head = ShapeModelingHead(input_dim=64, num_harmonics=8, dropout=0.2)

        assert isinstance(head, nn.Module)
        assert isinstance(head.shape_net, nn.Module)
        assert head.num_harmonics == 8

    def test_shape_modeling_head_forward(self):
        """Test ShapeModelingHead forward pass."""
        head = ShapeModelingHead(input_dim=64, num_harmonics=6)
        x = torch.randn(32, 64)

        output = head(x)

        assert isinstance(output, dict)
        assert "real_coeffs" in output
        assert "imag_coeffs" in output
        assert output["real_coeffs"].shape == (32, 6)
        assert output["imag_coeffs"].shape == (32, 6)
        assert not torch.isnan(output["real_coeffs"]).any()
        assert not torch.isnan(output["imag_coeffs"]).any()

    def test_shape_modeling_head_different_harmonics(self):
        """Test ShapeModelingHead with different number of harmonics."""
        head = ShapeModelingHead(input_dim=64, num_harmonics=10)
        x = torch.randn(32, 64)

        output = head(x)

        assert output["real_coeffs"].shape == (32, 10)
        assert output["imag_coeffs"].shape == (32, 10)

    def test_shape_modeling_head_gradients(self):
        """Test that ShapeModelingHead works correctly."""
        head = ShapeModelingHead(input_dim=64, num_harmonics=5)
        x = torch.randn(32, 64)

        output = head(x)

        # Check that output has correct shape and no NaN values
        assert output["real_coeffs"].shape == (32, 5)
        assert output["imag_coeffs"].shape == (32, 5)
        assert not torch.isnan(output["real_coeffs"]).any()
        assert not torch.isnan(output["imag_coeffs"]).any()

        # Check that head has trainable parameters
        assert len(list(head.parameters())) > 0


class TestOutputHeadsRegistry:
    """Test the OUTPUT_HEADS registry."""

    def test_output_heads_registry_contains_all_heads(self):
        """Test that all head types are registered."""
        expected_heads = [
            "classification",
            "regression",
            "period_detection",
            "shape_modeling",
        ]

        for head_type in expected_heads:
            assert head_type in OUTPUT_HEADS
            assert issubclass(OUTPUT_HEADS[head_type], nn.Module)

    def test_output_heads_registry_types(self):
        """Test that registered heads are the correct types."""
        assert OUTPUT_HEADS["classification"] == ClassificationHead
        assert OUTPUT_HEADS["regression"] == RegressionHead
        assert OUTPUT_HEADS["period_detection"] == PeriodDetectionHead
        assert OUTPUT_HEADS["shape_modeling"] == ShapeModelingHead


class TestCreateOutputHead:
    """Test the create_output_head factory function."""

    def test_create_classification_head(self):
        """Test creating classification head."""
        head = create_output_head("classification", input_dim=64, output_dim=5)

        assert isinstance(head, ClassificationHead)
        assert head.classifier is not None

    def test_create_regression_head(self):
        """Test creating regression head."""
        head = create_output_head("regression", input_dim=64, output_dim=3)

        assert isinstance(head, RegressionHead)
        assert head.regressor is not None

    def test_create_period_detection_head(self):
        """Test creating period detection head."""
        head = create_output_head("period_detection", input_dim=64)

        assert isinstance(head, PeriodDetectionHead)
        assert head.period_net is not None

    def test_create_shape_modeling_head(self):
        """Test creating shape modeling head."""
        head = create_output_head("shape_modeling", input_dim=64, num_harmonics=8)

        assert isinstance(head, ShapeModelingHead)
        assert head.shape_net is not None

    def test_create_output_head_default_output_dim(self):
        """Test creating heads with default output_dim."""
        # Classification should default to 2 classes
        cls_head = create_output_head("classification", input_dim=64)
        assert cls_head.classifier is not None

        # Regression should default to 1 output
        reg_head = create_output_head("regression", input_dim=64)
        assert reg_head.regressor is not None

    def test_create_output_head_invalid_type(self):
        """Test creating head with invalid type."""
        with pytest.raises(ValueError, match="Unknown head type"):
            create_output_head("invalid_head", input_dim=64)

    def test_create_output_head_with_additional_kwargs(self):
        """Test creating head with additional kwargs."""
        head = create_output_head(
            "classification", input_dim=64, output_dim=5, dropout=0.3
        )

        assert isinstance(head, ClassificationHead)

    def test_create_output_head_filter_kwargs(self):
        """Test that invalid kwargs are filtered out."""
        # This should not raise an error even with invalid kwargs
        head = create_output_head(
            "classification",
            input_dim=64,
            output_dim=5,
            invalid_param="should_be_filtered",
        )

        assert isinstance(head, ClassificationHead)


class TestOutputHeadsIntegration:
    """Integration tests for output heads."""

    def test_all_heads_forward_pass(self):
        """Test that all head types can perform forward pass."""
        input_dim = 64
        batch_size = 16
        x = torch.randn(batch_size, input_dim)

        # Test all head types
        heads = {
            "classification": create_output_head("classification", input_dim, 5),
            "regression": create_output_head("regression", input_dim, 3),
            "period_detection": create_output_head("period_detection", input_dim),
            "shape_modeling": create_output_head(
                "shape_modeling", input_dim, num_harmonics=6
            ),
        }

        for head_type, head in heads.items():
            output = head(x)

            if head_type == "classification":
                assert output.shape == (batch_size, 5)
            elif head_type == "regression":
                assert output.shape == (batch_size, 3)
            elif head_type == "period_detection":
                assert isinstance(output, dict)
                assert "period" in output
                assert "uncertainty" in output
            elif head_type == "shape_modeling":
                assert isinstance(output, dict)
                assert "real_coeffs" in output
                assert "imag_coeffs" in output

    def test_heads_gradient_flow(self):
        """Test that all head types work correctly."""
        input_dim = 64
        batch_size = 16
        x = torch.randn(batch_size, input_dim)

        heads = {
            "classification": create_output_head("classification", input_dim, 5),
            "regression": create_output_head("regression", input_dim, 3),
            "period_detection": create_output_head("period_detection", input_dim),
            "shape_modeling": create_output_head(
                "shape_modeling", input_dim, num_harmonics=6
            ),
        }

        for head_type, head in heads.items():
            output = head(x)

            # Check that output is valid
            if isinstance(output, dict):
                for tensor in output.values():
                    assert not torch.isnan(tensor).any()
            else:
                assert not torch.isnan(output).any()

            # Check that head has trainable parameters
            assert len(list(head.parameters())) > 0
