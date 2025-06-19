"""
Tests for Base Classes
=====================

Tests for BaseAstroGNN, BaseTemporalGNN, BaseTNGModel and FeatureFusion.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.base_gnn import (
    BaseAstroGNN,
    BaseTemporalGNN,
    BaseTNGModel,
    ConvType,
    FeatureFusion,
)


class TestBaseAstroGNN:
    """Test BaseAstroGNN functionality."""

    def test_initialization(self):
        """Test BaseAstroGNN initializes correctly."""
        model = BaseAstroGNN(input_dim=16, hidden_dim=64, conv_type="gcn", num_layers=2)

        assert isinstance(model, nn.Module)
        assert model.input_dim == 16
        assert model.hidden_dim == 64
        assert model.conv_type == "gcn"
        assert model.num_layers == 2
        assert hasattr(model, "input_projection")
        assert hasattr(model, "convs")
        assert hasattr(model, "norms")

    def test_all_conv_types(self):
        """Test all supported convolution types."""

        # Define supported conv types explicitly since ConvType is a Literal, not an enum
        conv_types: list[ConvType] = ["gcn", "gat", "sage", "transformer"]

        for conv_type in conv_types:
            model = BaseAstroGNN(
                input_dim=8,
                hidden_dim=32,
                conv_type=conv_type,
                num_layers=2,
            )

            x = torch.randn(5, 8)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

            output = model(x, edge_index)
            assert output.shape == (5, 32)  # hidden_dim, not output_dim
            assert not torch.isnan(output).any()

    def test_graph_forward_method(self):
        """Test graph_forward method specifically."""
        model = BaseAstroGNN(input_dim=10, hidden_dim=48, num_layers=3)

        x = torch.randn(6, 48)  # Already projected to hidden_dim
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        # Test normal forward
        output = model.graph_forward(x, edge_index)
        assert output.shape == (6, 48)

        # Test with intermediate outputs
        intermediate = model.graph_forward(x, edge_index, return_intermediate=True)
        assert isinstance(intermediate, list)
        assert len(intermediate) == 3  # num_layers


class TestBaseTemporalGNN:
    """Test BaseTemporalGNN functionality."""

    def test_initialization(self):
        """Test BaseTemporalGNN initializes correctly."""
        model = BaseTemporalGNN(
            input_dim=10,
            hidden_dim=32,
            conv_type="gat",
            recurrent_type="lstm",
            recurrent_layers=2,
        )

        assert isinstance(model, nn.Module)
        assert model.input_dim == 10
        assert model.hidden_dim == 32
        assert model.conv_type == "gat"
        assert model.recurrent_type == "lstm"
        assert model.recurrent_layers == 2
        assert hasattr(model, "rnn")

    def test_temporal_forward_pass(self):
        """Test forward pass with temporal data."""
        model = BaseTemporalGNN(
            input_dim=10, hidden_dim=32, conv_type="gcn", recurrent_type="lstm"
        )

        # Test temporal data
        x = torch.randn(5, 10)  # 5 nodes, 10 features
        temporal_features = torch.randn(5, 32)  # 5 nodes, hidden_dim features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        output = model(x, edge_index, temporal_features)
        assert output.shape == (5, 32)  # hidden_dim output
        assert not torch.isnan(output).any()

    def test_temporal_forward_method(self):
        """Test temporal_forward method specifically."""
        model = BaseTemporalGNN(input_dim=8, hidden_dim=24)

        # Test with multiple snapshots
        snapshots = [torch.randn(4, 24), torch.randn(4, 24), torch.randn(4, 24)]

        output = model.temporal_forward(snapshots)
        assert output.shape == (4, 24)

        # Test with single snapshot
        single_output = model.temporal_forward([snapshots[0]])
        assert torch.equal(single_output, snapshots[0])


class TestBaseTNGModel:
    """Test BaseTNGModel for cosmological simulations."""

    def test_initialization(self):
        """Test BaseTNGModel initializes correctly."""
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=64,
            cosmological_features=True,
            redshift_encoding=True,
        )

        assert isinstance(model, nn.Module)
        assert model.input_dim == 12
        assert model.hidden_dim == 64
        assert model.cosmological_features
        assert model.redshift_encoding
        assert hasattr(model, "redshift_encoder")
        assert hasattr(model, "cosmo_head")

    def test_forward_pass(self):
        """Test forward pass with cosmological data."""
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=64,
            cosmological_features=True,
            redshift_encoding=False,  # Disable for simpler test
        )

        x = torch.randn(8, 12)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

        output = model(x, edge_index)
        assert output.shape == (8, 64)  # hidden_dim output
        assert not torch.isnan(output).any()

    def test_cosmological_encoding(self):
        """Test cosmological encoding with redshift."""
        model = BaseTNGModel(input_dim=8, hidden_dim=32, redshift_encoding=True)

        x = torch.randn(6, 8)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        redshift = 2.5

        output = model(x, edge_index, redshift=redshift)
        assert output.shape == (6, 32)
        assert not torch.isnan(output).any()


class TestFeatureFusion:
    """Test FeatureFusion module."""

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = FeatureFusion(input_dims=[16, 32, 8], output_dim=64, dropout=0.1)

        features = [torch.randn(5, 16), torch.randn(5, 32), torch.randn(5, 8)]

        fused = fusion(features)
        assert fused.shape == (5, 64)
        assert not torch.isnan(fused).any()

    def test_different_input_combinations(self):
        """Test different input dimension combinations."""
        # Test with 2 inputs
        fusion_2 = FeatureFusion(input_dims=[10, 20], output_dim=32)
        features_2 = [torch.randn(3, 10), torch.randn(3, 20)]
        output_2 = fusion_2(features_2)
        assert output_2.shape == (3, 32)

        # Test with 4 inputs
        fusion_4 = FeatureFusion(input_dims=[8, 16, 12, 4], output_dim=48)
        features_4 = [
            torch.randn(7, 8),
            torch.randn(7, 16),
            torch.randn(7, 12),
            torch.randn(7, 4),
        ]
        output_4 = fusion_4(features_4)
        assert output_4.shape == (7, 48)

    def test_fusion_module_components(self):
        """Test internal components of fusion module."""
        fusion = FeatureFusion(input_dims=[16, 24], output_dim=40)

        assert hasattr(fusion, "fusion")
        assert isinstance(fusion.fusion, nn.Sequential)

        # Check that total input dimension is handled correctly
        total_input = sum([16, 24])  # 40
        features = [torch.randn(4, 16), torch.randn(4, 24)]

        # Should not raise dimension errors
        output = fusion(features)
        assert output.shape == (4, 40)


class TestBaseClassesIntegration:
    """Test integration between base classes."""

    def test_inheritance_chain(self):
        """Test that inheritance works correctly."""
        # BaseTemporalGNN should inherit from BaseAstroGNN
        temporal_model = BaseTemporalGNN(input_dim=8, hidden_dim=24)
        assert isinstance(temporal_model, BaseAstroGNN)
        assert hasattr(temporal_model, "graph_forward")
        assert hasattr(temporal_model, "temporal_forward")

        # BaseTNGModel should inherit from BaseTemporalGNN
        tng_model = BaseTNGModel(input_dim=8, hidden_dim=24)
        assert isinstance(tng_model, BaseTemporalGNN)
        assert isinstance(tng_model, BaseAstroGNN)
        assert hasattr(tng_model, "graph_forward")
        assert hasattr(tng_model, "temporal_forward")

    def test_parameter_consistency(self):
        """Test that parameters are consistent across inheritance."""
        input_dim, hidden_dim = 12, 48

        base_model = BaseAstroGNN(input_dim=input_dim, hidden_dim=hidden_dim)
        temporal_model = BaseTemporalGNN(input_dim=input_dim, hidden_dim=hidden_dim)
        tng_model = BaseTNGModel(input_dim=input_dim, hidden_dim=hidden_dim)

        # All should have same basic parameters
        for model in [base_model, temporal_model, tng_model]:
            assert model.input_dim == input_dim
            assert model.hidden_dim == hidden_dim
            assert hasattr(model, "input_projection")
            assert hasattr(model, "convs")

        # Test forward pass consistency
        x = torch.randn(5, input_dim)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

        base_output = base_model(x, edge_index)
        temporal_output = temporal_model(x, edge_index)
        tng_output = tng_model(x, edge_index)

        # All outputs should have same shape
        assert base_output.shape == temporal_output.shape == tng_output.shape
        assert base_output.shape == (5, hidden_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
