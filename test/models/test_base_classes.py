"""
Tests for Base Classes
=====================

Tests for BaseAstroGNN, BaseTemporalGNN, BaseTNGModel and FeatureFusion with new unified architecture.
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
from astro_lab.models.config import ModelConfig, EncoderConfig, GraphConfig, OutputConfig
from astro_lab.models.layers import LayerFactory


class TestBaseAstroGNN:
    """Test BaseAstroGNN functionality with new architecture."""

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

    def test_with_config_object(self):
        """Test BaseAstroGNN with Config object."""
        config = ModelConfig(
            name="test_base_gnn",
            description="Test BaseAstroGNN with config",
            graph=GraphConfig(
                conv_type="gcn",
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
            ),
        )

        # Create model using config
        model = BaseAstroGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type
        assert model.num_layers == config.graph.num_layers


class TestBaseTemporalGNN:
    """Test BaseTemporalGNN functionality with new architecture."""

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

    def test_with_config_object(self):
        """Test BaseTemporalGNN with Config object."""
        config = ModelConfig(
            name="test_temporal_gnn",
            description="Test BaseTemporalGNN with config",
            graph=GraphConfig(
                conv_type="gat",
                hidden_dim=32,
                num_layers=2,
                dropout=0.1,
            ),
        )

        # Create model using config
        model = BaseTemporalGNN(
            input_dim=10,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            recurrent_type="lstm",
            recurrent_layers=2,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type


class TestBaseTNGModel:
    """Test BaseTNGModel for cosmological simulations with new architecture."""

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

    def test_with_config_object(self):
        """Test BaseTNGModel with Config object."""
        config = ModelConfig(
            name="test_tng_model",
            description="Test BaseTNGModel with config",
            graph=GraphConfig(
                conv_type="transformer",
                hidden_dim=64,
                num_layers=3,
                dropout=0.1,
            ),
        )

        # Create model using config
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            cosmological_features=True,
            redshift_encoding=False,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type


class TestFeatureFusion:
    """Test FeatureFusion module with new architecture."""

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
        """Test individual components of fusion module."""
        fusion = FeatureFusion(
            input_dims=[16, 32], output_dim=64
        )
        # Test that fusion has expected components
        assert hasattr(fusion, "fusion")
        # Test forward pass
        features = [torch.randn(5, 16), torch.randn(5, 32)]
        output = fusion(features)
        assert output.shape == (5, 64)

    def test_with_config_object(self):
        """Test FeatureFusion with Config object."""
        config = ModelConfig(
            name="test_feature_fusion",
            description="Test FeatureFusion with config",
            encoder=EncoderConfig(
                use_photometry=True,
                use_astrometry=True,
                use_spectroscopy=False,
            ),
        )

        # Create fusion using config
        input_dims = []
        if config.encoder.use_photometry:
            input_dims.append(16)
        if config.encoder.use_astrometry:
            input_dims.append(32)
        if config.encoder.use_spectroscopy:
            input_dims.append(24)

        fusion = FeatureFusion(input_dims=input_dims, output_dim=64)

        # Test with features matching config
        features = [torch.randn(5, dim) for dim in input_dims]
        output = fusion(features)
        assert output.shape == (5, 64)


class TestLayerFactoryIntegration:
    """Test integration with LayerFactory."""

    def test_layer_factory_with_base_gnn(self):
        """Test LayerFactory creating layers for BaseAstroGNN."""
        factory = LayerFactory()

        # Create conv layer using factory
        conv_layer = factory.create_conv_layer("gcn", 16, 32)
        assert isinstance(conv_layer, nn.Module)

        # Test forward pass
        x = torch.randn(5, 16)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        output = conv_layer(x, edge_index)
        assert output.shape == (5, 32)

    def test_layer_factory_mlp_for_fusion(self):
        """Test LayerFactory creating MLPs for feature fusion."""
        factory = LayerFactory()

        # Create MLP using factory
        mlp = factory.create_mlp(
            64,  # input_dim
            [128],  # hidden_dims_or_output_dim
            32,  # output_dim
            activation="relu",
            dropout=0.1,
        )

        # Test forward pass
        x = torch.randn(5, 64)
        output = mlp(x)
        assert output.shape == (5, 32)


class TestBaseClassesIntegration:
    """Test integration between base classes and new architecture."""

    def test_inheritance_chain(self):
        """Test that base classes work with new Config objects."""
        config = ModelConfig(
            name="test_integration",
            description="Test integration",
            graph=GraphConfig(
                conv_type="gcn",
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
            ),
        )

        # Test BaseAstroGNN with config
        model = BaseAstroGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type

    def test_parameter_consistency(self):
        """Test that parameters are consistent across base classes."""
        config = ModelConfig(
            name="test_consistency",
            description="Test parameter consistency",
            graph=GraphConfig(
                conv_type="gat",
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
            ),
        )

        # Test multiple base classes with same config
        base_gnn = BaseAstroGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
        )

        temporal_gnn = BaseTemporalGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            recurrent_type="lstm",
            recurrent_layers=2,
        )

        assert base_gnn.hidden_dim == temporal_gnn.hidden_dim
        assert base_gnn.conv_type == temporal_gnn.conv_type
        assert base_gnn.num_layers == temporal_gnn.num_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
