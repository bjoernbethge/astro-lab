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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv_types: list[ConvType] = ["gcn", "gat", "sage", "transformer"]

        for conv_type in conv_types:
            model = BaseAstroGNN(
                input_dim=8,
                hidden_dim=32,
                conv_type=conv_type,
                num_layers=2,
                device=device,
            )

            x = torch.randn(5, 8).to(device)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)

            output = model(x, edge_index)
            assert output.shape == (5, 32)
            assert not torch.isnan(output).any()

    def test_graph_forward_method(self):
        """Test graph_forward method specifically."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseAstroGNN(input_dim=10, hidden_dim=48, num_layers=3, device=device)

        x = torch.randn(6, 48).to(device)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long).to(device)

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTemporalGNN(
            input_dim=10,
            hidden_dim=32,
            conv_type="gat",
            recurrent_type="lstm",
            recurrent_layers=2,
            device=device,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTemporalGNN(
            input_dim=10, hidden_dim=32, conv_type="gcn", recurrent_type="lstm", device=device
        )

        # Test temporal data
        x = torch.randn(5, 10).to(device)
        temporal_features = torch.randn(5, 32).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)

        output = model(x, edge_index, temporal_features)
        assert output.shape == (5, 32)
        assert not torch.isnan(output).any()

    def test_temporal_forward_method(self):
        """Test temporal_forward method specifically."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTemporalGNN(input_dim=8, hidden_dim=24, device=device)

        # Test with multiple snapshots
        snapshots = [torch.randn(4, 24).to(device), torch.randn(4, 24).to(device), torch.randn(4, 24).to(device)]

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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTemporalGNN(
            input_dim=10,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            recurrent_type="lstm",
            recurrent_layers=2,
            device=device,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type


class TestBaseTNGModel:
    """Test BaseTNGModel for cosmological simulations with new architecture."""

    def test_initialization(self):
        """Test BaseTNGModel initializes correctly."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=64,
            cosmological_features=True,
            redshift_encoding=True,
            device=device,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=64,
            cosmological_features=True,
            redshift_encoding=False,  # Disable for simpler test
            device=device,
        )

        x = torch.randn(8, 12).to(device)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long).to(device)

        output = model(x, edge_index)
        assert output.shape == (8, 64)
        assert not torch.isnan(output).any()

    def test_cosmological_encoding(self):
        """Test cosmological encoding with redshift."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTNGModel(input_dim=8, hidden_dim=32, redshift_encoding=True, device=device)

        x = torch.randn(6, 8).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseTNGModel(
            input_dim=12,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            cosmological_features=True,
            redshift_encoding=False,
            device=device,
        )

        assert model.hidden_dim == config.graph.hidden_dim
        assert model.conv_type == config.graph.conv_type


class TestFeatureFusion:
    """Test FeatureFusion module with new architecture."""

    def test_concat_fusion(self):
        """Test concatenation fusion."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fusion = FeatureFusion(input_dims=[16, 32, 8], output_dim=64, dropout=0.1).to(device)

        features = [torch.randn(5, 16).to(device), torch.randn(5, 32).to(device), torch.randn(5, 8).to(device)]

        fused = fusion(features)
        assert fused.shape == (5, 64)
        assert not torch.isnan(fused).any()

    def test_different_input_combinations(self):
        """Test different input dimension combinations."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Test with 2 inputs
        fusion_2 = FeatureFusion(input_dims=[10, 20], output_dim=32).to(device)
        features_2 = [torch.randn(3, 10).to(device), torch.randn(3, 20).to(device)]
        output_2 = fusion_2(features_2)
        assert output_2.shape == (3, 32)

        # Test with 4 inputs
        fusion_4 = FeatureFusion(input_dims=[8, 16, 12, 4], output_dim=48).to(device)
        features_4 = [
            torch.randn(7, 8).to(device),
            torch.randn(7, 16).to(device),
            torch.randn(7, 12).to(device),
            torch.randn(7, 4).to(device),
        ]
        output_4 = fusion_4(features_4)
        assert output_4.shape == (7, 48)

    def test_fusion_module_components(self):
        """Test individual components of fusion module."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fusion = FeatureFusion(
            input_dims=[16, 32], output_dim=64
        ).to(device)
        # Test that fusion has expected components
        assert hasattr(fusion, "fusion")
        # Test forward pass
        features = [torch.randn(5, 16).to(device), torch.randn(5, 32).to(device)]
        output = fusion(features)
        assert output.shape == (5, 64)

    def test_with_config_object(self):
        """Test creating a fusion module from a configuration object."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Mock config
        encoder_config = EncoderConfig(
            photometry_dim=16,
            astrometry_dim=8,
            spectroscopy_dim=32,
        )
        graph_config = GraphConfig(hidden_dim=64)

        input_dims = [encoder_config.photometry_dim, encoder_config.astrometry_dim, encoder_config.spectroscopy_dim]
        fusion = FeatureFusion(input_dims=input_dims, output_dim=graph_config.hidden_dim).to(device)

        assert fusion.fusion[0].in_features == sum(input_dims)
        assert fusion.fusion[0].out_features == graph_config.hidden_dim
        
        # Test with features matching config
        features = [torch.randn(5, dim).to(device) for dim in input_dims]
        output = fusion(features)
        assert output.shape == (5, 64)


class TestLayerFactoryIntegration:
    """Test integration with LayerFactory."""

    def test_layer_factory_with_base_gnn(self):
        """Test LayerFactory creating layers for BaseAstroGNN."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        conv_layer = LayerFactory.create_conv_layer("gcn", 16, 32).to(device)
        assert isinstance(conv_layer, nn.Module)

        # Test forward pass
        x = torch.randn(5, 16).to(device)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long).to(device)
        output = conv_layer(x, edge_index)
        assert output.shape == (5, 32)

    def test_layer_factory_mlp_for_fusion(self):
        """Test LayerFactory creating MLPs for feature fusion."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mlp = LayerFactory.create_mlp(
            64,  # input_dim
            [128],  # hidden_dims_or_output_dim
            32,  # output_dim
            activation="relu",
            dropout=0.1,
        ).to(device)

        # Test forward pass
        x = torch.randn(5, 64).to(device)
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BaseAstroGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            device=device,
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_gnn = BaseAstroGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            device=device,
        )

        temporal_gnn = BaseTemporalGNN(
            input_dim=16,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            recurrent_type="lstm",
            recurrent_layers=2,
            device=device,
        )

        assert base_gnn.hidden_dim == temporal_gnn.hidden_dim
        assert base_gnn.conv_type == temporal_gnn.conv_type
        assert base_gnn.num_layers == temporal_gnn.num_layers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
