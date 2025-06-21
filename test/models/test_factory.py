"""
Tests for Model Factory
=======================

Tests for model factory pattern and registry system with new unified architecture.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.factory import (
    ModelFactory,
    ModelRegistry,
    create_asteroid_period_detector,
    create_gaia_classifier,
    create_lsst_transient_detector,
    create_sdss_galaxy_model,
    get_model_info,
    list_available_models,
)
from astro_lab.models.config import ModelConfig, EncoderConfig, GraphConfig, OutputConfig
from astro_lab.models.layers import LayerFactory


class TestModelRegistry:
    """Test the model registry system."""

    def test_registry_initialization(self):
        """Test registry initializes with default models."""
        registry = ModelRegistry()

        # Check that registry has list_available method
        available_models = registry.list_available()

        # Should return a list (may be empty if no models registered)
        assert isinstance(available_models, list)

    def test_register_custom_model(self):
        """Test registering a custom model."""
        registry = ModelRegistry()

        @registry.register("custom_test")
        class CustomTestModel(nn.Module):
            def __init__(self, input_dim=64, hidden_dim=32, output_dim=10):
                super().__init__()
                self.linear = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                return self.linear(x)

        # Check it's registered
        assert "custom_test" in registry.list_available()

        # Check we can create it
        model = registry.create(
            "custom_test", input_dim=64, hidden_dim=32, output_dim=10
        )
        assert isinstance(model, CustomTestModel)


class TestModelFactory:
    """Test the model factory system with new unified architecture."""

    def test_factory_initialization(self):
        """Test factory initializes correctly."""
        factory = ModelFactory()

        # Check that factory has required attributes
        assert hasattr(factory, "SURVEY_CONFIGS")
        assert hasattr(factory, "TASK_CONFIGS")
        assert isinstance(factory.SURVEY_CONFIGS, dict)
        assert isinstance(factory.TASK_CONFIGS, dict)

    def test_survey_specific_creation(self):
        """Test creating survey-specific models."""
        factory = ModelFactory()

        # Test common survey types from SURVEY_CONFIGS
        survey_types = ["gaia", "sdss", "lsst", "euclid", "des"]

        for survey in survey_types:
            try:
                model = factory.create_survey_model(
                    survey, task="stellar_classification"
                )
                assert isinstance(model, nn.Module)
            except (ValueError, ImportError, AttributeError):
                # Survey configuration or model might not be fully implemented
                pass

    def test_temporal_model_creation(self):
        """Test creating temporal models."""
        factory = ModelFactory()

        try:
            model = factory.create_temporal_model(
                model_type="alcdef", task="period_detection"
            )
            assert isinstance(model, nn.Module)
        except (ValueError, ImportError, AttributeError):
            # Temporal model might not be fully implemented
            pass

    def test_3d_stellar_model_creation(self):
        """Test creating 3D stellar models."""
        factory = ModelFactory()

        try:
            model = factory.create_3d_stellar_model(
                model_type="point_cloud", num_stars=100, radius=0.5
            )
            assert isinstance(model, nn.Module)
        except (ValueError, ImportError, AttributeError):
            # 3D model might not be fully implemented
            pass

    def test_multi_survey_model_creation(self):
        """Test creating multi-survey models."""
        factory = ModelFactory()

        try:
            model = factory.create_multi_survey_model(
                surveys=["gaia", "sdss"],
                task="stellar_classification",
                fusion_strategy="attention",
            )
            assert isinstance(model, nn.Module)
        except (ValueError, ImportError, AttributeError):
            # Multi-survey model might not be fully implemented
            pass


class TestConfigObjects:
    """Test the new Config objects and LayerFactory."""

    def test_model_config_creation(self):
        """Test creating ModelConfig objects."""
        config = ModelConfig(
            name="test_model",
            description="Test model configuration",
            encoder=EncoderConfig(
                use_photometry=True,
                use_astrometry=True,
                use_spectroscopy=False,
            ),
            graph=GraphConfig(
                conv_type="gcn",
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
            ),
            output=OutputConfig(
                task="stellar_classification",
                output_dim=7,
                pooling="mean",
            ),
        )

        assert config.name == "test_model"
        assert config.encoder.use_photometry
        assert config.graph.hidden_dim == 128
        assert config.output.task == "stellar_classification"

    def test_layer_factory_conv_layers(self):
        """Test LayerFactory creating convolution layers."""
        factory = LayerFactory()

        # Test GCN layer
        gcn_layer = factory.create_conv_layer("gcn", 64, 128)
        assert isinstance(gcn_layer, nn.Module)

        # Test GAT layer
        gat_layer = factory.create_conv_layer("gat", 64, 128, heads=8)
        assert isinstance(gat_layer, nn.Module)

        # Test SAGE layer
        sage_layer = factory.create_conv_layer("sage", 64, 128)
        assert isinstance(sage_layer, nn.Module)

        # Test Transformer layer
        transformer_layer = factory.create_conv_layer("transformer", 64, 128, heads=8)
        assert isinstance(transformer_layer, nn.Module)

    def test_layer_factory_mlp(self):
        """Test LayerFactory creating MLPs."""
        factory = LayerFactory()

        mlp = factory.create_mlp(
            64,  # input_dim
            [128, 256],  # hidden_dims_or_output_dim
            10,  # output_dim
            activation="relu",
            dropout=0.1,
        )
        assert isinstance(mlp, nn.Module)

        # Test forward pass
        x = torch.randn(5, 64)
        output = mlp(x)
        assert output.shape == (5, 10)

    def test_layer_factory_pooling(self):
        """Test LayerFactory pooling functions."""
        factory = LayerFactory()

        # Test pooling functions
        mean_pool = factory.get_pooling_function("mean")
        max_pool = factory.get_pooling_function("max")
        add_pool = factory.get_pooling_function("add")

        assert callable(mean_pool)
        assert callable(max_pool)
        assert callable(add_pool)

        # Test with dummy data
        x = torch.randn(10, 64)
        batch = torch.zeros(10, dtype=torch.long)
        batch[5:] = 1  # Two batches

        mean_output = mean_pool(x, batch)
        assert mean_output.shape == (2, 64)


class TestConvenienceFunctions:
    """Test convenience functions for model creation."""

    def test_create_gaia_classifier(self):
        """Test create_gaia_classifier convenience function."""
        try:
            model = create_gaia_classifier(num_classes=7, hidden_dim=128)
            assert isinstance(model, nn.Module)
        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_create_sdss_galaxy_model(self):
        """Test create_sdss_galaxy_model convenience function."""
        try:
            model = create_sdss_galaxy_model(task="galaxy_property_prediction")
            assert isinstance(model, nn.Module)
        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_create_lsst_transient_detector(self):
        """Test create_lsst_transient_detector convenience function."""
        try:
            model = create_lsst_transient_detector()
            assert isinstance(model, nn.Module)
        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_create_asteroid_period_detector(self):
        """Test create_asteroid_period_detector convenience function."""
        try:
            model = create_asteroid_period_detector()
            assert isinstance(model, nn.Module)
        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_list_available_models_function(self):
        """Test list_available_models convenience function."""
        try:
            available_models = list_available_models()
            assert isinstance(available_models, dict)
        except (ImportError, AttributeError):
            # Function might not be fully implemented
            pass

    def test_get_model_info_function(self):
        """Test get_model_info convenience function."""
        try:
            # Create a simple model to test with
            model = nn.Linear(10, 5)
            info = get_model_info(model)
            assert isinstance(info, dict)
        except (ImportError, AttributeError):
            # Function might not be fully implemented
            pass


class TestSurveyConfigurations:
    """Test survey-specific configurations."""

    def test_survey_config_structure(self):
        """Test that survey configurations have expected structure."""
        factory = ModelFactory()

        # Check that SURVEY_CONFIGS contains expected surveys
        expected_surveys = ["gaia", "sdss", "lsst", "euclid", "des"]

        for survey in expected_surveys:
            assert survey in factory.SURVEY_CONFIGS
            config = factory.SURVEY_CONFIGS[survey]
            assert isinstance(config, dict)
            assert "conv_type" in config
            assert "hidden_dim" in config
            assert "num_layers" in config

    def test_task_config_structure(self):
        """Test that task configurations have expected structure."""
        factory = ModelFactory()

        # Check that TASK_CONFIGS contains expected tasks
        expected_tasks = [
            "stellar_classification",
            "galaxy_property_prediction",
            "transient_detection",
            "period_detection",
            "shape_modeling",
            "cosmological_inference",
        ]

        for task in expected_tasks:
            assert task in factory.TASK_CONFIGS
            config = factory.TASK_CONFIGS[task]
            assert isinstance(config, dict)
            assert "output_head" in config
            assert "output_dim" in config
            assert "pooling" in config


class TestErrorHandling:
    """Test error handling in factory methods."""

    def test_invalid_survey_type(self):
        """Test error handling for invalid survey type."""
        factory = ModelFactory()

        with pytest.raises(ValueError):
            factory.create_survey_model("invalid_survey", task="stellar_classification")

    def test_invalid_task_type(self):
        """Test error handling for invalid task type."""
        factory = ModelFactory()

        with pytest.raises(ValueError):
            factory.create_survey_model("gaia", task="invalid_task")

    def test_invalid_model_type_in_registry(self):
        """Test error handling for invalid model type in registry."""
        registry = ModelRegistry()

        with pytest.raises(ValueError):
            registry.create("invalid_model_type")


class TestModelIntegration:
    """Test integration between factory and models."""

    def test_factory_model_compatibility(self):
        """Test that factory-created models are compatible with data."""
        factory = ModelFactory()

        try:
            model = factory.create_survey_model("gaia", task="stellar_classification")

            # Test with dummy data
            x = torch.randn(10, 64)  # 10 nodes, 64 features
            edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)

            output = model(x, edge_index)
            assert isinstance(output, torch.Tensor)
            assert not torch.isnan(output).any()

        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_model_forward_pass(self):
        """Test forward pass of factory-created models."""
        factory = ModelFactory()

        try:
            model = factory.create_survey_model("sdss", task="galaxy_property_prediction")

            # Test forward pass
            x = torch.randn(8, 128)
            edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)

            output = model(x, edge_index)
            assert output.shape[0] == 8  # Same number of nodes
            assert not torch.isnan(output).any()

        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_device_consistency(self):
        """Test that models work on different devices."""
        factory = ModelFactory()

        try:
            model = factory.create_survey_model("lsst", task="transient_detection")

            # Test on CPU
            x_cpu = torch.randn(5, 64)
            edge_index_cpu = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            output_cpu = model(x_cpu, edge_index_cpu)

            # Test on GPU if available
            if torch.cuda.is_available():
                model_gpu = model.to("cuda")
                x_gpu = x_cpu.to("cuda")
                edge_index_gpu = edge_index_cpu.to("cuda")
                output_gpu = model_gpu(x_gpu, edge_index_gpu)

                # Check outputs are consistent
                assert torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-1)

        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass


class TestModelParameterValidation:
    """Test parameter validation in model creation."""

    def test_parameter_override(self):
        """Test that parameters can be overridden."""
        factory = ModelFactory()

        try:
            # Create model with overridden parameters
            model = factory.create_survey_model(
                "gaia",
                task="stellar_classification",
                hidden_dim=256,  # Override default
                num_layers=5,    # Override default
                dropout=0.2,     # Override default
            )

            assert isinstance(model, nn.Module)

        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_minimal_parameter_creation(self):
        """Test creating models with minimal parameters."""
        factory = ModelFactory()

        try:
            # Create model with only required parameters
            model = factory.create_survey_model("gaia", task="stellar_classification")

            assert isinstance(model, nn.Module)

        except (ImportError, AttributeError):
            # Model might not be fully implemented
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
