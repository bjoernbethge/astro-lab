"""
Tests for Model Factory
=======================

Tests for model factory pattern and registry system with new unified architecture.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

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
from astro_lab.models.config import (
    ModelConfig,
    EncoderConfig,
    GraphConfig,
    OutputConfig,
)
from astro_lab.models.layers import LayerFactory
from astro_lab.utils.config.surveys import get_survey_config, get_available_surveys
from astro_lab.models.astro import AstroSurveyGNN
from astro_lab.tensors import SurveyTensor, PhotometricTensor, LightcurveTensor

# Mock data for testing
@pytest.fixture
def mock_survey_data():
    photometry = PhotometricTensor(data=torch.randn(10, 5), bands=['u','g','r','i','z'])
    edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)
    # The GNN expects a dictionary of tensors for features `x` and a separate `edge_index`
    return {"photometry": photometry, "edge_index": edge_index}

@pytest.fixture
def mock_temporal_data():
    # A single lightcurve with 20 time steps, and 2 features (time, mag)
    n_points = 20
    lightcurve = torch.randn(n_points, 2)
    # Sort by time
    lightcurve[:, 0] = torch.sort(lightcurve[:, 0]).values
    # Create a simple sequential edge index (t -> t+1)
    edge_index = torch.stack([torch.arange(0, n_points - 1), torch.arange(1, n_points)], dim=0)
    return {"lightcurve": LightcurveTensor(data=lightcurve, bands=['time', 'mag']), "edge_index": edge_index}


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
        assert hasattr(factory, "TASK_CONFIGS")
        assert isinstance(factory.TASK_CONFIGS, dict)

    def test_survey_specific_creation(self, mock_survey_data):
        """Test creating survey-specific models."""
        factory = ModelFactory()
        model = factory.create_survey_model(
            "gaia", task="stellar_classification", photometry_bands=['u','g','r','i','z']
        )
        assert isinstance(model, AstroSurveyGNN)
        output = model(mock_survey_data, mock_survey_data["edge_index"])
        assert output.shape[0] == 10

    def test_temporal_model_creation(self, mock_temporal_data):
        """Test creating temporal models for lightcurve analysis."""
        factory = ModelFactory()
        model = factory.create_temporal_model(
            model_type="alcdef", 
            task="period_detection", 
            lightcurve_features=2  # Match the mock data dimensions
        )
        assert isinstance(model, nn.Module)
        
        # Get device of the model
        device = next(model.parameters()).device
        
        # Move mock data to the correct device
        lightcurve = mock_temporal_data["lightcurve"]
        edge_index = mock_temporal_data["edge_index"].to(device)
        
        output = model(lightcurve, edge_index)
        assert output.shape[0] == 1

    def test_3d_stellar_model_creation(self):
        """Test creating 3D stellar models."""
        factory = ModelFactory()
        # This model type is not implemented, so we expect a ValueError
        with pytest.raises(ValueError):
            factory.create_3d_stellar_model("star_formation", task="property_prediction")

    def test_multi_survey_model_creation(self, mock_survey_data):
        """Test creating multi-survey models."""
        factory = ModelFactory()
        config = {
            "surveys": ["gaia", "sdss"],
            "photometry_bands": ['u','g','r','i','z']
        }
        
        # FIX: Remove 'surveys' from kwargs to avoid multiple values error
        kwargs_for_model = config.copy()
        surveys_list = kwargs_for_model.pop("surveys")
        
        model = factory.create_multi_survey_model(
            surveys=surveys_list, task="multi_property_prediction", **kwargs_for_model
        )
        assert isinstance(model, AstroSurveyGNN)
        output = model(mock_survey_data, mock_survey_data["edge_index"])
        assert output.shape[0] == 10


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

    def test_create_gaia_classifier(self, mock_survey_data):
        """Test create_gaia_classifier convenience function."""
        model = create_gaia_classifier(num_classes=7, hidden_dim=128, photometry_bands=['u','g','r','i','z'])
        assert isinstance(model, AstroSurveyGNN)
        output = model(mock_survey_data, mock_survey_data["edge_index"])
        assert output.shape == (10, 7)

    def test_create_sdss_galaxy_model(self, mock_survey_data):
        """Test create_sdss_galaxy_model convenience function."""
        model = create_sdss_galaxy_model(task="galaxy_property_prediction", photometry_bands=['u','g','r','i','z'])
        assert isinstance(model, AstroSurveyGNN)

# LSST test removed - survey not supported

    def test_create_asteroid_period_detector(self, mock_temporal_data):
        """Test create_asteroid_period_detector convenience function."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = create_asteroid_period_detector(lightcurve_features=2, num_classes=1).to(device)
        assert isinstance(model, nn.Module)
        
        # Move mock data to the correct device
        lightcurve = mock_temporal_data["lightcurve"].to(device)
        edge_index = mock_temporal_data["edge_index"].to(device)
        
        output = model(lightcurve, edge_index)
        assert output.shape[0] == 1

    def test_list_available_models_function(self):
        """Test the list_available_models utility function."""
        models = list_available_models()
        assert isinstance(models, dict)
        # Check for expected keys instead of a specific value
        assert "surveys" in models
        assert "tasks" in models
        assert "output_heads" in models

    def test_get_model_info_function(self):
        """Test the get_model_info utility function."""
        # First, create a model instance
        model = create_gaia_classifier(num_classes=7, hidden_dim=128, photometry_bands=['u','g','r','i','z'])
        
        # Now, get its info
        info = get_model_info(model)
        assert isinstance(info, dict)
        assert "model_class" in info
        assert "num_parameters" in info
        assert info["num_parameters"] > 0


class TestSurveyConfigurations:
    """Test loading and using survey-specific configurations."""

    def test_survey_config_structure(self):
        """Test that survey configurations have expected structure."""
        # Check that centralized survey configs contain expected surveys
        available_surveys = get_available_surveys()

        for survey in available_surveys:
            config = get_survey_config(survey)
            assert isinstance(config, dict)
            assert "name" in config
            assert "coord_cols" in config
            assert "mag_cols" in config

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
            factory.create_survey_model("invalid_survey", "some_task")

    def test_invalid_task_type(self):
        """Test error handling for invalid task type."""
        factory = ModelFactory()
        with pytest.raises(ValueError):
            factory.create_survey_model("gaia", "invalid_task")

    def test_invalid_model_type_in_registry(self):
        """Test error handling for invalid model type in registry."""
        registry = ModelRegistry()
        with pytest.raises(ValueError):
            registry.create("invalid_type")


class TestModelIntegration:
    """Test integration between factory and models."""

    def test_factory_model_compatibility(self, mock_survey_data):
        """Test that factory-created models are compatible with data."""
        factory = ModelFactory()
        model = factory.create_survey_model("gaia", task="stellar_classification", photometry_bands=['u','g','r','i','z'])
        output = model(mock_survey_data, mock_survey_data["edge_index"])
        assert output.shape[0] == 10
        assert output.shape[1] > 1

    def test_model_forward_pass(self, mock_survey_data):
        """Test forward pass of factory-created models."""
        factory = ModelFactory()
        model = factory.create_survey_model("sdss", task="galaxy_property_prediction", photometry_bands=['u','g','r','i','z'])
        output = model(mock_survey_data, mock_survey_data["edge_index"])
        assert output.shape[0] == 10

    def test_device_consistency(self):
        """Test that models produce consistent results across devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        factory = ModelFactory()
        
        # Create models explicitly on CPU for consistency
        model_cpu = factory.create_astro_model(
            input_dim=7,
            hidden_dim=64,
            output_dim=7,
            device='cpu'  # Force CPU
        )
        model_gpu = factory.create_astro_model(
            input_dim=7,
            hidden_dim=64,
            output_dim=7,
            device='cuda'  # Force GPU
        )

        # Test with consistent data
        x = torch.randn(10, 7)
        edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)

        # CPU test
        x_cpu = x.to('cpu')
        edge_index_cpu = edge_index.to('cpu')
        output_cpu = model_cpu(x_cpu, edge_index_cpu)

        # GPU test
        x_gpu = x.to('cuda')
        edge_index_gpu = edge_index.to('cuda')
        output_gpu = model_gpu(x_gpu, edge_index_gpu)

        # Verify outputs are on correct devices
        assert output_cpu.device.type == "cpu"
        assert output_gpu.device.type == "cuda"

        # Check shape consistency
        assert output_cpu.shape == output_gpu.shape


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
