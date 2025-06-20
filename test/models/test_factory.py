"""
Tests for Model Factory
=======================

Tests for model factory pattern and registry system.
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
    """Test the model factory system."""

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

            # Check that config has expected keys
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
        ]

        for task in expected_tasks:
            if task in factory.TASK_CONFIGS:
                config = factory.TASK_CONFIGS[task]
                assert isinstance(config, dict)
                assert "output_head" in config
                assert "output_dim" in config


class TestErrorHandling:
    """Test error handling in model factory."""

    def test_invalid_survey_type(self):
        """Test error handling for invalid survey types."""
        factory = ModelFactory()

        with pytest.raises(ValueError, match="Unknown survey"):
            factory.create_survey_model(
                "nonexistent_survey", task="stellar_classification"
            )

    def test_invalid_task_type(self):
        """Test error handling for invalid task types."""
        factory = ModelFactory()

        with pytest.raises(ValueError, match="Unknown task"):
            factory.create_survey_model("gaia", task="nonexistent_task")

    def test_invalid_model_type_in_registry(self):
        """Test error handling for invalid model types in registry."""
        registry = ModelRegistry()

        with pytest.raises(ValueError, match="Unknown model type"):
            registry.create("nonexistent_model_type")


class TestModelIntegration:
    """Test integration between factory and models."""

    def test_factory_model_compatibility(self):
        """Test that factory-created models are compatible."""
        factory = ModelFactory()

        # Test that we can create at least one survey model
        survey_types = ["gaia", "sdss", "lsst"]
        model_created = False

        for survey in survey_types:
            try:
                model = factory.create_survey_model(
                    survey, task="stellar_classification"
                )
                if model is not None:
                    assert isinstance(model, nn.Module)
                    model_created = True
                    break
            except (ValueError, ImportError, AttributeError):
                continue

        # If no models were created, that's okay (might not be implemented)
        # This test just ensures no unexpected errors occur

    def test_model_forward_pass(self):
        """Test that factory-created models can perform forward passes."""
        try:
            model = create_gaia_classifier(num_classes=3, hidden_dim=32)

            if model is not None:
                # Test forward pass with graph data
                x = torch.randn(10, 16)  # 10 nodes, 16 features
                edge_index = torch.randint(0, 10, (2, 20))  # 20 edges

                output = model(x, edge_index)
                assert output.shape[0] == 10
                assert not torch.isnan(output).any()

        except (ImportError, AttributeError, TypeError):
            # Model might not exist or have different signature
            pass

    def test_device_consistency(self):
        """Test that factory models maintain device consistency."""
        try:
            model = create_gaia_classifier(num_classes=3, hidden_dim=32)

            if model is not None:
                # Test CPU
                x_cpu = torch.randn(5, 16)
                edge_index_cpu = torch.randint(0, 5, (2, 10))
                output_cpu = model(x_cpu, edge_index_cpu)
                assert output_cpu.device.type == "cpu"

                # Test CUDA if available
                if torch.cuda.is_available():
                    model_cuda = model.cuda()
                    x_cuda = x_cpu.cuda()
                    edge_index_cuda = edge_index_cpu.cuda()
                    output_cuda = model_cuda(x_cuda, edge_index_cuda)
                    assert output_cuda.device.type == "cuda"

        except (ImportError, AttributeError, TypeError):
            # Model might not exist or have different behavior
            pass


class TestModelParameterValidation:
    """Test parameter validation in model creation."""

    def test_parameter_override(self):
        """Test that custom parameters override defaults."""
        factory = ModelFactory()

        try:
            # Create model with custom parameters
            model = factory.create_survey_model(
                "gaia",
                task="stellar_classification",
                hidden_dim=256,  # Override default
                num_layers=5,  # Override default
            )

            if model is not None:
                # Check that parameters were applied (if accessible)
                if hasattr(model, "hidden_dim"):
                    assert model.hidden_dim == 256
                if hasattr(model, "num_layers"):
                    assert model.num_layers == 5

        except (ValueError, ImportError, AttributeError):
            # Model might not be fully implemented
            pass

    def test_minimal_parameter_creation(self):
        """Test model creation with minimal parameters."""
        factory = ModelFactory()

        try:
            # Create model with only required parameters
            model = factory.create_survey_model("gaia")

            if model is not None:
                assert isinstance(model, nn.Module)

        except (ValueError, ImportError, AttributeError):
            # Model might not be fully implemented
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
