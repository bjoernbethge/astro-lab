"""Tests for Model Factory Functions."""

import pytest
import torch
import torch.nn as nn

from astro_lab.models import (
    ALCDEFTemporalGNN,
    AstroPhotGNN,
    AstroSurveyGNN,
    ModelConfig,
    TemporalGCN,
    create_asteroid_period_detector,
    create_gaia_classifier,
    create_lsst_transient_detector,
    create_sdss_galaxy_model,
    get_predefined_config,
)
from astro_lab.models.components import (
    create_conv_layer,
    create_mlp,
    create_output_head,
)
from astro_lab.models.factories import (
    create_galaxy_modeler,
    create_lightcurve_classifier,
    create_temporal_graph_model,
)
from astro_lab.tensors import LightcurveTensorDict, PhotometricTensorDict


@pytest.fixture
def mock_survey_data():
    """Mock survey data for testing."""
    photometry_data = torch.randn(10, 5)
    edge_index = torch.randint(0, 10, (2, 20), dtype=torch.long)
    return {"photometry": photometry_data, "edge_index": edge_index}


@pytest.fixture
def mock_temporal_data():
    """Mock temporal data for testing."""
    n_points = 20
    lightcurve_data = torch.randn(n_points, 2)
    edge_index = torch.stack(
        [torch.arange(0, n_points - 1), torch.arange(1, n_points)], dim=0
    )
    return {
        "lightcurve": lightcurve_data,
        "edge_index": edge_index,
    }


class TestModelCreation:
    """Test basic model creation."""

    def test_astro_survey_gnn_creation(self):
        """Test creating AstroSurveyGNN."""
        model = AstroSurveyGNN(
            input_dim=10, hidden_dim=128, output_dim=7, task="classification"
        )
        assert isinstance(model, nn.Module)
        assert model.hidden_dim == 128
        assert model.output_dim == 7

    def test_temporal_gcn_creation(self):
        """Test creating TemporalGCN."""
        model = TemporalGCN(input_dim=5, hidden_dim=64, output_dim=1, task="regression")
        assert isinstance(model, nn.Module)

    def test_alcdef_temporal_gnn_creation(self):
        """Test creating ALCDEFTemporalGNN."""
        model = ALCDEFTemporalGNN(hidden_dim=64, task="period_detection")
        assert isinstance(model, nn.Module)

    def test_astrophot_gnn_creation(self):
        """Test creating AstroPhotGNN."""
        model = AstroPhotGNN(
            input_dim=20, hidden_dim=128, model_components=["sersic", "disk"]
        )
        assert isinstance(model, nn.Module)


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_gaia_classifier(self):
        """Test creating Gaia classifier model."""
        # Should raise error without num_classes
        with pytest.raises(ValueError, match="num_classes must be specified"):
            create_gaia_classifier()

        # Should work with num_classes
        model = create_gaia_classifier(num_classes=8)  # Gaia data has 8 classes
        assert isinstance(model, AstroSurveyGNN)
        assert model.get_num_parameters() > 0

    def test_create_sdss_galaxy_model(self):
        """Test creating SDSS galaxy model."""
        # Should raise error without output_dim
        with pytest.raises(ValueError, match="output_dim must be specified"):
            create_sdss_galaxy_model()

        # Should work with output_dim
        model = create_sdss_galaxy_model(output_dim=5)
        assert isinstance(model, AstroSurveyGNN)
        assert model.get_num_parameters() > 0

    def test_create_lsst_transient_detector(self):
        """Test creating LSST transient detector."""
        # Should raise error without num_classes
        with pytest.raises(ValueError, match="num_classes must be specified"):
            create_lsst_transient_detector()

        # Should work with num_classes
        model = create_lsst_transient_detector(num_classes=2)
        assert isinstance(model, AstroSurveyGNN)
        assert model.get_num_parameters() > 0

    def test_create_asteroid_period_detector(self):
        """Test asteroid period detector factory."""
        model = create_asteroid_period_detector()
        assert isinstance(model, ALCDEFTemporalGNN)
        assert model.task == "period_detection"

    def test_create_lightcurve_classifier(self):
        """Test lightcurve classifier factory."""
        model = create_lightcurve_classifier(num_classes=3)
        assert isinstance(model, ALCDEFTemporalGNN)
        assert model.task == "classification"

    def test_create_galaxy_modeler(self):
        """Test galaxy modeler factory."""
        model = create_galaxy_modeler(model_components=["sersic"])
        assert isinstance(model, AstroPhotGNN)
        assert "sersic" in model.model_components

    def test_create_temporal_graph_model(self):
        """Test temporal graph model factory."""
        model = create_temporal_graph_model(input_dim=10, output_dim=5)
        assert isinstance(model, TemporalGCN)


class TestConfiguration:
    """Test configuration system."""

    def test_model_config_creation(self):
        """Test creating ModelConfig."""
        config = ModelConfig(
            name="test_model", hidden_dim=256, conv_type="gat", task="classification"
        )
        assert config.name == "test_model"
        assert config.hidden_dim == 256
        assert config.conv_type == "gat"

    def test_get_predefined_config(self):
        """Test getting predefined configs."""
        config = get_predefined_config("gaia_classifier")
        assert isinstance(config, ModelConfig)
        assert config.name == "gaia_classifier"
        assert config.task == "classification"

    def test_config_to_dict(self):
        """Test config to dict conversion."""
        config = ModelConfig(name="test")
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test"


class TestComponents:
    """Test component functions."""

    def test_create_conv_layer(self):
        """Test conv layer creation."""
        # GCN
        gcn = create_conv_layer("gcn", 64, 128)
        assert isinstance(gcn, nn.Module)

        # GAT
        gat = create_conv_layer("gat", 64, 128, heads=8)
        assert isinstance(gat, nn.Module)

        # SAGE
        sage = create_conv_layer("sage", 64, 128)
        assert isinstance(sage, nn.Module)

    def test_create_mlp(self):
        """Test MLP creation."""
        mlp = create_mlp(64, 10, [128, 64])
        assert isinstance(mlp, nn.Module)

        # Test forward pass
        x = torch.randn(5, 64)
        output = mlp(x)
        assert output.shape == (5, 10)

    def test_create_output_head(self):
        """Test output head creation."""
        # Classification
        clf_head = create_output_head("classification", 128, 10)
        assert isinstance(clf_head, nn.Module)

        # Regression
        reg_head = create_output_head("regression", 128, 1)
        assert isinstance(reg_head, nn.Module)

        # Period detection
        period_head = create_output_head("period_detection", 128)
        assert isinstance(period_head, nn.Module)


class TestModelForwardPass:
    """Test model forward passes."""

    def test_survey_gnn_forward(self, mock_survey_data):
        """Test AstroSurveyGNN forward pass."""
        model = create_gaia_classifier(num_classes=7)

        # Test with PyG-style data
        x = torch.randn(10, 13)  # 10 nodes, 13 features
        edge_index = mock_survey_data["edge_index"]

        output = model(x, edge_index)
        assert output.shape == (10, 7)

    def test_temporal_forward(self, mock_temporal_data):
        """Test temporal model forward pass."""
        model = create_asteroid_period_detector()

        lightcurve = mock_temporal_data["lightcurve"]
        edge_index = mock_temporal_data["edge_index"]

        output = model(lightcurve, edge_index)
        # Period detection returns dict
        assert isinstance(output, dict)
        assert "period" in output
        assert "uncertainty" in output

    def test_galaxy_model_forward(self):
        """Test galaxy model forward pass."""
        model = create_galaxy_modeler()

        x = torch.randn(5, 20)  # 5 galaxies, 20 features
        edge_index = torch.randint(0, 5, (2, 10))

        output = model(x, edge_index)
        # Check actual output_dim from model
        expected_dim = model.output_dim
        assert output.shape == (1, expected_dim)

        # Test with component output
        output_components = model(x, edge_index, return_components=True)
        assert isinstance(output_components, dict)
        assert "sersic" in output_components
        assert "disk" in output_components
        assert "global" in output_components


class TestDeviceHandling:
    """Test device handling."""

    def test_model_device_cpu(self):
        """Test model on CPU."""
        model = create_gaia_classifier(device="cpu")
        assert next(model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_model_device_cuda(self):
        """Test model on CUDA."""
        model = create_gaia_classifier(device="cuda")
        assert next(model.parameters()).device.type == "cuda"

    def test_auto_device_movement(self):
        """Test automatic device movement in forward pass."""
        model = create_gaia_classifier()
        device = next(model.parameters()).device

        # Create data on different device
        x = torch.randn(10, 13)
        edge_index = torch.randint(0, 10, (2, 20))

        # Model should handle device movement
        output = model(x, edge_index)
        assert output.device == device


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_conv_type(self):
        """Test invalid convolution type."""
        with pytest.raises(ValueError):
            create_conv_layer("invalid_conv", 64, 128)

    def test_invalid_output_head(self):
        """Test invalid output head type."""
        with pytest.raises(ValueError):
            create_output_head("invalid_head", 128)

    def test_invalid_config_name(self):
        """Test invalid config name."""
        with pytest.raises(ValueError):
            get_predefined_config("invalid_config")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
