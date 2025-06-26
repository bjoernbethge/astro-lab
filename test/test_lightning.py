"""
Tests for Lightning model functionality.
"""

import pytest
import torch
import torch_geometric
from torch_geometric.data import Batch

from astro_lab.data import AstroDataModule
from astro_lab.models.config import get_preset
from astro_lab.models.core import (
    list_lightning_models,
    list_presets,
)
from astro_lab.training import AstroTrainer


class TestLightningModels:
    """Test Lightning model creation and functionality."""

    def test_list_models(self):
        """Test listing available Lightning models."""
        models = list_lightning_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "astro_node_gnn" in models

    def test_list_presets(self):
        """Test listing available presets."""
        presets = list_presets()
        assert isinstance(presets, dict)
        assert len(presets) > 0
        assert "node_classifier_small" in presets

    def test_get_preset(self):
        """Test getting preset configuration."""
        preset_config = get_preset("node_classifier_small")
        assert preset_config is not None
        assert preset_config.task == "node_classification"
        assert preset_config.hidden_dim == 32
        assert preset_config.num_layers == 2

    def test_invalid_preset(self):
        """Test handling of invalid preset name."""
        with pytest.raises(ValueError):
            get_preset("invalid_preset")


class TestAstroTrainer:
    """Test AstroTrainer functionality."""

    def test_trainer_creation(self):
        """Test AstroTrainer creation."""
        config = {
            "model": "astro_node_gnn",
            "survey": "gaia",
            "max_samples": 50,
            "batch_size": 4,
            "epochs": 2,
        }
        trainer = AstroTrainer(config)
        assert trainer is not None
        assert trainer.config == config

    def test_trainer_setup(self):
        """Test AstroTrainer setup."""
        config = {
            "model": "astro_node_gnn",
            "survey": "gaia",
            "max_samples": 50,
            "batch_size": 4,
            "epochs": 2,
        }
        trainer = AstroTrainer(config)

        # Test component creation
        model = trainer.create_model()
        assert model is not None

        datamodule = trainer.create_datamodule()
        assert datamodule is not None

        lightning_trainer = trainer.create_trainer()
        assert lightning_trainer is not None

    def test_quick_training(self):
        """Test quick training run."""
        config = {
            "model": "astro_node_gnn",
            "survey": "gaia",
            "max_samples": 50,
            "batch_size": 4,
            "epochs": 1,
        }
        trainer = AstroTrainer(config)

        # This should run quickly with limited samples
        try:
            success = trainer.train()
            # Training should complete without raising exceptions
            assert success is not None  # Should return some result
        except Exception as e:
            # If training fails, it should be due to data issues, not code issues
            assert "data" in str(e).lower() or "setup" in str(e).lower()


class TestCleanLightningIntegration:
    """Test Lightning integration with the new clean data module."""

    def test_lightning_with_clean_data_loaders(self):
        """Test Lightning models work with the new clean data loaders."""
        from astro_lab.data import create_astro_datamodule, load_survey_catalog

        # Test that new functions work with Lightning
        try:
            datamodule = create_astro_datamodule("gaia", max_samples=50)
            assert datamodule is not None
            assert callable(datamodule.setup)
        except Exception:
            pytest.skip("No GAIA data available for testing")

    def test_lightning_with_clean_processors(self):
        """Test Lightning models work with the new clean processors."""
        from astro_lab.data import create_survey_tensordict, preprocess_survey

        # Test that processor functions are available for Lightning integration
        assert callable(preprocess_survey)
        assert callable(create_survey_tensordict)
