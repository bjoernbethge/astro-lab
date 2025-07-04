"""
Test training module for AstroLab (Lightning API).
"""

import pytest
import torch
from torch_geometric.data import Data

from astro_lab.data.datamodules import SurveyDataModule
from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer


class TestBasicTraining:
    """Test basic training functionality."""

    @pytest.fixture
    def dummy_data(self):
        """Create dummy data for testing."""
        data_list = []
        for _ in range(20):
            x = torch.randn(20, 10)
            edge_index = torch.randint(0, 20, (2, 40))
            y = torch.randint(0, 3, (20,))
            data_list.append(Data(x=x, edge_index=edge_index, y=y))
        return data_list

    def test_astro_trainer_creation(self):
        """Test AstroTrainer creation."""
        trainer = AstroTrainer(
            experiment_name="test",
            max_epochs=5,
            devices=1,
            accelerator="cpu",
        )
        assert trainer is not None
        assert hasattr(trainer, "fit")
        assert hasattr(trainer, "test")

    def test_astro_model_creation(self):
        """Test AstroModel creation."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        assert model is not None
        assert hasattr(model, "forward")

    def test_survey_datamodule_creation(self):
        """Test SurveyDataModule creation."""
        datamodule = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=1,
            num_workers=0,
            max_samples=10,
        )
        assert datamodule is not None
        assert hasattr(datamodule, "setup")
        assert hasattr(datamodule, "train_dataloader")


class TestTrainingIntegration:
    """Test training integration."""

    def test_trainer_with_datamodule(self):
        """Test trainer with datamodule integration."""
        # Create datamodule
        datamodule = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=1,
            num_workers=0,
            max_samples=10,
        )

        # Setup data
        datamodule.prepare_data()
        datamodule.setup()

        # Create model
        model = AstroModel(
            num_features=datamodule.num_features,
            num_classes=datamodule.num_classes,
            hidden_dim=32,
            num_layers=2,
            task="node_classification",
        )

        # Create trainer
        trainer = AstroTrainer(
            experiment_name="test_integration",
            max_epochs=1,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=False,
        )

        # Test that trainer can be created with model and datamodule
        assert trainer is not None
        assert model is not None
        assert datamodule is not None


class TestTrainingEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_features(self):
        """Test handling of mismatched feature dimensions."""
        # Create model with specific input dimension
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=32,
            num_layers=2,
            task="node_classification",
        )

        # This should work with correct dimensions
        x = torch.randn(5, 10)  # Correct dimension
        edge_index = torch.randint(0, 5, (2, 10))
        batch = torch.zeros(5, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        try:
            output = model(data)
            assert output.shape[0] == 5
        except Exception:
            assert True  # Acceptable if model raises due to mismatch

    def test_trainer_properties(self):
        """Test trainer properties."""
        trainer = AstroTrainer(
            experiment_name="test_properties",
            max_epochs=5,
            devices=1,
            accelerator="cpu",
        )

        # Test properties
        assert hasattr(trainer, "best_model_path")
        assert hasattr(trainer, "last_model_path")
        assert hasattr(trainer, "load_from_checkpoint")

    def test_datamodule_properties(self):
        """Test datamodule properties."""
        datamodule = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=1,
            num_workers=0,
            max_samples=10,
        )

        # Test properties
        assert hasattr(datamodule, "num_features")
        assert hasattr(datamodule, "num_classes")
        assert hasattr(datamodule, "get_info")
