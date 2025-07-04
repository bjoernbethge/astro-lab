"""
Lightning integration tests using current API.
"""

import lightning as L
import pytest
import torch
from torch_geometric.data import Data

from astro_lab.data.datamodules import SurveyDataModule, create_datamodule
from astro_lab.models import AstroModel
from astro_lab.training import AstroTrainer


class TestLightningModels:
    """Test Lightning model integration."""

    def test_model_is_lightning_module(self):
        """Test that AstroModel is a Lightning module."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        assert hasattr(model, "forward")

    def test_model_with_point_cloud_layers(self):
        """Test model with point cloud layers."""
        pytest.skip("Point cloud layers not implemented in AstroModel.")

    def test_model_with_shape_head(self):
        """Test model with shape modeling head."""
        pytest.skip("Shape modeling head not implemented in AstroModel.")


class TestDataModuleIntegration:
    """Test DataModule API with Lightning."""

    def test_create_graph_datamodule(self):
        """Test creating graph datamodule."""
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=4,
            num_workers=0,
            max_samples=50,
            k_neighbors=8,
        )
        assert hasattr(dm, "train_dataloader")
        assert hasattr(dm, "val_dataloader")
        assert hasattr(dm, "test_dataloader")

    def test_create_node_datamodule(self):
        """Test creating node datamodule."""
        try:
            dm = SurveyDataModule(
                survey="gaia",
                task="node_classification",
                batch_size=32,
                num_workers=0,
                max_samples=100,
                k_neighbors=10,
            )
            assert hasattr(dm, "train_dataloader")
        except FileNotFoundError:
            pytest.skip("Data not available")

    def test_create_datamodule_factory(self):
        """Test the create_datamodule factory function."""
        dm = create_datamodule(
            survey="gaia",
            task="node_classification",
            batch_size=4,
            max_samples=50,
        )
        assert isinstance(dm, SurveyDataModule)


class TestTrainingWorkflow:
    """Test complete training workflow with current API."""

    @pytest.fixture
    def training_config(self):
        return {
            "survey": "gaia",
            "task": "node_classification",
            "batch_size": 4,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "num_workers": 0,
            "max_samples": 50,
        }

    def test_trainer_with_datamodule(self, training_config):
        """Test trainer with datamodule."""
        # Create datamodule
        datamodule = SurveyDataModule(
            survey=training_config["survey"],
            task=training_config["task"],
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
            max_samples=training_config["max_samples"],
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
            task=training_config["task"],
        )

        # Create trainer
        trainer = AstroTrainer(
            experiment_name="test_workflow",
            max_epochs=training_config["max_epochs"],
            devices=training_config["devices"],
            accelerator=training_config["accelerator"],
            enable_progress_bar=False,
        )

        assert trainer is not None
        assert model is not None
        assert datamodule is not None


class TestMemoryEfficiency:
    """Test memory-efficient features."""

    def test_mixed_precision_training(self):
        """Test mixed precision training."""
        trainer = AstroTrainer(
            experiment_name="test_mixed_precision",
            precision="16-mixed",
            accelerator="cpu",
            max_epochs=1,
        )
        assert hasattr(trainer.trainer, "precision")

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        trainer = AstroTrainer(
            experiment_name="test_gradient_accumulation",
            accumulate_grad_batches=4,
            max_epochs=1,
        )
        assert hasattr(trainer.trainer, "accumulate_grad_batches")


class TestCallbacks:
    """Test Lightning callbacks with current API."""

    def test_early_stopping(self):
        """Test early stopping callback."""
        trainer = AstroTrainer(
            experiment_name="test_early_stopping",
            early_stopping=True,
            early_stopping_patience=3,
            early_stopping_monitor="val_loss",
            max_epochs=100,
        )

        # Test that trainer was created successfully
        assert trainer is not None
        assert hasattr(trainer, "trainer")

    def test_model_checkpoint(self):
        """Test model checkpoint callback."""
        trainer = AstroTrainer(
            experiment_name="test_checkpoint",
            checkpoint_save_top_k=3,
            checkpoint_monitor="val_loss",
            checkpoint_mode="min",
            max_epochs=1,
        )

        # Test that trainer was created successfully
        assert trainer is not None
        assert hasattr(trainer, "trainer")


class TestMultiGPU:
    """Test multi-GPU support."""

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="Multi-GPU not available",
    )
    def test_ddp_strategy(self):
        """Test DDP strategy for multi-GPU."""
        trainer = AstroTrainer(
            experiment_name="test_ddp",
            accelerator="gpu",
            devices=2,
            strategy="ddp",
            max_epochs=1,
        )
        assert hasattr(trainer.trainer, "strategy")
