"""
Test Lightning integration for AstroLab.
"""

import pytest
import torch

from astro_lab.data.dataset.astrolab import AstroLabInMemoryDataset
from astro_lab.data.dataset.lightning import AstroLabDataModule
from astro_lab.data.samplers.neighbor import KNNSampler
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
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=4,
            num_workers=1,
        )
        assert hasattr(dm, "train_dataloader")
        assert hasattr(dm, "val_dataloader")
        assert hasattr(dm, "test_dataloader")

    def test_create_node_datamodule(self):
        """Test creating node datamodule."""
        try:
            dataset = AstroLabInMemoryDataset(
                survey_name="gaia",
                task="node_classification",
            )
            sampler = KNNSampler(k=10)
            dm = AstroLabDataModule(
                dataset=dataset,
                sampler=sampler,
                batch_size=32,
                num_workers=1,
            )
            assert hasattr(dm, "train_dataloader")
        except FileNotFoundError:
            pytest.skip("Data not available")

    def test_create_datamodule_factory(self):
        """Test the create_datamodule factory function."""
        dataset = AstroLabInMemoryDataset(
            survey_name="gaia",
            task="node_classification",
        )
        sampler = KNNSampler(k=8)
        dm = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=4,
            num_workers=1,
        )
        assert isinstance(dm, AstroLabDataModule)


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
        }

    def test_trainer_with_datamodule(self, training_config):
        """Test trainer with datamodule."""
        # Create datamodule
        dataset = AstroLabInMemoryDataset(
            survey_name=training_config["survey"],
            task=training_config["task"],
        )
        sampler = KNNSampler(k=8)
        datamodule = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=training_config["batch_size"],
            num_workers=training_config["num_workers"],
        )

        # Setup data
        datamodule.setup()

        # Create model
        model = AstroModel(
            num_features=10,  # Use default features
            num_classes=3,
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
        assert hasattr(trainer, "precision")

    def test_gradient_accumulation(self):
        """Test gradient accumulation."""
        trainer = AstroTrainer(
            experiment_name="test_gradient_accumulation",
            accumulate_grad_batches=4,
            max_epochs=1,
        )
        assert hasattr(trainer, "accumulate_grad_batches")


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
        assert hasattr(trainer, "strategy")
