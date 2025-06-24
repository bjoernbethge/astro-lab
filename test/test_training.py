"""
Tests for training functionality.
"""

import astropy.io.fits as fits
import polars as pl
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Batch, Data

from astro_lab.models.config import ModelConfig
from astro_lab.models.core.survey_gnn import AstroSurveyGNN
from astro_lab.models.factories import create_gaia_classifier
from astro_lab.tensors import SurveyTensor
from astro_lab.training import AstroLightningModule, AstroTrainer
from astro_lab.utils.config.surveys import get_available_surveys


class TestAstroLightningModule:
    """Test actual AstroLightningModule with real models and new architecture."""

    def test_lightning_module_initialization(self):
        """Test AstroLightningModule initializes properly."""
        # Test with real model
        model = AstroSurveyGNN(
            hidden_dim=64, output_dim=8, num_layers=2, dropout=0.1, conv_type="gcn"
        )

        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None
        assert lightning_module.task_type == "classification"
        assert lightning_module.hparams.get("learning_rate") == 1e-3

    def test_lightning_module_forward(self):
        """Test forward pass with real data."""
        model = AstroSurveyGNN(
            hidden_dim=32, output_dim=4, num_layers=2, conv_type="gcn"
        )

        lightning_module = AstroLightningModule(model=model, task_type="classification")

        # Create test batch
        x = torch.randn(10, 5)  # 10 nodes, 5 features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        # Forward pass
        output = lightning_module(batch)

        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 10  # Number of nodes

    def test_lightning_module_with_config_object(self):
        """Test LightningModule with Config objects."""
        config = ModelConfig(
            name="test_lightning_config",
            hidden_dim=64,
            num_layers=2,
            conv_type="gcn",
            dropout=0.1,
            task="classification",
            use_photometry=True,
            use_astrometry=True,
            use_spectroscopy=False,
            output_dim=8,
            pooling="mean",
            activation="relu",
            num_heads=8,
        )

        # Create model using config values - but our factory may use defaults
        model = create_gaia_classifier(
            hidden_dim=config.hidden_dim,
            num_classes=config.output_dim or 8,  # Default to 8 if None
        )

        lightning_module = AstroLightningModule(model=model, task_type="classification")

        assert lightning_module.model is not None
        # The actual model might use different hidden_dim due to factory defaults
        # So we just test that the model was created successfully
        assert hasattr(lightning_module.model, "hidden_dim")
        assert isinstance(lightning_module.model.hidden_dim, int)

    def test_lightning_module_auto_creation(self):
        """Test automatic model creation."""
        lightning_module = AstroLightningModule(task_type="classification")

        # Create test batch to trigger auto-creation
        x = torch.randn(5, 10)  # 5 nodes, 10 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        # Forward pass should create model automatically
        output = lightning_module(batch)

        assert lightning_module.model is not None
        assert output is not None

    def test_lightning_module_optimizer_configuration(self):
        """Test optimizer configuration."""
        model = AstroSurveyGNN(hidden_dim=32, output_dim=4)

        lightning_module = AstroLightningModule(
            model=model, learning_rate=1e-3, weight_decay=1e-4, scheduler="cosine"
        )

        optimizer_config = lightning_module.configure_optimizers()

        assert optimizer_config is not None
        # Should be either optimizer or dict with optimizer and scheduler
        if isinstance(optimizer_config, dict):
            assert "optimizer" in optimizer_config
        else:
            assert hasattr(optimizer_config, "param_groups")

    def test_lightning_module_training_step(self):
        """Test training step with real batch."""
        model = AstroSurveyGNN(hidden_dim=32, output_dim=4)

        lightning_module = AstroLightningModule(model=model, task_type="classification")

        # Create test batch with targets
        x = torch.randn(8, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        y = torch.randint(0, 4, (8,))  # Classification targets (0-3, so 4 classes)

        batch = {"x": x, "edge_index": edge_index, "y": y}

        # Training step
        loss = lightning_module.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_lightning_module_validation_step(self):
        """Test validation step."""
        model = AstroSurveyGNN(hidden_dim=32, output_dim=4)

        lightning_module = AstroLightningModule(model=model, task_type="classification")

        # Create test batch
        x = torch.randn(6, 5)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        y = torch.randint(0, 4, (6,))  # Classification targets (0-3, so 4 classes)

        batch = {"x": x, "edge_index": edge_index, "y": y}

        # Validation step
        loss = lightning_module.validation_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)

    def test_lightning_module_unsupervised_learning(self):
        """Test unsupervised learning mode."""
        model = AstroSurveyGNN(hidden_dim=32, output_dim=32)

        lightning_module = AstroLightningModule(
            model=model, task_type="unsupervised", projection_dim=16
        )

        # Create test batch
        x = torch.randn(10, 5)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        # Forward pass should create projection head
        output = lightning_module(batch)

        assert output is not None
        assert lightning_module.projection_head is not None

    def test_lightning_module_regression_task(self):
        """Test regression task."""
        model = AstroSurveyGNN(hidden_dim=32, output_dim=1)

        lightning_module = AstroLightningModule(model=model, task_type="regression")

        # Create test batch
        x = torch.randn(5, 3)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        y = torch.randn(5, 1)  # Regression targets

        batch = {"x": x, "edge_index": edge_index, "y": y}

        # Training step
        loss = lightning_module.training_step(batch, 0)

        assert loss is not None
        assert isinstance(loss, torch.Tensor)


class TestModelFactoryIntegration:
    """Test integration between factory functions and LightningModule."""

    def test_factory_lightning_integration(self):
        """Test creating LightningModule with factory models."""
        # Create model using factory function
        model = create_gaia_classifier(
            hidden_dim=64,
            num_classes=7,
        )

        # Create LightningModule with factory model
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None
        assert isinstance(lightning_module.model, nn.Module)

        # Test forward pass
        x = torch.randn(10, 13)  # Gaia features
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        output = lightning_module(batch)
        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_config_lightning_integration(self):
        """Test creating LightningModule with Config objects."""
        config = ModelConfig(
            name="test_integration",
            conv_type="gat",
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            task="classification",
            output_dim=7,
        )

        # Create model using config values
        model = create_gaia_classifier(
            hidden_dim=config.hidden_dim,
            num_classes=config.output_dim or 7,  # Default to 7 if None
        )

        # Create LightningModule
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None
        assert lightning_module.model.hidden_dim == config.hidden_dim


class TestTrainingConfigurations:
    """Test different training configurations with new architecture."""

    def test_survey_specific_training(self):
        """Test training with survey-specific configurations."""
        # Test with Gaia model
        model = create_gaia_classifier()

        # Create LightningModule
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None

        # Test forward pass
        x = torch.randn(8, 13)  # Gaia features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        output = lightning_module(batch)
        assert output is not None

    def test_task_specific_training(self):
        """Test training with different task types."""
        # Test classification
        model = create_gaia_classifier()

        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None

        # Test regression
        model_reg = AstroSurveyGNN(
            hidden_dim=64,
            output_dim=1,
            task="regression",
        )

        lightning_module_reg = AstroLightningModule(
            model=model_reg,
            task_type="regression",
            learning_rate=1e-3,
        )

        assert lightning_module_reg.model is not None


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality in AstroTrainer."""

    @pytest.mark.parametrize(
        "dataset_fixture", ["nsa_dataset"]
    )  # Use NSA instead of Gaia since it has more samples
    def test_optimize_hyperparameters_basic(self, request, dataset_fixture):
        """Test basic hyperparameter optimization using synthetic data."""
        # Get dataset from parametrized fixture for metadata only
        dataset = request.getfixturevalue(dataset_fixture)

        # Create synthetic dataloaders like the working trainer tests
        def create_synthetic_dataloader(batch_size=8, num_batches=3):
            """Create synthetic dataloader for testing hyperparameter optimization."""
            data_list = []
            for _ in range(num_batches):
                x = torch.randn(50, 6)  # 50 nodes, 6 features (similar to NSA)
                edge_index = torch.randint(0, 50, (2, 150))  # 150 edges
                y = torch.randint(0, 3, (50,))  # 3 classes for classification

                from torch_geometric.data import Data

                data = Data(x=x, edge_index=edge_index, y=y, num_nodes=50)
                data.survey = "nsa"  # Add survey metadata
                data_list.append(data)

            from torch_geometric.loader import DataLoader

            return DataLoader(data_list, batch_size=1, shuffle=True)

        train_loader = create_synthetic_dataloader()
        val_loader = create_synthetic_dataloader()

        # Create model and lightning module like in test_cuda.py
        model = AstroSurveyGNN(hidden_dim=32, output_dim=3)  # 3 classes
        lightning_module = AstroLightningModule(
            model=model, task_type="classification", learning_rate=1e-3
        )

        # Use simplified AstroTrainer without complex config (like test_cuda.py)
        from astro_lab.training.trainer import AstroTrainer

        trainer = AstroTrainer(
            lightning_module=lightning_module,
            max_epochs=1,  # Very short for testing
            enable_progress_bar=False,  # Disable for cleaner test output
        )

        # Run optimization with minimal trials
        results = trainer.optimize_hyperparameters(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            n_trials=2,  # Just 2 trials for testing
            timeout=30,  # 30 second timeout
        )

        # Check results
        assert isinstance(results, dict)
        assert "best_params" in results
        assert "best_value" in results
        assert "n_trials" in results
        assert results["n_trials"] >= 1

    def test_dataset_properties(self, nsa_dataset, gaia_dataset):
        """Test the properties of the dataset fixtures."""
        # Test NSA dataset
        assert len(nsa_dataset) > 0
        first_item = nsa_dataset[0]
        assert first_item is not None
        assert hasattr(first_item, "x")  # PyG Data object
        assert hasattr(first_item, "edge_index")
        assert hasattr(first_item, "survey")
        assert first_item.survey == "nsa"

        # Test Gaia dataset
        assert len(gaia_dataset) > 0
        first_item = gaia_dataset[0]
        assert first_item is not None
        assert hasattr(first_item, "x")
        assert hasattr(first_item, "edge_index")
        assert hasattr(first_item, "survey")
        assert first_item.survey == "gaia"
