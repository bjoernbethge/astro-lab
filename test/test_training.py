"""
Tests for astro_lab.training - Real Training Classes Only
========================================================

Tests only actual implemented training classes with new unified architecture.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch.utils.data import DataLoader, TensorDataset

from astro_lab.models.astro import AstroSurveyGNN
from astro_lab.models.factory import ModelFactory
from astro_lab.models.config import ModelConfig, EncoderConfig, GraphConfig, OutputConfig
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
        """Test model creation from Config object."""
        config = ModelConfig(
            name="test_lightning_config",
            description="Test lightning module with config",
            encoder=EncoderConfig(
                use_photometry=True,
                use_astrometry=True,
                use_spectroscopy=False,
            ),
            graph=GraphConfig(
                conv_type="gcn",
                hidden_dim=64,
                num_layers=2,
                dropout=0.1,
            ),
            output=OutputConfig(
                task="stellar_classification",
                output_dim=8,
                pooling="mean",
            ),
        )

        # Create model using ModelFactory with config
        model = ModelFactory.create_survey_model(
            survey="gaia",
            task="stellar_classification",
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
        )

        lightning_module = AstroLightningModule(
            model=model, task_type="classification"
        )

        assert lightning_module.model is not None
        assert lightning_module.model.hidden_dim == config.graph.hidden_dim

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
    """Test integration between ModelFactory and LightningModule."""

    def test_factory_lightning_integration(self):
        """Test creating LightningModule with ModelFactory models."""
        # Create model using ModelFactory
        model = ModelFactory.create_survey_model(
            survey="gaia",
            task="stellar_classification",
            hidden_dim=64,
            num_layers=2,
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
        x = torch.randn(10, 64)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        batch = Data(x=x, edge_index=edge_index)

        output = lightning_module(batch)
        assert output is not None
        assert isinstance(output, torch.Tensor)

    def test_config_lightning_integration(self):
        """Test creating LightningModule with Config objects."""
        config = ModelConfig(
            name="test_integration",
            description="Test config-lightning integration",
            graph=GraphConfig(
                conv_type="gat",
                hidden_dim=128,
                num_layers=3,
                dropout=0.1,
            ),
            output=OutputConfig(
                task="stellar_classification",
                output_dim=7,
                pooling="attention",
            ),
        )

        # Create model using config
        model = ModelFactory.create_survey_model(
            survey="gaia",
            task=config.output.task,
            hidden_dim=config.graph.hidden_dim,
            conv_type=config.graph.conv_type,
            num_layers=config.graph.num_layers,
            dropout=config.graph.dropout,
        )

        # Create LightningModule
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3,
        )

        assert lightning_module.model is not None
        assert lightning_module.model.hidden_dim == config.graph.hidden_dim


class TestTrainingConfigurations:
    """Test different training configurations with new architecture."""

    def test_survey_specific_training(self):
        """Test training with survey-specific configurations."""
        available_surveys = get_available_surveys()

        for survey in available_surveys:
            try:
                # Create survey-specific model
                model = ModelFactory.create_survey_model(
                    survey=survey,
                    task="stellar_classification",
                )

                # Create LightningModule
                lightning_module = AstroLightningModule(
                    model=model,
                    task_type="classification",
                    learning_rate=1e-3,
                )

                assert lightning_module.model is not None

                # Test forward pass
                x = torch.randn(8, 64)
                edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
                batch = Data(x=x, edge_index=edge_index)

                output = lightning_module(batch)
                assert output is not None

            except (ImportError, AttributeError):
                # Survey might not be fully implemented
                pass

    def test_task_specific_training(self):
        """Test training with different task types."""
        tasks = ["stellar_classification", "galaxy_property_prediction", "transient_detection"]

        for task in tasks:
            try:
                # Create task-specific model
                model = ModelFactory.create_survey_model(
                    survey="gaia",
                    task=task,
                )

                # Create LightningModule
                lightning_module = AstroLightningModule(
                    model=model,
                    task_type="classification" if "classification" in task else "regression",
                    learning_rate=1e-3,
                )

                assert lightning_module.model is not None

            except (ImportError, AttributeError):
                # Task might not be fully implemented
                pass


class TestHyperparameterOptimization:
    """Test hyperparameter optimization functionality in AstroTrainer."""

    def test_optimize_hyperparameters_basic(self, gaia_dataset):
        """Test basic hyperparameter optimization using Gaia dataset fixture."""
        # Create dataloaders from the fixture
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(gaia_dataset[:8], batch_size=2, shuffle=True)
        val_loader = DataLoader(gaia_dataset[8:10], batch_size=2, shuffle=False)
        
        # Create model and lightning module
        model = AstroSurveyGNN(hidden_dim=32, output_dim=4)
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification",
            learning_rate=1e-3
        )
        
        # Create trainer
        trainer = AstroTrainer(
            lightning_module=lightning_module,
            max_epochs=2,  # Very short for testing
            enable_progress_bar=False,
        )
        
        # Run optimization with minimal trials
        results = trainer.optimize_hyperparameters(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            n_trials=2,  # Just 2 trials for testing
            timeout=60,  # 1 minute timeout
        )
        
        # Check results
        assert isinstance(results, dict)
        assert "best_params" in results
        assert "best_value" in results
        assert "n_trials" in results
        assert results["n_trials"] >= 1

    def test_optimize_hyperparameters_custom_search_space(self, nsa_dataset):
        """Test optimization with custom search space using NSA dataset fixture."""
        # Create dataloaders from the fixture
        from torch_geometric.loader import DataLoader
        train_loader = DataLoader(nsa_dataset[:8], batch_size=2, shuffle=True)
        val_loader = DataLoader(nsa_dataset[8:10], batch_size=2, shuffle=False)
        
        # Create model
        model = AstroSurveyGNN(hidden_dim=64, output_dim=4)
        lightning_module = AstroLightningModule(
            model=model,
            task_type="classification"
        )
        
        trainer = AstroTrainer(
            lightning_module=lightning_module,
            max_epochs=1,
            enable_progress_bar=False,
        )
        
        # Define custom search space
        search_space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "hidden_dim": {"type": "int", "low": 32, "high": 128},
            "dropout": {"type": "float", "low": 0.1, "high": 0.3},
        }
        
        # Run optimization
        results = trainer.optimize_hyperparameters(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            n_trials=2,
            search_space=search_space,
        )
        
        # Check that custom parameters were optimized
        assert "learning_rate" in results["best_params"]
        assert "hidden_dim" in results["best_params"]
        assert "dropout" in results["best_params"]
