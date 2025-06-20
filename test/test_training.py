"""
Tests for astro_lab.training - Real Training Classes Only
========================================================

Tests only actual implemented training classes.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from astro_lab.models.astro import AstroSurveyGNN
from astro_lab.training import AstroLightningModule


class TestAstroLightningModule:
    """Test actual AstroLightningModule with real models."""

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

    def test_lightning_module_config_creation(self):
        """Test model creation from config."""
        config = {
            "type": "gaia_classifier",
            "params": {
                "hidden_dim": 64,
                "num_classes": 8,
            },
        }

        lightning_module = AstroLightningModule(
            model_config=config, task_type="classification"
        )

        assert lightning_module.model is not None

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
        y = torch.randint(0, 4, (8,))  # Classification targets

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
        y = torch.randint(0, 4, (6,))

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
