"""
Test models module for AstroLab (Lightning API).
"""

import pytest
import torch
from torch_geometric.data import Data

from astro_lab.models import AstroModel


class TestModelCreation:
    """Test model creation."""

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

    def test_astro_model_with_different_tasks(self):
        """Test AstroModel with different tasks."""
        tasks = ["node_classification", "graph_classification", "node_regression"]

        for task in tasks:
            model = AstroModel(
                num_features=10,
                num_classes=3,
                hidden_dim=64,
                num_layers=2,
                task=task,
            )
            assert model is not None
            assert model.task == task


class TestModelFunctionality:
    """Test model forward passes and features."""

    @pytest.fixture
    def sample_data(self):
        """Create sample graph data."""
        x = torch.randn(100, 10)
        edge_index = torch.randint(0, 100, (2, 200))
        batch = torch.zeros(100, dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)

    def test_node_classification_forward(self, sample_data):
        """Test node classification forward pass."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        out = model(sample_data)
        assert out.shape[0] == 100

    def test_graph_classification_forward(self, sample_data):
        """Test graph classification forward pass."""
        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="graph_classification",
        )
        out = model(sample_data)
        assert out.shape[1] == 3

    def test_node_regression_forward(self, sample_data):
        """Test node regression forward pass."""
        model = AstroModel(
            num_features=10,
            num_classes=1,  # Regression output
            hidden_dim=64,
            num_layers=2,
            task="node_regression",
        )
        out = model(sample_data)
        assert out.shape[0] == 100


class TestModelUtils:
    """Test model utilities."""

    def test_model_device_handling(self):
        """Test model device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = AstroModel(
            num_features=10,
            num_classes=3,
            hidden_dim=64,
            num_layers=2,
            task="node_classification",
        )
        model = model.cuda()

        # Test forward on GPU
        x = torch.randn(10, 10).cuda()
        edge_index = torch.randint(0, 10, (2, 20)).cuda()
        batch = torch.zeros(10, dtype=torch.long).cuda()
        data = Data(x=x, edge_index=edge_index, batch=batch)
        out = model(data)
        assert out.is_cuda


class TestModelConfigs:
    """Test model configurations."""

    @pytest.mark.parametrize(
        "task",
        ["node_classification", "graph_classification", "node_regression"],
    )
    def test_model_configs(self, task):
        """Test each task with different configs."""
        # Small config
        model = AstroModel(
            num_features=8,
            num_classes=2,
            hidden_dim=16,
            num_layers=2,
            task=task,
        )
        assert model is not None

        # Medium config
        model = AstroModel(
            num_features=32,
            num_classes=5,
            hidden_dim=64,
            num_layers=3,
            task=task,
        )
        assert model is not None

        # Large config
        model = AstroModel(
            num_features=128,
            num_classes=10,
            hidden_dim=256,
            num_layers=4,
            task=task,
        )
        assert model is not None
