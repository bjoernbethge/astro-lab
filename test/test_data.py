"""
Tests for AstroLab data module.

Uses real data to test the complete data pipeline with the new clean structure.
"""

from pathlib import Path

import pytest
import torch

from astro_lab.data import AstroDataModule
from astro_lab.data.datasets import SurveyGraphDataset
from astro_lab.data.graphs import create_knn_graph
from astro_lab.tensors import SurveyTensorDict


class TestAstroDataModule:
    """Test AstroDataModule with real data."""

    def test_datamodule_creation(self, astro_datamodule):
        """Test AstroDataModule creation."""
        assert astro_datamodule is not None
        assert astro_datamodule.survey == "gaia"
        assert astro_datamodule.max_samples == 100

    def test_datamodule_setup(self, astro_datamodule):
        """Test AstroDataModule setup with real data."""
        astro_datamodule.setup()

        # Check that data was loaded
        assert astro_datamodule._main_data is not None
        assert hasattr(astro_datamodule._main_data, "num_nodes")
        assert astro_datamodule._main_data.num_nodes > 0

        # Check dataset info
        assert astro_datamodule.num_features is not None
        assert astro_datamodule.num_classes is not None

    def test_datamodule_splits(self, astro_datamodule):
        """Test train/val/test splits."""
        astro_datamodule.setup()
        data = astro_datamodule._main_data

        # Check that splits exist
        assert hasattr(data, "train_mask")
        assert hasattr(data, "val_mask")
        assert hasattr(data, "test_mask")

        # Check that splits are boolean
        assert data.train_mask.dtype == torch.bool
        assert data.val_mask.dtype == torch.bool
        assert data.test_mask.dtype == torch.bool

        # Check that splits are mutually exclusive
        train_val_overlap = (data.train_mask & data.val_mask).sum()
        train_test_overlap = (data.train_mask & data.test_mask).sum()
        val_test_overlap = (data.val_mask & data.test_mask).sum()

        assert train_val_overlap == 0
        assert train_test_overlap == 0
        assert val_test_overlap == 0

    def test_datamodule_info(self, astro_datamodule):
        """Test dataset information."""
        astro_datamodule.setup()
        info = astro_datamodule.get_info()

        assert "survey" in info
        assert "num_nodes" in info
        assert "num_edges" in info
        assert "num_features" in info
        assert info["survey"] == "gaia"


class TestSurveyGraphDataset:
    """Test SurveyGraphDataset with real data."""

    def test_dataset_creation(self, survey_graph_dataset):
        """Test SurveyGraphDataset creation."""
        assert survey_graph_dataset is not None
        assert survey_graph_dataset.survey == "gaia"
        assert survey_graph_dataset.graph_method == "knn"

    def test_dataset_loading(self, survey_graph_dataset):
        """Test dataset loading with real data."""
        # Dataset should load data automatically
        assert len(survey_graph_dataset) > 0

        # Get first graph
        graph = survey_graph_dataset[0]
        assert graph is not None
        assert hasattr(graph, "num_nodes")
        assert graph.num_nodes > 0

    def test_dataset_info(self, survey_graph_dataset):
        """Test dataset information."""
        info = survey_graph_dataset.get_info()

        assert "survey" in info
        assert "graph_method" in info
        assert "num_nodes" in info
        assert "num_edges" in info
        assert info["survey"] == "gaia"
        assert info["graph_method"] == "knn"

    def test_survey_tensor_access(self, survey_graph_dataset):
        """Test access to underlying SurveyTensorDict."""
        try:
            survey_tensor = survey_graph_dataset.get_survey_tensor()
            assert isinstance(survey_tensor, SurveyTensorDict)
            assert "spatial" in survey_tensor
        except FileNotFoundError:
            pytest.skip("SurveyTensorDict not saved yet")


class TestGraphBuilders:
    """Test centralized graph builders."""

    def test_knn_graph_creation(self, sample_survey_tensor):
        """Test KNN graph creation."""
        graph = create_knn_graph(sample_survey_tensor, k_neighbors=8)

        # Check basic graph properties
        assert hasattr(graph, "num_nodes")
        assert hasattr(graph, "edge_index")
        assert hasattr(graph, "x")
        assert graph.graph_type == "knn"
        assert graph.k_neighbors == 8

    def test_graph_metadata(self, sample_survey_tensor):
        """Test graph metadata."""
        graph = create_knn_graph(sample_survey_tensor, k_neighbors=8)

        # Check metadata
        assert hasattr(graph, "graph_type")
        assert hasattr(graph, "k_neighbors")
        assert graph.graph_type == "knn"
        assert graph.k_neighbors == 8

    def test_graph_builder_parameter_validation(self, sample_survey_tensor):
        """Test graph builder parameter validation."""
        # Test with invalid k_neighbors
        with pytest.raises(Exception):
            create_knn_graph(sample_survey_tensor, k_neighbors=0)

        # Test with very large k_neighbors
        with pytest.raises(Exception):
            create_knn_graph(sample_survey_tensor, k_neighbors=1000)

        # Test with valid k_neighbors (should succeed)
        graph = create_knn_graph(sample_survey_tensor, k_neighbors=8)
        assert graph is not None
        assert hasattr(graph, "num_nodes")
        assert hasattr(graph, "edge_index")
        assert graph.graph_type == "knn"
        assert graph.k_neighbors == 8


class TestDataIntegration:
    """Test integration between data components."""

    def test_datamodule_with_survey_dataset(
        self, astro_datamodule, survey_graph_dataset
    ):
        """Test that DataModule and SurveyGraphDataset work together."""
        # Setup both
        astro_datamodule.setup()
        survey_graph_dataset._load_data()

        # Both should have data
        assert astro_datamodule._main_data is not None
        assert len(survey_graph_dataset) > 0

        # Both should have similar structure
        dm_graph = astro_datamodule._main_data
        sg_graph = survey_graph_dataset[0]

        assert hasattr(dm_graph, "num_nodes")
        assert hasattr(sg_graph, "num_nodes")
        assert hasattr(dm_graph, "edge_index")
        assert hasattr(sg_graph, "edge_index")

    def test_graph_builder_with_real_data(self, real_survey_tensor):
        """Test graph builders with real survey data."""
        # Test KNN
        knn_graph = create_knn_graph(real_survey_tensor, k_neighbors=8)
        assert hasattr(knn_graph, "num_nodes")
        assert knn_graph.num_nodes is not None
        assert knn_graph.num_nodes > 0
        assert hasattr(knn_graph, "edge_index")
        assert knn_graph.edge_index is not None
        assert (
            knn_graph.edge_index.shape[1] > 0
        )  # Check edge count via edge_index shape


class TestDataValidation:
    """Test data validation with real scenarios."""

    def test_graph_builder_parameter_validation(self, sample_survey_tensor):
        """Test graph builder parameter validation."""
        # Test with invalid k_neighbors
        with pytest.raises(Exception):
            create_knn_graph(sample_survey_tensor, k_neighbors=0)

        # Test with very large k_neighbors
        with pytest.raises(Exception):
            create_knn_graph(sample_survey_tensor, k_neighbors=1000)

        # Test with valid k_neighbors (should succeed)
        graph = create_knn_graph(sample_survey_tensor, k_neighbors=8)
        assert graph is not None
        assert hasattr(graph, "num_nodes")
        assert hasattr(graph, "edge_index")
        assert graph.graph_type == "knn"
        assert graph.k_neighbors == 8


class TestCleanDataModule:
    """Test the cleaned data module structure."""

    def test_new_loaders_available(self):
        """Test that new loader functions are available."""
        from astro_lab.data import download_survey, load_catalog, load_survey_catalog

        # Verify functions exist and are callable
        assert callable(load_catalog)
        assert callable(load_survey_catalog)
        assert callable(download_survey)

    def test_new_processors_available(self):
        """Test that new processor functions are available."""
        from astro_lab.data import create_survey_tensordict, preprocess_survey

        # Verify functions exist and are callable
        assert callable(preprocess_survey)
        assert callable(create_survey_tensordict)
