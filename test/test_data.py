"""
Test data module and datasets for AstroLab.
"""

from pathlib import Path

import polars as pl
import pytest
import torch
from torch_geometric.data import Data

from astro_lab.data import (
    SurveyCrossMatcher,
    SurveyDataModule,
    create_graph_from_survey,
    get_preprocessor,
    get_supported_surveys,
)
from astro_lab.models import AstroModel


class TestDataModuleAPI:
    """Test SurveyDataModule API."""

    def test_create_datamodule_graph_task(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="graph_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        assert hasattr(dm, "train_dataloader")
        assert hasattr(dm, "val_dataloader")
        assert hasattr(dm, "test_dataloader")

    def test_create_datamodule_node_task(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        assert hasattr(dm, "train_dataloader")

    def test_backend_selection(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        assert hasattr(dm, "train_dataloader")


class TestDatasets:
    """Test SurveyGraphDataset integration."""

    def test_survey_graph_dataset(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        loader = dm.train_dataloader()
        batch = next(iter(loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")

    def test_point_cloud_dataset(self):
        pytest.skip("Point cloud dataset integration not implemented in new API.")


class TestPyGLightningFeatures:
    """Test Lightning features with SurveyDataModule."""

    def test_automatic_splits(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "train_mask")

    def test_batch_structure(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        assert hasattr(batch, "x")
        assert hasattr(batch, "edge_index")


class TestClusteringUtils:
    def test_pyg_kmeans(self):
        pytest.skip("Clustering utilities not migrated to new API.")

    def test_spatial_clustering_fps(self):
        pytest.skip("Clustering utilities not migrated to new API.")


class TestNodeLevelTasks:
    def test_neighbor_sampling(self):
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=2,
            num_workers=0,
            max_samples=10,
        )
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))
        assert hasattr(batch, "x")


class TestPreprocessors:
    """Test survey preprocessors."""

    def test_gaia_preprocessor(self, tmp_path):
        """Test Gaia preprocessor."""
        # Create minimal test data
        test_data = pl.DataFrame(
            {
                "source_id": [1, 2, 3, 4, 5],
                "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
                "dec": [-5.0, 5.0, 15.0, 25.0, 35.0],
                "parallax": [2.0, 1.5, 1.0, 0.8, 0.5],
                "parallax_error": [0.1, 0.1, 0.1, 0.1, 0.1],
                "pmra": [1.0, -1.0, 0.5, -0.5, 0.0],
                "pmdec": [0.5, -0.5, 0.2, -0.2, 0.0],
                "phot_g_mean_mag": [10.0, 11.0, 12.0, 13.0, 14.0],
                "phot_bp_mean_mag": [10.5, 11.5, 12.5, 13.5, 14.5],
                "phot_rp_mean_mag": [9.5, 10.5, 11.5, 12.5, 13.5],
                "ruwe": [1.0, 1.1, 1.2, 1.3, 1.4],
            }
        )

        # Save test data
        test_file = tmp_path / "gaia_test.parquet"
        test_data.write_parquet(test_file)

        # Initialize preprocessor with test data
        preprocessor = get_preprocessor("gaia")
        preprocessor.raw_dir = tmp_path
        preprocessor.processed_dir = tmp_path

        # Mock the data finding
        preprocessor._find_data_file = lambda: test_file

        # Run preprocessing
        df_processed, graph = preprocessor.preprocess(max_samples=5)

        # Check processed dataframe
        assert len(df_processed) <= 5
        assert "x" in df_processed.columns
        assert "y" in df_processed.columns
        assert "z" in df_processed.columns
        assert "bp_rp" in df_processed.columns

        # Check graph
        assert isinstance(graph, Data)
        assert hasattr(graph, "x")
        assert hasattr(graph, "pos")
        assert hasattr(graph, "edge_index")
        assert graph.num_nodes == len(df_processed)

    def test_preprocessor_quality_filters(self):
        """Test quality filtering in preprocessors."""
        # Create data with quality issues
        test_data = pl.DataFrame(
            {
                "source_id": range(10),
                "ra": [10.0] * 10,
                "dec": [20.0] * 10,
                "parallax": [1.0, -1.0, 0.0, 0.5, 2.0, 3.0, 0.1, 0.01, 5.0, 1.0],
                "parallax_error": [0.2] * 10,
                "ruwe": [1.0, 1.5, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0],
            }
        )

        preprocessor = get_preprocessor("gaia")
        filtered = preprocessor.apply_quality_filters(test_data)

        # Should filter out negative parallax, high RUWE, etc.
        assert len(filtered) < 10
        assert all(filtered["parallax"] > 0)
        assert all(filtered["ruwe"] < 1.4)


class TestConverters:
    """Test data converters."""

    def test_create_graph_from_survey(self):
        """Test direct graph creation from survey data."""
        # Create test data with 3D coordinates
        test_data = pl.DataFrame(
            {
                "x": [10.0, 20.0, 30.0, 40.0, 50.0],
                "y": [5.0, -5.0, 15.0, -15.0, 0.0],
                "z": [1.0, 2.0, 3.0, 4.0, 5.0],
                "mag_g": [10.0, 11.0, 12.0, 13.0, 14.0],
                "color_bp_rp": [0.5, 1.0, 1.5, 2.0, 2.5],
            }
        )

        # Create graph
        graph = create_graph_from_survey(
            test_data,
            survey="gaia",
            k_neighbors=3,
            feature_cols=["mag_g", "color_bp_rp"],
        )

        # Check graph structure
        assert isinstance(graph, Data)
        assert graph.x.shape == (5, 2)  # 5 nodes, 2 features
        assert graph.pos.shape == (5, 3)  # 5 nodes, 3D positions
        assert graph.edge_index.shape[0] == 2  # edge list format
        assert graph.num_edges > 0
        assert graph.survey == "gaia"
        assert graph.feature_names == ["mag_g", "color_bp_rp"]


class TestCrossMatching:
    """Test cross-matching functionality."""

    def test_survey_cross_match(self):
        """Test matching between two surveys."""
        # Create two mock catalogs
        primary = pl.DataFrame(
            {
                "source_id": [1, 2, 3, 4, 5],
                "ra": [10.0, 20.0, 30.0, 40.0, 50.0],
                "dec": [5.0, 10.0, 15.0, 20.0, 25.0],
                "mag": [10.0, 11.0, 12.0, 13.0, 14.0],
            }
        )

        # Secondary with slight position offsets
        secondary = pl.DataFrame(
            {
                "object_id": [101, 102, 103],
                "ra": [10.0001, 20.0001, 35.0],  # First two match, third doesn't
                "dec": [5.0001, 10.0001, 15.0],
                "flux": [100.0, 80.0, 60.0],
            }
        )

        # Perform cross-match
        matcher = SurveyCrossMatcher(max_separation=1.0)  # 1 arcsec
        matched = matcher.match_surveys(primary, secondary)

        # Should find 2 matches
        assert len(matched) == 2
        assert "primary_source_id" in matched.columns
        assert "secondary_object_id" in matched.columns
        assert "separation_arcsec" in matched.columns

    def test_spatial_join(self):
        """Test 3D spatial joining."""
        # Create two catalogs with 3D positions
        df1 = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "x": [10.0, 20.0, 30.0],
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            }
        )

        df2 = pl.DataFrame(
            {
                "id": [101, 102, 103],
                "x": [11.0, 25.0, 100.0],  # First is close to df1[0], second to df1[1]
                "y": [0.0, 0.0, 0.0],
                "z": [0.0, 0.0, 0.0],
            }
        )

        matcher = SurveyCrossMatcher()
        joined = matcher.spatial_join(df1, df2, radius_pc=5.0)

        # Should find 1 match (11.0 is within 5 pc of 10.0)
        assert len(joined) >= 1
        assert "distance_pc" in joined.columns


class TestDataModule:
    """Test Lightning DataModule."""

    def test_survey_datamodule_setup(self):
        """Test SurveyDataModule initialization and setup."""
        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            batch_size=32,
            num_workers=0,
            max_samples=100,
            data_dir="./test_data",
        )

        # Check initialization
        assert dm.survey == "gaia"
        assert dm.task == "node_classification"
        assert dm.batch_size == 32

    def test_datamodule_transforms(self):
        """Test that transforms are properly applied."""
        from torch_geometric.transforms import NormalizeFeatures

        dm = SurveyDataModule(
            survey="gaia",
            task="node_classification",
            transform=NormalizeFeatures(),
            num_workers=0,
            max_samples=50,
        )

        assert dm.transform is not None
        assert any(isinstance(t, NormalizeFeatures) for t in dm.transform.transforms)


class TestSurveySupport:
    """Test survey support utilities."""

    def test_get_supported_surveys(self):
        """Test getting list of supported surveys."""
        surveys = get_supported_surveys()

        assert isinstance(surveys, list)
        assert "gaia" in surveys
        assert "sdss" in surveys
        assert len(surveys) > 5  # Should have many surveys

    def test_get_preprocessor_all_surveys(self):
        """Test that all surveys have preprocessors."""
        surveys = get_supported_surveys()

        for survey in surveys:
            preprocessor = get_preprocessor(survey)
            assert preprocessor is not None
            assert hasattr(preprocessor, "preprocess")
            assert hasattr(preprocessor, "apply_quality_filters")


class TestMemoryEfficiency:
    """Test memory-efficient processing."""

    def test_large_dataset_handling(self):
        """Test that large datasets can be processed in chunks."""
        # Create large mock dataset
        n_objects = 10000
        test_data = pl.DataFrame(
            {
                "source_id": range(n_objects),
                "ra": [i * 0.01 for i in range(n_objects)],
                "dec": [i * 0.01 - 50 for i in range(n_objects)],
                "parallax": [1.0 + i * 0.0001 for i in range(n_objects)],
                "mag": [10.0 + i * 0.001 for i in range(n_objects)],
            }
        )

        # Process with max_samples limit
        graph = create_graph_from_survey(
            test_data.head(1000),  # Only process first 1000
            survey="gaia",
            k_neighbors=5,
        )

        assert graph.num_nodes == 1000
        assert graph.x.shape[0] == 1000
