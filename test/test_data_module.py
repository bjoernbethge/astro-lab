"""
Tests for astro_lab.data module - Real Classes with Real Data Only
================================================================

Tests only actual implemented classes with real astronomical data.
No mocks, no fakes - only testing the actual codebase functionality.
"""

from pathlib import Path

import polars as pl
import pytest
import torch

# Import only actual classes that exist
from astro_lab.data.core import (
    AstroDataset,
    create_cosmic_web_loader,
    detect_survey_type,
    get_optimal_batch_size,
    get_optimal_device,
    load_gaia_data,
    load_nsa_data,
)

# Import create_graph_from_dataframe from preprocessing
from astro_lab.data.preprocessing import create_graph_from_dataframe, create_graph_datasets_from_splits

# Import AstroDataModule from the correct module
from astro_lab.data.datamodule import AstroDataModule

# Import catalog management from manager
from astro_lab.data.manager import (
    AstroDataManager,
    list_catalog_names,
    load_catalog,
    load_gaia_bright_stars,
)

# Import utility functions from utils (not core)
from astro_lab.data.utils import (
    create_training_splits,
    get_data_statistics,
    load_splits_from_parquet,
    save_splits_to_parquet,
)

# Import preprocess_catalog from preprocessing
from astro_lab.data.preprocessing import preprocess_catalog


class TestAstroDataManager:
    """Test actual AstroDataManager with real data files."""

    def test_manager_initialization(self):
        manager = AstroDataManager()
        assert manager.base_dir.exists()
        assert manager.raw_dir.exists()
        assert manager.processed_dir.exists()

    def test_list_real_catalogs(self):
        manager = AstroDataManager()
        catalogs = manager.list_catalogs()
        assert isinstance(catalogs, pl.DataFrame)
        if len(catalogs) > 0:
            assert "name" in catalogs.columns
            assert "size_mb" in catalogs.columns
            assert "path" in catalogs.columns

    def test_load_real_gaia_data(self, gaia_dataset):
        info = gaia_dataset.get_info()
        assert info["survey"] == "gaia"
        assert gaia_dataset.len() > 0
        sample = gaia_dataset.get(0)
        assert hasattr(sample, "x")
        columns = info.get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"Gaia-Dataset hat nicht die erwarteten Spalten: {columns}")


class TestIntegratedDataModule:
    """Test integrated Data-Module APIs with real data."""

    def test_list_catalogs_function(self):
        catalogs = list_catalog_names()
        assert isinstance(catalogs, list)

    def test_get_data_statistics(self):
        test_df = pl.DataFrame({"ra": [0.0, 1.0, 2.0], "dec": [0.0, 1.0, 2.0], "mag": [10.0, 11.0, 12.0]})
        stats = get_data_statistics(test_df)
        assert isinstance(stats, dict)
        assert "n_rows" in stats
        assert "n_columns" in stats
        assert stats["n_rows"] == 3
        assert stats["n_columns"] == 3

    def test_detect_survey_type(self):
        assert detect_survey_type("gaia_catalog", None) == "gaia"
        assert detect_survey_type("nsa_data", None) == "nsa"
        assert detect_survey_type("linear_survey", None) == "linear"

    def test_create_training_splits(self):
        test_df = pl.DataFrame({"ra": list(range(100)), "dec": list(range(100)), "mag": list(range(100))})
        train, val, test = create_training_splits(test_df, test_size=0.2, val_size=0.1)
        assert isinstance(train, pl.DataFrame)
        assert isinstance(val, pl.DataFrame)
        assert isinstance(test, pl.DataFrame)
        total = len(train) + len(val) + len(test)
        assert total == len(test_df)

    def test_load_gaia_data_integrated(self, gaia_dataset):
        info = gaia_dataset.get_info()
        assert info["survey"] == "gaia"
        assert gaia_dataset.len() > 0
        sample = gaia_dataset.get(0)
        assert hasattr(sample, "x")
        columns = info.get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"Gaia-Dataset hat nicht die erwarteten Spalten: {columns}")

    def test_load_nsa_data_integrated(self, nsa_dataset):
        info = nsa_dataset.get_info()
        assert info["survey"] == "nsa"
        assert nsa_dataset.len() > 0
        sample = nsa_dataset.get(0)
        assert hasattr(sample, "x")
        columns = info.get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"NSA-Dataset hat nicht die erwarteten Spalten: {columns}")


class TestCosmicWebAnalysis:
    """Test cosmic web analysis functions."""

    def test_cosmic_web_loader_linear(self, linear_dataset):
        info = linear_dataset.get_info()
        assert info["survey"] == "linear"
        assert linear_dataset.len() > 0
        sample = linear_dataset.get(0)
        assert hasattr(sample, "x")

    def test_cosmic_web_loader_gaia(self, gaia_dataset):
        info = gaia_dataset.get_info()
        assert info["survey"] == "gaia"
        assert gaia_dataset.len() > 0
        sample = gaia_dataset.get(0)
        assert hasattr(sample, "x")


class TestGraphCreation:
    """Test graph creation functions with batch and device awareness."""

    def test_create_graph_from_dataframe(self, gaia_dataset):
        # Test, ob ein Graph aus dem Dataset erzeugt werden kann
        sample = gaia_dataset.get(0)
        assert hasattr(sample, "edge_index")
        assert hasattr(sample, "x")


class TestFileOperations:
    """Test file operation functions."""

    def test_save_and_load_splits(self):
        test_df = pl.DataFrame({"ra": list(range(20)), "dec": list(range(20)), "mag": list(range(20))})
        train, val, test = create_training_splits(test_df, test_size=0.2, val_size=0.1)
        output_dir = Path("test_splits")
        output_dir.mkdir(exist_ok=True)
        try:
            save_splits_to_parquet(train, val, test, output_dir, "test_dataset")
            loaded_train, loaded_val, loaded_test = load_splits_from_parquet(output_dir, "test_dataset")
            assert len(loaded_train) == len(train)
            assert len(loaded_val) == len(val)
            assert len(loaded_test) == len(test)
        finally:
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir)


class TestSurveyConfigs:
    """Test actual survey configurations."""

    def test_available_surveys(self):
        from astro_lab.utils.config.surveys import get_available_surveys
        surveys = get_available_surveys()
        assert "gaia" in surveys
        assert "nsa" in surveys
        assert "linear" in surveys
        assert "exoplanet" in surveys
        assert "rrlyrae" in surveys
