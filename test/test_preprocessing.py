"""
Tests for the AstroLab preprocessing system.

Tests for data processing, splits, and the modern preprocessing pipeline.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pytest
import torch

# Import only actual classes that exist
from astro_lab.data.core import (
    AstroDataset,
    create_cosmic_web_loader,
    create_graph_datasets_from_splits,
    create_graph_from_dataframe,
    detect_survey_type,
    get_optimal_batch_size,
    get_optimal_device,
    load_gaia_data,
    load_nsa_data,
)

# Import AstroDataModule from the correct module
from astro_lab.data.datamodule import AstroDataModule


class TestPreprocessingUtils:
    """Tests for preprocessing utility functions."""

    def test_detect_survey_type(self):
        """Test survey type detection."""
        # Test Gaia detection
        survey_type = detect_survey_type("gaia_catalog", None)
        assert survey_type == "gaia"

        # Test NSA detection
        survey_type = detect_survey_type("nsa_data", None)
        assert survey_type == "nsa"

    def test_get_optimal_device(self):
        """Test optimal device detection."""
        device = get_optimal_device()
        device_str = str(device)
        assert device_str in ["cpu", "cuda"] or device_str.startswith("cuda")

    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        batch_size = get_optimal_batch_size(1000)
        assert isinstance(batch_size, int)
        assert batch_size > 0


class TestAstroDataset:
    """Tests for AstroDataset functionality."""

    def test_astro_dataset_creation(self, gaia_dataset):
        """Test AstroDataset creation."""
        assert len(gaia_dataset) > 0
        assert hasattr(gaia_dataset, "data")

    def test_astro_dataset_loading(self, gaia_dataset):
        """Test AstroDataset loading."""
        first_item = gaia_dataset[0]
        assert first_item is not None
        assert hasattr(first_item, "x")


class TestAstroDataModule:
    """Tests for AstroDataModule functionality."""

    def test_astro_data_module_creation(self, gaia_dataset):
        """Test AstroDataModule creation."""
        datamodule = AstroDataModule(
            survey="gaia",
            train_dataset=gaia_dataset,
            val_dataset=gaia_dataset,
            test_dataset=gaia_dataset,
            batch_size=32,
        )
        assert datamodule.batch_size == 32

    def test_astro_data_module_dataloaders(self, gaia_dataset):
        """Test AstroDataModule dataloaders."""
        datamodule = AstroDataModule(
            survey="gaia",
            train_dataset=gaia_dataset,
            val_dataset=gaia_dataset,
            test_dataset=gaia_dataset,
            batch_size=32,
        )
        
        train_loader = datamodule.train_dataloader()
        assert train_loader is not None
