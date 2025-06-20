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
from astro_lab.data import (
    SURVEY_CONFIGS,
    AstroDataModule,
    AstroDataset,
    load_gaia_data,
    load_nsa_data,
)
from astro_lab.data.manager import AstroDataManager, load_gaia_bright_stars


class TestAstroDataManager:
    """Test actual AstroDataManager with real data files."""

    def test_manager_initialization(self):
        """Test AstroDataManager initializes with real directory structure."""
        manager = AstroDataManager()
        assert manager.base_dir.exists()
        assert manager.raw_dir.exists()
        assert manager.processed_dir.exists()

    def test_list_real_catalogs(self):
        """Test listing actual catalog files that exist."""
        manager = AstroDataManager()
        catalogs = manager.list_catalogs()

        assert isinstance(catalogs, pl.DataFrame)
        # If we have real data, verify structure
        if len(catalogs) > 0:
            assert "name" in catalogs.columns
            assert "size_mb" in catalogs.columns
            assert "path" in catalogs.columns

    @pytest.mark.skipif(
        not Path("data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet").exists(),
        reason="Real Gaia data not available",
    )
    def test_load_real_gaia_data(self):
        """Test loading actual Gaia bright star catalog."""
        df = load_gaia_bright_stars(magnitude_limit=10.0)

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

        # Check actual Gaia columns exist
        gaia_columns = ["ra", "dec", "phot_g_mean_mag"]
        for col in gaia_columns:
            if col in df.columns:
                assert not df[col].is_null().all()


class TestAstroDataset:
    """Test actual AstroDataset with real survey data."""

    @pytest.mark.skipif(
        not Path("data/raw/gaia").exists(), reason="Gaia data not available"
    )
    def test_real_gaia_dataset(self):
        """Test AstroDataset with real Gaia data."""
        dataset = AstroDataset(survey="gaia", max_samples=50)

        assert len(dataset) > 0
        assert dataset.survey == "gaia"

        # Test actual PyG data object
        item = dataset[0]
        assert hasattr(item, "x")
        if hasattr(item, "x"):
            assert item.x.shape[1] > 0

        # Test dataset info with real data
        info = dataset.get_info()
        assert isinstance(info, dict)
        assert "survey" in info
        assert info["survey"] == "gaia"

    @pytest.mark.skipif(
        not any(Path("data").rglob("nsa*")), reason="NSA data not available"
    )
    def test_real_nsa_dataset(self):
        """Test AstroDataset with real NSA data."""
        dataset = AstroDataset(survey="nsa", max_samples=30)

        assert len(dataset) > 0
        assert dataset.survey == "nsa"

        # Test actual data structure
        item = dataset[0]
        assert hasattr(item, "x")


class TestRealDataLoading:
    """Test actual data loading functions with real data."""

    @pytest.mark.skipif(
        not Path("data/raw/gaia").exists(), reason="Gaia data not available"
    )
    def test_load_gaia_dataset(self):
        """Test load_gaia_data function with real data."""
        dataset = load_gaia_data(max_samples=20, return_tensor=False)

        assert isinstance(dataset, AstroDataset)
        assert dataset.survey == "gaia"
        assert len(dataset) <= 20
        assert len(dataset) > 0

    @pytest.mark.skipif(
        not Path("data/raw/gaia").exists(), reason="Gaia data not available"
    )
    def test_load_gaia_tensor(self):
        """Test load_gaia_data with tensor output."""
        try:
            from astro_lab.tensors import SurveyTensor

            tensor_data = load_gaia_data(max_samples=25, return_tensor=True)
            assert isinstance(tensor_data, SurveyTensor)
            assert tensor_data.survey_name == "gaia"
            assert len(tensor_data) <= 25

        except ImportError:
            pytest.skip("Tensor integration not available")


class TestAstroDataModule:
    """Test actual AstroDataModule with real data."""

    @pytest.mark.skipif(
        not Path("data/raw/gaia").exists(), reason="Gaia data not available"
    )
    def test_real_datamodule_setup(self):
        """Test AstroDataModule with real Gaia data."""
        datamodule = AstroDataModule(
            survey="gaia", max_samples=100, batch_size=16, val_split=0.2, test_split=0.1
        )

        # Test setup process
        datamodule.setup()

        # Test dataloaders exist
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        test_loader = datamodule.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None

        # Test actual batch
        batch = next(iter(train_loader))
        assert hasattr(batch, "x")
        # Batch might contain all data if dataset is small
        assert batch.x.shape[0] > 0


class TestSurveyConfigs:
    """Test actual survey configurations."""

    def test_survey_configs_structure(self):
        """Test SURVEY_CONFIGS has correct structure."""
        assert isinstance(SURVEY_CONFIGS, dict)
        assert len(SURVEY_CONFIGS) > 0
        assert "gaia" in SURVEY_CONFIGS

        # Test Gaia config structure
        gaia_config = SURVEY_CONFIGS["gaia"]
        required_keys = ["coord_cols", "mag_cols", "data_release"]
        for key in required_keys:
            assert key in gaia_config

    def test_available_surveys(self):
        """Test which surveys have actual data available."""
        data_dir = Path("data")
        if not data_dir.exists():
            pytest.skip("Data directory not available")

        available_surveys = []
        for survey_name in SURVEY_CONFIGS.keys():
            survey_path = data_dir / "raw" / survey_name
            if survey_path.exists() and any(survey_path.glob("*.parquet")):
                available_surveys.append(survey_name)

        # Print available surveys for debugging
        if available_surveys:
            print(f"Available surveys: {available_surveys}")


class TestDataFileIntegrity:
    """Test integrity of actual data files."""

    @pytest.mark.skipif(
        not Path("data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet").exists(),
        reason="Gaia mag10.0 file not available",
    )
    def test_gaia_file_integrity(self):
        """Test actual Gaia data file is readable and valid."""
        file_path = Path("data/raw/gaia/gaia_dr3_bright_all_sky_mag10.0.parquet")
        df = pl.read_parquet(file_path)

        # Basic integrity
        assert len(df) > 0
        assert len(df.columns) > 0

        # Astronomical data validation
        if "ra" in df.columns and "dec" in df.columns:
            ra_min = df["ra"].min()
            ra_max = df["ra"].max()
            dec_min = df["dec"].min()
            dec_max = df["dec"].max()

            if ra_min is not None and ra_max is not None:
                assert float(ra_min) >= 0 and float(ra_max) <= 360
            if dec_min is not None and dec_max is not None:
                assert float(dec_min) >= -90 and float(dec_max) <= 90

    @pytest.mark.skipif(
        not Path("data/nsa_processed.parquet").exists(), reason="NSA file not available"
    )
    def test_nsa_file_integrity(self):
        """Test actual NSA data file is readable."""
        file_path = Path("data/nsa_processed.parquet")
        df = pl.read_parquet(file_path)

        assert len(df) > 0
        assert len(df.columns) > 0
