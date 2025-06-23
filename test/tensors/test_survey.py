"""
Tests for SurveyTensor and related functionality.
"""

import pytest
import torch
import polars as pl
import astropy.io.fits as fits

from astro_lab.data import AstroDataset
from astro_lab.tensors import SurveyTensor, PhotometricTensor


class TestSurveyTensor:
    """Test SurveyTensor functionality and dataset integration."""

    def test_survey_tensor_creation(self):
        """Test basic SurveyTensor creation."""
        # Create sample astronomical data
        n_objects = 100
        n_features = 10
        data = torch.randn(n_objects, n_features)

        # Create column mapping
        column_mapping = {f"feature_{i}": i for i in range(n_features)}

        # Create SurveyTensor
        survey = SurveyTensor(
            data=data,
            survey_name="test_survey",
            column_mapping=column_mapping,
            data_release="v1.0",
        )

        assert survey.survey_name == "test_survey"
        assert survey.get_metadata("data_release") == "v1.0"
        assert survey.shape == (n_objects, n_features)
        assert len(survey.column_mapping) == n_features

    def test_survey_tensor_validation(self):
        """Test SurveyTensor validation."""
        data = torch.randn(50, 5)

        # Missing survey_name should raise error
        with pytest.raises(ValueError, match="survey_name"):
            SurveyTensor(data=data, survey_name="")

    def test_survey_tensor_metadata(self):
        """Test SurveyTensor metadata handling."""
        data = torch.randn(20, 8)
        column_mapping = {f"feature_{i}": i for i in range(8)}

        survey = SurveyTensor(
            data=data,
            survey_name="gaia",
            column_mapping=column_mapping,
            data_release="DR3",
            filter_system="gaia",
            magnitude_limit=12.0,
        )

        assert survey.get_metadata("filter_system") == "gaia"
        assert survey.get_metadata("magnitude_limit") == 12.0

    def test_survey_tensor_column_access(self):
        """Test column access methods."""
        data = torch.randn(30, 5)
        columns = ["ra", "dec", "mag_g", "mag_r", "parallax"]
        column_mapping = {col: i for i, col in enumerate(columns)}

        survey = SurveyTensor(
            data=data, survey_name="test", column_mapping=column_mapping
        )

        # Test column access
        ra_data = survey.get_column("ra")
        assert ra_data.shape == (30,)
        assert torch.equal(ra_data, data[:, 0])

    def test_survey_tensor_photometric_integration(self):
        """Test PhotometricTensor integration."""
        # Create data with photometric bands
        n_objects = 50
        bands = ["u", "g", "r", "i", "z"]
        mag_cols = [f"modelMag_{b}" for b in bands]
        data = torch.randn(n_objects, len(bands) + 2)  # +2 for ra, dec

        column_mapping = {"ra": 0, "dec": 1}
        for i, col in enumerate(mag_cols):
            column_mapping[col] = i + 2

        survey = SurveyTensor(
            data=data, survey_name="sdss", column_mapping=column_mapping
        )

        # Test photometric tensor creation
        phot_tensor = survey.get_photometric_tensor()
        assert phot_tensor is not None
        assert phot_tensor.bands == bands
        # Fix linter error by checking data shape instead of protocol shape
        assert phot_tensor.data.shape == (n_objects, len(bands))

    def test_survey_tensor_spatial_integration(self):
        """Test Spatial3DTensor integration."""
        # Create data with spatial coordinates
        n_objects = 40
        data = torch.randn(n_objects, 5)

        column_mapping = {"ra": 0, "dec": 1, "parallax": 2, "pmra": 3, "pmdec": 4}

        survey = SurveyTensor(
            data=data, survey_name="gaia", column_mapping=column_mapping
        )

        # Test spatial tensor creation
        spatial_tensor = survey.get_spatial_tensor()
        assert spatial_tensor is not None
        from astro_lab.tensors import Spatial3DTensor
        assert isinstance(spatial_tensor, Spatial3DTensor)
        assert spatial_tensor.shape[0] == n_objects

    def test_survey_tensor_statistics(self):
        """Test survey tensor statistics computation."""
        n_objects = 100
        n_features = 8

        data = torch.randn(n_objects, n_features)
        survey = SurveyTensor(
            data,
            survey_name="test_survey",
            column_mapping={"ra": 0, "dec": 1, "mag": 2},
        )

        # Test basic statistics - use built-in tensor methods
        assert survey.shape == (n_objects, n_features)
        assert len(survey) == n_objects

        # Test metadata access
        assert survey.get_metadata("survey_name") == "test_survey"

        # Check that column_mapping is now a direct attribute (Pydantic field)
        dumped = survey.model_dump()
        assert "column_mapping" in dumped
        assert "metadata" not in dumped or "column_mapping" not in dumped.get("metadata", {})

    def test_nested_tensor_access(self):
        """Test accessing data from nested tensors."""
        # Create a nested PhotometricTensor
        phot_data = torch.randn(10, 5)
        bands = ["u", "g", "r", "i", "z"]
        phot_tensor = PhotometricTensor(data=phot_data, bands=bands)

        # Create a SurveyTensor wrapping the PhotometricTensor
        survey_tensor = SurveyTensor(data=phot_tensor, survey_name="SDSS")

        # The data should be extracted from the PhotometricTensor
        assert isinstance(survey_tensor.data, torch.Tensor)
        torch.testing.assert_close(survey_tensor.data, phot_data)
        
        # The bands should be preserved
        assert survey_tensor.bands == bands
        
        # Should be able to get photometric tensor back
        phot_retrieved = survey_tensor.get_photometric_tensor()
        if phot_retrieved is not None:
            assert phot_retrieved.bands == bands


class TestSurveyTensorDatasetIntegration:
    """Test SurveyTensor integration with dataset classes."""

    @pytest.mark.slow
    def test_gaia_survey_tensor_integration(self, gaia_dataset):
        """Test Gaia dataset SurveyTensor integration."""
        assert len(gaia_dataset) > 0
        first_item = gaia_dataset[0]
        assert first_item is not None, "Dataset returned None for first item"
        assert hasattr(first_item, "x")  # PyG Data object
        # Zusätzliche Prüfung auf standardisierte Spalten
        columns = gaia_dataset.get_info().get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"Gaia-Dataset hat nicht die erwarteten Spalten: {columns}")

    @pytest.mark.slow
    def test_nsa_survey_tensor_integration(self, nsa_dataset):
        """Test NSA dataset SurveyTensor integration."""
        assert len(nsa_dataset) > 0
        first_item = nsa_dataset[0]
        assert first_item is not None
        assert hasattr(first_item, "x")  # PyG Data object
        columns = nsa_dataset.get_info().get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"NSA-Dataset hat nicht die erwarteten Spalten: {columns}")

    @pytest.mark.slow
    def test_exoplanet_survey_tensor_integration(self, exoplanet_dataset):
        """Test Exoplanet dataset SurveyTensor integration."""
        assert len(exoplanet_dataset) > 0
        first_item = exoplanet_dataset[0]
        assert first_item is not None
        assert hasattr(first_item, "x")  # PyG Data object
        columns = exoplanet_dataset.get_info().get("columns", [])
        assert "ra" in columns and "dec" in columns, (
            f"Exoplanet-Dataset hat nicht die erwarteten Spalten: {columns}")

    def test_cross_survey_operations(self, gaia_dataset, nsa_dataset):
        """Test cross-survey tensor operations."""
        assert len(gaia_dataset) > 0
        assert len(nsa_dataset) > 0
        gaia_cols = gaia_dataset.get_info().get("columns", [])
        nsa_cols = nsa_dataset.get_info().get("columns", [])
        assert "ra" in gaia_cols and "dec" in gaia_cols
        assert "ra" in nsa_cols and "dec" in nsa_cols

    def test_survey_tensor_quality_cuts(self):
        """Test survey tensor quality cuts."""
        n_objects = 100
        data = torch.randn(n_objects, 5)

        survey = SurveyTensor(
            data,
            survey_name="test_survey",
            column_mapping={"ra": 0, "dec": 1, "mag": 2},
        )

        # Test basic filtering using torch operations
        magnitude_col = 2
        bright_mask = survey.data[:, magnitude_col] < 0  # Arbitrary cut
        n_bright = bright_mask.sum().item()

        assert isinstance(n_bright, int)
        assert n_bright <= n_objects

    def test_survey_tensor_matching(self):
        """Test survey tensor matching."""
        n_objects = 50
        data1 = torch.randn(n_objects, 5)
        data2 = torch.randn(n_objects, 5)

        survey1 = SurveyTensor(
            data1, survey_name="survey1", column_mapping={"ra": 0, "dec": 1}
        )
        survey2 = SurveyTensor(
            data2, survey_name="survey2", column_mapping={"ra": 0, "dec": 1}
        )

        # Test basic coordinate access
        coords1 = survey1.data[:, :2]  # ra, dec
        coords2 = survey2.data[:, :2]  # ra, dec

        assert coords1.shape == (n_objects, 2)
        assert coords2.shape == (n_objects, 2)
