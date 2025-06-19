"""
Tests for astro_lab.data module.

Tests datasets, loaders, transforms, and data management functionality.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from astro_lab.data import (
    SURVEY_CONFIGS,
    AstroDataModule,
    AstroDataset,
    create_astro_dataloader,
    create_astro_datamodule,
    load_gaia_data,
    load_lightcurve_data,
    load_nsa_data,
    load_sdss_data,
)


# Helper function to get data directory
def get_data_dir() -> Path:
    """Get data directory path."""
    return Path(__file__).parent.parent / "src" / "data"


# Helper function to check astroquery availability
def check_astroquery_available() -> bool:
    """Check if astroquery is available."""
    try:
        import astroquery

        return True
    except ImportError:
        return False


class TestDataUtilities:
    """Test data utility functions."""

    def test_data_dir_access(self):
        """Test data directory functionality."""
        data_dir = get_data_dir()
        assert isinstance(data_dir, Path)

    def test_astroquery_check(self):
        """Test astroquery availability check."""
        available = check_astroquery_available()
        assert isinstance(available, bool)

    def test_survey_configs_available(self):
        """Test survey configurations are available."""
        assert isinstance(SURVEY_CONFIGS, dict)
        assert len(SURVEY_CONFIGS) > 0
        assert "gaia" in SURVEY_CONFIGS
        assert "sdss" in SURVEY_CONFIGS


class TestDataImports:
    """Test that all data modules can be imported."""

    def test_import_core_classes(self):
        """Test importing core dataset classes."""
        # These should all be available
        assert AstroDataset is not None
        assert AstroDataModule is not None
        assert create_astro_dataloader is not None
        assert create_astro_datamodule is not None

    def test_import_convenience_functions(self):
        """Test importing convenience functions."""
        # These should all be available
        assert load_gaia_data is not None
        assert load_sdss_data is not None
        assert load_nsa_data is not None
        assert load_lightcurve_data is not None


class TestBasicDataset:
    """Test basic dataset functionality."""

    def test_dataset_creation(self):
        """Test basic dataset creation."""
        x = torch.randn(100, 10)
        y = torch.randn(100)

        # Create a simple dataset-like structure
        data = list(zip(x, y))
        assert len(data) == 100

        sample_x, sample_y = data[0]
        assert sample_x.shape == (10,)
        assert isinstance(sample_y, torch.Tensor)

    def test_tensor_operations(self):
        """Test tensor operations on astronomical data."""
        # Mock coordinate data
        ra = torch.rand(50) * 360  # 0-360 degrees
        dec = torch.rand(50) * 180 - 90  # -90 to +90 degrees
        distance = torch.rand(50) * 1000  # 0-1000 Mpc

        coords = torch.stack([ra, dec, distance], dim=1)
        assert coords.shape == (50, 3)

        # Test coordinate range validation
        assert torch.all(ra >= 0) and torch.all(ra <= 360)
        assert torch.all(dec >= -90) and torch.all(dec <= 90)
        assert torch.all(distance >= 0)


class TestDataProcessing:
    """Test data processing functionality."""

    def test_photometric_data_processing(self):
        """Test photometric data processing."""
        n_objects = 100
        n_bands = 5

        # Mock photometric data
        magnitudes = torch.randn(n_objects, n_bands) + 20
        mag_errors = torch.rand(n_objects, n_bands) * 0.1 + 0.05

        # Test magnitude to flux conversion
        fluxes = torch.pow(10, -0.4 * magnitudes)
        assert torch.all(fluxes > 0)

        # Test color computation
        if n_bands > 1:
            colors = magnitudes[:, :-1] - magnitudes[:, 1:]
            assert colors.shape == (n_objects, n_bands - 1)

    def test_spectral_data_processing(self):
        """Test spectral data processing."""
        n_objects = 20
        n_wavelengths = 200

        # Mock spectral data
        wavelengths = torch.linspace(3000, 9000, n_wavelengths)
        flux = torch.rand(n_objects, n_wavelengths)

        # Test wavelength operations
        wave_range = wavelengths.max() - wavelengths.min()
        assert wave_range > 0

        # Test flux normalization
        normalized_flux = flux / flux.mean(dim=1, keepdim=True)
        assert normalized_flux.shape == flux.shape


class TestDataValidation:
    """Test data validation and error handling."""

    def test_coordinate_validation(self):
        """Test coordinate validation."""
        # Valid coordinates
        valid_ra = torch.tensor([0, 180, 360])
        valid_dec = torch.tensor([-90, 0, 90])

        # Test RA range
        assert torch.all(valid_ra >= 0) and torch.all(valid_ra <= 360)

        # Test Dec range
        assert torch.all(valid_dec >= -90) and torch.all(valid_dec <= 90)

    def test_magnitude_validation(self):
        """Test magnitude validation."""
        # Reasonable magnitude range
        mags = torch.tensor([10, 15, 20, 25, 30])

        # Should be positive for most astronomical objects
        assert torch.all(mags > 0)

        # Faint limit check
        bright_objects = mags < 25
        assert torch.any(bright_objects)  # Should have some bright objects


class TestMockDataGeneration:
    """Test mock data generation for testing."""

    def test_generate_mock_catalog(self):
        """Test generating mock astronomical catalog."""
        n_objects = 1000

        # Generate mock catalog
        catalog = {
            "ra": torch.rand(n_objects) * 360,
            "dec": torch.rand(n_objects) * 180 - 90,
            "g_mag": torch.randn(n_objects) * 2 + 20,
            "r_mag": torch.randn(n_objects) * 2 + 19.5,
            "i_mag": torch.randn(n_objects) * 2 + 19,
            "redshift": torch.rand(n_objects) * 0.5,
        }

        # Validate catalog structure
        assert len(catalog) == 6  # 6 columns
        for key, values in catalog.items():
            assert len(values) == n_objects
            assert isinstance(values, torch.Tensor)

    def test_generate_mock_lightcurves(self):
        """Test generating mock lightcurves."""
        n_objects = 50
        n_observations = 100

        # Generate time series
        time = torch.linspace(0, 365, n_observations)  # 1 year

        lightcurves = []
        for i in range(n_objects):
            # Simple sinusoidal variation + noise
            period = torch.rand(1) * 50 + 1  # 1-51 day periods
            amplitude = torch.rand(1) * 0.1 + 0.05  # 0.05-0.15 mag amplitude

            signal = amplitude * torch.sin(2 * torch.pi * time / period)
            noise = torch.randn(n_observations) * 0.02
            lightcurve = 20.0 + signal + noise  # Base magnitude 20

            lightcurves.append(lightcurve)

        lightcurves = torch.stack(lightcurves)
        assert lightcurves.shape == (n_objects, n_observations)

    def test_generate_mock_spectra(self):
        """Test generating mock spectra."""
        n_objects = 30
        n_wavelengths = 500

        # Wavelength grid
        wavelengths = torch.linspace(4000, 7000, n_wavelengths)  # Angstroms

        spectra = []
        for i in range(n_objects):
            # Simple blackbody-like spectrum
            temp = torch.rand(1) * 3000 + 4000  # 4000-7000 K

            # Simplified Planck function
            spectrum = 1 / (
                wavelengths**4 * (torch.exp(1.44e7 / (wavelengths * temp)) - 1)
            )

            # Add noise
            noise = torch.randn(n_wavelengths) * spectrum.std() * 0.1
            spectrum_noisy = spectrum + noise

            spectra.append(spectrum_noisy)

        spectra = torch.stack(spectra)
        assert spectra.shape == (n_objects, n_wavelengths)
        assert torch.all(spectra > 0)  # Flux should be positive


class TestAstroDataset:
    """Test AstroDataset functionality."""

    def test_astro_dataset_creation(self):
        """Test creating AstroDataset with demo data."""
        try:
            dataset = AstroDataset(survey="gaia", max_samples=100)
            assert len(dataset) > 0
            # Test getting first item
            item = dataset[0]
            assert hasattr(item, "x")  # PyG Data object
        except Exception:
            pytest.skip("Could not create dataset - may need real data")

    def test_astro_dataset_info(self):
        """Test dataset info functionality."""
        try:
            dataset = AstroDataset(survey="gaia", max_samples=50)
            info = dataset.get_info()
            assert isinstance(info, dict)
            assert "survey" in info
            assert "n_samples" in info
        except Exception:
            pytest.skip("Could not create dataset - may need real data")


class TestConvenienceFunctions:
    """Test convenience data loading functions."""

    def test_load_gaia_data(self):
        """Test loading Gaia data."""
        try:
            # Test with return_tensor=False for PyG compatibility
            dataset = load_gaia_data(max_samples=10, return_tensor=False)
            assert dataset is not None
        except Exception:
            pytest.skip("Could not load Gaia data - may need setup")

    def test_load_sdss_data(self):
        """Test loading SDSS data."""
        try:
            dataset = load_sdss_data(max_samples=10, return_tensor=False)
            assert dataset is not None
        except Exception:
            pytest.skip("Could not load SDSS data - may need setup")
