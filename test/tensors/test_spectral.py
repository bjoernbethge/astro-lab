"""
Tests for SpectralTensor.
"""

import pickle

import pytest
import torch

from astro_lab.tensors.spectral import SpectralTensor


class TestSpectralTensor:
    """Test spectral tensor functionality."""

    def test_spectral_creation(self):
        """Test spectral tensor creation."""
        n_wavelengths = 100
        flux = torch.randn(5, n_wavelengths)  # 5 spectra
        wavelengths = torch.linspace(4000, 7000, n_wavelengths)

        spectral = SpectralTensor(
            flux, wavelengths=wavelengths, flux_units="erg/s/cm2/A", redshift=0.1
        )

        assert spectral.n_wavelengths == n_wavelengths
        assert spectral.redshift == 0.1
        assert spectral.flux_units == "erg/s/cm2/A"
        assert torch.equal(spectral.wavelengths, wavelengths)

    def test_wavelength_validation(self):
        """Test wavelength array validation."""
        flux = torch.randn(5, 100)
        wavelengths = torch.linspace(4000, 7000, 50)  # Wrong size

        with pytest.raises(ValueError, match="must match wavelength array length"):
            SpectralTensor(flux, wavelengths=wavelengths)

    def test_spectral_properties(self):
        """Test spectral tensor properties."""
        flux = torch.randn(3, 200)
        wavelengths = torch.linspace(3000, 9000, 200)

        spectral = SpectralTensor(flux, wavelengths=wavelengths)

        # Test wavelength range
        wave_min, wave_max = spectral.wavelength_range
        assert wave_min == 3000.0
        assert wave_max == 9000.0

        # Test delta wavelength
        delta_wave = spectral.delta_wavelength
        assert len(delta_wave) == 199  # N-1 differences

    def test_redshift_operations(self):
        """Test redshift-related operations."""
        flux = torch.randn(2, 100)
        wavelengths = torch.linspace(4000, 7000, 100)

        spectral = SpectralTensor(flux, wavelengths=wavelengths, redshift=0.5)

        # Test redshift correction methods exist
        assert hasattr(spectral, "apply_redshift")
        assert hasattr(spectral, "rest_wavelengths")

    def test_tensor_pickling(self):
        """Test tensor serialization with pickle."""
        data = torch.randn(3, 3)
        tensor = SpectralTensor(data, wavelengths=torch.linspace(4000, 7000, 3))

        # Test pickle serialization
        tensor_bytes = pickle.dumps(tensor)
        tensor_loaded = pickle.loads(tensor_bytes)

        # Test that loaded tensor has same properties
        assert tensor_loaded.shape == tensor.shape
        torch.testing.assert_close(tensor_loaded.data, tensor.data)

        # Test state dict
        tensor_dict = tensor.model_dump()
        assert isinstance(tensor_dict, dict)
        assert "data" in tensor_dict
