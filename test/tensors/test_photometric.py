"""
Tests for PhotometricTensor.
"""

import pytest
import torch

from astro_lab.tensors.photometric import PhotometricTensor


class TestPhotometricTensor:
    """Test photometric tensor functionality."""

    def test_photometric_creation(self):
        """Test photometric tensor creation."""
        n_objects = 20
        bands = ["u", "g", "r", "i", "z"]
        magnitudes = torch.randn(n_objects, len(bands))
        errors = torch.rand(n_objects, len(bands)) * 0.1

        phot = PhotometricTensor(
            magnitudes, bands=bands, measurement_errors=errors, photometric_system="AB"
        )

        assert phot.bands == bands
        assert phot.n_bands == len(bands)
        assert phot.photometric_system == "AB"
        assert phot.is_magnitude is True
        if phot.measurement_errors is not None:
            assert torch.equal(phot.measurement_errors, errors)

    def test_band_validation(self):
        """Test band validation."""
        magnitudes = torch.randn(10, 5)

        # No bands provided
        with pytest.raises(ValueError, match="requires band names"):
            PhotometricTensor(magnitudes, bands=[])

        # Wrong number of bands
        with pytest.raises(ValueError, match="doesn't match number of bands"):
            PhotometricTensor(
                magnitudes, bands=["g", "r", "i"]
            )  # Only 3 bands for 5-dim data

    def test_error_validation(self):
        """Test measurement error validation."""
        magnitudes = torch.randn(10, 3)
        bands = ["g", "r", "i"]
        wrong_errors = torch.randn(10, 2)  # Wrong shape

        with pytest.raises(ValueError, match="doesn't match data shape"):
            PhotometricTensor(magnitudes, bands=bands, measurement_errors=wrong_errors)

    def test_photometric_properties(self):
        """Test photometric tensor properties."""
        magnitudes = torch.randn(15, 4)
        bands = ["B", "V", "R", "I"]

        phot = PhotometricTensor(magnitudes, bands=bands, is_magnitude=False)

        assert phot.is_magnitude is False  # Flux mode
        assert phot.extinction_coefficients == {}  # Default empty

        # Test color computation methods exist
        assert hasattr(phot, "compute_colors")
        assert hasattr(phot, "to_flux")
