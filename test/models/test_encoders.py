"""
Tests for Enhanced Encoders
===========================

Tests for enhanced encoder classes with improved error handling.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any
from pydantic import ValidationError

from astro_lab.models.encoders import (
    BaseEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
    AstrometryEncoder,
    LightcurveEncoder,
    create_encoder,
    ENCODER_REGISTRY,
)
from astro_lab.tensors import (
    LightcurveTensor,
    PhotometricTensor,
    Spatial3DTensor,
    SpectralTensor,
    SurveyTensor,
)


class TestBaseEncoder:
    """Test the base encoder class."""

    def test_initialization(self):
        """Test BaseEncoder initialization."""
        encoder = BaseEncoder(input_dim=16, output_dim=32)
        assert encoder.input_dim == 16
        assert encoder.output_dim == 32
        assert isinstance(encoder.layers, nn.Module)

    def test_device_detection(self):
        """Test device detection."""
        # Test CPU device
        cpu_encoder = BaseEncoder(input_dim=8, output_dim=16, device=torch.device("cpu"))
        assert cpu_encoder.device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            cuda_encoder = BaseEncoder(input_dim=8, output_dim=16, device=cuda_device)
            assert cuda_encoder.device.type == "cuda"


class TestPhotometryEncoder:
    """Test the photometry encoder."""

    def test_initialization(self):
        """Test PhotometryEncoder initialization."""
        encoder = PhotometryEncoder(input_dim=5, output_dim=64)
        assert encoder.output_dim == 64

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with a real tensor."""
        encoder = PhotometryEncoder(input_dim=5, output_dim=48)
        magnitudes = torch.randn(10, 5)
        bands = ["u", "g", "r", "i", "z"]
        photometric_tensor = PhotometricTensor(data=magnitudes, bands=bands)
        output = encoder(photometric_tensor)
        assert output.shape == (10, 48)

    def test_different_band_numbers(self):
        """Test with different numbers of bands."""
        encoder = PhotometryEncoder(input_dim=3, output_dim=32)
        magnitudes = torch.randn(5, 3)
        bands = ["g", "r", "i"]
        photometric_tensor = PhotometricTensor(data=magnitudes, bands=bands)
        output = encoder(photometric_tensor)
        assert output.shape == (5, 32)

    def test_color_computation(self):
        """Test color computation within the encoder."""
        encoder = PhotometryEncoder(input_dim=2, output_dim=64)
        magnitudes = torch.tensor([[14.5, 13.8], [15.1, 14.2]])
        photometric_tensor = PhotometricTensor(data=magnitudes, bands=["g", "r"])
        output = encoder(photometric_tensor)
        assert output.shape == (2, 64)

    def test_dimension_mismatch_handling(self):
        """Test handling of mismatched input dimensions."""
        encoder = PhotometryEncoder(input_dim=5, output_dim=64)
        with pytest.raises(ValueError):
             # This should fail because the number of bands (5) doesn't match the data dim (4)
             PhotometricTensor(data=torch.randn(10, 4), bands=['u','g','r','i','z'])

    def test_single_object_handling(self):
        """Test handling of a single lightcurve (no batch)."""
        encoder = LightcurveEncoder(input_dim=2, hidden_dim=32, output_dim=48)
        
        # Create a single lightcurve (no batch dimension)
        lightcurve_data = torch.randn(50, 2) # time, mag
        # Ensure time is monotonically increasing for the validator
        lightcurve_data[:, 0] = torch.sort(lightcurve_data[:, 0]).values
        
        lightcurve_tensor = LightcurveTensor(data=lightcurve_data, bands=['time', 'mag'])
        
        # The encoder should handle this gracefully by unsqueezing and squeezing
        output = encoder(lightcurve_tensor)
        assert output.shape == (1, 50, 48) # batch, seq_len, out_dim


class TestAstrometryEncoder:
    """Test the astrometry encoder."""

    def test_initialization(self):
        """Test AstrometryEncoder initialization."""
        encoder = AstrometryEncoder(input_dim=3, output_dim=64)
        assert encoder.output_dim == 64

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with a real tensor."""
        encoder = AstrometryEncoder(input_dim=3, output_dim=32)
        coordinates = torch.randn(10, 3)
        spatial_tensor = Spatial3DTensor(data=coordinates)
        output = encoder(spatial_tensor)
        assert output.shape == (10, 32)

    def test_different_coordinate_dimensions(self):
        """Test with different coordinate dimensions."""
        with pytest.raises(ValueError):
            coordinates = torch.randn(10, 6)  # Invalid: 6 dimensions
            Spatial3DTensor(data=coordinates)

    def test_proper_motion_handling(self):
        """Test handling of proper motion data."""
        with pytest.raises(ValueError):
            # Proper motion + parallax = 5 dimensions
            data = torch.randn(10, 5)
            Spatial3DTensor(data=data, frame="icrs")


class TestSpectroscopyEncoder:
    """Test the spectroscopy encoder."""

    def test_initialization(self):
        """Test SpectroscopyEncoder initialization."""
        encoder = SpectroscopyEncoder(input_dim=100, output_dim=128)
        assert encoder.output_dim == 128

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with a real tensor."""
        encoder = SpectroscopyEncoder(input_dim=100, output_dim=64)
        flux = torch.randn(10, 100)
        wavelengths = torch.linspace(4000, 8000, 100)
        spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)
        output = encoder(spectral_tensor)
        assert output.shape == (10, 64)

    def test_different_spectrum_lengths(self):
        """Test with different spectrum lengths."""
        encoder = SpectroscopyEncoder(input_dim=50, output_dim=96)
        flux = torch.randn(5, 50)
        wavelengths = torch.linspace(4000, 6000, 50)
        spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)
        output = encoder(spectral_tensor)
        assert output.shape == (5, 96)

    def test_survey_tensor_compatibility(self):
        """Test that the encoder can handle a SpectralTensor."""
        encoder = SpectroscopyEncoder(input_dim=100, output_dim=128)
        spectral_data = torch.randn(10, 100)
        wavelengths = torch.linspace(3000, 9000, 100)
        spectral_tensor = SpectralTensor(data=spectral_data, wavelengths=wavelengths)
        
        output = encoder(spectral_tensor)
        assert output.shape == (10, 128)


class TestLightcurveEncoder:
    """Test the lightcurve encoder."""

    def test_initialization(self):
        """Test LightcurveEncoder initialization."""
        encoder = LightcurveEncoder(input_dim=2, hidden_dim=64, output_dim=96)
        assert encoder.output_dim == 96

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with a real tensor."""
        encoder = LightcurveEncoder(input_dim=2, hidden_dim=32, output_dim=64)
        times = torch.sort(torch.rand(20)).values.unsqueeze(1)
        magnitudes = torch.randn(20, 1)
        data = torch.cat([times, magnitudes], dim=1)
        lightcurve_tensor = LightcurveTensor(data=data)
        
        # The encoder expects a batch dimension, so we add one
        output = encoder(lightcurve_tensor.data.unsqueeze(0))
        assert output.shape == (1, 64)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        encoder = LightcurveEncoder(input_dim=2, hidden_dim=64, output_dim=48)
        times = torch.linspace(0, 100, 50).unsqueeze(1)
        magnitudes = torch.randn(50, 1)
        data = torch.cat([times, magnitudes], dim=1)
        lightcurve_tensor = LightcurveTensor(data=data)
        
        output = encoder(lightcurve_tensor.data.unsqueeze(0))
        assert output.shape == (1, 48)

    def test_different_feature_dimensions(self):
        """Test with different feature dimensions."""
        encoder = LightcurveEncoder(input_dim=3, hidden_dim=128, output_dim=72)
        times = torch.linspace(0, 50, 20).unsqueeze(1)
        mags_and_features = torch.randn(20, 2)
        data = torch.cat([times, mags_and_features], dim=1)
        lightcurve_tensor = LightcurveTensor(data=data)

        output = encoder(lightcurve_tensor.data.unsqueeze(0))
        assert output.shape == (1, 72)


class TestEncoderIntegration:
    """Test integration of different encoders."""

    def test_encoder_compatibility(self):
        """Test compatibility between different encoders."""
        output_dim = 128
        photometry_encoder = PhotometryEncoder(input_dim=5, output_dim=output_dim)
        astrometry_encoder = AstrometryEncoder(input_dim=3, output_dim=output_dim)

        photometry_data = PhotometricTensor(
            data=torch.randn(10, 5), bands=["u", "g", "r", "i", "z"]
        )
        astrometry_data = Spatial3DTensor(data=torch.randn(10, 3))

        phot_output = photometry_encoder(photometry_data)
        astro_output = astrometry_encoder(astrometry_data)

        assert phot_output.shape == (10, output_dim)
        assert astro_output.shape == (10, output_dim)

    def test_multi_modal_encoding(self):
        """Test encoding of multiple modalities."""
        output_dim = 64
        photometry_encoder = PhotometryEncoder(input_dim=5, output_dim=output_dim)
        spectroscopy_encoder = SpectroscopyEncoder(input_dim=100, output_dim=output_dim)

        photometry_data = PhotometricTensor(
            data=torch.randn(10, 5), bands=["u", "g", "r", "i", "z"]
        )
        flux = torch.randn(10, 100)
        wavelengths = torch.linspace(4000, 8000, 100)
        spectroscopy_data = SpectralTensor(data=flux, wavelengths=wavelengths)

        phot_output = photometry_encoder(photometry_data)
        spec_output = spectroscopy_encoder(spectroscopy_data)

        # Example of feature fusion
        fused_features = torch.cat([phot_output, spec_output], dim=1)
        assert fused_features.shape == (10, output_dim * 2)

    def test_device_consistency(self):
        """Test that encoders handle device placement correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for device consistency test")
        
        device = torch.device("cuda")
        encoder = PhotometryEncoder(input_dim=5, output_dim=32, device=device)
        
        photometry_data = PhotometricTensor(
            data=torch.randn(10, 5).to(device), bands=["u", "g", "r", "i", "z"]
        )
        
        output = encoder(photometry_data)
        assert output.device.type == "cuda"


class TestEncoderErrorHandling:
    """Test error handling capabilities of the encoders."""

    def test_empty_batch_handling(self):
        """Test that encoders handle empty tensors gracefully."""
        # Using pydantic validation, this should raise a ValueError on creation
        with pytest.raises(ValidationError):
            Spatial3DTensor(data=torch.empty(0, 3))

    def test_dimension_mismatch_handling(self):
        """Test handling of input dimension mismatches."""
        encoder = PhotometryEncoder(input_dim=5, hidden_dim=64, output_dim=128)
        small_data = PhotometricTensor(data=torch.randn(6, 1), bands=["g"])
        with pytest.raises(RuntimeError):
            encoder(small_data)

    def test_single_object_handling(self):
        """Test handling of a single object (batch size 1)."""
        encoder = LightcurveEncoder(input_dim=1, hidden_dim=32, output_dim=32)
        lightcurve_tensor = LightcurveTensor(data=torch.randn(50, 1), times=torch.randn(50))
        output = encoder(lightcurve_tensor)
        assert output.shape == (1, 32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
