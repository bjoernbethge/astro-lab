"""Tests for Simplified Encoders."""

import pytest
import torch
import torch.nn as nn

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
)


class TestBaseEncoder:
    """Test the base encoder class."""

    def test_initialization(self):
        """Test BaseEncoder initialization."""
        encoder = BaseEncoder(input_dim=16, output_dim=32)
        assert encoder.input_dim == 16
        assert encoder.output_dim == 32
        assert isinstance(encoder.encoder, nn.Module)

    def test_device_detection(self):
        """Test device detection."""
        # Test CPU device
        cpu_encoder = BaseEncoder(input_dim=8, output_dim=16, device='cpu')
        assert next(cpu_encoder.parameters()).device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_encoder = BaseEncoder(input_dim=8, output_dim=16, device='cuda')
            assert next(cuda_encoder.parameters()).device.type == "cuda"
            
    def test_forward_pass(self):
        """Test forward pass."""
        encoder = BaseEncoder(input_dim=10, output_dim=20)
        x = torch.randn(5, 10)
        output = encoder(x)
        assert output.shape == (5, 20)


class TestPhotometryEncoder:
    """Test the photometry encoder."""

    def test_initialization(self):
        """Test PhotometryEncoder initialization."""
        encoder = PhotometryEncoder(output_dim=64)
        assert encoder.output_dim == 64
        assert encoder.input_dim == 5  # Default 5 bands

    def test_forward_pass_with_tensor(self):
        """Test forward pass with tensor."""
        encoder = PhotometryEncoder(output_dim=48)
        magnitudes = torch.randn(10, 5)
        output = encoder(magnitudes)
        assert output.shape == (10, 48)
        
    def test_forward_pass_with_photometric_tensor(self):
        """Test forward pass with PhotometricTensor."""
        encoder = PhotometryEncoder(output_dim=48)
        magnitudes = torch.randn(10, 5)
        bands = ["u", "g", "r", "i", "z"]
        photometric_tensor = PhotometricTensor(data=magnitudes, bands=bands)
        output = encoder(photometric_tensor)
        assert output.shape == (10, 48)

    def test_different_band_numbers(self):
        """Test with different numbers of bands."""
        encoder = PhotometryEncoder(output_dim=32, input_dim=3)
        magnitudes = torch.randn(5, 3)
        output = encoder(magnitudes)
        assert output.shape == (5, 32)

    def test_single_object_handling(self):
        """Test handling of a single object."""
        encoder = PhotometryEncoder(output_dim=32)
        photometry_data = torch.randn(5)  # Single object, 5 bands
        output = encoder(photometry_data)
        assert output.shape == (1, 32)


class TestAstrometryEncoder:
    """Test the astrometry encoder."""

    def test_initialization(self):
        """Test AstrometryEncoder initialization."""
        encoder = AstrometryEncoder(output_dim=64)
        assert encoder.output_dim == 64
        assert encoder.input_dim == 5  # Default: ra, dec, parallax, pmra, pmdec

    def test_forward_pass_with_tensor(self):
        """Test forward pass with tensor."""
        encoder = AstrometryEncoder(output_dim=32)
        astrometry_data = torch.randn(10, 5)
        output = encoder(astrometry_data)
        assert output.shape == (10, 32)
        
    def test_forward_pass_with_spatial_tensor(self):
        """Test forward pass with Spatial3DTensor."""
        encoder = AstrometryEncoder(output_dim=32, input_dim=3)
        coordinates = torch.randn(10, 3)
        spatial_tensor = Spatial3DTensor(data=coordinates)
        output = encoder(spatial_tensor)
        assert output.shape == (10, 32)


class TestSpectroscopyEncoder:
    """Test the spectroscopy encoder."""

    def test_initialization(self):
        """Test SpectroscopyEncoder initialization."""
        encoder = SpectroscopyEncoder(output_dim=128)
        assert encoder.output_dim == 128
        assert encoder.input_dim == 3  # Default: teff, logg, feh

    def test_forward_pass_with_tensor(self):
        """Test forward pass with tensor."""
        encoder = SpectroscopyEncoder(output_dim=64, input_dim=100)
        spectral_data = torch.randn(10, 100)
        output = encoder(spectral_data)
        assert output.shape == (10, 64)
        
    def test_forward_pass_with_spectral_tensor(self):
        """Test forward pass with SpectralTensor."""
        encoder = SpectroscopyEncoder(output_dim=64, input_dim=100)
        flux = torch.randn(10, 100)
        wavelengths = torch.linspace(4000, 8000, 100)
        spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)
        output = encoder(spectral_tensor)
        assert output.shape == (10, 64)


class TestLightcurveEncoder:
    """Test the lightcurve encoder."""

    def test_initialization(self):
        """Test LightcurveEncoder initialization."""
        encoder = LightcurveEncoder(hidden_dim=64, output_dim=96)
        assert encoder.output_dim == 96
        assert encoder.hidden_dim == 64
        assert encoder.input_dim == 1  # Default: magnitude values

    def test_forward_pass_with_tensor(self):
        """Test forward pass with tensor."""
        encoder = LightcurveEncoder(hidden_dim=32, output_dim=64)
        # Single sequence of 20 time points
        lightcurve_data = torch.randn(20)
        output = encoder(lightcurve_data)
        assert output.shape == (1, 64)  # (batch=1, output_dim)
        
    def test_forward_pass_with_batch(self):
        """Test forward pass with batched data."""
        encoder = LightcurveEncoder(hidden_dim=32, output_dim=64)
        # Batch of 5 sequences, each with 20 time points
        lightcurve_data = torch.randn(5, 20)
        output = encoder(lightcurve_data)
        assert output.shape == (5, 64)
        
    def test_forward_pass_with_lightcurve_tensor(self):
        """Test forward pass with LightcurveTensor."""
        encoder = LightcurveEncoder(input_dim=2, hidden_dim=32, output_dim=64)
        times = torch.sort(torch.rand(20)).values.unsqueeze(1)
        magnitudes = torch.randn(20, 1)
        data = torch.cat([times, magnitudes], dim=1)
        lightcurve_tensor = LightcurveTensor(data=data)
        
        output = encoder(lightcurve_tensor)
        assert output.shape == (1, 64)

    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        encoder = LightcurveEncoder(hidden_dim=64, output_dim=48)
        # Longer sequence
        lightcurve_data = torch.randn(50)
        output = encoder(lightcurve_data)
        assert output.shape == (1, 48)


class TestEncoderFactory:
    """Test encoder factory function."""
    
    def test_create_encoder(self):
        """Test creating encoders via factory."""
        # Photometry
        phot_encoder = create_encoder('photometry', output_dim=64)
        assert isinstance(phot_encoder, PhotometryEncoder)
        
        # Astrometry
        astro_encoder = create_encoder('astrometry', output_dim=32)
        assert isinstance(astro_encoder, AstrometryEncoder)
        
        # Spectroscopy
        spec_encoder = create_encoder('spectroscopy', output_dim=128)
        assert isinstance(spec_encoder, SpectroscopyEncoder)
        
        # Lightcurve
        lc_encoder = create_encoder('lightcurve', hidden_dim=64, output_dim=96)
        assert isinstance(lc_encoder, LightcurveEncoder)
        
    def test_invalid_encoder_type(self):
        """Test error on invalid encoder type."""
        with pytest.raises(ValueError):
            create_encoder('invalid_encoder', output_dim=64)
            
    def test_encoder_registry(self):
        """Test encoder registry."""
        assert 'photometry' in ENCODER_REGISTRY
        assert 'astrometry' in ENCODER_REGISTRY
        assert 'spectroscopy' in ENCODER_REGISTRY
        assert 'lightcurve' in ENCODER_REGISTRY


class TestEncoderIntegration:
    """Test integration scenarios."""

    def test_multi_modal_encoding(self):
        """Test encoding of multiple modalities."""
        output_dim = 64
        photometry_encoder = PhotometryEncoder(output_dim=output_dim)
        spectroscopy_encoder = SpectroscopyEncoder(output_dim=output_dim, input_dim=100)

        photometry_data = torch.randn(10, 5)
        spectroscopy_data = torch.randn(10, 100)

        phot_output = photometry_encoder(photometry_data)
        spec_output = spectroscopy_encoder(spectroscopy_data)

        # Feature fusion
        fused_features = torch.cat([phot_output, spec_output], dim=1)
        assert fused_features.shape == (10, output_dim * 2)

    def test_device_consistency(self):
        """Test device handling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        encoder = PhotometryEncoder(output_dim=32, device='cuda')
        photometry_data = torch.randn(10, 5)  # CPU tensor
        
        # Encoder should handle device movement
        output = encoder(photometry_data)
        assert output.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
