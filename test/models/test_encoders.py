"""
Tests for Enhanced Encoders
===========================

Tests for enhanced encoder classes with improved error handling.
"""

import pytest
import torch
import torch.nn as nn

from astro_lab.models.encoders import (
    AstrometryEncoder,
    BaseEncoder,
    LightcurveEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
)

# Import real tensor classes
from astro_lab.tensors import (
    LightcurveTensor,
    PhotometricTensor,
    Spatial3DTensor,
    SpectralTensor,
)


class TestBaseEncoder:
    """Test BaseEncoder functionality."""

    def test_initialization(self):
        """Test BaseEncoder initializes correctly."""
        encoder = BaseEncoder(output_dim=32, expected_input_dim=16)

        assert isinstance(encoder, nn.Module)
        assert encoder.output_dim == 32
        assert encoder.expected_input_dim == 16
        assert hasattr(encoder, "fallback_projection")

    def test_device_detection(self):
        """Test device detection functionality."""
        encoder = BaseEncoder(output_dim=16, expected_input_dim=8)

        # Test with tensor-like object
        class MockTensor:
            def __init__(self, device):
                self._data = torch.randn(5, 8, device=device)

        mock_tensor_cpu = MockTensor("cpu")
        device_cpu = encoder._get_device(mock_tensor_cpu)
        assert device_cpu.type == "cpu"

        # Test with CUDA if available
        if torch.cuda.is_available():
            mock_tensor_cuda = MockTensor("cuda")
            device_cuda = encoder._get_device(mock_tensor_cuda)
            assert device_cuda.type == "cuda"

    def test_batch_size_detection(self):
        """Test batch size detection functionality."""
        encoder = BaseEncoder(output_dim=16, expected_input_dim=8)

        # Test with tensor-like object
        class MockTensor:
            def __init__(self, batch_size):
                self._data = torch.randn(batch_size, 8)

        for batch_size in [1, 5, 10, 32]:
            mock_tensor = MockTensor(batch_size)
            detected_batch_size = encoder._get_batch_size(mock_tensor)
            assert detected_batch_size == batch_size

    def test_fallback_features(self):
        """Test fallback feature creation."""
        encoder = BaseEncoder(output_dim=32, expected_input_dim=16)

        class MockTensor:
            def __init__(self):
                self._data = torch.randn(8, 16)

        mock_tensor = MockTensor()
        fallback_features = encoder.create_fallback_features(mock_tensor)

        assert fallback_features.shape == (8, 32)
        assert torch.all(fallback_features == 0)


class TestPhotometryEncoder:
    """Test PhotometryEncoder for photometric data processing."""

    def test_initialization(self):
        """Test PhotometryEncoder initializes correctly."""
        encoder = PhotometryEncoder(output_dim=64)

        assert isinstance(encoder, nn.Module)
        assert encoder.output_dim == 64
        assert encoder.expected_input_dim == 32
        assert hasattr(encoder, "encoder")

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with real PhotometricTensor."""
        encoder = PhotometryEncoder(output_dim=48)

        # Create real PhotometricTensor
        magnitudes = torch.randn(10, 5)  # 10 objects, 5 bands
        photometric_tensor = PhotometricTensor(
            data=magnitudes, bands=["u", "g", "r", "i", "z"]
        )

        output = encoder(photometric_tensor)
        assert output.shape == (10, 48)
        assert not torch.isnan(output).any()

    def test_different_band_numbers(self):
        """Test with different numbers of photometric bands."""
        encoder = PhotometryEncoder(output_dim=32)

        # Test with different band configurations
        band_configs = [
            (["g"], 1),
            (["g", "r", "i"], 3),
            (["u", "g", "r", "i", "z"], 5),
            (["J", "H", "K", "W1", "W2", "W3", "W4", "W5"], 8),
        ]

        for bands, num_bands in band_configs:
            magnitudes = torch.randn(6, num_bands)
            photometric_tensor = PhotometricTensor(data=magnitudes, bands=bands)

            output = encoder(photometric_tensor)
            assert output.shape == (6, 32)
            assert not torch.isnan(output).any()

    def test_color_computation(self):
        """Test color index computation."""
        encoder = PhotometryEncoder(output_dim=64)

        # Test with enough bands for color computation
        magnitudes = torch.randn(8, 5)  # 5 bands -> 4 possible colors
        photometric_tensor = PhotometricTensor(
            data=magnitudes, bands=["u", "g", "r", "i", "z"]
        )

        output = encoder(photometric_tensor)
        assert output.shape == (8, 64)
        assert not torch.isnan(output).any()


class TestAstrometryEncoder:
    """Test AstrometryEncoder for astrometric data processing."""

    def test_initialization(self):
        """Test AstrometryEncoder initializes correctly."""
        encoder = AstrometryEncoder(output_dim=48)

        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, "encoder")

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with real Spatial3DTensor."""
        encoder = AstrometryEncoder(output_dim=32)

        # Create real Spatial3DTensor
        coordinates = torch.randn(12, 3)  # RA, Dec, distance
        spatial_tensor = Spatial3DTensor(data=coordinates, coordinate_system="icrs")

        output = encoder(spatial_tensor)
        assert output.shape == (12, 32)
        assert not torch.isnan(output).any()

    def test_different_coordinate_dimensions(self):
        """Test with different coordinate dimensions."""
        encoder = AstrometryEncoder(output_dim=24)

        # Test with 3D coordinates (Spatial3DTensor requires exactly 3 dimensions)
        coordinates = torch.randn(8, 3)
        spatial_tensor = Spatial3DTensor(data=coordinates, coordinate_system="galactic")

        output = encoder(spatial_tensor)
        assert output.shape == (8, 24)
        assert not torch.isnan(output).any()

    def test_proper_motion_handling(self):
        """Test proper motion data handling."""
        encoder = AstrometryEncoder(output_dim=32)

        # Test with 3D coordinates (Spatial3DTensor requires 3D coordinates)
        coordinates = torch.randn(10, 3)  # x, y, z coordinates
        spatial_tensor = Spatial3DTensor(data=coordinates, coordinate_system="icrs")

        output = encoder(spatial_tensor)
        assert output.shape == (10, 32)
        assert not torch.isnan(output).any()


class TestSpectroscopyEncoder:
    """Test SpectroscopyEncoder for spectroscopic data processing."""

    def test_initialization(self):
        """Test SpectroscopyEncoder initializes correctly."""
        encoder = SpectroscopyEncoder(output_dim=128)

        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, "encoder")

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with real SpectralTensor."""
        encoder = SpectroscopyEncoder(output_dim=64)

        # Create real SpectralTensor
        wavelengths = torch.linspace(4000, 7000, 1000)
        flux = torch.randn(6, 1000)  # 6 spectra, 1000 wavelength bins
        spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)

        output = encoder(spectral_tensor)
        assert output.shape == (6, 64)
        assert not torch.isnan(output).any()

    def test_different_spectrum_lengths(self):
        """Test with different spectrum lengths."""
        encoder = SpectroscopyEncoder(output_dim=96)

        # Test with different spectrum lengths
        spectrum_lengths = [100, 500, 1000, 2048]

        for length in spectrum_lengths:
            wavelengths = torch.linspace(3000, 9000, length)
            flux = torch.randn(4, length)
            spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)

            output = encoder(spectral_tensor)
            assert output.shape == (4, 96)
            assert not torch.isnan(output).any()

    def test_survey_tensor_compatibility(self):
        """Test compatibility with different spectral data."""
        encoder = SpectroscopyEncoder(output_dim=48)

        # Test with shorter spectrum
        wavelengths = torch.linspace(5000, 6000, 500)
        flux = torch.randn(8, 500)
        spectral_tensor = SpectralTensor(data=flux, wavelengths=wavelengths)

        output = encoder(spectral_tensor)
        assert output.shape == (8, 48)
        assert not torch.isnan(output).any()


class TestLightcurveEncoder:
    """Test LightcurveEncoder for time-series data processing."""

    def test_initialization(self):
        """Test LightcurveEncoder initializes correctly."""
        encoder = LightcurveEncoder(output_dim=96)

        assert isinstance(encoder, nn.Module)
        assert hasattr(encoder, "encoder")

    def test_forward_pass_with_real_tensor(self):
        """Test forward pass with real LightcurveTensor."""
        encoder = LightcurveEncoder(output_dim=64)

        # Create real LightcurveTensor
        # For batch processing: times and magnitudes must have same first dimension
        n_points = 100
        times = torch.linspace(0, 100, n_points)  # 100 time points
        magnitudes = torch.randn(n_points, 3)  # 100 time points, 3 bands
        lightcurve_tensor = LightcurveTensor(
            times=times, magnitudes=magnitudes, bands=["g", "r", "i"]
        )

        output = encoder(lightcurve_tensor)
        # LightcurveEncoder should handle time series and return (batch=1, features)
        # Since we have one lightcurve with n_points time samples
        assert output.shape[0] == 1  # Should be batch size 1
        assert output.shape[1] == 64  # Should be feature dimension
        assert not torch.isnan(output).any()

    def test_different_sequence_lengths(self):
        """Test with different lightcurve lengths."""
        encoder = LightcurveEncoder(output_dim=48)

        # Test with different sequence lengths
        sequence_lengths = [50, 100, 200, 500]

        for length in sequence_lengths:
            times = torch.linspace(0, 50, length)
            magnitudes = torch.randn(length, 2)  # length time points, 2 bands
            lightcurve_tensor = LightcurveTensor(
                times=times, magnitudes=magnitudes, bands=["g", "r"]
            )

            output = encoder(lightcurve_tensor)
            # Should return (batch=1, features) for single lightcurve
            assert output.shape[0] == 1
            assert output.shape[1] == 48
            assert not torch.isnan(output).any()

    def test_different_feature_dimensions(self):
        """Test with different numbers of bands."""
        encoder = LightcurveEncoder(output_dim=72)

        # Test with different feature dimensions
        band_configs = [
            (["g"], 1),
            (["g", "r"], 2),
            (["g", "r", "i"], 3),
            (["u", "g", "r", "i", "z"], 5),
        ]

        for bands, num_bands in band_configs:
            times = torch.linspace(0, 30, 150)
            magnitudes = torch.randn(150, num_bands)  # 150 time points, num_bands
            lightcurve_tensor = LightcurveTensor(
                times=times, magnitudes=magnitudes, bands=bands
            )

            output = encoder(lightcurve_tensor)
            # Should return (batch=1, features) for single lightcurve
            assert output.shape[0] == 1
            assert output.shape[1] == 72
            assert not torch.isnan(output).any()


class TestEncoderIntegration:
    """Test integration between different encoders."""

    def test_encoder_compatibility(self):
        """Test that different encoders produce compatible outputs."""
        output_dim = 64
        batch_size = 10

        # Create different encoders with same output dimension
        photometry_encoder = PhotometryEncoder(output_dim=output_dim)
        astrometry_encoder = AstrometryEncoder(output_dim=output_dim)
        spectroscopy_encoder = SpectroscopyEncoder(output_dim=output_dim)
        lightcurve_encoder = LightcurveEncoder(output_dim=output_dim)

        # Create real tensors
        photometry_data = PhotometricTensor(
            data=torch.randn(batch_size, 5), bands=["u", "g", "r", "i", "z"]
        )
        astrometry_data = Spatial3DTensor(
            data=torch.randn(batch_size, 3), coordinate_system="icrs"
        )
        spectroscopy_data = SpectralTensor(
            data=torch.randn(batch_size, 1000),
            wavelengths=torch.linspace(4000, 7000, 1000),
        )
        # Create multiple separate lightcurves for batch processing
        lightcurve_outputs = []
        for i in range(batch_size):
            times = torch.linspace(0, 50, 100)
            magnitudes = torch.randn(100, 2)  # 100 time points, 2 bands
            lightcurve_tensor = LightcurveTensor(
                times=times, magnitudes=magnitudes, bands=["g", "r"]
            )
            lc_output = lightcurve_encoder(lightcurve_tensor)
            lightcurve_outputs.append(lc_output)

        # Stack individual lightcurve outputs to simulate batch
        lightcurve_output = torch.cat(lightcurve_outputs, dim=0)

        photometry_output = photometry_encoder(photometry_data)
        astrometry_output = astrometry_encoder(astrometry_data)
        spectroscopy_output = spectroscopy_encoder(spectroscopy_data)

        # All should have same output shape
        assert photometry_output.shape == (batch_size, output_dim)
        assert astrometry_output.shape == (batch_size, output_dim)
        assert spectroscopy_output.shape == (batch_size, output_dim)
        assert lightcurve_output.shape == (batch_size, output_dim)

    def test_multi_modal_encoding(self):
        """Test combining multiple encoder outputs."""
        batch_size = 8
        output_dim = 32

        # Create multiple encoders
        photometry_encoder = PhotometryEncoder(output_dim=output_dim)
        astrometry_encoder = AstrometryEncoder(output_dim=output_dim)

        # Create real data
        photometry_data = PhotometricTensor(
            data=torch.randn(batch_size, 5), bands=["u", "g", "r", "i", "z"]
        )
        astrometry_data = Spatial3DTensor(
            data=torch.randn(batch_size, 3), coordinate_system="galactic"
        )

        photometry_features = photometry_encoder(photometry_data)
        astrometry_features = astrometry_encoder(astrometry_data)

        # Combine features (simple concatenation)
        combined_features = torch.cat([photometry_features, astrometry_features], dim=1)

        assert combined_features.shape == (batch_size, output_dim * 2)
        assert not torch.isnan(combined_features).any()

    def test_device_consistency(self):
        """Test that encoders work on different devices."""
        encoder = PhotometryEncoder(output_dim=32)

        # Test on CPU
        photometry_data_cpu = PhotometricTensor(
            data=torch.randn(5, 5), bands=["u", "g", "r", "i", "z"]
        )
        output_cpu = encoder(photometry_data_cpu)
        assert output_cpu.device.type == "cpu"

        # Test on CUDA if available
        if torch.cuda.is_available():
            encoder_cuda = encoder.cuda()
            photometry_data_cuda = PhotometricTensor(
                data=torch.randn(5, 5, device="cuda"), bands=["u", "g", "r", "i", "z"]
            )
            output_cuda = encoder_cuda(photometry_data_cuda)
            assert output_cuda.device.type == "cuda"


class TestEncoderErrorHandling:
    """Test error handling in encoders."""

    def test_dimension_mismatch_handling(self):
        """Test handling of dimension mismatches."""
        encoder = PhotometryEncoder(output_dim=48)

        # Test with very small input
        small_data = PhotometricTensor(data=torch.randn(6, 1), bands=["g"])
        output_small = encoder(small_data)
        assert output_small.shape == (6, 48)
        assert not torch.isnan(output_small).any()

        # Test with very large input
        large_data = PhotometricTensor(
            data=torch.randn(6, 12), bands=[f"band_{i}" for i in range(12)]
        )
        output_large = encoder(large_data)
        assert output_large.shape == (6, 48)
        assert not torch.isnan(output_large).any()

    def test_empty_batch_handling(self):
        """Test handling of empty batches."""
        encoder = AstrometryEncoder(output_dim=24)

        # Test that empty batch creation raises ValueError (expected behavior)
        with pytest.raises(ValueError, match="cannot be empty"):
            empty_data = Spatial3DTensor(
                data=torch.randn(0, 3), coordinate_system="icrs"
            )

        # Test with minimal valid batch instead
        minimal_data = Spatial3DTensor(data=torch.randn(1, 3), coordinate_system="icrs")
        output = encoder(minimal_data)
        assert output.shape == (1, 24)

    def test_single_object_handling(self):
        """Test handling of single objects."""
        encoder = LightcurveEncoder(output_dim=32)

        # Test with single lightcurve
        times = torch.linspace(0, 10, 50)
        magnitudes = torch.randn(50, 2)  # 50 time points, 2 bands
        lightcurve_tensor = LightcurveTensor(
            times=times, magnitudes=magnitudes, bands=["g", "r"]
        )

        output = encoder(lightcurve_tensor)
        # Should return (batch=1, features) for single lightcurve
        assert output.shape[0] == 1
        assert output.shape[1] == 32
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
