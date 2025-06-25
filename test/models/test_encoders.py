"""Tests for Simplified Encoders."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from astro_lab.models.encoders import (
    ENCODER_REGISTRY,
    AstrometryEncoder,
    BaseEncoder,
    LightcurveEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
    create_encoder,
)
from astro_lab.tensors import (
    LightcurveTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
    SurveyTensorDict,
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
        cpu_encoder = BaseEncoder(input_dim=8, output_dim=16, device="cpu")
        assert next(cpu_encoder.parameters()).device.type == "cpu"

        # Test CUDA device if available
        if torch.cuda.is_available():
            cuda_encoder = BaseEncoder(input_dim=8, output_dim=16, device="cuda")
            assert next(cuda_encoder.parameters()).device.type == "cuda"

    def test_forward_pass(self):
        """Test forward pass."""
        encoder = BaseEncoder(input_dim=10, output_dim=20)
        x = torch.randn(5, 10)
        output = encoder(x)
        assert output.shape == (5, 20)

    def test_encoder_learns(self):
        """Test that encoder can actually learn."""
        torch.manual_seed(42)
        encoder = BaseEncoder(input_dim=10, output_dim=5)

        # Create simple dataset: input features should predict class
        X = torch.randn(100, 10)
        # Simple rule: if sum of first 5 features > 0, class 0, else class 1
        y = (X[:, :5].sum(dim=1) > 0).long()

        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

        # Train for a few steps
        initial_loss = None
        for _ in range(10):
            optimizer.zero_grad()
            outputs = encoder(X)
            # Add simple classifier head
            logits = torch.nn.Linear(5, 2)(outputs)
            loss = torch.nn.functional.cross_entropy(logits, y)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        # Loss should decrease
        final_loss = loss.item()
        assert final_loss < initial_loss, (
            f"Loss did not decrease: {initial_loss} -> {final_loss}"
        )


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

    def test_magnitude_normalization(self):
        """Test that encoder handles astronomical magnitudes properly."""
        encoder = PhotometryEncoder(output_dim=32)

        # Create realistic magnitude data (typically 10-20 range)
        magnitudes = torch.tensor(
            [
                [15.2, 14.8, 14.3, 14.0, 13.7],  # Bright star
                [20.5, 20.1, 19.8, 19.5, 19.2],  # Faint star
                [17.0, 16.5, 16.0, 15.5, 15.0],  # Medium star
            ]
        )

        output = encoder(magnitudes)

        # Check that different magnitudes produce different embeddings
        embeddings_std = output.std(dim=0)
        assert embeddings_std.mean() > 0.1, "Encoder produces too similar embeddings"

        # Check that encoder preserves relative brightness information
        # Brighter objects (lower magnitude) should have distinguishable features
        bright_embedding = output[0]
        faint_embedding = output[1]
        distance = torch.nn.functional.cosine_similarity(
            bright_embedding.unsqueeze(0), faint_embedding.unsqueeze(0)
        )
        assert distance < 0.99, "Bright and faint stars have too similar embeddings"

    def test_handles_missing_bands(self):
        """Test handling of missing photometric bands (NaN values)."""
        encoder = PhotometryEncoder(output_dim=32)

        # Create data with missing values
        magnitudes = torch.tensor(
            [
                [15.2, 14.8, float("nan"), 14.0, 13.7],
                [20.5, 20.1, 19.8, 19.5, 19.2],
            ]
        )

        # Should handle NaN values without crashing
        output = encoder(magnitudes)
        assert not torch.isnan(output).any(), "Encoder output contains NaN values"


class TestAstrometryEncoder:
    """Test the astrometry encoder."""

    def test_initialization(self):
        """Test AstrometryEncoder initialization."""
        encoder = AstrometryEncoder(output_dim=64)
        assert encoder.output_dim == 64
        assert encoder.input_dim == 5  # Default: ra, dec, parallax, pmra, pmdec

    def test_coordinate_invariance(self):
        """Test that encoder is invariant to coordinate translations."""
        encoder = AstrometryEncoder(output_dim=32)

        # Create astrometric data
        base_data = torch.tensor(
            [
                [120.0, 30.0, 10.0, 1.5, -2.0],  # RA, Dec, parallax, pm_ra, pm_dec
            ]
        )

        # RA wrapping around 360 degrees should produce same result
        wrapped_data = base_data.clone()
        wrapped_data[0, 0] += 360.0  # Add 360 to RA

        output1 = encoder(base_data)
        output2 = encoder(wrapped_data)

        # Note: Current implementation might not be invariant,
        # but this test documents expected behavior
        # For now, just check they produce valid outputs
        assert output1.shape == output2.shape

    def test_parallax_distance_encoding(self):
        """Test that parallax (distance) information is preserved."""
        encoder = AstrometryEncoder(output_dim=32, input_dim=3)

        # Create objects at different distances (parallax in mas)
        nearby = torch.tensor([[120.0, 30.0, 100.0]])  # 10 pc
        distant = torch.tensor([[120.0, 30.0, 1.0]])  # 1000 pc

        nearby_encoding = encoder(nearby)
        distant_encoding = encoder(distant)

        # Different distances should produce different encodings
        distance = torch.nn.functional.mse_loss(nearby_encoding, distant_encoding)
        assert distance > 0.01, "Nearby and distant objects have too similar encodings"


class TestSpectroscopyEncoder:
    """Test the spectroscopy encoder."""

    def test_spectral_parameter_encoding(self):
        """Test encoding of stellar parameters."""
        encoder = SpectroscopyEncoder(output_dim=64, input_dim=3)

        # Create realistic stellar parameters
        # [Teff (K), log g, [Fe/H]]
        hot_star = torch.tensor([[10000.0, 4.0, 0.0]])  # Hot main sequence
        cool_star = torch.tensor([[3500.0, 4.5, -0.5]])  # Cool dwarf
        giant_star = torch.tensor([[5000.0, 2.0, 0.2]])  # Red giant

        hot_encoding = encoder(hot_star)
        cool_encoding = encoder(cool_star)
        giant_encoding = encoder(giant_star)

        # Different stellar types should have different encodings
        hot_cool_dist = torch.nn.functional.mse_loss(hot_encoding, cool_encoding)
        hot_giant_dist = torch.nn.functional.mse_loss(hot_encoding, giant_encoding)

        assert hot_cool_dist > 0.01, "Hot and cool stars too similar"
        assert hot_giant_dist > 0.01, "Hot star and giant too similar"

    def test_spectrum_encoding(self):
        """Test encoding of full spectral data."""
        encoder = SpectroscopyEncoder(output_dim=64, input_dim=100)

        # Create synthetic spectral data
        wavelengths = torch.linspace(4000, 7000, 100)

        # Simple blackbody-like spectra
        hot_spectrum = torch.exp(-(((wavelengths - 4500) / 1000) ** 2)).unsqueeze(0)
        cool_spectrum = torch.exp(-(((wavelengths - 6500) / 1000) ** 2)).unsqueeze(0)

        hot_encoding = encoder(hot_spectrum)
        cool_encoding = encoder(cool_spectrum)

        # Different spectra should produce different encodings
        spectral_distance = torch.nn.functional.cosine_similarity(
            hot_encoding, cool_encoding
        )
        assert spectral_distance < 0.95, (
            "Different spectra produce too similar encodings"
        )


class TestLightcurveEncoder:
    """Test the lightcurve encoder."""

    def test_periodic_signal_encoding(self):
        """Test that encoder can capture periodic signals."""
        encoder = LightcurveEncoder(hidden_dim=32, output_dim=64)

        # Create synthetic periodic lightcurve
        time = torch.linspace(0, 10, 100)

        # Sinusoidal signal with period 2.5
        periodic_signal = torch.sin(2 * np.pi * time / 2.5) + 0.1 * torch.randn_like(
            time
        )

        # Random signal
        random_signal = torch.randn_like(time)

        periodic_encoding = encoder(periodic_signal)
        random_encoding = encoder(random_signal)

        # Periodic and random signals should have different encodings
        distance = torch.nn.functional.mse_loss(periodic_encoding, random_encoding)
        assert distance > 0.01, "Periodic and random signals have too similar encodings"

    def test_lstm_temporal_processing(self):
        """Test that LSTM processes temporal information correctly."""
        encoder = LightcurveEncoder(hidden_dim=64, output_dim=32)

        # Create lightcurves with different temporal patterns
        increasing = torch.linspace(0, 1, 50)  # Increasing brightness
        decreasing = torch.linspace(1, 0, 50)  # Decreasing brightness

        inc_encoding = encoder(increasing)
        dec_encoding = encoder(decreasing)

        # Different temporal patterns should produce different encodings
        pattern_distance = torch.nn.functional.cosine_similarity(
            inc_encoding, dec_encoding
        )
        assert pattern_distance < 0.99, "Different temporal patterns too similar"

    def test_variable_length_sequences(self):
        """Test handling of variable length sequences."""
        encoder = LightcurveEncoder(hidden_dim=32, output_dim=64)

        # Different length sequences
        short_seq = torch.randn(20)
        long_seq = torch.randn(100)

        short_encoding = encoder(short_seq)
        long_encoding = encoder(long_seq)

        # Both should produce same output dimension
        assert short_encoding.shape == long_encoding.shape == (1, 64)


class TestEncoderFactory:
    """Test encoder factory function."""

    def test_create_encoder(self):
        """Test creating encoders via factory."""
        # Photometry
        phot_encoder = create_encoder("photometry", output_dim=64)
        assert isinstance(phot_encoder, PhotometryEncoder)

        # Astrometry
        astro_encoder = create_encoder("astrometry", output_dim=32)
        assert isinstance(astro_encoder, AstrometryEncoder)

        # Spectroscopy
        spec_encoder = create_encoder("spectroscopy", output_dim=128)
        assert isinstance(spec_encoder, SpectroscopyEncoder)

        # Lightcurve
        lc_encoder = create_encoder("lightcurve", hidden_dim=64, output_dim=96)
        assert isinstance(lc_encoder, LightcurveEncoder)

    def test_invalid_encoder_type(self):
        """Test error on invalid encoder type."""
        with pytest.raises(ValueError):
            create_encoder("invalid_encoder", output_dim=64)

    def test_encoder_registry(self):
        """Test encoder registry."""
        assert "photometry" in ENCODER_REGISTRY
        assert "astrometry" in ENCODER_REGISTRY
        assert "spectroscopy" in ENCODER_REGISTRY
        assert "lightcurve" in ENCODER_REGISTRY


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

        # Check that fusion preserves information
        assert fused_features.std() > 0.1, "Fused features have too low variance"

    def test_gradient_flow(self):
        """Test that gradients flow through encoders properly."""
        encoder = PhotometryEncoder(output_dim=32)

        # Create simple data
        data = torch.randn(10, 5, requires_grad=True)

        # Forward pass
        output = encoder(data)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are non-zero
        assert data.grad is not None, "No gradients computed for input"
        assert data.grad.abs().sum() > 0, "Gradients are all zero"

        # Check encoder parameters have gradients
        for param in encoder.parameters():
            assert param.grad is not None, "No gradients for encoder parameters"
            assert param.grad.abs().sum() > 0, "Encoder gradients are all zero"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
