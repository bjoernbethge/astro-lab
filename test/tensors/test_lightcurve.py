"""
Tests for LightcurveTensor.
"""

import pytest
import torch

from astro_lab.tensors.lightcurve import LightcurveTensor


class TestLightcurveTensor:
    """Test lightcurve tensor functionality."""

    def test_lightcurve_creation(self):
        """Test lightcurve tensor creation."""
        n_observations = 100
        times = torch.sort(torch.rand(n_observations) * 1000)[0]  # Sort time
        magnitudes = torch.randn(n_observations)

        lc = LightcurveTensor(
            times=times, magnitudes=magnitudes, time_unit="days", magnitude_system="AB"
        )

        assert lc.time_unit == "days"
        assert lc.magnitude_system == "AB"
        assert len(lc.times) == n_observations

    def test_time_series_validation(self):
        """Test time series validation."""
        # Test validation - empty times should fail
        times = torch.tensor([])
        magnitudes = torch.randn(100)
        with pytest.raises(ValueError):
            LightcurveTensor(times=times, magnitudes=magnitudes)

    def test_lightcurve_properties(self):
        """Test lightcurve properties."""
        times = torch.linspace(0, 100, 50)
        magnitudes = torch.sin(2 * torch.pi * times / 10) + torch.randn(50) * 0.1

        lc = LightcurveTensor(times=times, magnitudes=magnitudes)

        # Test time range
        time_range = lc.times.max() - lc.times.min()
        assert abs(time_range - 100.0) < 1e-5

        # Test methods exist
        assert hasattr(lc, "compute_period_folded")
        assert hasattr(lc, "compute_statistics")
