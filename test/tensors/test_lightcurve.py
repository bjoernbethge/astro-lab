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
        times = torch.sort(torch.rand(n_observations) * 1000)[0].unsqueeze(1)
        magnitudes = torch.randn(n_observations).unsqueeze(1)
        data = torch.cat([times, magnitudes], dim=1)

        lc = LightcurveTensor(
            data=data, meta={"time_unit": "days", "magnitude_system": "AB"}
        )

        assert lc.meta["time_unit"] == "days"
        assert lc.meta["magnitude_system"] == "AB"
        assert len(lc.times) == n_observations

    def test_time_series_validation(self):
        """Test time series validation."""
        # Test validation - unsorted times should fail
        times = torch.tensor([1.0, 3.0, 2.0]).unsqueeze(1)
        magnitudes = torch.randn(3).unsqueeze(1)
        data = torch.cat([times, magnitudes], dim=1)
        
        with pytest.raises(ValueError, match="monotonically increasing"):
            LightcurveTensor(data=data)

    def test_lightcurve_properties(self):
        """Test lightcurve tensor basic properties."""
        # Create lightcurve with time spanning from 0 to 100
        times = torch.linspace(0, 100, 100).unsqueeze(1)  # Time from 0 to 100
        magnitudes = torch.randn(100, 1)
        data = torch.cat([times, magnitudes], dim=1)  # [time, magnitude]
        
        lightcurve = LightcurveTensor(data=data)

        assert lightcurve.n_observations == 100
        # Check time span calculation
        time_span = lightcurve.time_span
        assert abs(time_span - 100.0) < 1e-5  # Should be exactly 100.0

        # Test methods exist
        assert hasattr(lightcurve, "phase_fold")
        assert hasattr(lightcurve, "compute_statistics")
