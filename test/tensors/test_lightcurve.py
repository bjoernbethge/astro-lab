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
        """Test lightcurve properties."""
        times = torch.linspace(0, 100, 50).unsqueeze(1)
        magnitudes = torch.sin(2 * torch.pi * times / 10) + torch.randn(50).unsqueeze(1)
        data = torch.cat([times, magnitudes], dim=1)

        lc = LightcurveTensor(data=data)

        # Test time range
        time_span, _ = lc.get_time_range()
        assert abs(time_span - 100.0) < 1e-5

        # Test methods exist
        assert hasattr(lc, "phase_fold")
        assert hasattr(lc, "compute_statistics")
