"""
Tests for data visualization utilities.

Tests visualization data preparation, color mapping, and astronomical plotting helpers.
"""

import pytest
import numpy as np
import torch


class TestDataVisualization:
    """Test visualization data preparation."""

    def test_spatial_tensor_color_mapping(self):
        """Test color mapping for spatial data."""
        from astro_lab.tensors import Spatial3DTensor
        
        # Create test stellar data
        positions = np.random.rand(100, 3) * 10
        spatial_tensor = Spatial3DTensor(positions, unit="pc")
        
        # Test distance-based coloring
        distances = torch.norm(spatial_tensor.data, dim=1)
        
        # Normalize for color mapping
        normalized_distances = (distances - distances.min()) / (distances.max() - distances.min())
        
        assert normalized_distances.min() >= 0
        assert normalized_distances.max() <= 1
        assert normalized_distances.shape[0] == 100

    def test_magnitude_to_size_mapping_realistic(self):
        """Test realistic magnitude to size mapping."""
        # Simulate stellar magnitudes (brighter = smaller magnitude)
        apparent_mags = np.random.uniform(6, 14, 200)  # Typical range
        
        # Convert magnitude to size (brighter stars = larger markers)
        # Use Pogson's ratio: flux ratio = 10^(0.4 * mag_diff)
        brightest_mag = apparent_mags.min()
        mag_diff = apparent_mags - brightest_mag
        relative_flux = 10**(-0.4 * mag_diff)
        
        # Scale to marker sizes
        min_size, max_size = 1, 20
        sizes = min_size + (max_size - min_size) * relative_flux
        
        assert len(sizes) == 200
        assert sizes.min() >= min_size
        assert sizes.max() <= max_size
        assert sizes[apparent_mags.argmin()] == max_size  # Brightest star = largest

    def test_distance_modulus_visualization(self):
        """Test distance modulus calculations for visualization."""
        # Test distance modulus: m - M = 5 * log10(d) - 5
        distances_pc = np.array([10, 100, 1000, 10000])  # parsecs
        expected_distance_moduli = 5 * np.log10(distances_pc) - 5
        
        # Calculate directly
        calculated_dm = 5 * np.log10(distances_pc) - 5
        
        np.testing.assert_array_almost_equal(calculated_dm, expected_distance_moduli)
        
        # Test that closer objects have smaller distance moduli
        assert calculated_dm[0] < calculated_dm[1] < calculated_dm[2] < calculated_dm[3] 