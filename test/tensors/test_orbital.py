"""
Tests for OrbitTensor.
"""

import pytest
import torch

from astro_lab.tensors.orbital import OrbitTensor


class TestOrbitTensor:
    """Test orbital tensor functionality."""

    def test_orbital_creation(self):
        """Test orbital tensor creation."""
        # Orbital elements: [a, e, i, Ω, ω, ν] for N objects
        n_objects = 10
        orbital_elements = torch.rand(n_objects, 6)
        orbital_elements[:, 1] = torch.clamp(
            orbital_elements[:, 1], 0, 0.9
        )  # Eccentricity < 1

        orbital = OrbitTensor(
            orbital_elements,
            element_type="keplerian",
            attractor="earth",
        )

        assert orbital.element_type == "keplerian"
        assert orbital.attractor == "earth"
        assert orbital.mu > 0  # Has gravitational parameter

    def test_orbital_validation(self):
        """Test orbital element validation."""
        # Wrong number of elements
        with pytest.raises(ValueError):
            OrbitTensor(torch.randn(10, 5))  # Should be 6 elements

    def test_orbital_properties(self):
        """Test orbital tensor properties."""
        elements = torch.rand(5, 6)
        elements[:, 1] = 0.5  # Set eccentricity

        orbital = OrbitTensor(elements)

        # Test methods exist
        assert hasattr(orbital, "to_cartesian")
        assert hasattr(orbital, "propagate")
        assert hasattr(orbital, "orbital_period")
