"""
Tests for CosmologyCalculator - now integrated with tensors.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import CosmologyCalculator


class TestCosmologyCalculator:
    """Test CosmologyCalculator - now integrated with tensors."""

    def test_init_default(self):
        """Test default initialization."""
        calc = CosmologyCalculator()
        assert calc.H0 == 70.0
        assert calc.Omega_m == 0.3
        assert calc.Omega_Lambda == 0.7

    def test_hubble_parameter_tensor(self):
        """Test Hubble parameter with PyTorch tensors."""
        calc = CosmologyCalculator()
        z_tensor = torch.tensor([0.0, 0.5, 1.0])

        h_tensor = calc.hubble_parameter(z_tensor)
        assert isinstance(h_tensor, torch.Tensor)
        assert h_tensor[0] == 70.0  # z=0 case
        assert torch.all(h_tensor[1:] > 70.0)  # Higher z cases

    def test_comoving_distance_tensor(self):
        """Test comoving distance with tensors."""
        calc = CosmologyCalculator()
        z_tensor = torch.tensor([0.0, 1.0])

        d_tensor = calc.comoving_distance(z_tensor)

        assert isinstance(d_tensor, torch.Tensor)
        assert d_tensor.shape == (2,)
        assert abs(d_tensor[0].item()) < 1e-6  # z=0 case
        assert d_tensor[1].item() > 0  # z=1 case

    def test_angular_diameter_distance(self):
        """Test angular diameter distance with tensors."""
        calc = CosmologyCalculator()
        z = torch.tensor([0.5, 1.0, 2.0])

        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)
        
        expected = d_c / (1 + z)
        assert torch.all(torch.abs(d_a - expected) < 1e-6)

    def test_age_of_universe(self):
        """Test age calculation with tensors."""
        calc = CosmologyCalculator()

        z_tensor = torch.tensor([0.0, 1.0, 2.0])
        ages = calc.age_of_universe(z_tensor)
        
        assert isinstance(ages, torch.Tensor)
        assert ages.shape == (3,)
        assert ages[0] > ages[1] > ages[2]  # Universe was younger at higher z
        assert ages[0] > 13.0 # Approx age at z=0 in Gyr
        assert ages[2] > 0

    def test_cosmology_consistency(self):
        """Test that different distance measures are consistent."""
        calc = CosmologyCalculator(H0=67.4, Omega_m=0.315, Omega_Lambda=0.685)

        z = torch.tensor([0.1, 0.5, 1.0, 2.0])
        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)
        d_l = calc.luminosity_distance(z)

        # Check relationships
        assert torch.all(torch.abs(d_a - d_c / (1 + z)) < 1e-5)
        assert torch.all(torch.abs(d_l - d_c * (1 + z)) < 1e-5)
