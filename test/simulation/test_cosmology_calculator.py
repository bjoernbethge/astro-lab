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

        # Fix linter errors by handling different return types
        if isinstance(d_tensor, torch.Tensor):
            assert d_tensor.shape[0] == 2
            assert abs(d_tensor[0].item()) < 1e-6  # z=0 case
            assert d_tensor[1].item() > 0  # z=1 case
        elif isinstance(d_tensor, np.ndarray):
            assert len(d_tensor) == 2
            assert abs(d_tensor[0]) < 1e-6  # z=0 case
            assert d_tensor[1] > 0  # z=1 case
        else:
            # Single float case
            assert isinstance(d_tensor, float)

    def test_angular_diameter_distance(self):
        """Test angular diameter distance."""
        calc = CosmologyCalculator()
        z = 1.0

        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)

        # Fix linter errors by handling different return types
        if isinstance(d_c, (torch.Tensor, np.ndarray)):
            expected = d_c / (1 + z)
            if isinstance(d_a, (torch.Tensor, np.ndarray)):
                diff = np.abs(d_a - expected)
                if isinstance(diff, (torch.Tensor, np.ndarray)):
                    assert np.all(diff < 1e-6)
                else:
                    assert diff < 1e-6
        else:
            expected = d_c / (1 + z)
            assert abs(d_a - expected) < 1e-6

    def test_age_of_universe(self):
        """Test age calculation."""
        calc = CosmologyCalculator()

        age_z0 = calc.age_of_universe(0.0)
        age_z1 = calc.age_of_universe(1.0)

        # Fix linter errors by handling different return types
        def get_scalar_value(val):
            if isinstance(val, torch.Tensor):
                return val.item()
            elif isinstance(val, np.ndarray):
                return float(val) if val.ndim == 0 else val[0]
            else:
                return float(val)

        age_z0_scalar = get_scalar_value(age_z0)
        age_z1_scalar = get_scalar_value(age_z1)

        assert age_z0_scalar > age_z1_scalar  # Universe was younger at higher z
        assert age_z0_scalar > 0.5  # Reasonable age range
        assert age_z1_scalar > 0.0

    def test_cosmology_consistency(self):
        """Test cosmology calculations are consistent."""
        calc = CosmologyCalculator(H0=67.4, Omega_m=0.315, Omega_Lambda=0.685)

        z = 1.0
        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)
        d_l = calc.luminosity_distance(z)

        # Helper function to handle different return types
        def get_scalar_value(val):
            if isinstance(val, torch.Tensor):
                return val.item()
            elif isinstance(val, np.ndarray):
                return float(val) if val.ndim == 0 else val[0]
            else:
                return float(val)

        d_c_scalar = get_scalar_value(d_c)
        d_a_scalar = get_scalar_value(d_a)
        d_l_scalar = get_scalar_value(d_l)

        # Check relationships
        assert abs(d_a_scalar - d_c_scalar / (1 + z)) < 1e-6
        assert abs(d_l_scalar - d_c_scalar * (1 + z)) < 1e-6
