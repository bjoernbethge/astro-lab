"""
Integration tests for the complete tensor-based simulation system.
"""

import pytest
import torch

from astro_lab.tensors import CosmologyCalculator, SimulationTensor


class TestIntegration:
    """Integration tests for the complete tensor-based system."""

    def test_tng50_workflow(self):
        """Test typical TNG50 analysis workflow."""
        # Simulate TNG50 gas particles
        positions = torch.randn(1000, 3) * 10000  # TNG50-like scale
        masses = torch.distributions.Exponential(1.0).sample((1000, 1))
        potentials = torch.randn(1000, 1)
        features = torch.cat([masses, potentials], dim=1)

        # Create simulation tensor
        sim_tensor = SimulationTensor(
            positions=positions,
            features=features,
            simulation_name="TNG50",
            particle_type="gas",
            redshift=0.1,
            box_size=35000.0,
        )

        # Test cosmological calculations with proper type handling
        age = sim_tensor.cosmology.age_of_universe(sim_tensor.redshift)

        # Fix linter error by handling different return types
        if hasattr(age, "item"):
            age_scalar = age.item()
        elif hasattr(age, "__getitem__"):
            age_scalar = float(age[0]) if len(age) > 0 else float(age)
        else:
            age_scalar = float(age)

        assert age_scalar > 10.0  # Reasonable age

        # Test center of mass
        com = sim_tensor.calculate_center_of_mass()
        assert com.shape == (3,)

        # Test subset selection (high mass particles)
        high_mass_mask = features[:, 0] > features[:, 0].median()
        subset = sim_tensor.get_particle_subset(high_mass_mask)
        assert subset.num_particles < 1000
