"""
Integration tests for the complete tensor-based simulation system.
"""

import pytest
import torch

from astro_lab.tensors import CosmologyCalculator, SimulationTensor


class TestIntegration:
    """Integration tests for the complete tensor-based system."""

    @pytest.mark.integration
    def test_tng50_workflow(self):
        """Test a complete TNG50 simulation workflow."""
        # Create a mock TNG50-like simulation
        positions = torch.randn(100, 3) * 1000  # 100 particles in a 1000 unit box
        masses = torch.exp(torch.randn(100, 1) * 2 + 10)  # Log-normal mass distribution
        
        data = torch.cat([positions, masses], dim=1)
        
        sim_tensor = SimulationTensor(
            data=data,
            simulation_name="TNG50",
            particle_type="gas",
            box_size=35000.0,
            redshift=0.2,
            feature_names=["mass"]
        )
        
        # Test cosmological calculations using the internal calculator
        age = sim_tensor._cosmology_calculator.age_of_universe(sim_tensor.redshift)
        assert age > 0
        assert age < 13.8  # Less than age of universe at z=0
        
        # Test graph construction
        edge_index = torch.randint(0, 100, (2, 200))
        sim_tensor.edge_index = edge_index
        
        pyg_data = sim_tensor.to_torch_geometric()
        assert pyg_data.pos.shape == (100, 3)
        assert pyg_data.x.shape == (100, 1)
        assert pyg_data.edge_index.shape == (2, 200)
