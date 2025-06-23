"""
Tests for SimulationTensor - the new tensor-based simulation architecture.
"""

import numpy as np
import pytest
import torch
from torch_geometric.data import Data as PyGData

from astro_lab.tensors import CosmologyCalculator, SimulationTensor


class TestSimulationTensor:
    """Test SimulationTensor - the new tensor-based simulation architecture."""

    def test_init_basic(self):
        """Test basic initialization of the SimulationTensor."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(data=positions)
        assert torch.equal(sim_tensor.positions, positions)
        assert sim_tensor.features is None
        assert sim_tensor.num_particles == 10

    def test_init_with_features(self):
        """Test initialization with additional features."""
        positions = torch.randn(10, 3)
        features = torch.randn(10, 2)
        data = torch.cat([positions, features], dim=1)
        sim_tensor = SimulationTensor(
            data=data,
            feature_names=["mass", "temperature"]
        )
        assert sim_tensor.num_particles == 10
        assert sim_tensor.features.shape == (10, 2)
        assert sim_tensor.feature_names == ["mass", "temperature"]

    def test_init_with_edges(self):
        """Test initialization with graph edge indices."""
        positions = torch.randn(10, 3)
        edge_index = torch.randint(0, 10, (2, 20))
        sim_tensor = SimulationTensor(
            data=positions,
            edge_index=edge_index
        )
        assert torch.equal(sim_tensor.edge_index, edge_index)

    def test_cosmology_integration(self):
        """Test that cosmological parameters are correctly integrated."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(data=positions, redshift=1.0)
        assert "age_universe_gyr" in sim_tensor.meta
        assert sim_tensor.meta["age_universe_gyr"] > 0
        assert sim_tensor.meta["hubble_param"] > 0

    def test_periodic_distance(self):
        """Test periodic distance calculation."""
        positions = torch.tensor([[1.0, 1.0, 1.0], [99.0, 99.0, 99.0]])
        sim_tensor = SimulationTensor(data=positions, box_size=100.0)
        # The shortest distance should wrap around the box
        dist = sim_tensor.periodic_distance(0, 1)
        assert np.isclose(dist, np.sqrt(12.0), atol=1e-6)

    def test_periodic_boundaries_application(self):
        """Test application of periodic boundaries."""
        positions = torch.tensor([[101.0, -5.0, 50.0]])
        sim_tensor = SimulationTensor(data=positions, box_size=100.0)
        wrapped_pos = sim_tensor.apply_periodic_boundaries(positions)
        expected_pos = torch.tensor([[1.0, 95.0, 50.0]])
        assert torch.allclose(wrapped_pos, expected_pos)

    def test_particle_subset(self):
        """Test selecting a subset of particles."""
        positions = torch.randn(10, 3)
        features = torch.arange(10).unsqueeze(1)
        data = torch.cat([positions, features], dim=1)
        sim_tensor = SimulationTensor(data=data, feature_names=["id"])
        
        mask = sim_tensor.features[:, 0] < 5
        subset = sim_tensor.get_particle_subset(mask)
        assert subset.num_particles == 5
        assert torch.all(subset.features[:, 0] < 5)
        # Check that metadata is preserved
        assert subset.simulation_name == sim_tensor.simulation_name

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        positions = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        masses = torch.tensor([[1.0], [1.0]])
        data = torch.cat([positions, masses], dim=1)
        sim_tensor = SimulationTensor(data=data, feature_names=["mass"])
        
        com = sim_tensor.calculate_center_of_mass(mass_feature_idx=0)
        assert torch.allclose(com, torch.tensor([2.0, 0.0, 0.0]))

    def test_torch_geometric_conversion(self):
        """Test conversion to and from PyTorch Geometric Data objects."""
        positions = torch.randn(10, 3)
        features = torch.randn(10, 2)
        edge_index = torch.randint(0, 10, (2, 20))
        data = torch.cat([positions, features], dim=1)
        
        sim_tensor = SimulationTensor(
            data=data,
            edge_index=edge_index,
            feature_names=["f1", "f2"],
            simulation_name="TestSim",
            redshift=0.2
        )
        
        pyg_data = sim_tensor.to_torch_geometric()
        assert torch.equal(pyg_data.pos, sim_tensor.positions)
        assert torch.equal(pyg_data.x, sim_tensor.features)
        assert pyg_data.simulation_name == "TestSim"

        new_sim_tensor = SimulationTensor.from_torch_geometric(pyg_data)
        assert torch.equal(new_sim_tensor.data, sim_tensor.data)
        assert new_sim_tensor.simulation_name == "TestSim"
        assert new_sim_tensor.redshift == 0.2
        assert "age_universe_gyr" in new_sim_tensor.meta

    def test_from_torch_geometric(self):
        """Test creating a SimulationTensor from a PyG Data object."""
        pyg_data = PyGData(
            pos=torch.randn(15, 3),
            x=torch.randn(15, 1),
            edge_index=torch.randint(0, 15, (2, 30)),
            simulation_name="FromPyG",
            redshift=0.8
        )
        sim_tensor = SimulationTensor.from_torch_geometric(pyg_data, feature_names=['mass'])
        assert sim_tensor.num_particles == 15
        assert sim_tensor.features.shape == (15, 1)
        assert sim_tensor.simulation_name == "FromPyG"
        assert sim_tensor.redshift == 0.8
        assert "hubble_param" in sim_tensor.meta
        
    def test_redshift_update(self):
        """Test the update_redshift method."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(data=positions, redshift=0.0)
        age_at_z0 = sim_tensor.meta["age_universe_gyr"]
        
        updated_sim = sim_tensor.update_redshift(new_redshift=1.0)
        age_at_z1 = updated_sim.meta["age_universe_gyr"]
        
        assert updated_sim.redshift == 1.0
        assert age_at_z1 < age_at_z0
        # Ensure it returns a new instance
        assert id(updated_sim) != id(sim_tensor)

    def test_memory_info(self):
        """Test the memory_info method."""
        positions = torch.randn(100, 3)
        sim_tensor = SimulationTensor(data=positions)
        info = sim_tensor.memory_info()
        assert "total_size_mb" in info
        assert "device" in info
        assert float(info["total_size_mb"]) > 0

    def test_invalid_data_shape(self):
        """Test that initialization fails with invalid data shape."""
        with pytest.raises(ValueError):
            SimulationTensor(data=torch.randn(10, 2)) # Needs at least 3 pos columns
