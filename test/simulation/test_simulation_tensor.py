"""
Tests for SimulationTensor - the new tensor-based simulation architecture.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import CosmologyCalculator, SimulationTensor


class TestSimulationTensor:
    """Test SimulationTensor - the new tensor-based simulation architecture."""

    def test_init_basic(self):
        """Test basic SimulationTensor initialization."""
        positions = torch.randn(100, 3)
        sim_tensor = SimulationTensor(data=positions)

        assert sim_tensor.num_particles == 100
        assert sim_tensor.simulation_name == "TNG50"
        assert sim_tensor.particle_type == "gas"
        assert sim_tensor.box_size == 35000.0
        assert sim_tensor.redshift == 0.0

    def test_init_with_features(self):
        """Test initialization with particle features."""
        positions = torch.randn(100, 3)
        features = torch.randn(100, 2)  # mass, potential
        data = torch.cat([positions, features], dim=1)
        feature_names = ["mass", "potential"]

        sim_tensor = SimulationTensor(
            data=data, particle_type="stars", redshift=1.0, feature_names=feature_names
        )

        assert sim_tensor.features is not None
        assert sim_tensor.features.shape == (100, 2)
        assert sim_tensor.particle_type == "stars"
        assert sim_tensor.redshift == 1.0

    def test_init_with_edges(self):
        """Test initialization with graph edges."""
        positions = torch.randn(50, 3)
        edge_index = torch.randint(0, 50, (2, 200))

        sim_tensor = SimulationTensor(
            data=positions, edge_index=edge_index, simulation_name="Illustris"
        )

        assert sim_tensor.edge_index is not None
        assert sim_tensor.edge_index.shape == (2, 200)
        assert sim_tensor.simulation_name == "Illustris"

    def test_cosmology_integration(self):
        """Test integrated cosmology calculations."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(data=positions, redshift=1.0)

        # Check cosmology calculator is created internally
        assert hasattr(sim_tensor, "_cosmology_calculator")
        assert isinstance(sim_tensor._cosmology_calculator, CosmologyCalculator)

        # Check derived cosmological quantities in meta
        assert abs(sim_tensor.get_metadata("scale_factor") - 0.5) < 1e-6 # 1/(1+z)
        assert sim_tensor.get_metadata("age_universe_gyr") > 0
        assert sim_tensor.get_metadata("hubble_param") > 0

    def test_periodic_distance(self):
        """Test periodic boundary distance calculations."""
        positions = torch.tensor([[1.0, 1.0, 1.0], [34999.0, 34999.0, 34999.0]])
        sim_tensor = SimulationTensor(data=positions, box_size=35000.0)

        # Should wrap around - distance ≈ 2*sqrt(3)
        dist = sim_tensor.periodic_distance(0, 1)
        expected = 2 * np.sqrt(3)
        assert abs(dist - expected) < 0.1

    def test_periodic_boundaries_application(self):
        """Test applying periodic boundary conditions."""
        positions = torch.tensor([[36000.0, -1000.0, 17500.0]])
        sim_tensor = SimulationTensor(data=positions, box_size=35000.0)

        wrapped = sim_tensor.apply_periodic_boundaries(positions)
        assert torch.all(wrapped >= 0)
        assert torch.all(wrapped < 35000.0)

    def test_particle_subset(self):
        """Test getting particle subsets."""
        positions = torch.randn(100, 3)
        features = torch.randn(100, 2)
        data = torch.cat([positions, features], dim=1)
        sim_tensor = SimulationTensor(data=data, feature_names=["mass", "velocity"])

        # Create mask for first 50 particles
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:50] = True

        subset = sim_tensor.get_particle_subset(mask)
        assert subset.num_particles == 50
        assert subset.features.shape == (50, 2)

    def test_center_of_mass(self):
        """Test center of mass calculation."""
        # Create simple test case
        positions = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        masses = torch.tensor([[1.0], [1.0]])  # Equal masses
        data = torch.cat([positions, masses], dim=1)

        sim_tensor = SimulationTensor(data=data, feature_names=["mass"])
        com = sim_tensor.calculate_center_of_mass(mass_feature_idx=0)

        # Should be at [1.0, 0.0, 0.0]
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(com, expected)

    def test_torch_geometric_conversion(self):
        """Test conversion to PyTorch Geometric format."""
        positions = torch.randn(20, 3)
        features = torch.randn(20, 2)
        edge_index = torch.randint(0, 20, (2, 50))
        data = torch.cat([positions, features], dim=1)

        sim_tensor = SimulationTensor(
            data=data, feature_names=["f1", "f2"], edge_index=edge_index
        )

        data = sim_tensor.to_torch_geometric()

        assert hasattr(data, "pos")
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert data.pos.shape == (20, 3)
        assert data.x.shape == (20, 2)
        assert data.edge_index.shape == (2, 50)

    def test_from_torch_geometric(self):
        """Test creation from PyTorch Geometric data."""
        try:
            import torch_geometric
            from torch_geometric.data import Data

            data = Data(
                pos=torch.randn(15, 3),
                x=torch.randn(15, 3),
                edge_index=torch.randint(0, 15, (2, 30)),
            )
            data.simulation_name = "test"
            data.redshift = 0.5

            sim_tensor = SimulationTensor.from_torch_geometric(data)

            assert sim_tensor.num_particles == 15
            assert sim_tensor.simulation_name == "test"
            assert sim_tensor.redshift == 0.5

        except ImportError:
            pytest.skip("PyTorch Geometric not available")

    def test_redshift_update(self):
        """Test updating redshift and derived quantities."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(data=positions, redshift=0.0)

        initial_age = sim_tensor.get_metadata("age_universe_gyr")

        sim_tensor.update_redshift(1.0)

        assert sim_tensor.redshift == 1.0
        assert sim_tensor.get_metadata("scale_factor") == 0.5
        new_age = sim_tensor.get_metadata("age_universe_gyr")
        assert new_age < initial_age  # Universe was younger at z=1

    def test_memory_info(self):
        """Test memory information."""
        positions = torch.randn(100, 3)
        sim_tensor = SimulationTensor(data=positions)

        mem_info = sim_tensor.memory_info()

        assert "total_size_gb" in mem_info
        assert isinstance(mem_info["total_size_gb"], str)
        assert float(mem_info["total_size_gb"]) > 0
