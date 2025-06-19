"""
Test Simulation Tensors - Tensor-Based Architecture
==================================================

Tests the new tensor-based simulation architecture:
- SimulationTensor for cosmological simulation data
- Integrated CosmologyCalculator 
- Visualization integration (PyVista/Blender)
- Memory-efficient operations

All simulation functionality is now tensor-based, eliminating
the need for separate utility modules.
"""

import pytest
import numpy as np
import torch

from astro_lab.tensors import SimulationTensor, CosmologyCalculator


class TestSimulationTensor:
    """Test SimulationTensor - the new tensor-based simulation architecture."""

    def test_init_basic(self):
        """Test basic SimulationTensor initialization."""
        positions = torch.randn(100, 3)
        sim_tensor = SimulationTensor(positions)
        
        assert sim_tensor.num_particles == 100
        assert sim_tensor.simulation_name == "TNG50"
        assert sim_tensor.particle_type == "gas"
        assert sim_tensor.box_size == 35000.0
        assert sim_tensor.redshift == 0.0

    def test_init_with_features(self):
        """Test initialization with particle features."""
        positions = torch.randn(100, 3)
        features = torch.randn(100, 2)  # mass, potential
        
        sim_tensor = SimulationTensor(
            positions=positions,
            features=features,
            particle_type="stars",
            redshift=1.0
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
            positions=positions,
            edge_index=edge_index,
            simulation_name="Illustris"
        )
        
        assert sim_tensor.edge_index is not None
        assert sim_tensor.edge_index.shape == (2, 200)
        assert sim_tensor.simulation_name == "Illustris"

    def test_cosmology_integration(self):
        """Test integrated cosmology calculations."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(positions, redshift=1.0)
        
        # Check cosmology calculator
        cosmo = sim_tensor.cosmology
        assert isinstance(cosmo, CosmologyCalculator)
        
        # Check derived cosmological quantities
        assert sim_tensor.get_metadata("scale_factor") == 0.5  # 1/(1+z)
        assert sim_tensor.get_metadata("age_universe_gyr") > 0
        assert sim_tensor.get_metadata("hubble_param") > 0

    def test_periodic_distance(self):
        """Test periodic boundary distance calculations."""
        positions = torch.tensor([[1.0, 1.0, 1.0], [34999.0, 34999.0, 34999.0]])
        sim_tensor = SimulationTensor(positions, box_size=35000.0)
        
        # Should wrap around - distance ≈ 2*sqrt(3)
        dist = sim_tensor.periodic_distance(0, 1)
        expected = 2 * np.sqrt(3)
        assert abs(dist.item() - expected) < 0.1

    def test_periodic_boundaries_application(self):
        """Test applying periodic boundary conditions."""
        positions = torch.tensor([[36000.0, -1000.0, 17500.0]])
        sim_tensor = SimulationTensor(positions, box_size=35000.0)
        
        wrapped = sim_tensor.apply_periodic_boundaries(positions)
        assert 0 <= wrapped[0, 0] < 35000.0  # Wrapped x
        assert 0 <= wrapped[0, 1] < 35000.0  # Wrapped y

    def test_particle_subset(self):
        """Test getting particle subsets."""
        positions = torch.randn(100, 3)
        features = torch.randn(100, 2)
        sim_tensor = SimulationTensor(positions, features=features)
        
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
        
        sim_tensor = SimulationTensor(positions, features=masses)
        com = sim_tensor.calculate_center_of_mass()
        
        # Should be at [1.0, 0.0, 0.0]
        expected = torch.tensor([1.0, 0.0, 0.0])
        assert torch.allclose(com, expected)

    def test_torch_geometric_conversion(self):
        """Test conversion to PyTorch Geometric format."""
        positions = torch.randn(20, 3)
        features = torch.randn(20, 2)
        edge_index = torch.randint(0, 20, (2, 50))
        
        sim_tensor = SimulationTensor(
            positions=positions,
            features=features,
            edge_index=edge_index
        )
        
        data = sim_tensor.to_torch_geometric()
        
        assert hasattr(data, 'pos')
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
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
                edge_index=torch.randint(0, 15, (2, 30))
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
        sim_tensor = SimulationTensor(positions, redshift=0.0)
        
        initial_age = sim_tensor.get_metadata("age_universe_gyr")
        
        sim_tensor.update_redshift(1.0)
        
        assert sim_tensor.redshift == 1.0
        assert sim_tensor.get_metadata("scale_factor") == 0.5
        new_age = sim_tensor.get_metadata("age_universe_gyr")
        assert new_age < initial_age  # Universe was younger at z=1


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
        # Note: this returns numpy, but should work
        assert len(d_tensor) == 2
        assert abs(d_tensor[0]) < 1e-6  # z=0 case
        assert d_tensor[1] > 0  # z=1 case

    def test_angular_diameter_distance(self):
        """Test angular diameter distance."""
        calc = CosmologyCalculator()
        z = 1.0
        
        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)
        
        expected = d_c / (1 + z)
        assert abs(d_a - expected) < 1e-6

    def test_age_of_universe(self):
        """Test age calculation."""
        calc = CosmologyCalculator()
        
        age_z0 = calc.age_of_universe(0.0)
        age_z1 = calc.age_of_universe(1.0)
        
        assert age_z0 > age_z1  # Universe was younger at higher z
        assert age_z0 > 0.5  # Reasonable age range
        assert age_z1 > 0.0


class TestVisualizationIntegration:
    """Test visualization features integrated into tensors."""

    def test_to_pyvista(self):
        """Test PyVista conversion."""
        try:
            import pyvista as pv
            
            positions = torch.randn(50, 3)
            masses = torch.randn(50)
            
            sim_tensor = SimulationTensor(positions, features=masses.unsqueeze(1))
            mesh = sim_tensor.to_pyvista(scalars=masses)
            
            assert isinstance(mesh, pv.PolyData)
            assert mesh.n_points == 50
            assert 'scalars' in mesh.point_data
            
        except ImportError:
            pytest.skip("PyVista not available")

    def test_to_blender(self):
        """Test Blender conversion (mock test)."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(positions)
        
        # This will return None without Blender, but shouldn't crash
        result = sim_tensor.to_blender(name="test_sim")
        # Just check it doesn't crash - actual functionality needs Blender

    def test_memory_info(self):
        """Test memory information."""
        positions = torch.randn(100, 3)
        sim_tensor = SimulationTensor(positions)
        
        mem_info = sim_tensor.memory_info()
        
        assert 'device' in mem_info
        assert 'dtype' in mem_info
        assert 'shape' in mem_info
        assert 'data_ptr' in mem_info
        assert mem_info['shape'] == [100, 3]


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
            box_size=35000.0
        )
        
        # Test cosmological calculations
        age = sim_tensor.cosmology.age_of_universe(sim_tensor.redshift)
        assert age > 10.0  # Reasonable age
        
        # Test center of mass
        com = sim_tensor.calculate_center_of_mass()
        assert com.shape == (3,)
        
        # Test subset selection (high mass particles)
        high_mass_mask = features[:, 0] > features[:, 0].median()
        subset = sim_tensor.get_particle_subset(high_mass_mask)
        assert subset.num_particles < 1000

    def test_cosmology_consistency(self):
        """Test cosmology calculations are consistent."""
        calc = CosmologyCalculator(H0=67.4, Omega_m=0.315, Omega_Lambda=0.685)
        
        z = 1.0
        d_c = calc.comoving_distance(z)
        d_a = calc.angular_diameter_distance(z)
        d_l = calc.luminosity_distance(z)
        
        # Check relationships
        assert abs(d_a - d_c / (1 + z)) < 1e-6
        assert abs(d_l - d_c * (1 + z)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__]) 