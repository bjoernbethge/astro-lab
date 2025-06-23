"""
Tests for visualization features integrated into tensors.
"""

import pytest
import torch

from astro_lab.tensors import SimulationTensor


class TestVisualizationIntegration:
    """Test visualization features integrated into tensors."""

    def test_to_pyvista(self):
        """Test conversion to a PyVista PolyData object."""
        positions = torch.rand(100, 3) * 1000
        masses = torch.rand(100)
        # Combine positions and features for the 'data' argument
        data = torch.cat([positions, masses.unsqueeze(1)], dim=1)
        sim_tensor = SimulationTensor(data=data, feature_names=["mass"])
        
        poly_data = sim_tensor.to_pyvista(scalars="mass")
        assert poly_data is not None
        assert poly_data.n_points == 100
        assert "mass" in poly_data.point_data

    def test_to_blender(self):
        """Test conversion to Blender-compatible dictionary."""
        positions = torch.rand(50, 3)
        sim_tensor = SimulationTensor(data=positions)
        blender_data = sim_tensor.to_blender()
        assert isinstance(blender_data, dict)
        assert blender_data["name"] == "simulation"
        assert len(blender_data["positions"]) == 50
