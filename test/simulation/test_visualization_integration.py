"""
Tests for visualization features integrated into tensors.
"""

import pytest
import torch

from astro_lab.tensors import SimulationTensor


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
            assert "scalars" in mesh.point_data

        except ImportError:
            pytest.skip("PyVista not available")

    def test_to_blender(self):
        """Test Blender conversion (mock test)."""
        positions = torch.randn(10, 3)
        sim_tensor = SimulationTensor(positions)

        # This will return None without Blender, but shouldn't crash
        result = sim_tensor.to_blender(name="test_sim")
        # Just check it doesn't crash - actual functionality needs Blender
