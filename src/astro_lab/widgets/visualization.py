"""
Visualization Module - Multi-backend astronomical data visualization
==================================================================

Provides visualization capabilities using multiple backends:
- Open3D: Interactive 3D visualization
- PyVista: Scientific visualization
- Blender: Advanced 3D rendering and animation
"""

import logging
from typing import Any, Optional

import numpy as np
import torch

from ..tensors import SurveyTensorDict

logger = logging.getLogger(__name__)


class VisualizationModule:
    """
    Multi-backend visualization for astronomical data.

    Supports Open3D, PyVista, and Blender backends for interactive
    visualization of astronomical datasets.
    """

    def __init__(self) -> None:
        """Initialize the visualization module."""
        self.plotter = None

    def plot_to_open3d(
        self,
        survey_tensor: SurveyTensorDict,
        plot_type: str = "scatter_3d",
        **config: Any,
    ) -> Any:
        """
        Visualize data using Open3D backend.

        Args:
            survey_tensor: SurveyTensorDict with spatial and photometric data
            plot_type: Type of plot ('scatter_3d', 'point_cloud')
            **config: Additional Open3D-specific configuration

        Returns:
            Open3D visualization object
        """
        try:
            import open3d as o3d

            logger.info("Visualizing with Open3D...")

            # Extract coordinates
            spatial_tensor = survey_tensor.get_spatial_tensor()
            coords_3d = spatial_tensor.data.numpy()

            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords_3d)

            # Add colors if photometric data available
            try:
                phot_tensor = survey_tensor.get_photometric_tensor()
                colors = self._create_colors_from_photometry(phot_tensor)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            except Exception as e:
                logger.warning(
                    f"Could not add photometric colors: {e}. Using default colors."
                )
                # Use default colors based on position
                colors = self._create_position_colors(coords_3d)
                pcd.colors = o3d.utility.Vector3dVector(colors)

            return pcd

        except ImportError:
            logger.warning("Open3D not available. Install with: pip install open3d")
            return None

    def plot_to_pyvista(
        self,
        survey_tensor: SurveyTensorDict,
        plot_type: str = "scatter_3d",
        **config: Any,
    ) -> Any:
        """
        Visualize data using PyVista backend.

        Args:
            survey_tensor: SurveyTensorDict with spatial and photometric data
            plot_type: Type of plot ('scatter_3d', 'point_cloud')
            **config: Additional PyVista-specific configuration

        Returns:
            PyVista plotter object
        """
        try:
            import pyvista as pv

            logger.info("Visualizing with PyVista...")

            # Extract coordinates
            spatial_tensor = survey_tensor.get_spatial_tensor()
            coords_3d = spatial_tensor.data.numpy()

            # Create point cloud
            cloud = pv.PolyData(coords_3d)

            # Add colors if photometric data available
            try:
                phot_tensor = survey_tensor.get_photometric_tensor()
                colors = self._create_colors_from_photometry(phot_tensor)
                cloud.point_data["colors"] = colors
            except Exception as e:
                logger.warning(f"Could not get photometric data for coloring: {e}")
                # Use default colors based on position
                colors = self._create_position_colors(coords_3d)
                cloud.point_data["colors"] = colors

            # Create plotter
            plotter = pv.Plotter()
            plotter.add_points(cloud, scalars="colors", rgb=True, point_size=5)

            self.plotter = plotter
            return plotter

        except ImportError:
            logger.warning("PyVista not available. Install with: pip install pyvista")
            return None

    def plot_to_blender(
        self,
        survey_tensor: SurveyTensorDict,
        plot_type: str = "scatter_3d",
        **config: Any,
    ) -> Any:
        """
        Visualize data using Blender backend.

        Args:
            survey_tensor: SurveyTensorDict with spatial and photometric data
            plot_type: Type of plot ('scatter_3d', 'point_cloud')
            **config: Additional Blender-specific configuration

        Returns:
            Blender scene object or None if Blender not available
        """
        try:
            from ..utils.bpy import AstroLabApi

            logger.info("Visualizing with Blender...")

            # Extract coordinates
            spatial_tensor = survey_tensor.get_spatial_tensor()
            coords_3d = spatial_tensor.data.numpy()

            logger.info(f"Would send {coords_3d.shape[0]} points to Blender.")

            # Create Blender API instance
            api = AstroLabApi()

            # Use the advanced visualization suite to create a scene
            scene = api.advanced.create_galaxy_showcase()

            # For now, just return the scene since adding individual points
            # requires more complex integration with the 3D plotter
            return scene

        except ImportError:
            logger.warning("Blender not available. Install Blender and astro_lab[bpy]")
            return None

    def show(self, *args: Any, **kwargs: Any) -> None:
        """
        Show the last created interactive visualization.

        Args:
            *args: Positional arguments passed to visualization backend
            **kwargs: Keyword arguments passed to visualization backend
        """
        if self.plotter is not None:
            self.plotter.show(*args, **kwargs)
        else:
            logger.warning("No plotter available to show.")

    def select_backend(self, survey_tensor: SurveyTensorDict) -> str:
        """
        Automatically select the best visualization backend.

        Args:
            survey_tensor: SurveyTensorDict with data

        Returns:
            Recommended backend name ('open3d', 'pyvista', 'blender')
        """
        n_points = len(survey_tensor.data)

        # Simple heuristic based on data size
        if n_points < 10000:
            return "open3d"  # Good for small datasets
        elif n_points < 100000:
            return "pyvista"  # Good for medium datasets
        else:
            return "blender"  # Good for large datasets

    def _create_colors_from_photometry(self, phot_tensor: Any) -> np.ndarray:
        """
        Create colors from photometric data.

        Args:
            phot_tensor: PhotometricTensorDict with magnitude data

        Returns:
            RGB colors array
        """
        # Simple color mapping based on magnitude
        magnitudes = phot_tensor.data.mean(dim=1).numpy()

        # Normalize to [0, 1]
        mag_norm = (magnitudes - magnitudes.min()) / (
            magnitudes.max() - magnitudes.min()
        )

        # Create RGB colors (blue to red)
        colors = np.zeros((len(magnitudes), 3))
        colors[:, 0] = mag_norm  # Red
        colors[:, 2] = 1 - mag_norm  # Blue

        return colors

    def _create_position_colors(self, coords: np.ndarray) -> np.ndarray:
        """
        Create colors based on spatial position.

        Args:
            coords: 3D coordinates array

        Returns:
            RGB colors array
        """
        # Normalize coordinates to [0, 1]
        coords_norm = (coords - coords.min(axis=0)) / (
            coords.max(axis=0) - coords.min(axis=0)
        )

        # Use coordinates as RGB
        colors = np.clip(coords_norm, 0, 1)

        return colors
