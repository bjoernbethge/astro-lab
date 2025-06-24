"""
Visualization Module - Backend-specific visualization methods
===========================================================

Provides visualization methods for different backends:
- Open3D for large point clouds
- PyVista for interactive 3D visualization
- Blender for advanced rendering
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
import matplotlib.pyplot

from ..tensors.survey import SurveyTensor

logger = logging.getLogger(__name__)


class VisualizationModule:
    """
    Backend-specific visualization methods.
    """
    
    def __init__(self):
        self.plotter = None
    
    def plot_to_open3d(self, survey_tensor: SurveyTensor, plot_type: str = "scatter_3d", **config: Any) -> Any:
        """
        Visualize data using Open3D backend.
        
        Args:
            survey_tensor: SurveyTensor with data
            plot_type: Type of plot
            **config: Additional configuration
            
        Returns:
            Open3D PointCloud object
        """
        import open3d as o3d
        
        logger.info("Visualizing with Open3D...")
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords_3d = spatial_tensor.data.cpu().numpy()
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords_3d)
        
        # Add colors based on photometric data
        try:
            phot_tensor = survey_tensor.get_photometric_tensor()
            colors_raw = phot_tensor.data[:, 0].cpu().numpy()
            
            if np.ptp(colors_raw) > 0:
                colors_norm = (colors_raw - np.min(colors_raw)) / np.ptp(colors_raw)
            else:
                colors_norm = np.ones_like(colors_raw)
            
            cmap = matplotlib.pyplot.get_cmap(config.get("cmap", "plasma"))
            colors_rgb = cmap(colors_norm)[:, :3]
            pcd.colors = o3d.utility.Vector3dVector(colors_rgb)
            
        except Exception as e:
            logger.warning(f"Could not get photometric data for coloring: {e}. Using default color.")
        
        # Show visualization
        o3d.visualization.draw_geometries([pcd], window_name=f"AstroLab - {survey_tensor.survey_name}")
        return pcd
    
    def plot_to_pyvista(self, survey_tensor: SurveyTensor, plot_type: str = "scatter_3d", **config: Any) -> Any:
        """
        Visualize data using PyVista backend.
        
        Args:
            survey_tensor: SurveyTensor with data
            plot_type: Type of plot
            **config: Additional configuration
            
        Returns:
            PyVista Plotter object
        """
        import pyvista as pv
        
        logger.info("Visualizing with PyVista...")
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords_3d = spatial_tensor.data.cpu().numpy()
        
        point_cloud = pv.PolyData(coords_3d)
        
        # Add photometric data as scalars
        try:
            phot_tensor = survey_tensor.get_photometric_tensor()
            for i, band in enumerate(phot_tensor.bands):
                point_cloud[band] = phot_tensor.data[:, i].cpu().numpy()
            
            scalar_to_show = phot_tensor.bands[0]
        except Exception as e:
            logger.warning(f"Could not get photometric data for coloring: {e}")
            scalar_to_show = None
        
        # Create plotter
        self.plotter = pv.Plotter(window_size=config.get("window_size", [1024, 768]))
        self.plotter.set_background("black")
        self.plotter.add_mesh(
            point_cloud,
            scalars=scalar_to_show,
            cmap=config.get("cmap", "viridis"),
            render_points_as_spheres=True,
            point_size=config.get("point_size", 5.0),
        )
        
        return self.plotter
    
    def plot_to_blender(self, survey_tensor: SurveyTensor, plot_type: str = "scatter_3d", **config: Any) -> Any:
        """
        Visualize data using Blender backend.
        
        Args:
            survey_tensor: SurveyTensor with data
            plot_type: Type of plot
            **config: Additional configuration
            
        Returns:
            Blender object or coordinates
        """
        logger.info("Visualizing with Blender...")
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords_3d = spatial_tensor.data.cpu().numpy()
        
        # This would integrate with the Blender API
        # For now, just return coordinates
        logger.info(f"Would send {coords_3d.shape[0]} points to Blender.")
        return coords_3d
    
    def show(self, *args, **kwargs: Any):
        """
        Show the last created interactive visualization.
        """
        if self.plotter is not None:
            self.plotter.show(*args, **kwargs)
        else:
            logger.warning("No plotter available to show.")
    
    def select_backend(self, survey_tensor: SurveyTensor) -> str:
        """
        Select the optimal visualization backend based on data size.
        
        Args:
            survey_tensor: SurveyTensor with data
            
        Returns:
            Backend name ('open3d', 'pyvista', 'blender')
        """
        num_points = len(survey_tensor.data)
        
        if num_points > 50_000:
            return "open3d"
        else:
            return "pyvista" 