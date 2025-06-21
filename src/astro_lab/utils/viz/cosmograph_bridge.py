"""
CosmographBridge - Simple integration of Cosmograph with AstroLab tensors.

Provides a clean interface to create interactive graph visualizations
from AstroLab spatial tensors and survey data using Polars for data handling.
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import polars as pl

# Suppress NumPy warnings for Cosmograph compatibility
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="A module that was compiled using NumPy 1.x")
    from cosmograph import cosmo


class CosmographBridge:
    """
    Bridge class to create Cosmograph visualizations from AstroLab data.
    
    Provides simple methods to convert spatial tensors and survey data
    into interactive graph visualizations using Cosmograph.
    Uses Polars for efficient data handling and converts to pandas only when needed.
    """
    
    def __init__(self):
        """Initialize the Cosmograph bridge."""
        self.default_config = {
            'background_color': '#000011',
            'simulation_gravity': 0.1,
            'simulation_repulsion': 0.2,
            'show_labels': True,
            'show_top_labels_limit': 10,
            'curved_links': True,
            'curved_link_weight': 0.3
        }
    
    def from_spatial_tensor(self, 
                          tensor, 
                          radius: float = 5.0,
                          point_color: str = '#ffd700',
                          link_color: str = '#666666',
                          **kwargs) -> Any:
        """
        Create Cosmograph visualization from Spatial3DTensor.
        
        Args:
            tensor: AstroLab Spatial3DTensor
            radius: Radius for neighbor graph creation (default: 5.0)
            point_color: Color for points (default: gold)
            link_color: Color for links (default: gray)
            **kwargs: Additional Cosmograph parameters
            
        Returns:
            Cosmograph widget
        """
        # Extract coordinates
        coords = tensor.cartesian.cpu().numpy()
        
        # Create neighbor graph
        edges = tensor._create_radius_graph(tensor.cartesian, radius=radius).t().cpu().numpy()
        
        # Create Polars DataFrames
        points_df = pl.DataFrame({
            'id': [f'node_{i}' for i in range(len(coords))],
            'x': coords[:, 0],
            'y': coords[:, 1],
            'z': coords[:, 2],
            'distance': np.linalg.norm(coords, axis=1)
        })
        
        links_df = pl.DataFrame({
            'source': [f'node_{edges[i, 0]}' for i in range(len(edges))],
            'target': [f'node_{edges[i, 1]}' for i in range(len(edges))],
            'distance': np.random.uniform(1.0, radius, len(edges))
        })
        
        # Convert to pandas for Cosmograph (required by cosmograph)
        points_pandas = points_df.to_pandas()
        links_pandas = links_df.to_pandas()
        
        # Merge configs
        config = {**self.default_config, **kwargs}
        
        # Remove point_color from config to avoid duplicate parameter
        config.pop('point_color', None)
        
        # Use link_width_by and link_width_scale instead of link_width_range
        return cosmo(
            points=points_pandas,
            links=links_pandas,
            point_id_by='id',
            point_x_by='x',
            point_y_by='y',
            point_color=point_color,
            point_size_range=[2, 6],
            link_source_by='source',
            link_target_by='target',
            link_color=link_color,
            link_width_by='distance',
            link_width_scale=1.0,
            **config
        )
    
    def from_cosmic_web_results(self, 
                              results: Dict[str, Any],
                              survey_name: str,
                              radius: float = 5.0,
                              **kwargs) -> Any:
        """
        Create Cosmograph visualization from cosmic web analysis results.
        
        Args:
            results: Results from create_cosmic_web_loader
            survey_name: Name of survey (gaia, sdss, nsa, tng50, etc.)
            radius: Radius for neighbor graph creation
            **kwargs: Additional Cosmograph parameters
            
        Returns:
            Cosmograph widget
        """
        # Extract coordinates from cosmic web results
        if 'coordinates' in results:
            coords = np.array(results['coordinates'])
        elif 'spatial_tensor' in results:
            # If spatial_tensor is available, use it
            tensor = results['spatial_tensor']
            coords = tensor.cartesian.cpu().numpy()
        else:
            raise ValueError("No coordinates found in cosmic web results")
        
        # Set colors based on survey type
        color_map = {
            'gaia': '#ffd700',      # Gold for stars
            'sdss': '#4a90e2',      # Blue for galaxies
            'nsa': '#e24a4a',       # Red for NSA
            'tng50': '#00ff00',     # Green for simulation
            'tng': '#00ff00',       # Green for simulation
            'linear': '#ff8800',    # Orange for asteroids
            'exoplanet': '#ff00ff'  # Magenta for exoplanets
        }
        
        point_color = color_map.get(survey_name, '#ffffff')
        
        return self.from_coordinates(
            coords,
            radius=radius,
            point_color=point_color,
            **kwargs
        )
    
    def from_survey_data(self, 
                        data: Dict[str, Any],
                        survey_name: str,
                        radius: float = 5.0,
                        **kwargs) -> Any:
        """
        Create Cosmograph visualization from survey data.
        
        Args:
            data: Survey data dictionary with 'spatial_tensor' key
            survey_name: Name of survey (gaia, sdss, nsa, tng50)
            radius: Radius for neighbor graph creation
            **kwargs: Additional Cosmograph parameters
            
        Returns:
            Cosmograph widget
        """
        # Check if this is cosmic web results
        if 'coordinates' in data or 'spatial_tensor' in data:
            return self.from_cosmic_web_results(data, survey_name, radius, **kwargs)
        
        # Fallback to old method
        if 'spatial_tensor' not in data:
            raise ValueError("Survey data must contain 'spatial_tensor' key")
        
        # Set colors based on survey type
        color_map = {
            'gaia': '#ffd700',      # Gold for stars
            'sdss': '#4a90e2',      # Blue for galaxies
            'nsa': '#e24a4a',       # Red for NSA
            'tng50': '#00ff00'      # Green for simulation
        }
        
        point_color = color_map.get(survey_name, '#ffffff')
        
        return self.from_spatial_tensor(
            data['spatial_tensor'],
            radius=radius,
            point_color=point_color,
            **kwargs
        )
    
    def from_coordinates(self, 
                        coords: np.ndarray,
                        edges: Optional[np.ndarray] = None,
                        radius: float = 5.0,
                        **kwargs) -> Any:
        """
        Create Cosmograph visualization from raw coordinates.
        
        Args:
            coords: Nx3 array of 3D coordinates
            edges: Optional Nx2 array of edge indices
            radius: Radius for neighbor graph if edges not provided
            **kwargs: Additional Cosmograph parameters
            
        Returns:
            Cosmograph widget
        """
        if edges is None:
            # Create simple neighbor graph using distance
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            
            edge_list = []
            for i in range(len(coords)):
                for j in range(1, len(indices[i])):
                    if distances[i][j] <= radius:
                        edge_list.append([i, indices[i][j]])
            edges = np.array(edge_list, dtype=int)
        
        # Extract coordinates explicitly
        x_coords = coords[:, 0].tolist()
        y_coords = coords[:, 1].tolist()
        z_coords = coords[:, 2].tolist()
        
        # Create Polars DataFrames
        points_df = pl.DataFrame({
            'id': [f'point_{i}' for i in range(len(coords))],
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        })
        
        if edges.size > 0:
            sources = [f'point_{src}' for src in edges[:, 0]]
            targets = [f'point_{tgt}' for tgt in edges[:, 1]]
        else:
            sources = []
            targets = []
        links_df = pl.DataFrame({
            'source': sources,
            'target': targets
        })
        
        # Convert to pandas for Cosmograph (required by cosmograph)
        points_pandas = points_df.to_pandas()
        links_pandas = links_df.to_pandas()
        
        # Merge configs
        config = {**self.default_config, **kwargs}
        
        # Extract point_color from kwargs or use default
        point_color = kwargs.get('point_color', '#ffffff')
        link_color = kwargs.get('link_color', '#666666')
        
        # Remove these from config to avoid duplicate parameters
        config.pop('point_color', None)
        config.pop('link_color', None)
        
        return cosmo(
            points=points_pandas,
            links=links_pandas,
            point_id_by='id',
            point_x_by='x',
            point_y_by='y',
            point_color=point_color,
            point_size_range=[2, 6],
            link_source_by='source',
            link_target_by='target',
            link_color=link_color,
            link_width_scale=1.0,
            **config
        )

    def from_polars_dataframe(self,
                             df: pl.DataFrame,
                             x_col: str = 'x',
                             y_col: str = 'y',
                             z_col: str = 'z',
                             id_col: Optional[str] = None,
                             radius: float = 5.0,
                             **kwargs) -> Any:
        """
        Create Cosmograph visualization directly from Polars DataFrame.
        
        Args:
            df: Polars DataFrame with coordinate columns
            x_col: Column name for x coordinates
            y_col: Column name for y coordinates
            z_col: Column name for z coordinates
            id_col: Column name for point IDs (optional)
            radius: Radius for neighbor graph creation
            **kwargs: Additional Cosmograph parameters
            
        Returns:
            Cosmograph widget
        """
        # Extract coordinates
        coords = df.select([x_col, y_col, z_col]).to_numpy()
        
        # Create IDs if not provided
        if id_col is None or id_col not in df.columns:
            ids = [f'point_{i}' for i in range(len(df))]
        else:
            ids = df[id_col].to_list()
        
        # Create neighbor graph
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        edge_list = []
        for i in range(len(coords)):
            for j in range(1, len(indices[i])):
                if distances[i][j] <= radius:
                    edge_list.append([i, indices[i][j]])
        edges = np.array(edge_list, dtype=int)
        
        # Extract coordinates explicitly
        x_coords = coords[:, 0].tolist()
        y_coords = coords[:, 1].tolist()
        z_coords = coords[:, 2].tolist()
        
        # Create Polars DataFrames
        points_df = pl.DataFrame({
            'id': ids,
            'x': x_coords,
            'y': y_coords,
            'z': z_coords
        })
        
        if edges.size > 0:
            sources = [ids[src] for src in edges[:, 0]]
            targets = [ids[tgt] for tgt in edges[:, 1]]
        else:
            sources = []
            targets = []
        links_df = pl.DataFrame({
            'source': sources,
            'target': targets
        })
        
        # Convert to pandas for Cosmograph
        points_pandas = points_df.to_pandas()
        links_pandas = links_df.to_pandas()
        
        # Merge configs
        config = {**self.default_config, **kwargs}
        
        # Extract colors from kwargs or use defaults
        point_color = kwargs.get('point_color', '#ffffff')
        link_color = kwargs.get('link_color', '#666666')
        
        # Remove these from config to avoid duplicate parameters
        config.pop('point_color', None)
        config.pop('link_color', None)
        
        return cosmo(
            points=points_pandas,
            links=links_pandas,
            point_id_by='id',
            point_x_by='x',
            point_y_by='y',
            point_color=point_color,
            point_size_range=[2, 6],
            link_source_by='source',
            link_target_by='target',
            link_color=link_color,
            link_width_scale=1.0,
            **config
        )


# Convenience function
def create_cosmograph_visualization(data_source, **kwargs):
    """
    Convenience function to create Cosmograph visualization.
    
    Args:
        data_source: Spatial3DTensor, survey data dict, cosmic web results, coordinates array, or Polars DataFrame
        **kwargs: Additional parameters for CosmographBridge
        
    Returns:
        Cosmograph widget
    """
    bridge = CosmographBridge()
    
    if hasattr(data_source, 'cartesian'):
        # Spatial3DTensor
        return bridge.from_spatial_tensor(data_source, **kwargs)
    elif isinstance(data_source, dict):
        # Check if it's cosmic web results
        if 'coordinates' in data_source or 'spatial_tensor' in data_source:
            survey_name = kwargs.pop('survey_name', 'unknown')
            return bridge.from_cosmic_web_results(data_source, survey_name, **kwargs)
        else:
            raise ValueError("Dictionary must contain 'coordinates' or 'spatial_tensor' key")
    elif isinstance(data_source, np.ndarray):
        # Raw coordinates
        return bridge.from_coordinates(data_source, **kwargs)
    elif isinstance(data_source, pl.DataFrame):
        # Polars DataFrame
        return bridge.from_polars_dataframe(data_source, **kwargs)
    else:
        raise ValueError("Unsupported data source type") 