"""
Plotly Bridge for AstroLab visualization.
Provides web-based 3D visualization of astronomical data.
"""

import logging
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


def create_plotly_visualization(survey_tensor: Any, **config) -> Any:
    """
    Create Plotly visualization for SurveyTensorDict.

    Args:
        survey_tensor: SurveyTensorDict object
        **config: Configuration options

    Returns:
        Plotly figure object
    """
    # Extract spatial coordinates from SurveyTensorDict
    if hasattr(survey_tensor, "spatial") and "coordinates" in survey_tensor["spatial"]:
        coords = survey_tensor["spatial"]["coordinates"]
        logger.info(f"✅ Extracted 3D coordinates for Plotly: {coords.shape}")
    elif hasattr(survey_tensor, "get_spatial_tensor"):
        # Fallback for old API
        spatial_tensor = survey_tensor.get_spatial_tensor()
        if hasattr(spatial_tensor, "cartesian"):
            coords = spatial_tensor.cartesian
            logger.info(
                f"✅ Extracted 3D coordinates for Plotly (legacy): {coords.shape}"
            )
        else:
            raise ValueError("SurveyTensorDict has no cartesian coordinates")
    else:
        raise ValueError("SurveyTensorDict has no spatial data")

    # Limit points for web visualization
    max_points = config.get("max_points", 10000)
    if coords.shape[0] > max_points:
        logger.info(f"Sampling {max_points} points for Plotly visualization")
        indices = torch.randperm(coords.shape[0])[:max_points]
        coords = coords[indices]

    # Convert to numpy
    coords_np = coords.cpu().numpy()

    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords_np[:, 0],
                y=coords_np[:, 1],
                z=coords_np[:, 2],
                mode="markers",
                marker=dict(
                    size=config.get("point_size", 2),
                    opacity=config.get("opacity", 0.8),
                    color=coords_np[:, 2],  # Color by z-value
                    colorscale="Viridis",
                    showscale=True,
                ),
                name=config.get("name", "AstroLab Data"),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        title=config.get("title", "AstroLab 3D Visualization"),
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"
        ),
        width=config.get("width", 800),
        height=config.get("height", 600),
    )

    # Show if requested
    if config.get("show", True):
        fig.show()

    return fig


def plot_cosmic_web_3d(
    spatial_tensor: Any,
    cluster_labels: Optional[np.ndarray] = None,
    title: str = "Cosmic Web Structure",
    point_size: int = 3,
    show_clusters: bool = True,
    opacity: float = 0.8,
    width: int = 900,
    height: int = 700,
    **kwargs,
) -> go.Figure:
    """
    Create 3D visualization of cosmic web structure.
    
    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        cluster_labels: Cluster assignments (-1 for noise)
        title: Plot title
        point_size: Point size
        show_clusters: Color by clusters
        opacity: Point opacity
        width: Figure width
        height: Figure height
        
    Returns:
        Plotly figure
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"].cpu().numpy()
    elif hasattr(spatial_tensor, "data"):
        coords = spatial_tensor.data.cpu().numpy()
    else:
        coords = spatial_tensor.cpu().numpy()
    
    # Prepare data for plotting
    if cluster_labels is not None and show_clusters:
        # Color by cluster
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        
        traces = []
        
        # Create colormap
        import matplotlib.cm as cm
        colormap = cm.get_cmap('tab20')
        
        # Plot each cluster
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            cluster_coords = coords[mask]
            
            if label == -1:
                # Noise points
                color = 'lightgray'
                name = 'Isolated'
                opacity_cluster = opacity * 0.3
                size = point_size * 0.7
            else:
                # Cluster points
                color_rgb = colormap(i / max(n_clusters, 1))[:3]
                color = f'rgb({int(color_rgb[0]*255)},{int(color_rgb[1]*255)},{int(color_rgb[2]*255)})'
                name = f'Cluster {label}'
                opacity_cluster = opacity
                size = point_size
                
            trace = go.Scatter3d(
                x=cluster_coords[:, 0],
                y=cluster_coords[:, 1],
                z=cluster_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=opacity_cluster,
                    line=dict(width=0),
                ),
                name=name,
                text=[f'{name} Point {i}' for i in range(len(cluster_coords))],
                hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
            )
            traces.append(trace)
            
    else:
        # Single trace without clustering
        trace = go.Scatter3d(
            x=coords[:, 0],
            y=coords[:, 1],
            z=coords[:, 2],
            mode='markers',
            marker=dict(
                size=point_size,
                color=coords[:, 2],  # Color by z-coordinate
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title='Z Coordinate'),
            ),
            text=[f'Point {i}' for i in range(len(coords))],
            hovertemplate='Point %{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
        )
        traces = [trace]
        
    # Create figure
    fig = go.Figure(data=traces)
    
    # Update layout
    unit = kwargs.get('unit', 'pc')
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title=f'X ({unit})',
            yaxis_title=f'Y ({unit})',
            zaxis_title=f'Z ({unit})',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        width=width,
        height=height,
        showlegend=show_clusters,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template="plotly_dark" if kwargs.get('dark_mode', True) else "plotly_white",
    )
    
    return fig


def plot_density_heatmap(
    spatial_tensor: Any,
    density_counts: torch.Tensor,
    radius: float = 50.0,
    title: str = "Local Density Map",
    **kwargs,
) -> go.Figure:
    """
    Create density heatmap visualization.
    
    Args:
        spatial_tensor: Spatial coordinates
        density_counts: Number of neighbors for each point
        radius: Radius used for density calculation
        title: Plot title
        
    Returns:
        Plotly figure
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"].cpu().numpy()
    else:
        coords = spatial_tensor.cpu().numpy()
    
    density = density_counts.cpu().numpy()
    
    # Create figure
    fig = go.Figure()
    
    # 3D scatter with density coloring
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=density,
            colorscale='Hot',
            opacity=0.8,
            colorbar=dict(
                title=f'Neighbors<br>within {radius} pc',
                titleside='right',
            ),
        ),
        text=[f'Density: {d}' for d in density],
        hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}',
        name='Local Density',
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        scene=dict(
            xaxis_title='X (pc)',
            yaxis_title='Y (pc)',
            zaxis_title='Z (pc)',
            aspectmode='data',
        ),
        width=kwargs.get('width', 800),
        height=kwargs.get('height', 700),
        template="plotly_dark" if kwargs.get('dark_mode', True) else "plotly_white",
    )
    
    return fig


def plot_multi_scale_clustering(
    spatial_tensor: Any,
    clustering_results: Dict[str, Dict[str, Any]],
    title: str = "Multi-Scale Clustering Analysis",
    **kwargs,
) -> go.Figure:
    """
    Create subplot visualization comparing clustering at different scales.
    
    Args:
        spatial_tensor: Spatial coordinates
        clustering_results: Results from multi-scale clustering
        title: Main title
        
    Returns:
        Plotly figure with subplots
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"].cpu().numpy()
    else:
        coords = spatial_tensor.cpu().numpy()
    
    # Get scales
    scales = sorted(clustering_results.keys())
    n_scales = len(scales)
    
    # Determine subplot layout
    n_cols = min(n_scales, 3)
    n_rows = (n_scales + n_cols - 1) // n_cols
    
    # Create subplot titles
    subplot_titles = []
    for scale in scales:
        stats = clustering_results[scale]
        n_clusters = stats.get('n_clusters', 0)
        grouped_frac = stats.get('grouped_fraction', 0)
        subplot_titles.append(f"{scale}<br>{n_clusters} clusters ({grouped_frac:.0%} grouped)")
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles,
        specs=[[{'type': 'scatter3d'} for _ in range(n_cols)] for _ in range(n_rows)],
        vertical_spacing=0.1,
        horizontal_spacing=0.05,
    )
    
    # Plot each scale
    for idx, scale in enumerate(scales):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        results = clustering_results[scale]
        
        if 'labels' in results:
            labels = results['labels']
            if hasattr(labels, 'cpu'):
                labels = labels.cpu().numpy()
        else:
            continue
            
        # Color by cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])
        
        for label in unique_labels:
            mask = labels == label
            cluster_coords = coords[mask]
            
            if label == -1:
                color = 'lightgray'
                name = 'Isolated'
                size = 2
            else:
                # Use color from colorscale
                color_idx = label % 10  # Cycle through 10 colors
                color = f'hsl({color_idx * 36}, 70%, 50%)'
                name = f'C{label}'
                size = 3
                
            fig.add_trace(
                go.Scatter3d(
                    x=cluster_coords[:, 0],
                    y=cluster_coords[:, 1],
                    z=cluster_coords[:, 2],
                    mode='markers',
                    marker=dict(size=size, color=color),
                    name=name,
                    showlegend=False,
                ),
                row=row, col=col
            )
    
    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor='center'),
        height=400 * n_rows,
        width=400 * n_cols,
        showlegend=False,
        template="plotly_dark" if kwargs.get('dark_mode', True) else "plotly_white",
    )
    
    # Update all 3D axes
    for i in range(n_scales):
        row = i // n_cols + 1
        col = i % n_cols + 1
        fig.update_scenes(
            dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='data',
            ),
            row=row, col=col
        )
    
    return fig
