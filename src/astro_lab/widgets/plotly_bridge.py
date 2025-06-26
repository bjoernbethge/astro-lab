"""
Plotly Bridge for AstroLab visualization.
Provides web-based 3D visualization of astronomical data.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import plotly.graph_objects as go
import torch

logger = logging.getLogger(__name__)


def create_plotly_visualization(survey_tensor: Any, **config) -> Any:
    """
    Create Plotly visualization for SurveyTensor.

    Args:
        survey_tensor: SurveyTensor object
        **config: Configuration options

    Returns:
        Plotly figure object
    """
    # Extract spatial coordinates
    if hasattr(survey_tensor, "get_spatial_tensor"):
        spatial_tensor = survey_tensor.get_spatial_tensor()
        if hasattr(spatial_tensor, "cartesian"):
            coords = spatial_tensor.cartesian
            logger.info(f"✅ Extracted 3D coordinates for Plotly: {coords.shape}")
        else:
            raise ValueError("SurveyTensor has no cartesian coordinates")
    else:
        raise ValueError("SurveyTensor has no spatial tensor")

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
