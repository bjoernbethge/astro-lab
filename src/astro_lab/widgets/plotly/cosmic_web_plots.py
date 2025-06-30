"""
Cosmic Web Plotly visualizations for astro-lab.

Provides specialized cosmic web plots with proper astronomical context.
"""

import logging
from typing import Any, Dict, Optional

# Astropy imports for astronomical correctness
import numpy as np
import plotly.graph_objects as go
import torch
from astropy.visualization import quantity_support
from plotly.subplots import make_subplots

# Enable quantity support
quantity_support()

logger = logging.getLogger(__name__)


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
    Create 3D cosmic web visualization with Plotly.

    Args:
        spatial_tensor: Spatial coordinates tensor
        cluster_labels: Cluster assignments
        title: Plot title
        point_size: Point size
        show_clusters: Whether to show cluster colors
        opacity: Point opacity
        width: Plot width
        height: Plot height
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"]
    elif hasattr(spatial_tensor, "data"):
        coords = spatial_tensor.data
    else:
        coords = spatial_tensor

    # Convert to numpy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.array(coords)

    # Ensure 3D coordinates
    if coords_np.shape[1] == 2:
        # 2D coordinates - add z=0
        x, y = coords_np[:, 0], coords_np[:, 1]
        z = np.zeros_like(x)
    elif coords_np.shape[1] >= 3:
        x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]
    else:
        raise ValueError(f"Unsupported coordinate dimensions: {coords_np.shape[1]}")

    # Create 3D scatter plot
    fig = go.Figure()

    if show_clusters and cluster_labels is not None:
        # Color by cluster
        unique_labels = np.unique(cluster_labels)
        colors = np.zeros((len(cluster_labels), 3))

        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            # Use different colors for different clusters
            hue = i / len(unique_labels)
            # HSV to RGB conversion
            colors[mask] = [hue, 0.8, 0.8]

        # Convert to RGB strings
        color_strings = [
            f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})"
            for c in colors
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color=color_strings,
                    opacity=opacity,
                ),
                text=[
                    f"Object {i}<br>Cluster: {cluster_labels[i]}<br>X: {x[i]:.1f}"
                    f"<br>Y: {y[i]:.1f}<br>Z: {z[i]:.1f}"
                    for i in range(len(x))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Cosmic Web Structures",
            )
        )
    else:
        # scatter
        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=dict(
                    size=point_size,
                    color="blue",
                    opacity=opacity,
                ),
                text=[
                    f"Object {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}<br>Z: {z[i]:.1f}"
                    for i in range(len(x))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Cosmic Web Points",
            )
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="X [Mpc]",
            yaxis_title="Y [Mpc]",
            zaxis_title="Z [Mpc]",
        ),
        width=width,
        height=height,
        template="plotly_white",
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
        spatial_tensor: Spatial coordinates tensor
        density_counts: Density field tensor
        radius: Search radius
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"]
    elif hasattr(spatial_tensor, "data"):
        coords = spatial_tensor.data
    else:
        coords = spatial_tensor

    # Convert to numpy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.array(coords)

    if isinstance(density_counts, torch.Tensor):
        density_np = density_counts.cpu().numpy()
    else:
        density_np = np.array(density_counts)

    # Project to 2D if 3D
    if coords_np.shape[1] >= 3:
        x, y = coords_np[:, 0], coords_np[:, 1]
    else:
        x, y = coords_np[:, 0], coords_np[:, 1]

    # Create heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=kwargs.get("point_size", 5),
                color=density_np,
                colorscale="Viridis",
                opacity=kwargs.get("opacity", 0.7),
                showscale=True,
                colorbar=dict(title="Density Count"),
            ),
            text=[
                f"Object {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}"
                f"<br>Density: {density_np[i]:.1f}"
                for i in range(len(x))
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Density Map",
        )
    )

    # Update layout
    fig.update_layout(
        title=dict(
            text=f"{title} (Radius: {radius:.1f} Mpc)",
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        xaxis_title="X [Mpc]",
        yaxis_title="Y [Mpc]",
        width=kwargs.get("width", 900),
        height=kwargs.get("height", 700),
        template="plotly_white",
    )

    return fig


def plot_multi_scale_clustering(
    spatial_tensor: Any,
    clustering_results: Dict[str, Dict[str, Any]],
    title: str = "Multi-Scale Clustering Analysis",
    **kwargs,
) -> go.Figure:
    """
    Create multi-scale clustering visualization.

    Args:
        spatial_tensor: Spatial coordinates tensor
        clustering_results: Dictionary of clustering results
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract coordinates
    if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
        coords = spatial_tensor["coordinates"]
    elif hasattr(spatial_tensor, "data"):
        coords = spatial_tensor.data
    else:
        coords = spatial_tensor

    # Convert to numpy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.array(coords)

    # Project to 2D if 3D
    if coords_np.shape[1] >= 3:
        x, y = coords_np[:, 0], coords_np[:, 1]
    else:
        x, y = coords_np[:, 0], coords_np[:, 1]

    # Create subplots for different scales
    scales = list(clustering_results.keys())
    n_scales = len(scales)

    if n_scales == 0:
        raise ValueError("No clustering results provided")

    # Create subplot layout
    if n_scales <= 2:
        fig = make_subplots(
            rows=1,
            cols=n_scales,
            subplot_titles=[f"Scale: {scale}" for scale in scales],
            specs=[[{"type": "scatter"}] * n_scales],
        )
    else:
        cols = min(3, n_scales)
        rows = (n_scales + cols - 1) // cols
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"Scale: {scale}" for scale in scales],
            specs=[[{"type": "scatter"} for _ in range(cols)] for _ in range(rows)],
        )

    # Add traces for each scale
    for i, (scale, result) in enumerate(clustering_results.items()):
        if "labels" not in result:
            continue

        labels = result["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Create colors for clusters
        unique_labels = np.unique(labels)
        colors = np.zeros((len(labels), 3))

        for j, label in enumerate(unique_labels):
            mask = labels == label
            hue = j / len(unique_labels)
            colors[mask] = [hue, 0.8, 0.8]

        # Convert to RGB strings
        color_strings = [
            f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})"
            for c in colors
        ]

        # Determine subplot position
        if n_scales <= 2:
            row, col = 1, i + 1
        else:
            row = i // cols + 1
            col = i % cols + 1

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 3),
                    color=color_strings,
                    opacity=kwargs.get("opacity", 0.7),
                ),
                text=[
                    f"Object {j}<br>Cluster: {labels[j]}<br>X: {x[j]:.1f}"
                    f"<br>Y: {y[j]:.1f}"
                    for j in range(len(x))
                ],
                hovertemplate="%{text}<extra></extra>",
                name=f"Scale {scale}",
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=16),
        ),
        width=kwargs.get("width", 1200),
        height=kwargs.get("height", 800),
        template="plotly_white",
    )

    # Update axes labels
    for i in range(1, n_scales + 1):
        if n_scales <= 2:
            row, col = 1, i
        else:
            row = (i - 1) // cols + 1
            col = (i - 1) % cols + 1

        fig.update_xaxes(title_text="X [Mpc]", row=row, col=col)
        fig.update_yaxes(title_text="Y [Mpc]", row=row, col=col)

    return fig


__all__ = [
    "plot_cosmic_web_3d",
    "plot_density_heatmap",
    "plot_multi_scale_clustering",
]
