"""
Stellar Plotly visualizations for astro-lab.

Provides specialized stellar evolution, galaxy cluster, and exoplanet system plots.
"""

import logging
from typing import Any

# Astropy imports for astronomical correctness
import numpy as np
import plotly.graph_objects as go
import torch
from astropy.visualization import quantity_support

# Enable quantity support
quantity_support()

logger = logging.getLogger(__name__)


def plot_stellar_evolution(
    stellar_data: Any,
    title: str = "Stellar Evolution",
    **kwargs,
) -> go.Figure:
    """
    Create stellar evolution visualization.

    Args:
        stellar_data: Stellar evolution data
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract stellar data
    if hasattr(stellar_data, "stellar_evolution"):
        data = stellar_data["stellar_evolution"]
    else:
        data = stellar_data

    # Try to extract common stellar parameters
    if "effective_temperature" in data and "luminosity" in data:
        teff = data["effective_temperature"]
        lum = data["luminosity"]
        x_data = teff
        y_data = lum
        x_label = "Effective Temperature [K]"
        y_label = "Luminosity [L☉]"
        plot_type = "hr_diagram"
    elif "age" in data and "mass" in data:
        age = data["age"]
        mass = data["mass"]
        x_data = age
        y_data = mass
        x_label = "Age [Gyr]"
        y_label = "Mass [M☉]"
        plot_type = "mass_evolution"
    elif "radius" in data and "mass" in data:
        radius = data["radius"]
        mass = data["mass"]
        x_data = mass
        y_data = radius
        x_label = "Mass [M☉]"
        y_label = "Radius [R☉]"
        plot_type = "mass_radius"
    else:
        # Fallback to generic scatter
        keys = list(data.keys())
        if len(keys) >= 2:
            x_data = data[keys[0]]
            y_data = data[keys[1]]
            x_label = keys[0]
            y_label = keys[1]
            plot_type = "generic"
        else:
            raise ValueError("Insufficient stellar data for visualization")

    # Convert to numpy if needed
    if isinstance(x_data, torch.Tensor):
        x_np = x_data.cpu().numpy()
    else:
        x_np = np.array(x_data)

    if isinstance(y_data, torch.Tensor):
        y_np = y_data.cpu().numpy()
    else:
        y_np = np.array(y_data)

    # Create plot
    fig = go.Figure()

    # Color by additional parameter if available
    color_data = None
    if "metallicity" in data:
        color_data = data["metallicity"]
        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()
        color_title = "Metallicity [Z]"
    elif "surface_gravity" in data:
        color_data = data["surface_gravity"]
        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()
        color_title = "Surface Gravity [log g]"

    if color_data is not None:
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 5),
                    color=color_data,
                    colorscale="Viridis",
                    opacity=kwargs.get("opacity", 0.7),
                    showscale=True,
                    colorbar=dict(title=color_title),
                ),
                text=[
                    f"Star {i}<br>{x_label}: {x_np[i]:.2e}"
                    f"<br>{y_label}: {y_np[i]:.2e}<br>Color: {color_data[i]:.3f}"
                    for i in range(len(x_np))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Stellar Evolution",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 5),
                    color="red",
                    opacity=kwargs.get("opacity", 0.7),
                ),
                text=[
                    f"Star {i}<br>{x_label}: {x_np[i]:.2e}<br>{y_label}: {y_np[i]:.2e}"
                    for i in range(len(x_np))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Stellar Evolution",
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
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=kwargs.get("width", 900),
        height=kwargs.get("height", 700),
        template="plotly_white",
    )

    # Special handling for HR diagram
    if plot_type == "hr_diagram":
        fig.update_xaxes(autorange="reversed")  # Temperature decreases to the right
        fig.update_yaxes(type="log")  # Log scale for luminosity

    return fig


def plot_galaxy_cluster(
    cluster_data: Any,
    title: str = "Galaxy Cluster",
    **kwargs,
) -> go.Figure:
    """
    Create galaxy cluster visualization.

    Args:
        cluster_data: Galaxy cluster data
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract cluster data
    if hasattr(cluster_data, "cluster"):
        data = cluster_data["cluster"]
    else:
        data = cluster_data

    # Try to extract spatial coordinates
    if "coordinates" in data:
        coords = data["coordinates"]
    elif "positions" in data:
        coords = data["positions"]
    elif "spatial" in data and "coordinates" in data["spatial"]:
        coords = data["spatial"]["coordinates"]
    else:
        raise ValueError("No spatial coordinates found in cluster data")

    # Convert to numpy
    if isinstance(coords, torch.Tensor):
        coords_np = coords.cpu().numpy()
    else:
        coords_np = np.array(coords)

    # Handle different coordinate dimensions
    if coords_np.shape[1] == 2:
        x, y = coords_np[:, 0], coords_np[:, 1]
        z = np.zeros_like(x)
        plot_3d = False
    elif coords_np.shape[1] >= 3:
        x, y, z = coords_np[:, 0], coords_np[:, 1], coords_np[:, 2]
        plot_3d = kwargs.get("plot_3d", True)
    else:
        raise ValueError(f"Unsupported coordinate dimensions: {coords_np.shape[1]}")

    # Create plot
    if plot_3d:
        fig = go.Figure()

        # Color by additional parameter if available
        color_data = None
        if "magnitudes" in data:
            mags = data["magnitudes"]
            if "g_mag" in mags:
                color_data = mags["g_mag"]
                color_title = "g Magnitude [mag]"
            elif "r_mag" in mags:
                color_data = mags["r_mag"]
                color_title = "r Magnitude [mag]"

        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()

        if color_data is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("point_size", 3),
                        color=color_data,
                        colorscale="Viridis",
                        opacity=kwargs.get("opacity", 0.7),
                        showscale=True,
                        colorbar=dict(title=color_title),
                    ),
                    text=[
                        f"Galaxy {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}"
                        f"<br>Z: {z[i]:.1f}<br>Mag: {color_data[i]:.2f}"
                        for i in range(len(x))
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    name="Galaxies",
                )
            )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("point_size", 3),
                        color="blue",
                        opacity=kwargs.get("opacity", 0.7),
                    ),
                    text=[
                        f"Galaxy {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}<br>Z: {z[i]:.1f}"
                        for i in range(len(x))
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    name="Galaxies",
                )
            )

        # Update layout for 3D
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
            width=kwargs.get("width", 900),
            height=kwargs.get("height", 700),
            template="plotly_white",
        )
    else:
        # 2D plot
        fig = go.Figure()

        # Color by additional parameter if available
        color_data = None
        if "magnitudes" in data:
            mags = data["magnitudes"]
            if "g_mag" in mags:
                color_data = mags["g_mag"]
                color_title = "g Magnitude [mag]"
            elif "r_mag" in mags:
                color_data = mags["r_mag"]
                color_title = "r Magnitude [mag]"

        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()

        if color_data is not None:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("point_size", 5),
                        color=color_data,
                        colorscale="Viridis",
                        opacity=kwargs.get("opacity", 0.7),
                        showscale=True,
                        colorbar=dict(title=color_title),
                    ),
                    text=[
                        f"Galaxy {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}"
                        f"<br>Mag: {color_data[i]:.2f}"
                        for i in range(len(x))
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    name="Galaxies",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("point_size", 5),
                        color="blue",
                        opacity=kwargs.get("opacity", 0.7),
                    ),
                    text=[
                        f"Galaxy {i}<br>X: {x[i]:.1f}<br>Y: {y[i]:.1f}"
                        for i in range(len(x))
                    ],
                    hovertemplate="%{text}<extra></extra>",
                    name="Galaxies",
                )
            )

        # Update layout for 2D
        fig.update_layout(
            title=dict(
                text=title,
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


def plot_exoplanet_system(
    exoplanet_data: Any,
    title: str = "Exoplanet System",
    **kwargs,
) -> go.Figure:
    """
    Create exoplanet system visualization.

    Args:
        exoplanet_data: Exoplanet system data
        title: Plot title
        **kwargs: Additional parameters

    Returns:
        Plotly figure object
    """
    # Extract exoplanet data
    if hasattr(exoplanet_data, "exoplanet"):
        data = exoplanet_data["exoplanet"]
    else:
        data = exoplanet_data

    # Try to extract orbital parameters
    if "semi_major_axis" in data and "eccentricity" in data:
        a = data["semi_major_axis"]
        e = data["eccentricity"]
        x_data = a
        y_data = e
        x_label = "Semi-Major Axis [AU]"
        y_label = "Eccentricity"
        plot_type = "orbital_parameters"
    elif "period" in data and "mass" in data:
        period = data["period"]
        mass = data["mass"]
        x_data = period
        y_data = mass
        x_label = "Orbital Period [days]"
        y_label = "Planet Mass [M⊕]"
        plot_type = "period_mass"
    elif "radius" in data and "mass" in data:
        radius = data["radius"]
        mass = data["mass"]
        x_data = mass
        y_data = radius
        x_label = "Planet Mass [M⊕]"
        y_label = "Planet Radius [R⊕]"
        plot_type = "mass_radius"
    else:
        # Fallback to generic scatter
        keys = list(data.keys())
        if len(keys) >= 2:
            x_data = data[keys[0]]
            y_data = data[keys[1]]
            x_label = keys[0]
            y_label = keys[1]
            plot_type = "generic"
        else:
            raise ValueError("Insufficient exoplanet data for visualization")

    # Convert to numpy if needed
    if isinstance(x_data, torch.Tensor):
        x_np = x_data.cpu().numpy()
    else:
        x_np = np.array(x_data)

    if isinstance(y_data, torch.Tensor):
        y_np = y_data.cpu().numpy()
    else:
        y_np = np.array(y_data)

    # Create plot
    fig = go.Figure()

    # Color by additional parameter if available
    color_data = None
    if "temperature" in data:
        color_data = data["temperature"]
        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()
        color_title = "Temperature [K]"
    elif "density" in data:
        color_data = data["density"]
        if isinstance(color_data, torch.Tensor):
            color_data = color_data.cpu().numpy()
        color_title = "Density [g/cm³]"

    if color_data is not None:
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 8),
                    color=color_data,
                    colorscale="Viridis",
                    opacity=kwargs.get("opacity", 0.7),
                    showscale=True,
                    colorbar=dict(title=color_title),
                ),
                text=[
                    f"Planet {i}<br>{x_label}: {x_np[i]:.2e}"
                    f"<br>{y_label}: {y_np[i]:.2e}"
                    for i in range(len(x_np))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Exoplanets",
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=x_np,
                y=y_np,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 8),
                    color="green",
                    opacity=kwargs.get("opacity", 0.7),
                ),
                text=[
                    f"Planet {i}<br>{x_label}: {x_np[i]:.2e}"
                    f"<br>{y_label}: {y_np[i]:.2e}"
                    for i in range(len(x_np))
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Exoplanets",
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
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=kwargs.get("width", 900),
        height=kwargs.get("height", 700),
        template="plotly_white",
    )

    # Special handling for different plot types
    if plot_type == "orbital_parameters":
        fig.update_xaxes(type="log")  # Log scale for semi-major axis
    elif plot_type == "period_mass":
        fig.update_xaxes(type="log")  # Log scale for period
        fig.update_yaxes(type="log")  # Log scale for mass

    return fig


__all__ = [
    "plot_stellar_evolution",
    "plot_galaxy_cluster",
    "plot_exoplanet_system",
]
