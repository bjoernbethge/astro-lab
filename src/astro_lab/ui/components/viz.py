"""
Simple Visualization Component
=============================

Real visualization using actual AstroLab widgets.
"""

import marimo as mo
import numpy as np
import polars as pl

from astro_lab.widgets.cosmograph_bridge import CosmographBridge
from astro_lab.widgets.plotly.bridge import AstronomicalPlotlyBridge


def create_viz_component():
    """Create visualization component for cosmic web page."""
    return create_visualizer()


def create_visualizer():
    """Create real visualization interface."""

    # Visualization options
    backend = mo.ui.dropdown(
        options={
            "plotly": "Plotly 3D Scatter",
            "cosmograph": "Cosmograph Interactive Network",
        },
        value="plotly",
        label="Visualization Backend",
    )

    node_size = mo.ui.slider(0.5, 10, 2, label="Point size")

    color_by = mo.ui.dropdown(
        options={
            "auto": "Auto-detect color",
            "magnitude": "By magnitude",
            "distance": "By distance",
            "cluster": "By cluster",
        },
        value="auto",
        label="Color scheme",
    )

    # Create button
    create_btn = mo.ui.button(label="🎨 Create Visualization", kind="success")

    # UI
    ui = mo.vstack(
        [mo.md("### 🎨 Visualization"), backend, node_size, color_by, create_btn]
    )

    # Store configuration for use by other components
    viz_config = None
    status = mo.md(
        "⏳ **Ready to create visualization** - Configure options and click 'Create Visualization'"
    )

    if create_btn.value:
        viz_config = {
            "backend": backend.value,
            "node_size": node_size.value,
            "color_by": color_by.value,
            "ready": True,
        }

        status = mo.callout(
            f"✅ Visualization configured: {backend.value} with {color_by.value} coloring",
            kind="success",
        )

    return ui, viz_config, status


def create_plotly_viz(data, color_by=None, node_size=2.0):
    """Create Plotly 3D scatter plot using real data."""
    try:
        if isinstance(data, pl.DataFrame):
            # Extract coordinates
            if all(col in data.columns for col in ["x", "y", "z"]):
                x = data["x"].to_numpy()
                y = data["y"].to_numpy()
                z = data["z"].to_numpy()
            else:
                raise ValueError("Data must have x, y, z columns for plotting")

            # Extract color data
            color_data = None
            if color_by == "magnitude" and "magnitude" in data.columns:
                color_data = data["magnitude"].to_numpy()
            elif color_by == "distance" and "distance_pc" in data.columns:
                color_data = data["distance_pc"].to_numpy()
            elif color_by == "auto" and "brightness" in data.columns:
                color_data = data["brightness"].to_numpy()
        else:
            # Assume numpy array
            x, y, z = data[:, 0], data[:, 1], data[:, 2]
            color_data = None

        # Create plot using AstroLab widget
        bridge = AstronomicalPlotlyBridge()
        fig = bridge._create_3d_scatter(
            coords=np.column_stack([x, y, z]),
            colors=color_data,
            point_size=node_size,
            title="3D Astronomical Data Visualization",
        )

        return mo.ui.plotly(fig)

    except Exception as e:
        return mo.callout(f"Plotly visualization error: {str(e)}", kind="danger")


def create_cosmic_web_viz(data, analysis_results=None):
    """Create cosmic web visualization using Cosmograph."""
    try:
        bridge = CosmographBridge()

        if analysis_results and "scale_results" in analysis_results:
            # Use analysis results to create network
            scale = list(analysis_results["scale_results"].keys())[0]  # Use first scale
            labels = analysis_results["scale_results"][scale]["cluster_labels"]

            if isinstance(data, pl.DataFrame):
                # Add cluster labels to data
                data_with_clusters = data.with_columns(pl.Series("cluster", labels))

                viz = bridge.create_visualization(
                    data_with_clusters,
                    x_col="x",
                    y_col="y",
                    z_col="z",
                    cluster_col="cluster",
                    node_size=2.0,
                )
            else:
                # Create simple visualization
                viz = bridge.create_visualization(
                    coordinates=data, labels=labels, node_size=2.0
                )
        else:
            # Simple data visualization
            if isinstance(data, pl.DataFrame):
                viz = bridge.create_visualization(
                    data, x_col="x", y_col="y", z_col="z", node_size=2.0
                )
            else:
                viz = bridge.create_visualization(coordinates=data, node_size=2.0)

        return viz

    except Exception as e:
        return mo.callout(f"Cosmograph visualization error: {str(e)}", kind="danger")


def create_blender_cosmic_web_scene(
    coordinates: "np.ndarray", output_path: str = "cosmic_web.png", render: bool = True
):
    """Create Blender scene using AlbPy (simplified interface)."""
    try:
        from astro_lab.widgets.albpy import generate_cosmic_web_scene

        # Generate scene and use the result
        result = generate_cosmic_web_scene(
            coordinates=coordinates, output_path=output_path, render=render
        )

        return mo.callout(
            f"✅ Blender scene generated with {len(coordinates)} points. Status: {result.get('status', 'unknown')}",
            kind="success",
        )

    except ImportError:
        return mo.callout(
            "❌ Blender/AlbPy not available. Install Blender and enable AlbPy.",
            kind="warning",
        )
    except Exception as e:
        return mo.callout(f"❌ Blender scene error: {str(e)}", kind="danger")
