"""
Simple Visualization Component
=============================

Functional visualization interface.
"""

import marimo as mo

from astro_lab.ui.components.visualizer.universal import create_3d_scatter
from astro_lab.widgets.albpy import generate_cosmic_web_scene
from astro_lab.widgets.cosmograph_bridge import CosmographBridge


def create_visualizer():
    """Create simple visualization interface."""

    # Visualization options
    backend = mo.ui.dropdown(
        options=["cosmograph", "plotly", "pyvista"], value="cosmograph", label="Backend"
    )

    node_size = mo.ui.slider(0.5, 10, 2, step=0.5, label="Point size")

    show_edges = mo.ui.checkbox(True, label="Show edges")

    # Create button
    create_btn = mo.ui.button("Create Visualization", kind="primary")

    # UI
    ui = mo.vstack(
        [mo.md("### 🎨 Visualization"), backend, node_size, show_edges, create_btn]
    )

    # Placeholder for viz
    viz = mo.md("")
    status = mo.md("")

    if create_btn.value:
        status = mo.callout(
            f"✅ Would create {backend.value} visualization", kind="success"
        )

    return ui, viz, status


def create_cosmic_web_viz(data, analysis_results=None):
    """Create cosmic web visualization using Cosmograph."""
    try:
        bridge = CosmographBridge()

        if analysis_results:
            # Use analysis results
            viz = bridge.from_cosmic_web_results(analysis_results, survey_name="auto")
        else:
            # Just visualize data
            viz = bridge.from_dataframe(
                data, x_col="x", y_col="y", z_col="z", node_size=2.0
            )

        return viz

    except Exception as e:
        return mo.callout(f"Visualization error: {str(e)}", kind="danger")


def create_plotly_viz(data, color_by=None):
    """Create Plotly 3D scatter plot."""
    try:
        # Extract coordinates
        if hasattr(data, "select"):
            x = data["x"].to_numpy()
            y = data["y"].to_numpy()
            z = data["z"].to_numpy()
        else:
            x, y, z = data[:, 0], data[:, 1], data[:, 2]

        # Create plot
        fig = create_3d_scatter(x=x, y=y, z=z, color=color_by, title="3D Visualization")

        return mo.ui.plotly(fig)

    except Exception as e:
        return mo.callout(f"Plotly error: {str(e)}", kind="danger")


def create_blender_cosmic_web_scene(
    survey: str, max_samples: int = 10000, render: bool = True
):
    """
    Create a Blender-based cosmic web scene using the AlbPy generator.
    Args:
        survey: Survey name (e.g. 'gaia', 'sdss', 'nsa', 'tng50', 'exoplanet')
        max_samples: Maximum number of data points
        render: Whether to render the scene
    """
    generate_cosmic_web_scene(survey, max_samples=max_samples, render=render)
