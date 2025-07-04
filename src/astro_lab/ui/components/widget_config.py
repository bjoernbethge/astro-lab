"""
Widget Configuration
===================

Configuration for all available visualization backends and their options.
"""

import marimo as mo


def get_available_backends():
    """Get all available visualization backends from the widgets module."""
    # These are the actual backends supported by astro_lab.widgets
    return ["auto", "pyvista", "open3d", "blender", "cosmograph", "plotly"]


def get_backend_plot_types(backend: str):
    """Get available plot types for a specific backend from actual implementations."""
    if backend == "auto":
        return ["auto"]

    # Get actual plot types from the bridge implementations
    plot_types = {
        "pyvista": ["scatter_3d", "point_cloud", "volume", "surface"],
        "open3d": ["point_cloud", "mesh", "lines"],
        "blender": ["scatter_3d", "particles", "mesh", "volume"],
        "cosmograph": ["interactive_3d", "network", "timeline"],
        "plotly": [
            "scatter",  # Basic scatter plot
            "density",  # Density visualization
            "clustering",  # Clustering results
            "cosmic_web",  # Cosmic web analysis
            "histogram",  # Distribution plots
            "heatmap",  # Correlation heatmap
            "scatter_3d",  # 3D scatter
            "scatter_2d",  # 2D scatter
            "sky_map",  # Sky map projection
            "hr_diagram",  # Hertzsprung-Russell diagram
            "color_magnitude",  # Color-magnitude diagram
            "magnitude_histogram",  # Magnitude distribution
            "cluster_analysis",  # Cluster analysis results
            "multi_scale",  # Multi-scale analysis
        ],
    }
    return plot_types.get(backend, ["auto"])


def get_backend_color_options(backend: str):
    """Get available color options for a specific backend."""
    # These are based on actual tensor fields and properties
    color_options = {
        "pyvista": ["distance", "magnitude", "redshift", "cluster", "density"],
        "open3d": ["distance", "magnitude", "cluster"],
        "blender": ["distance", "magnitude", "redshift", "cluster", "material"],
        "cosmograph": ["distance", "magnitude", "redshift", "cluster", "time"],
        "plotly": [
            "auto",  # Automatic coloring
            "cluster",  # Cluster labels
            "magnitude",  # Magnitude values
            "redshift",  # Redshift values
            "distance",  # Distance values
            "density",  # Local density
            "type",  # Object type
            "scale",  # Analysis scale
        ],
        "auto": ["auto"],
    }
    return color_options.get(backend, ["auto"])


def get_backend_size_options(backend: str):
    """Get available size options for a specific backend."""
    # These are based on actual tensor fields and properties
    size_options = {
        "pyvista": ["magnitude", "redshift", "cluster_size", "distance", "fixed"],
        "open3d": ["magnitude", "cluster_size", "fixed"],
        "blender": ["magnitude", "redshift", "cluster_size", "distance", "fixed"],
        "cosmograph": ["magnitude", "redshift", "cluster_size", "time", "fixed"],
        "plotly": [
            "fixed",  # Fixed size
            "magnitude",  # Magnitude-based size
            "redshift",  # Redshift-based size
            "cluster_size",  # Cluster size
            "distance",  # Distance-based size
            "density",  # Density-based size
        ],
        "auto": ["auto"],
    }
    return size_options.get(backend, ["auto"])


def create_widget_config():
    """Create widget configuration UI."""
    backends = get_available_backends()

    return mo.ui.dictionary(
        {
            "backend": mo.ui.dropdown(
                options=backends, value="plotly", label="Visualization Backend"
            ),
            "plot_type": mo.ui.dropdown(
                options=get_backend_plot_types("plotly"),
                value="scatter",
                label="Plot Type",
            ),
            "color_by": mo.ui.dropdown(
                options=get_backend_color_options("plotly"),
                value="auto",
                label="Color By",
            ),
            "size_by": mo.ui.dropdown(
                options=get_backend_size_options("plotly"),
                value="fixed",
                label="Size By",
            ),
            "node_size": mo.ui.slider(start=1, stop=20, value=2, label="Node Size"),
            "opacity": mo.ui.slider(
                start=0.1, stop=1.0, value=0.8, step=0.1, label="Opacity"
            ),
            "show_edges": mo.ui.checkbox(value=False, label="Show Edges"),
            "interactive": mo.ui.checkbox(value=True, label="Interactive Mode"),
            "fullscreen": mo.ui.checkbox(value=False, label="Fullscreen"),
            "auto_scale": mo.ui.checkbox(value=True, label="Auto Scale"),
            "show_legend": mo.ui.checkbox(value=True, label="Show Legend"),
        }
    )
