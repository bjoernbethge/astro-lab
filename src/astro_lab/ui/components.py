"""
AstroLab UI Components - Marimo UI Components
============================================

Provides Marimo UI components for the AstroLab dashboard.
"""

import logging
from typing import Any, Dict, Optional

import marimo as mo
import torch

from ..widgets import AstroLabWidget

logger = logging.getLogger(__name__)

# Global widget instance for actual functionality
_astro_widget = None
try:
    _astro_widget = AstroLabWidget()
    WIDGETS_AVAILABLE = True
except Exception:
    _astro_widget = None
    WIDGETS_AVAILABLE = False


def ui_quick_setup() -> mo.ui.dictionary:
    """Quick setup component for common workflows."""
    return mo.ui.dictionary(
        {
            "experiment_name": mo.ui.text(
                label="Experiment Name",
                placeholder="e.g., gaia_stellar_v1",
            ),
            "survey": mo.ui.dropdown(
                label="Survey",
                options=["gaia", "sdss", "nsa", "linear", "tng50"],
                value="gaia",
            ),
            "model": mo.ui.dropdown(
                label="Model",
                options=["gaia_classifier", "survey_gnn", "point_cloud_gnn"],
                value="gaia_classifier",
            ),
            "quick_setup": mo.ui.button(label="🚀 Quick Setup"),
        },
        label="🚀 Quick Setup",
    )


def ui_data_controls() -> mo.ui.dictionary:
    """Data loading and management controls."""
    return mo.ui.dictionary(
        {
            "survey": mo.ui.dropdown(
                label="Survey",
                options=["gaia", "sdss", "nsa", "linear", "tng50"],
                value="gaia",
            ),
            "max_samples": mo.ui.slider(
                label="Max Samples",
                start=1000,
                stop=100000,
                step=1000,
                value=25000,
            ),
            "use_cache": mo.ui.checkbox(
                label="Use Cache",
                value=True,
            ),
            "load_data": mo.ui.button(label="📊 Load Data"),
            "preview_data": mo.ui.button(label="👁️ Preview Data"),
        },
        label="📊 Data Controls",
    )


def ui_visualization_controls() -> mo.ui.dictionary:
    """Visualization controls using AstroLab widget backends."""
    backend_options = ["plotly", "matplotlib", "bokeh"]

    # Add widget backends if available
    if WIDGETS_AVAILABLE:
        backend_options.extend(["open3d", "pyvista", "blender"])

    return mo.ui.dictionary(
        {
            "backend": mo.ui.dropdown(
                label="Backend",
                options=backend_options,
                value="plotly",
            ),
            "plot_type": mo.ui.dropdown(
                label="Plot Type",
                options=["scatter", "scatter_3d", "histogram", "heatmap", "density"],
                value="scatter",
            ),
            "max_points": mo.ui.slider(
                label="Max Points",
                start=1000,
                stop=100000,
                step=5000,
                value=25000,
            ),
            "interactive": mo.ui.checkbox(
                label="Interactive",
                value=True,
            ),
            "enable_3d": mo.ui.checkbox(
                label="Enable 3D",
                value=True,
            ),
            "create_plot": mo.ui.button(label="📈 Create Plot"),
        },
        label="🎨 Visualization",
    )


def ui_analysis_controls() -> mo.ui.dictionary:
    """Analysis controls using AstroLab widget functionality."""
    return mo.ui.dictionary(
        {
            "method": mo.ui.dropdown(
                label="Analysis Method",
                options=["clustering", "density", "structure", "neighbors"],
                value="clustering",
            ),
            "clustering_algorithm": mo.ui.dropdown(
                label="Clustering Algorithm",
                options=["dbscan", "kmeans", "agglomerative"],
                value="dbscan",
            ),
            "k_neighbors": mo.ui.slider(
                label="K-Neighbors",
                start=3,
                stop=50,
                step=1,
                value=10,
            ),
            "eps": mo.ui.number(
                label="DBSCAN Eps",
                value=10.0,
                start=0.1,
                stop=100.0,
                step=0.1,
            ),
            "min_samples": mo.ui.slider(
                label="Min Samples",
                start=3,
                stop=20,
                step=1,
                value=5,
            ),
            "use_gpu": mo.ui.checkbox(
                label="Use GPU",
                value=torch.cuda.is_available(),
            ),
            "run_analysis": mo.ui.button(label="🔬 Run Analysis"),
        },
        label="🔬 Analysis",
    )


def ui_model_controls() -> mo.ui.dictionary:
    """Model training controls."""
    return mo.ui.dictionary(
        {
            "model_type": mo.ui.dropdown(
                label="Model Type",
                options=["gaia_classifier", "survey_gnn", "point_cloud_gnn"],
                value="gaia_classifier",
            ),
            "task": mo.ui.dropdown(
                label="Task",
                options=["classification", "regression", "clustering"],
                value="classification",
            ),
            "batch_size": mo.ui.slider(
                label="Batch Size",
                start=8,
                stop=128,
                step=8,
                value=32,
            ),
            "epochs": mo.ui.slider(
                label="Epochs",
                start=1,
                stop=100,
                step=1,
                value=10,
            ),
            "learning_rate": mo.ui.number(
                label="Learning Rate",
                value=0.001,
                start=0.0001,
                stop=0.1,
                step=0.0001,
            ),
            "prepare_data": mo.ui.button(label="🔧 Prepare Data for Model"),
            "train_model": mo.ui.button(label="🏋️ Train Model"),
        },
        label="🤖 Model Training",
    )


def ui_graph_controls() -> mo.ui.dictionary:
    """Graph analysis controls using AstroLab widget."""
    return mo.ui.dictionary(
        {
            "graph_type": mo.ui.dropdown(
                label="Graph Type",
                options=["spatial", "knn", "radius", "delaunay"],
                value="spatial",
            ),
            "k_neighbors": mo.ui.slider(
                label="K-Neighbors",
                start=3,
                stop=50,
                step=1,
                value=10,
            ),
            "radius": mo.ui.number(
                label="Radius",
                value=1.0,
                start=0.1,
                stop=10.0,
                step=0.1,
            ),
            "edge_weight": mo.ui.dropdown(
                label="Edge Weight",
                options=["distance", "inverse_distance", "none"],
                value="distance",
            ),
            "use_gpu": mo.ui.checkbox(
                label="Use GPU",
                value=torch.cuda.is_available(),
            ),
            "create_graph": mo.ui.button(label="🕸️ Create Graph"),
            "analyze_graph": mo.ui.button(label="📊 Analyze Graph"),
        },
        label="🕸️ Graph Analysis",
    )


def ui_system_status() -> mo.ui.dictionary:
    """System status and information."""
    # Get system information
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"

    return mo.ui.dictionary(
        {
            "device": mo.ui.text(
                label="Device",
                value=device_info,
                disabled=True,
            ),
            "widgets_available": mo.ui.text(
                label="Widgets Available",
                value="Yes" if WIDGETS_AVAILABLE else "No",
                disabled=True,
            ),
            "memory_usage": mo.ui.text(
                label="Memory Usage",
                value="Checking...",
                disabled=True,
            ),
            "refresh_status": mo.ui.button(label="🔄 Refresh Status"),
        },
        label="💻 System Status",
    )


def ui_quick_actions() -> mo.ui.dictionary:
    """Quick action buttons for common tasks."""
    return mo.ui.dictionary(
        {
            "load_gaia": mo.ui.button(label="🌟 Load Gaia"),
            "load_sdss": mo.ui.button(label="🔭 Load SDSS"),
            "load_tng50": mo.ui.button(label="🌌 Load TNG50"),
            "create_3d_plot": mo.ui.button(label="🎨 Create 3D Plot"),
            "run_clustering": mo.ui.button(label="🔬 Run Clustering"),
            "export_results": mo.ui.button(label="💾 Export Results"),
        },
        label="⚡ Quick Actions",
    )


def handle_component_actions(components: Dict[str, mo.ui.dictionary]) -> Optional[str]:
    """Handle actions from UI components."""
    # This function would handle the actual logic for component actions
    # For now, it's a placeholder
    return "Action handled"


# Export all UI components
__all__ = [
    "ui_quick_setup",
    "ui_data_controls",
    "ui_visualization_controls",
    "ui_analysis_controls",
    "ui_model_controls",
    "ui_graph_controls",
    "ui_system_status",
    "ui_quick_actions",
    "handle_component_actions",
    "WIDGETS_AVAILABLE",
]
