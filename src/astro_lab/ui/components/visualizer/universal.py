"""
Universal Visualizer
===================

Main visualizer that routes to appropriate backend.
"""

from typing import Any, Dict, Optional
import marimo as mo

from .base import BaseVisualizer
from .cosmograph_viz import CosmographVisualizer
from .pyvista_viz import PyVistaVisualizer
from .plotly_viz import PlotlyVisualizer
from .blender_viz import BlenderVisualizer


class UniversalVisualizer:
    """Universal visualizer that can handle any data format and backend."""

    def __init__(self):
        self.backend = "cosmograph"
        self.visualizers = {
            "cosmograph": CosmographVisualizer(),
            "pyvista": PyVistaVisualizer(),
            "plotly": PlotlyVisualizer(),
            "blender": BlenderVisualizer(),
        }
        self.base = BaseVisualizer()  # For utility methods

    def visualize(
        self,
        data: Any,
        backend: Optional[str] = None,
        style: str = "cosmic_web",
        analysis_results: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> mo.Html:
        """Create visualization with automatic data format detection."""
        # Use specified backend or default
        backend = backend or self.backend

        # Get style preset
        style_params = self.visualizers[backend].style_presets.get(style, {})
        style_params.update(kwargs)  # Override with custom params

        # Detect data format and extract coordinates
        coords, metadata = self.base.extract_coordinates(data)

        if coords is None:
            return mo.callout(
                "Unable to extract coordinates from data",
                kind="danger"
            )

        # Add analysis results if available
        if analysis_results:
            metadata.update(analysis_results)

        # Get appropriate visualizer
        if backend not in self.visualizers:
            return mo.callout(f"Unknown backend: {backend}", kind="danger")

        visualizer = self.visualizers[backend]

        # Create visualization
        return visualizer.create_visualization(coords, metadata, style_params)

    def create_control_panel(self) -> mo.Html:
        """Create visualization control panel."""
        # Backend selector
        backend_selector = mo.ui.dropdown(
            options={
                "cosmograph": "Cosmograph - Interactive GPU-accelerated",
                "pyvista": "PyVista - Scientific 3D rendering",
                "plotly": "Plotly - Interactive web plots",
                "blender": "Blender - High-quality rendering",
            },
            value=self.backend,
            label="Visualization Backend",
        )

        # Style preset selector
        style_selector = mo.ui.dropdown(
            options={
                "cosmic_web": "Cosmic Web - Large-scale structure",
                "clusters": "Clusters - Dense regions",
                "filaments": "Filaments - Connected structures",
                "exploration": "Exploration - General purpose",
            },
            value="cosmic_web",
            label="Style Preset",
        )

        # Visual parameters
        point_size = mo.ui.slider(
            start=0.5, stop=10.0, value=2.0, step=0.5,
            label="Point Size",
        )

        node_opacity = mo.ui.slider(
            start=0.1, stop=1.0, value=0.8, step=0.1,
            label="Node Opacity",
        )

        link_opacity = mo.ui.slider(
            start=0.0, stop=1.0, value=0.5, step=0.1,
            label="Link Opacity",
        )

        # Color scheme
        color_scheme = mo.ui.dropdown(
            options=["survey", "magnitude", "velocity", "cluster", "density"],
            value="survey",
            label="Color Scheme",
        )

        # Display options
        show_grid = mo.ui.checkbox(value=True, label="Show Grid")
        show_axes = mo.ui.checkbox(value=True, label="Show Axes")
        show_stats = mo.ui.checkbox(value=True, label="Show Statistics")

        # Camera controls
        camera_preset = mo.ui.dropdown(
            options=["auto", "front", "top", "iso", "free"],
            value="auto",
            label="Camera Position",
        )

        # Export options
        export_format = mo.ui.dropdown(
            options=["png", "html", "gltf", "ply", "obj"],
            value="png",
            label="Export Format",
        )

        export_btn = mo.ui.button("Export Visualization", kind="secondary")

        return mo.vstack([
            mo.md("### Visualization Controls"),
            backend_selector,
            style_selector,
            mo.ui.tabs({
                "Appearance": mo.vstack([
                    point_size,
                    node_opacity,
                    link_opacity,
                    color_scheme,
                ]),
                "Display": mo.vstack([
                    show_grid,
                    show_axes,
                    show_stats,
                    camera_preset,
                ]),
                "Export": mo.vstack([
                    export_format,
                    export_btn,
                ]),
            }),
        ])
