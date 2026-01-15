"""
PyVista Visualizer
==================

Scientific 3D visualization using PyVista.
"""

import numpy as np
from typing import Dict, Any
import marimo as mo

from astro_lab.widgets.alpv import (
    create_pyvista_visualization,
    create_cosmic_web_visualization,
)
from .base import BaseVisualizer


class PyVistaVisualizer(BaseVisualizer):
    """PyVista-based visualizer for scientific rendering."""

    def create_visualization(
        self,
        coords: np.ndarray,
        metadata: Dict[str, Any],
        params: Dict[str, Any]
    ) -> mo.Html:
        """Create PyVista visualization."""
        # Check for cosmic web features
        if any(key in metadata for key in ["cluster_labels", "filament_edges"]):
            return create_cosmic_web_visualization(
                coords,
                clusters=metadata.get("cluster_labels"),
                filaments=metadata.get("filament_edges"),
                point_size=params.get("node_size", 2.0),
                opacity=params.get("node_opacity", 0.8),
                show_axes=True,
                show_grid=params.get("show_grid", True),
                camera_position="iso",
            )
        else:
            return create_pyvista_visualization(
                coords,
                scalars=metadata.get("magnitude"),
                point_size=params.get("node_size", 2.0),
                opacity=params.get("node_opacity", 0.8),
                cmap=params.get("color_scheme", "viridis"),
                show_axes=True,
            )
