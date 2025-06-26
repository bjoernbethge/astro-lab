"""AstroLab Widgets - Visualization and analysis widgets for astronomical data."""

from .astro_lab import AstroLabWidget
from .cosmograph_bridge import CosmographBridge

# Visualization functions (moved from utils.viz)
from .graph import cluster_and_analyze
from .plotly_bridge import create_plotly_visualization
from .tng50 import TNG50Visualizer

__all__ = [
    "AstroLabWidget",
    # Graph analysis
    "cluster_and_analyze",
    # Visualization backends
    "create_plotly_visualization",
    "CosmographBridge",
    "TNG50Visualizer",
]

# Complete widget architecture:
# - AstroLabWidget: Main entry point
# - Visualization backends: plotly_bridge, cosmograph_bridge, tensor_bridge
# - Graph analysis: graph.py with clustering and analysis
# - Specialized: tng50.py for simulation visualization
