"""AstroLab Widgets - Visualization and analysis widgets for astronomical data."""

from .astro_lab import AstroLabWidget
from .cosmograph_bridge import CosmographBridge

# Visualization functions (moved from utils.viz)
from .graph import cluster_and_analyze, analyze_cosmic_web_structure
from .plotly_bridge import create_plotly_visualization, plot_cosmic_web_3d
from .tng50 import TNG50Visualizer

__all__ = [
    "AstroLabWidget",
    # Graph analysis
    "cluster_and_analyze",
    "analyze_cosmic_web_structure",
    # Visualization backends
    "create_plotly_visualization",
    "plot_cosmic_web_3d",
    "CosmographBridge",
    "TNG50Visualizer",
]

# Complete widget architecture:
# - AstroLabWidget: Main entry point
# - Visualization backends: plotly_bridge, cosmograph_bridge, tensor_bridge
# - Graph analysis: graph.py with clustering and cosmic web analysis
# - Specialized: tng50.py for simulation visualization
