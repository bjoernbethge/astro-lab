"""
UI Components for AstroLab
=========================

Core UI components for the dashboard.
"""

# State management
from . import state, system_info
from .analyzer import create_analyzer, run_cosmic_web_analysis

# Core components
from .data_loader import create_data_loader

# Visualizer components
from .visualizer import (
    CosmographVisualizer,
    PlotlyVisualizer,
    PyVistaVisualizer,
    UniversalVisualizer,
)
from .viz import create_cosmic_web_viz, create_plotly_viz, create_visualizer

__all__ = [
    "state",
    "system_info",
    "create_data_loader",
    "create_analyzer",
    "run_cosmic_web_analysis",
    "create_visualizer",
    "create_cosmic_web_viz",
    "create_plotly_viz",
    "UniversalVisualizer",
    "CosmographVisualizer",
    "PyVistaVisualizer",
    "PlotlyVisualizer",
]
