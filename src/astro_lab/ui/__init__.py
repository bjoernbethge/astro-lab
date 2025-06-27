"""
AstroLab UI Package
==================

Modern Marimo-based user interface for astronomical data analysis.

The UI provides:
- Interactive data exploration and visualization
- GPU-accelerated analysis tools
- AI-powered assistance
- SQL console for data queries
- Model training and evaluation
- Real-time system monitoring
"""

from .components import (
    handle_dashboard_events,
    ui_ai_assistant,
    ui_analysis_center,
    ui_data_explorer,
    ui_graph_analyzer,
    ui_model_lab,
    ui_results_gallery,
    ui_sql_console,
    ui_system_monitor,
    ui_visualization_studio,
    ui_workflow_builder,
)
from .dashboard import (
    AstroLabDashboard,
    create_astrolab_dashboard,
)

# Version info
__version__ = "2.0.0"
__author__ = "AstroLab Team"

# Main exports
__all__ = [
    # Dashboard
    "AstroLabDashboard",
    "create_astrolab_dashboard",
    # Core Components
    "ui_data_explorer",
    "ui_visualization_studio",
    "ui_analysis_center",
    "ui_model_lab",
    "ui_ai_assistant",
    "ui_sql_console",
    "ui_graph_analyzer",
    "ui_system_monitor",
    "ui_results_gallery",
    "ui_workflow_builder",
    # Utilities
    "handle_dashboard_events",
]
