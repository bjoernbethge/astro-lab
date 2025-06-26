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

from .dashboard import (
    AstroLabDashboard,
    create_astrolab_dashboard,
    create_mobile_dashboard,
    create_presentation_mode,
)

from .components import (
    ui_data_explorer,
    ui_visualization_studio,
    ui_analysis_center,
    ui_model_lab,
    ui_ai_assistant,
    ui_sql_console,
    ui_graph_analyzer,
    ui_system_monitor,
    ui_results_gallery,
    ui_workflow_builder,
    handle_dashboard_events,
)

from .settings import (
    ui_experiment_settings,
    ui_data_settings,
    ui_model_settings,
    ui_visualization_settings,
    save_config,
    load_config,
    get_config_files,
)

# Version info
__version__ = "2.0.0"
__author__ = "AstroLab Team"

# Main exports
__all__ = [
    # Dashboard
    "AstroLabDashboard",
    "create_astrolab_dashboard",
    "create_mobile_dashboard",
    "create_presentation_mode",
    
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
    
    # Settings
    "ui_experiment_settings",
    "ui_data_settings",
    "ui_model_settings",
    "ui_visualization_settings",
    
    # Utilities
    "save_config",
    "load_config",
    "get_config_files",
    "handle_dashboard_events",
]
