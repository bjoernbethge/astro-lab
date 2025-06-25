"""
AstroLab UI Module
=================

Clean marimo-based user interface for AstroLab.
Integrated with the AstroLab widget system for advanced functionality.

## Quick Start

```python
import marimo as mo
from astro_lab.ui import create_astrolab_dashboard

# Create full dashboard
dashboard = create_astrolab_dashboard()

# Or use individual components
from astro_lab.ui import (
    ui_config_loader,
    ui_quick_setup,
    ui_model_selector,
    ui_graph_controls,
    handle_component_actions,
    WIDGETS_AVAILABLE
)

config_loader = ui_config_loader()
quick_setup = ui_quick_setup()

# Check if advanced widget features are available
if WIDGETS_AVAILABLE:
    graph_controls = ui_graph_controls()
```

## Integration

- **ConfigLoader**: Uses real AstroLab configuration system
- **data_config**: Integrates with path management
- **Model Configs**: Works with actual model configurations
- **Training Configs**: Supports predefined training setups
- **AstroLab Widgets**: Advanced visualization and analysis (if available)
"""

import logging
from typing import Any

from .components import (
    WIDGETS_AVAILABLE,
    handle_component_actions,
    ui_analysis_controls,
    ui_data_controls,
    ui_graph_controls,
    ui_model_controls,
    ui_quick_actions,
    ui_quick_setup,
    ui_system_status,
    ui_visualization_controls,
)
from .dashboard import (
    AstroLabDashboard,
    create_analysis_dashboard,
    create_astrolab_dashboard,
    create_config_dashboard,
    create_minimal_dashboard,
    create_widget_showcase,
    dashboard,
)
from .settings import (
    UIConfigManager,
    handle_config_actions,
    ui_config,
    ui_config_loader,
    ui_config_status,
    ui_data_paths,
    ui_experiment_manager,
    ui_model_selector,
    ui_survey_selector,
    ui_training_selector,
)

# Configure logging
logger = logging.getLogger(__name__)

# Version and metadata
__version__ = "2.0.0"
__description__ = "Clean marimo UI for AstroLab with widget integration"


def create_dashboard(dashboard_type: str = "full") -> Any:
    """
    Create a dashboard of specified type.

    Args:
        dashboard_type: One of "full", "minimal", "config", "analysis", "widgets"

    Returns:
        Marimo UI component

    Raises:
        ValueError: If dashboard_type is not supported
    """
    dashboard_factories = {
        "full": create_astrolab_dashboard,
        "minimal": create_minimal_dashboard,
        "config": create_config_dashboard,
        "analysis": create_analysis_dashboard,
        "widgets": create_widget_showcase,
    }

    if dashboard_type not in dashboard_factories:
        raise ValueError(
            f"Unknown dashboard type: {dashboard_type}. Supported: {list(dashboard_factories.keys())}"
        )

    return dashboard_factories[dashboard_type]()


# Export all public components
__all__ = [
    # Core classes
    "UIConfigManager",
    "ui_config",
    "AstroLabDashboard",
    "dashboard",
    # Configuration components
    "ui_config_loader",
    "ui_data_paths",
    "ui_survey_selector",
    "ui_model_selector",
    "ui_training_selector",
    "ui_experiment_manager",
    "ui_config_status",
    # UI Components
    "ui_quick_setup",
    "ui_data_controls",
    "ui_visualization_controls",
    "ui_analysis_controls",
    "ui_model_controls",
    "ui_graph_controls",
    "ui_system_status",
    "ui_quick_actions",
    # Event handlers
    "handle_config_actions",
    "handle_component_actions",
    # Dashboard factories
    "create_astrolab_dashboard",
    "create_minimal_dashboard",
    "create_config_dashboard",
    "create_analysis_dashboard",
    "create_widget_showcase",
    "create_dashboard",
    # Widget availability
    "WIDGETS_AVAILABLE",
]

# Supported dashboard types
DASHBOARD_TYPES = ["full", "minimal", "config", "analysis", "widgets"]

# UI Theme options
UI_THEMES = ["light", "dark", "auto"]

# Backend options
VISUALIZATION_BACKENDS = ["plotly", "matplotlib", "bokeh"]
if WIDGETS_AVAILABLE:
    VISUALIZATION_BACKENDS.extend(["open3d", "pyvista", "blender"])

MODEL_BACKENDS = ["pytorch"]

logger.info("🌟 AstroLab UI Module loaded successfully!")
logger.info(f"   📊 Dashboard types: {', '.join(DASHBOARD_TYPES)}")
logger.info(f"   🎨 Visualization backends: {', '.join(VISUALIZATION_BACKENDS)}")
logger.info(f"   🤖 Model backends: {', '.join(MODEL_BACKENDS)}")
logger.info("   🚀 Ready for interactive astronomical data analysis!")
