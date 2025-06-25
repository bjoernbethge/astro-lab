"""
AstroLab Dashboard
=================

Clean marimo dashboard interface for AstroLab.
Integrated with AstroLab widget system for full functionality.
"""

from typing import Any, Dict, Optional

import marimo as mo

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
from .settings import (
    handle_config_actions,
    ui_config_loader,
    ui_config_status,
    ui_data_paths,
    ui_experiment_manager,
    ui_model_selector,
    ui_survey_selector,
    ui_training_selector,
)


class AstroLabDashboard:
    """Complete AstroLab dashboard with integrated components."""

    def __init__(self):
        """Initialize the dashboard."""
        self.current_data = None
        self.status_message = "Ready"

    def create_main_tabs(self) -> mo.ui.tabs:
        """Create main dashboard tabs."""

        # Data & Analysis Tab
        data_tab = mo.vstack(
            [
                mo.md("## 📊 Data Management"),
                ui_data_controls(),
                mo.md("## 🔬 Analysis"),
                ui_analysis_controls(),
                mo.md("## 📈 Visualization"),
                ui_visualization_controls(),
            ]
        )

        #  Analysis Tab (with graph functionality)
        advanced_tab = mo.vstack(
            [
                mo.md("## 🕸️ Graph Analysis"),
                ui_graph_controls(),
                mo.md("## 🤖 Model Training"),
                ui_model_controls(),
                mo.md("## 📊 System Status"),
                ui_system_status(),
            ]
        )

        # Configuration Tab
        config_tab = mo.vstack(
            [
                mo.md("## 🚀 Quick Setup"),
                ui_quick_setup(),
                mo.md("## 📂 Configuration"),
                ui_config_loader(),
                mo.md("## 📊 Survey Selection"),
                ui_survey_selector(),
                mo.md("## 🤖 Model Selection"),
                ui_model_selector(),
            ]
        )

        # Management Tab
        management_tab = mo.vstack(
            [
                mo.md("## 🧪 Experiment Management"),
                ui_experiment_manager(),
                mo.md("## 📁 Data Paths"),
                ui_data_paths(),
                mo.md("## ℹ️ Configuration Status"),
                ui_config_status(),
            ]
        )

        return mo.ui.tabs(
            {
                "🚀 Main": data_tab,
                "🔬 ": advanced_tab,
                "⚙️ Config": config_tab,
                "🔧 Manage": management_tab,
            }
        )

    def create_sidebar(self) -> mo.vstack:
        """Create dashboard sidebar."""
        widget_status = "🟢 Available" if WIDGETS_AVAILABLE else "🔴 Not Available"

        return mo.vstack(
            [
                mo.md("# 🌟 AstroLab"),
                mo.md("* Astronomical Data Analysis*"),
                mo.md("---"),
                mo.md("### ⚡ Quick Actions"),
                ui_quick_actions(),
                mo.md("### 📊 System Status"),
                ui_system_status(),
                mo.md(f"**AstroLab Widgets:** {widget_status}"),
                mo.md("---"),
                mo.md("### 📚 Resources"),
                mo.md("""
- [Documentation](https://astro-lab.readthedocs.io)
- [GitHub](https://github.com/astro-lab/astro-lab)
- [Examples](examples/)
            """),
            ]
        )

    def create_welcome_screen(self) -> mo.vstack:
        """Create welcome screen."""
        features_text = """
## Modern Astronomical Data Analysis Platform

AstroLab provides comprehensive tools for astronomical data analysis,
machine learning, and interactive visualization.

### 🚀 Getting Started

1. **Configure**: Set up your experiment and data paths
2. **Load Data**: Choose a survey and load astronomical data
3. **Visualize**: Create interactive plots and visualizations
4. **Analyze**: Run clustering, classification, and analysis
5. **Model**: Train machine learning models on your data

### 📊 Supported Surveys
- **Gaia**: European Space Agency's stellar survey
- **SDSS**: Sloan Digital Sky Survey
- **NSA**: NASA-Sloan Atlas
- **LINEAR**: Linear Asteroid Survey
- **TNG50**: IllustrisTNG simulation data

### 🧰 Available Tools
- Interactive plotting with Plotly, Matplotlib, Bokeh"""

        if WIDGETS_AVAILABLE:
            features_text += """
- ** 3D visualization** with Open3D, PyVista, Blender
- **GPU-accelerated analysis** and clustering
- **Graph-based analysis** with PyTorch Geometric
- **High-performance neighbor finding**"""
        else:
            features_text += """
- Graph-based analysis and clustering
- Neural networks and machine learning"""

        features_text += """
- GPU acceleration support
            """

        return mo.vstack(
            [
                mo.md("# 🌟 Welcome to AstroLab"),
                mo.md(features_text),
                mo.md("### 🎯 Quick Start"),
                mo.hstack(
                    [
                        mo.ui.button(label="📊 Load Sample Data"),
                        mo.ui.button(label="🎨 Create Plot"),
                        mo.ui.button(label="🤖 Train Model"),
                    ]
                ),
            ]
        )

    def create_full_dashboard(self) -> mo.vstack:
        """Create the complete dashboard layout."""
        header = mo.md("# 🌟 AstroLab Dashboard")

        welcome = self.create_welcome_screen()
        main_tabs = self.create_main_tabs()
        sidebar = self.create_sidebar()

        # Main content area
        main_content = mo.hstack(
            [
                mo.vstack([welcome, main_tabs]),
                sidebar,
            ]
        )

        return mo.vstack(
            [
                header,
                mo.md("---"),
                main_content,
            ]
        )

    def handle_interactions(self, components: Dict[str, Any]) -> Optional[str]:
        """Handle all dashboard interactions."""

        # Handle component actions
        component_result = handle_component_actions(components)
        if component_result:
            self.status_message = component_result
            return component_result

        # Handle config actions
        config_result = handle_config_actions(components)
        if config_result:
            self.status_message = "✅ Configuration action completed"
            return self.status_message

        return None


# Global dashboard instance
dashboard = AstroLabDashboard()


def create_astrolab_dashboard() -> mo.vstack:
    """Create the complete AstroLab dashboard."""
    return dashboard.create_full_dashboard()


def create_minimal_dashboard() -> mo.vstack:
    """Create a minimal dashboard for quick analysis."""
    return mo.vstack(
        [
            mo.md("# 🌟 AstroLab - Quick Analysis"),
            mo.md("## 📊 Data Loading"),
            ui_data_controls(),
            mo.md("## 📈 Visualization"),
            ui_visualization_controls(),
            mo.md("## 🔬 Analysis"),
            ui_analysis_controls(),
            mo.md("## ⚡ Quick Actions"),
            ui_quick_actions(),
        ]
    )


def create_config_dashboard() -> mo.vstack:
    """Create a configuration-focused dashboard."""
    return mo.vstack(
        [
            mo.md("# ⚙️ AstroLab Configuration"),
            mo.ui.tabs(
                {
                    "🚀 Quick": ui_quick_setup(),
                    "📂 Config": ui_config_loader(),
                    "📊 Survey": ui_survey_selector(),
                    "🤖 Model": ui_model_selector(),
                    "🏋️ Training": ui_training_selector(),
                    "🧪 Experiment": ui_experiment_manager(),
                    "📁 Paths": ui_data_paths(),
                    "ℹ️ Status": ui_config_status(),
                }
            ),
        ]
    )


def create_analysis_dashboard() -> mo.vstack:
    """Create an analysis-focused dashboard with widget integration."""
    return mo.vstack(
        [
            mo.md("# 🔬 AstroLab Analysis Dashboard"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("### 📊 Data"),
                            ui_data_controls(),
                            mo.md("### 🔬 Analysis"),
                            ui_analysis_controls(),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("### 📈 Visualization"),
                            ui_visualization_controls(),
                            mo.md("### 🕸️ Graph Analysis"),
                            ui_graph_controls(),
                        ]
                    ),
                ]
            ),
            mo.md("## 🤖 Model Training"),
            ui_model_controls(),
            mo.md("## 📊 Status & Actions"),
            mo.hstack(
                [
                    ui_system_status(),
                    ui_quick_actions(),
                ]
            ),
        ]
    )


def create_widget_showcase() -> mo.vstack:
    """Create a dashboard showcasing AstroLab widget capabilities."""
    if not WIDGETS_AVAILABLE:
        return mo.vstack(
            [
                mo.md("# ❌ AstroLab Widgets Not Available"),
                mo.md(
                    "Please install the required dependencies for widget functionality."
                ),
            ]
        )

    return mo.vstack(
        [
            mo.md("# 🌟 AstroLab Widget Showcase"),
            mo.md("*Demonstrating advanced visualization and analysis capabilities*"),
            mo.md("## 🎨  Visualization"),
            mo.md("**Backends:** Open3D, PyVista, Blender integration"),
            ui_visualization_controls(),
            mo.md("## 🕸️ Graph Analysis"),
            mo.md(
                "**Features:** GPU-accelerated neighbor finding, PyTorch Geometric integration"
            ),
            ui_graph_controls(),
            mo.md("## 🔬  Analysis"),
            mo.md(
                "**Capabilities:** GPU clustering, density analysis, structure analysis"
            ),
            ui_analysis_controls(),
            mo.md("## 📊 System Information"),
            ui_system_status(),
        ]
    )


__all__ = [
    "AstroLabDashboard",
    "dashboard",
    "create_astrolab_dashboard",
    "create_minimal_dashboard",
    "create_config_dashboard",
    "create_analysis_dashboard",
    "create_widget_showcase",
]
