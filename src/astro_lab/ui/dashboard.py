"""
AstroLab Dashboard - Modern Marimo UI
====================================

A modern, interactive dashboard for astronomical data analysis using
the latest Marimo features (2025) and actual AstroLab modules.
"""

import marimo as mo
from typing import Any, Dict, Optional
import logging

# Import the new modular UI components
from .modules import (
    # Data modules
    data_explorer,
    data_loader,
    catalog_manager,
    # Training modules
    training_dashboard,
    model_selector,
    experiment_tracker,
    # Visualization modules
    plot_creator,
    graph_creator,
    clustering_visualizer,
    cosmograph_viewer,
    # Analysis modules
    analysis_panel,
    clustering_tool,
    statistics_viewer,
    subgraph_sampler,
    # Monitoring modules
    system_monitor,
    gpu_monitor,
    mlflow_dashboard,
    training_monitor,
)

# Import settings from ui folder
from .settings import (
    ui_experiment_settings as experiment_settings,
    ui_data_settings as data_settings,
    ui_model_settings as model_settings,
    ui_visualization_settings as visualization_settings,
)

# Create advanced settings function
def advanced_settings() -> mo.Html:
    """Advanced settings panel."""
    return mo.vstack([
        mo.md("## ⚙️ Advanced Settings"),
        mo.md("*Advanced configuration options are available in the individual setting panels.*"),
        mo.callout(
            "Use the other settings tabs for specific configuration options.",
            kind="info"
        ),
    ])

logger = logging.getLogger(__name__)


class AstroLabDashboard:
    """Modern AstroLab dashboard with reactive UI components using actual AstroLab modules."""
    
    def __init__(self):
        """Initialize the dashboard."""
        # mo.state returns (value, setter) tuple
        self.state, self.set_state = mo.state({
            "current_tab": "data",
            "theme": "dark",
            "notifications": [],
        })
        
    def create_header(self) -> mo.Html:
        """Create modern header with navigation."""
        return mo.Html("""
        <div style="background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                    padding: 1.5rem 2rem; border-bottom: 3px solid #667eea;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <h1 style="margin: 0; font-size: 2.5rem; color: white;">
                        🌟 AstroLab
                    </h1>
                    <span style="color: #e0e0e0; font-size: 1rem;">
                        Astronomical Data Analysis Platform
                    </span>
                </div>
                <div style="display: flex; gap: 1rem; align-items: center;">
                    <span style="color: white;">v2.0</span>
                    <button style="background: rgba(255,255,255,0.2); color: white; 
                                   border: none; padding: 0.5rem 1rem; border-radius: 8px; 
                                   cursor: pointer;">
                        📚 Docs
                    </button>
                </div>
            </div>
        </div>
        """)
        
    def create_main_interface(self) -> mo.Html:
        """Create the main interface with tabs using actual AstroLab modules."""
        
        # Data tab with actual data modules
        data_tab = mo.vstack([
            mo.tabs({
                "📂 Explorer": data_explorer(),
                "🔄 Loader": data_loader(),
                "📁 Catalogs": catalog_manager(),
            })
        ])
        
        # Training tab with actual training modules
        training_tab = mo.vstack([
            mo.tabs({
                "🚀 Dashboard": training_dashboard(),
                "🎯 Models": model_selector(),
                "📊 Experiments": experiment_tracker(),
            })
        ])
        
        # Visualization tab with actual visualization modules
        viz_tab = mo.vstack([
            mo.tabs({
                "🎨 Plots": plot_creator(),
                "🕸️ Graphs": graph_creator(),
                "🎯 Clustering": clustering_visualizer(),
                "🌌 Cosmograph": cosmograph_viewer(),
            })
        ])
        
        # Analysis tab with actual analysis modules
        analysis_tab = mo.vstack([
            mo.tabs({
                "🔬 Analysis": analysis_panel(),
                "🎯 Clustering": clustering_tool(),
                "📊 Statistics": statistics_viewer(),
                "🎲 Subgraphs": subgraph_sampler(),
            })
        ])
        
        # Monitoring tab with actual monitoring modules
        monitor_tab = mo.vstack([
            mo.tabs({
                "💻 System": system_monitor(),
                "🎮 GPU": gpu_monitor(),
                "📈 Training": training_monitor(),
                "🔍 MLflow": mlflow_dashboard(),
            })
        ])
        
        # Settings tab with actual settings modules
        settings_tab = mo.vstack([
            mo.tabs({
                "🧪 Experiment": experiment_settings(),
                "📊 Data": data_settings(),
                "🤖 Models": model_settings(),
                "🎨 Visualization": visualization_settings(),
                "⚙️ Advanced": advanced_settings(),
            })
        ])
        
        # Create main tabs
        main_tabs = mo.ui.tabs({
            "📊 Data": data_tab,
            "🤖 Training": training_tab,
            "🎨 Visualization": viz_tab,
            "🔬 Analysis": analysis_tab,
            "📈 Monitoring": monitor_tab,
            "⚙️ Settings": settings_tab,
        })
        
        return main_tabs
        
    def create_sidebar(self) -> mo.Html:
        """Create sidebar with quick status and actions."""
        
        # Quick status
        status_card = mo.Html("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;">
            <h3 style="margin: 0 0 0.5rem 0;">📊 Quick Status</h3>
            <div style="font-size: 0.9rem; line-height: 1.6;">
                <div>🟢 System: Ready</div>
                <div>📂 Data: Not loaded</div>
                <div>🤖 Model: Not trained</div>
                <div>🎮 GPU: Available</div>
            </div>
        </div>
        """)
        
        # Quick actions
        actions_card = mo.vstack([
            mo.md("### ⚡ Quick Actions"),
            mo.ui.button("🌟 Load Gaia Sample", full_width=True),
            mo.ui.button("🔭 Load SDSS Sample", full_width=True),
            mo.ui.button("🎨 Quick 3D Plot", full_width=True),
            mo.ui.button("🤖 Train Fast Model", full_width=True),
        ])
        
        # Recent activity
        activity_card = mo.Html("""
        <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px;">
            <h3 style="margin: 0 0 0.5rem 0;">📝 Recent Activity</h3>
            <div style="font-size: 0.85rem; line-height: 1.5;">
                <div>• Dashboard initialized</div>
                <div>• GPU detected: RTX 4070</div>
                <div>• MLflow connected</div>
            </div>
        </div>
        """)
        
        return mo.Html(f"""
        <div style="width: 300px; padding: 1.5rem; background: #1a1a2e; height: 100%; 
                    overflow-y: auto; border-left: 1px solid #333;">
            {status_card._repr_html_()}
            {actions_card._repr_html_()}
            {activity_card._repr_html_()}
        </div>
        """)
        
    def create_footer(self) -> mo.Html:
        """Create footer with system info."""
        return mo.Html("""
        <div style="padding: 1rem 2rem; background: #16213e; border-top: 1px solid #333;
                    display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; gap: 2rem; font-size: 0.9rem;">
                <span>🐍 Python 3.12</span>
                <span>🔥 PyTorch 2.5</span>
                <span>⚡ Lightning 2.5</span>
                <span>📊 Marimo 0.14</span>
            </div>
            <div style="display: flex; gap: 1rem; align-items: center;">
                <a href="https://github.com/astro-lab" style="color: #667eea;">GitHub</a>
                <span>|</span>
                <a href="https://docs.astro-lab.io" style="color: #667eea;">Documentation</a>
            </div>
        </div>
        """)
        
    def create_full_dashboard(self) -> mo.Html:
        """Create the complete modern dashboard."""
        
        # Add CSS
        css = mo.Html("""
        <style>
            body {
                margin: 0;
                background: #0f0f23;
                color: #e0e0e0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            /* Tab styling */
            .marimo-tabs {
                background: transparent !important;
            }
            
            .marimo-tab-button {
                color: #a0a0a0 !important;
                border-bottom: 2px solid transparent !important;
                transition: all 0.2s !important;
            }
            
            .marimo-tab-button:hover {
                color: #667eea !important;
                background: rgba(102, 126, 234, 0.1) !important;
            }
            
            .marimo-tab-button[data-selected="true"] {
                color: #667eea !important;
                border-bottom-color: #667eea !important;
            }
            
            /* Button styling */
            button {
                transition: all 0.2s;
            }
            
            button:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
            
            /* Card styling */
            .card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            /* Scrollbar styling */
            ::-webkit-scrollbar {
                width: 8px;
                height: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
            }
            
            ::-webkit-scrollbar-thumb {
                background: rgba(102, 126, 234, 0.3);
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: rgba(102, 126, 234, 0.5);
            }
        </style>
        """)
        
        # Main layout components
        header = self.create_header()
        main_interface = self.create_main_interface()
        sidebar = self.create_sidebar()
        footer = self.create_footer()
        
        # Combine into full layout
        dashboard = mo.Html(f"""
        <div style="height: 100vh; display: flex; flex-direction: column; background: #0f0f23;">
            {header._repr_html_()}
            <div style="flex: 1; display: flex; overflow: hidden;">
                <div style="flex: 1; padding: 2rem; overflow-y: auto;">
                    {main_interface._repr_html_()}
                </div>
                {sidebar._repr_html_()}
            </div>
            {footer._repr_html_()}
        </div>
        """)
        
        return mo.vstack([css, dashboard])


# Create singleton instance
_dashboard = AstroLabDashboard()


def create_astrolab_dashboard() -> mo.Html:
    """Create the modern AstroLab dashboard."""
    return _dashboard.create_full_dashboard()


def create_compact_dashboard() -> mo.Html:
    """Create a compact version for smaller screens."""
    return mo.vstack([
        mo.md("# 🌟 AstroLab Compact"),
        mo.tabs({
            "📊 Data": mo.vstack([
                data_explorer(),
                data_loader(),
            ]),
            "🤖 Training": training_dashboard(),
            "🎨 Viz": plot_creator(),
            "🔬 Analysis": analysis_panel(),
            "📈 Monitor": mo.vstack([
                system_monitor(),
                gpu_monitor(),
            ]),
        })
    ])


def create_demo_dashboard() -> mo.Html:
    """Create a demo dashboard for presentations."""
    return mo.vstack([
        mo.Html("""
        <div style="text-align: center; padding: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
            <h1 style="font-size: 3rem; margin: 0;">🌟 AstroLab Demo</h1>
            <p style="font-size: 1.5rem; margin-top: 1rem;">
                Modern Astronomical Data Analysis with Graph Neural Networks
            </p>
        </div>
        """),
        mo.tabs({
            "1️⃣ Load Data": data_explorer(),
            "2️⃣ Visualize": plot_creator(),
            "3️⃣ Train Model": training_dashboard(),
            "4️⃣ Analyze": analysis_panel(),
        }),
    ])


__all__ = [
    "AstroLabDashboard",
    "create_astrolab_dashboard",
    "create_compact_dashboard",
    "create_demo_dashboard",
]
