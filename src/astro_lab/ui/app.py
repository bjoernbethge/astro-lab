"""
AstroLab UI Application
======================

Functional astronomical data analysis dashboard.
"""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="full")


@app.cell
def init():
    """Initialize imports and state."""
    import marimo as mo
    import torch

    # Import components
    from astro_lab.ui.components import state, system_info

    # Import pages
    from astro_lab.ui.pages import (
        analysis,
        config,
        cosmic_web,
        data,
        training,
        visualization,
    )

    # Create application state
    app_state = state.create_state()

    return (
        mo,
        torch,
        data,
        cosmic_web,
        analysis,
        visualization,
        training,
        config,
        state,
        system_info,
        app_state,
    )


@app.cell
def header(mo, system_info):
    """Create application header."""
    # Get system info
    sys_info = system_info.get_system_info()

    # Create header
    header = mo.vstack(
        [
            mo.md("# 🌌 **AstroLab**"),
            mo.md("*Astro GNN Laboratory for Cosmic Web Exploration*"),
            mo.hstack(
                [
                    mo.stat("GPU", "✅" if sys_info["gpu_available"] else "❌"),
                    mo.stat(
                        "Memory",
                        f"{sys_info['memory_used']:.1f}/{sys_info['memory_total']:.1f} GB",
                    ),
                    mo.stat("Surveys", str(len(sys_info["available_surveys"]))),
                ]
            ),
        ]
    )

    return header, sys_info


@app.cell
def navigation(mo):
    """Create navigation."""
    tabs = mo.ui.tabs(
        {
            "📡 Data": "data",
            "🌌 Cosmic Web": "cosmic_web",
            "🔬 Analysis": "analysis",
            "🎨 Visualization": "visualization",
            "🏋️ Training": "training",
            "⚙️ Config": "config",
        }
    )

    return (tabs,)


@app.cell
def main_content(
    mo, tabs, data, cosmic_web, analysis, visualization, training, config, app_state
):
    """Main content routing."""

    # Get current tab
    current_tab = tabs.value

    # Route to appropriate page
    if current_tab == "data":
        content = data.create_page(app_state)
    elif current_tab == "cosmic_web":
        content = cosmic_web.create_page(app_state)
    elif current_tab == "analysis":
        content = analysis.create_page(app_state)
    elif current_tab == "visualization":
        content = visualization.create_page(app_state)
    elif current_tab == "training":
        content = training.create_page(app_state)
    elif current_tab == "config":
        content = config.create_page(app_state)
    else:
        content = mo.md("# Page not found")

    return content, current_tab


@app.cell
def footer(mo):
    """Create footer."""
    footer = mo.hstack(
        [
            mo.md("*AstroLab v1.0.0*"),
            mo.md("•"),
            mo.md("[GitHub](https://github.com/bjoernbethge/astro-lab)"),
            mo.md("•"),
            mo.md("*© 2025 Astro Graph Agent Team*"),
        ],
        justify="center",
    )

    return (footer,)


@app.cell
def layout(mo, header, navigation, main_content, footer):
    """Assemble the application."""

    # Main layout
    app_layout = mo.vstack(
        [header, mo.md("---"), navigation, main_content, mo.md("---"), footer], gap=2
    )

    return (app_layout,)


@app.cell
def display(app_layout):
    """Display the application."""
    return app_layout


@app.cell
def css(mo):
    """CSS customization."""
    mo.Html("""
    <style>
    /* AstroLab Theme */
    :root {
        --astro-gold: #FFD700;
        --astro-blue: #4169E1;
    }
    
    h1, h2 {
        background: linear-gradient(45deg, var(--astro-gold), var(--astro-blue));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    button[data-kind="primary"], button[data-kind="success"] {
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }
    
    button[data-kind="primary"]:hover, button[data-kind="success"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """)

    return


if __name__ == "__main__":
    app.run()
