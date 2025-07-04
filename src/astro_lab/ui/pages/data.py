"""
Data Page
=========

Simple data loading page with Marimo.
"""

import marimo as mo

from astro_lab.ui.components.data_loader import create_data_loader
from astro_lab.ui.components.viz import create_plotly_viz


def create_page(app_state=None):
    """Create the data loading page."""

    # Data loader
    loader_ui, loader_status, loaded_data = create_data_loader()

    # Update app state if data was loaded
    if loaded_data is not None and app_state:
        app_state.loaded_data = loaded_data
        app_state.n_objects = len(loaded_data) if hasattr(loaded_data, "__len__") else 0
        app_state.status = f"Loaded {app_state.n_objects:,} objects"

    # Preview section
    preview = mo.md("")
    if app_state and app_state.loaded_data is not None:
        # Show basic info
        info = mo.vstack(
            [
                mo.md("### 📊 Data Loaded"),
                mo.stat("Objects", f"{app_state.n_objects:,}"),
                mo.stat("Type", type(app_state.loaded_data).__name__),
            ]
        )

        # Try to create preview viz
        try:
            viz = create_plotly_viz(app_state.loaded_data)
            preview = mo.vstack([info, viz])
        except Exception:
            preview = info

    # Layout
    return mo.vstack([mo.md("## 📡 Data Loading"), loader_ui, loader_status, preview])
