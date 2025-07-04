"""
Visualization Page
=================

Visualization page using the proper visualizer components.
"""

import marimo as mo
from astro_lab.ui.components.visualizer import UniversalVisualizer


def create_page(app_state=None):
    """Create the visualization page."""

    # Check if data is loaded
    if app_state is None or app_state.loaded_data is None:
        return mo.vstack([
            mo.md("## 🎨 Visualization"),
            mo.callout(
                "Please load data first in the Data tab to create visualizations.",
                kind="warning"
            ),
            mo.ui.button("Go to Data Tab", kind="primary")
        ])

    # Create visualizer
    visualizer = UniversalVisualizer()

    # Control panel
    controls = visualizer.create_control_panel()

    # Create visualization button
    create_btn = mo.ui.button("🎨 Create Visualization", kind="success", full_width=True)

    # Visualization display
    viz_display = mo.md("")
    status = mo.md("")

    if create_btn.value:
        try:
            # Get control values (simplified for now)
            backend = "cosmograph"  # Default
            style = "cosmic_web"

            # Use analysis results if available
            analysis_results = None
            if hasattr(app_state, 'analysis_result'):
                analysis_results = app_state.analysis_result

            # Create visualization
            viz = visualizer.visualize(
                app_state.loaded_data,
                backend=backend,
                style=style,
                analysis_results=analysis_results,
                node_size=2.0,
                node_opacity=0.8
            )

            viz_display = viz
            status = mo.callout(
                f"✅ Created {backend} visualization with {app_state.n_objects:,} objects",
                kind="success"
            )

            # Update app state
            if app_state:
                app_state.visualization_result = viz
                app_state.viz_backend = backend

        except Exception as e:
            status = mo.callout(
                f"❌ Visualization error: {str(e)}",
                kind="danger"
            )

    # Layout
    return mo.vstack([
        mo.md("## 🎨 Visualization"),
        mo.hstack([
            # Left: Controls
            mo.vstack([
                controls,
                create_btn
            ], align="stretch"),
            # Right: Visualization
            mo.vstack([
                status,
                viz_display
            ], align="stretch")
        ], widths=[1, 2])
    ])
