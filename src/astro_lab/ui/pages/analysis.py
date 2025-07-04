"""
Analysis Page
============

Simple analysis page with working functionality.
"""

import marimo as mo
from astro_lab.ui.components.analyzer import create_analyzer, run_cosmic_web_analysis
from astro_lab.ui.components.viz import create_cosmic_web_viz


def create_page(app_state=None):
    """Create the analysis page."""

    # Check if data is loaded
    if app_state is None or app_state.loaded_data is None:
        return mo.vstack([
            mo.md("## 🔬 Analysis"),
            mo.callout(
                "Please load data first!",
                kind="warning"
            )
        ])

    # Create analyzer
    analyzer_ui, analyzer_status, analysis_params = create_analyzer()

    # Results section
    results_section = mo.md("")

    if analysis_params:
        # Run actual analysis
        try:
            results = run_cosmic_web_analysis(
                app_state.loaded_data,
                scales=analysis_params["scales"],
                min_samples=analysis_params["min_samples"]
            )

            # Update app state
            app_state.analysis_result = results

            # Show results
            results_info = mo.vstack([
                mo.md("### 📊 Results"),
                mo.stat("Total objects", f"{results.get('n_objects', 0):,}"),
                mo.stat("Analysis complete", "✅"),
            ])

            # Create visualization
            viz = create_cosmic_web_viz(app_state.loaded_data, results)

            results_section = mo.vstack([results_info, viz])

        except Exception as e:
            results_section = mo.callout(
                f"Analysis error: {str(e)}",
                kind="danger"
            )

    # Layout
    return mo.vstack([
        mo.md("## 🔬 Analysis"),
        analyzer_ui,
        analyzer_status,
        results_section
    ])
