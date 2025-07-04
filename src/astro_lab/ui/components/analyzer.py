"""
Analyzer Component
=================

Simple analysis interface for AstroLab.
"""

import marimo as mo
import numpy as np

from astro_lab.data.analysis.cosmic_web import analyze_cosmic_web


def create_analyzer():
    """Create simple analysis interface."""

    # Analysis parameters
    scales = mo.ui.array(
        [
            mo.ui.slider(5, 100, 10, label="Small scale (pc)"),
            mo.ui.slider(10, 200, 50, label="Medium scale (pc)"),
            mo.ui.slider(50, 500, 200, label="Large scale (pc)"),
        ]
    )

    min_samples = mo.ui.slider(3, 20, 5, label="Min samples")

    method = mo.ui.dropdown(
        options=["dbscan", "hdbscan", "optics"], value="dbscan", label="Algorithm"
    )

    # Run button
    run_btn = mo.ui.button(label="Run Analysis", kind="success")

    # UI
    ui = mo.vstack(
        [mo.md("### 🔬 Cosmic Web Analysis"), scales, min_samples, method, run_btn]
    )

    # Results
    result = None
    status = mo.md("")

    if run_btn.value:
        # Get scales values
        scale_values = [s.value for s in scales]

        # Run analysis placeholder
        status = mo.callout(
            f"✅ Analysis would run with scales: {scale_values}, min_samples: {min_samples.value}",
            kind="success",
        )

        # Would return actual results here
        result = {
            "scales": scale_values,
            "min_samples": min_samples.value,
            "method": method.value,
        }

    return ui, status, result


def run_cosmic_web_analysis(data, scales, min_samples=5):
    """Actually run cosmic web analysis."""
    try:
        # Extract coordinates
        if hasattr(data, "select"):
            # DataFrame
            coords = data.select(["x", "y", "z"]).to_numpy()
        else:
            # Assume numpy array
            coords = np.array(data)

        # Run analysis
        results = analyze_cosmic_web(coordinates=coords, scales=scales)

        return results
    except Exception as e:
        return {"error": str(e)}
