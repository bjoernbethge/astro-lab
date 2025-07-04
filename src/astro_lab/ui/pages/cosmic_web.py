"""
Cosmic Web Page
==============

Specialized cosmic web analysis page.
"""

import marimo as mo

from astro_lab.ui.components.analyzer import run_cosmic_web_analysis
from astro_lab.ui.components.viz import (
    create_blender_cosmic_web_scene,
    create_cosmic_web_viz,
)


def create_page(app_state=None):
    """Create the cosmic web analysis page."""

    # Check if data is loaded
    if (
        app_state is None
        or not hasattr(app_state, "loaded_data")
        or app_state.loaded_data is None
    ):
        return mo.vstack(
            [
                mo.md("## 🌌 Cosmic Web Analysis"),
                mo.callout(
                    "Please load data first in the Data tab to run cosmic web analysis.",
                    kind="warning",
                ),
                mo.ui.button("Go to Data Tab", kind="primary"),
            ]
        )

    # Multi-scale parameters
    scales = mo.ui.array(
        [
            mo.ui.slider(1, 100, 5, label="Small scale (pc)"),
            mo.ui.slider(10, 200, 25, label="Medium scale (pc)"),
            mo.ui.slider(20, 500, 100, label="Large scale (pc)"),
            mo.ui.slider(50, 1000, 500, label="Cosmic scale (pc)"),
        ]
    )

    # Clustering parameters
    min_samples = mo.ui.slider(3, 20, 5, label="Min samples per cluster")

    method = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN - Density-based clustering",
            "hdbscan": "HDBSCAN - Hierarchical density clustering",
            "optics": "OPTICS - Ordering points clustering",
        },
        value="dbscan",
        label="Clustering Algorithm",
    )

    # Visualization options
    show_clusters = mo.ui.checkbox(True, label="Show clusters")
    show_filaments = mo.ui.checkbox(True, label="Show filaments")
    show_voids = mo.ui.checkbox(True, label="Show voids")

    # Run button
    run_btn = mo.ui.button("🚀 Analyze Cosmic Web", kind="success", full_width=True)

    # Results section
    results_section = mo.md("")
    viz_section = mo.md("")

    if run_btn.value:
        try:
            # Get scale values
            scale_values = [s.value for s in scales.value]

            # Run analysis
            results = run_cosmic_web_analysis(
                app_state.loaded_data,
                scales=scale_values,
                min_samples=min_samples.value,
            )

            # Update app state
            app_state.analysis_result = results

            # Display results
            results_info = mo.vstack(
                [
                    mo.md("### 📊 Cosmic Web Structure Found"),
                    mo.hstack(
                        [
                            mo.stat("Objects", f"{results.get('n_objects', 0):,}"),
                            mo.stat(
                                "Structures", f"{len(scale_values)} scales analyzed"
                            ),
                        ]
                    ),
                ]
            )

            # Multi-scale results
            scale_results = []
            for i, scale in enumerate(scale_values):
                scale_results.append(
                    mo.md(f"**{scale} pc scale**: Analysis complete ✅")
                )

            results_section = mo.vstack(
                [results_info, mo.md("### Multi-Scale Analysis"), *scale_results]
            )

            # Create visualization
            viz = create_cosmic_web_viz(app_state.loaded_data, results)
            viz_section = mo.vstack([mo.md("### 🎨 Cosmic Web Visualization"), viz])

        except Exception as e:
            results_section = mo.callout(f"Analysis error: {str(e)}", kind="danger")

    # Survey dropdown
    survey_dropdown = mo.ui.dropdown(
        options={
            "gaia": "Gaia (Stars)",
            "sdss": "SDSS (Galaxies)",
            "nsa": "NSA (Galaxies)",
            "tng50": "TNG50 (Simulation)",
            "exoplanet": "Exoplanet (Stars)",
        },
        value="gaia",
        label="Survey for Blender Scene",
    )
    blender_btn = mo.ui.button("🪐 Generate Blender Scene", kind="success")
    blender_status = mo.md("")
    if blender_btn.value:
        create_blender_cosmic_web_scene(
            survey_dropdown.value, max_samples=10000, render=True
        )
        blender_status = mo.callout("Blender scene generated!", kind="success")

    # Layout
    return mo.vstack(
        [
            mo.md("## 🌌 Cosmic Web Analysis"),
            mo.hstack(
                [
                    # Left: Controls
                    mo.vstack(
                        [
                            mo.md("### Parameters"),
                            scales,
                            min_samples,
                            method,
                            mo.md("### Display Options"),
                            show_clusters,
                            show_filaments,
                            show_voids,
                            run_btn,
                            survey_dropdown,
                            blender_btn,
                            blender_status,
                        ],
                        align="stretch",
                    ),
                    # Right: Results
                    mo.vstack([results_section, viz_section], align="stretch"),
                ],
                widths=[1, 2],
            ),
        ]
    )
