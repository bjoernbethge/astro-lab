"""
Analyzer Component
=================

Real analysis interface using actual AstroLab cosmic web analyzer.
"""

import marimo as mo
import polars as pl

from astro_lab.data.analysis.structures import CosmicWebAnalyzer


def create_analyzer():
    """Create real analysis interface using CosmicWebAnalyzer."""

    # Analysis parameters
    scales = mo.ui.array(
        [
            mo.ui.slider(1, 50, 5, label="Small scale (pc)"),
            mo.ui.slider(10, 100, 25, label="Medium scale (pc)"),
            mo.ui.slider(50, 500, 100, label="Large scale (pc)"),
        ]
    )

    min_samples = mo.ui.slider(3, 20, 5, label="Min samples")

    method = mo.ui.dropdown(
        options={
            "dbscan": "DBSCAN - Density-based clustering",
            "hdbscan": "HDBSCAN - Hierarchical clustering",
            "optics": "OPTICS - Ordering points clustering",
        },
        value="dbscan",
        label="Clustering Algorithm",
    )

    # Run button
    run_btn = mo.ui.button(label="🚀 Run Cosmic Web Analysis", kind="success")

    # UI
    ui = mo.vstack(
        [mo.md("### 🔬 Cosmic Web Analysis"), scales, min_samples, method, run_btn]
    )

    # Results
    result = None
    status = mo.md("")

    if run_btn.value:
        # Get scales values
        scale_values = [s.value for s in scales.value if hasattr(s, "value")]

        status = mo.callout(
            f"✅ Analysis configured: scales={scale_values}, min_samples={min_samples.value}, method={method.value}",
            kind="success",
        )

        # Store result for use by other components
        result = {
            "scales": scale_values,
            "min_samples": min_samples.value,
            "method": method.value,
            "configured": True,
        }

    return ui, status, result


def run_cosmic_web_analysis(data, scales, min_samples=5, method="dbscan"):
    """Run real cosmic web analysis using CosmicWebAnalyzer."""
    try:
        # Create analyzer
        analyzer = CosmicWebAnalyzer()

        # Extract coordinates based on data type
        if isinstance(data, pl.DataFrame):
            # Check if we have 3D coordinates
            if all(col in data.columns for col in ["x", "y", "z"]):
                coordinates = data.select(["x", "y", "z"]).to_numpy()
            elif all(col in data.columns for col in ["ra", "dec", "distance_pc"]):
                # Convert spherical to cartesian
                from astro_lab.data.transforms.astronomical import (
                    spherical_to_cartesian,
                )

                x, y, z = spherical_to_cartesian(
                    data["ra"].to_numpy(),
                    data["dec"].to_numpy(),
                    data["distance_pc"].to_numpy(),
                )
                coordinates = [[x[i], y[i], z[i]] for i in range(len(x))]
            else:
                raise ValueError(
                    "Data must have either (x,y,z) or (ra,dec,distance_pc) columns"
                )
        else:
            # Assume numpy array or similar
            coordinates = data

        # Run analysis for each scale
        results = {
            "n_objects": len(coordinates),
            "scales": scales,
            "method": method,
            "min_samples": min_samples,
            "scale_results": {},
        }

        # Convert coordinates to torch tensor
        import torch

        coordinates_tensor = torch.tensor(coordinates, dtype=torch.float32)

        # Run comprehensive cosmic web analysis
        analysis_result = analyzer.analyze_cosmic_web(
            coordinates=coordinates_tensor,
            scales=scales,
        )

        # Extract results from the analysis
        results["cosmic_web"] = analysis_result.get("combined", {})
        results["filaments"] = analysis_result.get("filaments", {})
        results["structures"] = analysis_result.get("structures", {})

        # Calculate overall statistics
        cosmic_web = results["cosmic_web"].get("cosmic_web", {})
        results["total_structures"] = (
            len(cosmic_web.get("filaments", []))
            + len(cosmic_web.get("clusters", []))
            + len(cosmic_web.get("voids", []))
            + len(cosmic_web.get("walls", []))
        )

        # Overall statistics
        total_clustered = sum(
            r["n_clusters"] for r in results["scale_results"].values()
        )
        results["total_structures"] = total_clustered
        results["analysis_complete"] = True

        return results

    except Exception as e:
        return {"error": str(e), "analysis_complete": False}
