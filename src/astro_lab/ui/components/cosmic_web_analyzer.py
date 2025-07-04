"""
Cosmic Web Analyzer Component
============================

Interactive cosmic web analysis with real-time visualization.
"""

import marimo as mo
from typing import Dict, Any, List
import matplotlib.pyplot as plt

from astro_lab.data.analysis.cosmic_web import CosmicWebAnalyzer
from astro_lab.widgets.alcg import create_cosmic_web_cosmograph
from astro_lab.ui.components import state


class CosmicWebAnalyzerComponent:
    """Advanced cosmic web analysis with interactive controls."""

    def __init__(self):
        self.analyzer = CosmicWebAnalyzer()
        self.current_analysis = None
        self.analysis_params = {}

    def create_interface(self) -> mo.Html:
        """Create cosmic web analysis interface."""
        current_state = state.get_state()

        if not current_state.get("loaded_data"):
            return mo.callout(
                "Please load data first to perform cosmic web analysis",
                kind="neutral"
            )

        # Create layout with visualization prominent
        return mo.vstack([
            mo.md("## 🌌 Cosmic Web Analysis"),
            # Main visualization area
            self._create_visualization_area(),
            # Controls below in a compact layout
            mo.hstack([
                self._create_scale_controls(),
                self._create_method_controls(),
                self._create_display_controls(),
            ], widths=[1, 1, 1]),
            # Results summary
            self._create_results_summary(),
        ], gap=2)

    def _create_visualization_area(self) -> mo.Html:
        """Create main visualization area."""
        current_state = state.get_state()
        analysis_result = current_state.get("cosmic_web_analysis")

        if not analysis_result:
            # Show placeholder with instructions
            return mo.center(
                mo.vstack([
                    mo.md("### 🔭 Cosmic Web Visualization"),
                    mo.md("Configure parameters and run analysis to see results"),
                    mo.ui.button("Run Analysis", kind="primary"),
                ], align="center")
            )

        # Create cosmic web visualization
        try:
            viz = create_cosmic_web_cosmograph(
                analysis_result.get("spatial_tensor"),
                analysis_results=analysis_result,
                show_clusters=True,
                show_filaments=True,
                show_voids=True,
                node_size_scale=2.0,
                link_opacity=0.5,
            )

            # Add overlay controls
            overlay = self._create_visualization_overlay()

            return mo.vstack([
                viz,
                overlay,
            ], gap=0)

        except Exception as e:
            return mo.callout(f"Visualization error: {str(e)}", kind="danger")

    def _create_scale_controls(self) -> mo.Html:
        """Create multi-scale analysis controls."""
        # Preset scale configurations
        scale_presets = mo.ui.dropdown(
            options={
                "stellar": "Stellar (1-50 pc)",
                "local": "Local Group (10-200 pc)",
                "galactic": "Galactic (50-1000 pc)",
                "cosmic": "Cosmic (100-10000 pc)",
                "custom": "Custom Scales",
            },
            value="local",
            label="Scale Preset",
        )

        # Custom scale inputs
        custom_scales = mo.ui.array([
            mo.ui.number(
                start=0.1, stop=100, value=5, step=0.1,
                label="Small Scale (pc)"
            ),
            mo.ui.number(
                start=1, stop=1000, value=25, step=1,
                label="Medium Scale (pc)"
            ),
            mo.ui.number(
                start=10, stop=10000, value=100, step=10,
                label="Large Scale (pc)"
            ),
        ])

        # Adaptive scaling
        adaptive = mo.ui.checkbox(
            value=True,
            label="Adaptive Scaling",
        )

        return mo.vstack([
            mo.md("### 📏 Scale Analysis"),
            scale_presets,
            mo.ui.accordion({
                "Custom Scales": custom_scales,
            }),
            adaptive,
        ])

    def _create_method_controls(self) -> mo.Html:
        """Create analysis method controls."""
        # Structure detection methods
        detection_method = mo.ui.dropdown(
            options={
                "density": "Density Field - Grid-based analysis",
                "mst": "MST - Minimum spanning tree",
                "morse": "Morse Theory - Topological analysis",
                "hessian": "Hessian - Curvature-based",
                "persistence": "Persistent Homology",
                "hybrid": "Hybrid - Combined methods",
            },
            value="hybrid",
            label="Detection Method",
        )

        # Clustering algorithm
        clustering = mo.ui.dropdown(
            options={
                "dbscan": "DBSCAN - Density-based",
                "hdbscan": "HDBSCAN - Hierarchical",
                "optics": "OPTICS - Ordering points",
                "spectral": "Spectral - Graph-based",
            },
            value="hdbscan",
            label="Clustering Algorithm",
        )

        # Parameters
        min_samples = mo.ui.slider(
            start=3, stop=50, value=10, step=1,
            label="Min Samples",
        )

        return mo.vstack([
            mo.md("### 🔬 Analysis Methods"),
            detection_method,
            clustering,
            min_samples,
        ])

    def _create_display_controls(self) -> mo.Html:
        """Create display controls."""
        # Structure visibility
        show_structures = mo.ui.multiselect(
            options=["clusters", "filaments", "sheets", "voids", "nodes"],
            value=["clusters", "filaments"],
            label="Show Structures",
        )

        # Color scheme
        color_scheme = mo.ui.dropdown(
            options={
                "structure": "By Structure Type",
                "density": "By Density",
                "scale": "By Scale",
                "connectivity": "By Connectivity",
            },
            value="structure",
            label="Color Scheme",
        )

        # Animation
        animate = mo.ui.checkbox(
            value=False,
            label="Animate Evolution",
        )

        return mo.vstack([
            mo.md("### 🎨 Display Options"),
            show_structures,
            color_scheme,
            animate,
        ])

    def _create_visualization_overlay(self) -> mo.Html:
        """Create overlay controls for visualization."""
        # Zoom controls
        zoom_controls = mo.hstack([
            mo.ui.button("➖", kind="secondary", small=True),
            mo.ui.button("🔄", kind="secondary", small=True),
            mo.ui.button("➕", kind="secondary", small=True),
        ])

        # Info display
        info = mo.ui.text(
            value="",
            label="Selection Info",
            disabled=True,
        )

        return mo.hstack([
            zoom_controls,
            info,
        ], justify="space-between")

    def _create_results_summary(self) -> mo.Html:
        """Create analysis results summary."""
        current_state = state.get_state()
        results = current_state.get("cosmic_web_analysis")

        if not results:
            return mo.empty()

        # Structure counts
        structure_counts = mo.hstack([
            mo.stat("Clusters", f"{results.get('n_clusters', 0):,}"),
            mo.stat("Filaments", f"{results.get('n_filaments', 0):,}"),
            mo.stat("Sheets", f"{results.get('n_sheets', 0):,}"),
            mo.stat("Voids", f"{results.get('n_voids', 0):,}"),
        ])

        # Scale analysis
        scale_results = self._create_scale_results(results)

        # Topology metrics
        topology = self._create_topology_metrics(results)

        return mo.vstack([
            mo.md("### 📊 Analysis Results"),
            structure_counts,
            mo.tabs({
                "Scale Analysis": scale_results,
                "Topology": topology,
                "Statistics": self._create_statistics(results),
            }),
        ])

    def _create_scale_results(self, results: Dict[str, Any]) -> mo.Html:
        """Create scale analysis results."""
        scales = results.get("scales", [])
        scale_data = results.get("scale_analysis", {})

        if not scale_data:
            return mo.md("No scale analysis available")

        # Create scale comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Clustering vs scale
        ax1.plot(scales, [scale_data[s]["n_clusters"] for s in scales], 'o-')
        ax1.set_xlabel("Scale (pc)")
        ax1.set_ylabel("Number of Clusters")
        ax1.set_title("Clustering vs Scale")
        ax1.set_xscale("log")

        # Connectivity vs scale
        ax2.plot(scales, [scale_data[s]["connectivity"] for s in scales], 's-')
        ax2.set_xlabel("Scale (pc)")
        ax2.set_ylabel("Connectivity")
        ax2.set_title("Network Connectivity vs Scale")
        ax2.set_xscale("log")

        plt.tight_layout()

        return mo.vstack([
            mo.matplotlib(fig),
            mo.ui.table({
                "Scale (pc)": scales,
                "Clusters": [scale_data[s]["n_clusters"] for s in scales],
                "Filaments": [scale_data[s]["n_filaments"] for s in scales],
                "Connectivity": [f"{scale_data[s]['connectivity']:.3f}" for s in scales],
            }),
        ])

    def _create_topology_metrics(self, results: Dict[str, Any]) -> mo.Html:
        """Create topology metrics display."""
        topology = results.get("topology", {})

        if not topology:
            return mo.md("No topology analysis available")

        metrics = mo.ui.table({
            "Metric": [
                "Euler Characteristic",
                "Betti 0 (Components)",
                "Betti 1 (Loops)",
                "Betti 2 (Voids)",
                "Genus",
                "Persistence Entropy",
            ],
            "Value": [
                topology.get("euler_characteristic", "N/A"),
                topology.get("betti_0", "N/A"),
                topology.get("betti_1", "N/A"),
                topology.get("betti_2", "N/A"),
                topology.get("genus", "N/A"),
                f"{topology.get('persistence_entropy', 0):.3f}",
            ],
        })

        # Persistence diagram if available
        if "persistence_diagram" in topology:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot persistence diagram
            births = topology["persistence_diagram"]["births"]
            deaths = topology["persistence_diagram"]["deaths"]

            ax.scatter(births, deaths, alpha=0.6)
            ax.plot([0, max(deaths)], [0, max(deaths)], 'k--', alpha=0.3)
            ax.set_xlabel("Birth")
            ax.set_ylabel("Death")
            ax.set_title("Persistence Diagram")

            return mo.vstack([
                metrics,
                mo.matplotlib(fig),
            ])

        return metrics

    def _create_statistics(self, results: Dict[str, Any]) -> mo.Html:
        """Create statistical summary."""
        stats = results.get("statistics", {})

        if not stats:
            return mo.md("No statistics available")

        # Group statistics by category
        general_stats = {
            "Total Objects": stats.get("n_total", 0),
            "Analyzed Objects": stats.get("n_analyzed", 0),
            "Coverage": f"{stats.get('coverage', 0):.1%}",
            "Mean Density": f"{stats.get('mean_density', 0):.3f}",
            "Density Contrast": f"{stats.get('density_contrast', 0):.2f}",
        }

        structure_stats = {
            "Cluster Fraction": f"{stats.get('cluster_fraction', 0):.1%}",
            "Filament Fraction": f"{stats.get('filament_fraction', 0):.1%}",
            "Void Fraction": f"{stats.get('void_fraction', 0):.1%}",
            "Mean Cluster Size": f"{stats.get('mean_cluster_size', 0):.1f}",
            "Mean Filament Length": f"{stats.get('mean_filament_length', 0):.1f} pc",
        }

        return mo.vstack([
            mo.md("#### General Statistics"),
            mo.ui.table({"Metric": list(general_stats.keys()), "Value": list(general_stats.values())}),
            mo.md("#### Structure Statistics"),
            mo.ui.table({"Metric": list(structure_stats.keys()), "Value": list(structure_stats.values())}),
        ])

    async def run_analysis(
        self,
        data: Any,
        scales: List[float],
        method: str,
        clustering_algo: str,
        min_samples: int,
        show_structures: List[str],
    ) -> Dict[str, Any]:
        """Run cosmic web analysis."""
        try:
            # Update state
            state.update_state(analysis_status="Running cosmic web analysis...")

            # Run analysis
            results = self.analyzer.analyze(
                data,
                scales=scales,
                method=method,
                clustering_algorithm=clustering_algo,
                min_samples=min_samples,
                compute_topology=True,
                verbose=True,
            )

            # Add visualization parameters
            results["show_structures"] = show_structures
            results["method"] = method
            results["scales"] = scales

            # Update state
            state.update_state(
                cosmic_web_analysis=results,
                analysis_status="Analysis complete",
            )

            return results

        except Exception as e:
            state.update_state(
                analysis_status=f"Analysis error: {str(e)}",
            )
            raise


# Convenience function
def create_cosmic_web_analyzer() -> mo.Html:
    """Create cosmic web analyzer component."""
    analyzer = CosmicWebAnalyzerComponent()
    return analyzer.create_interface()
