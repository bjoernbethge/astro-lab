"""
Plotly Bridge for AstroLab - Statistical and Interactive Astronomical Plots
===========================================================================

Complete implementation of Plotly visualization backend for astronomical data.
"""

import logging
from typing import Any, Dict, List

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots

from astro_lab.tensors import (
    AnalysisTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)

logger = logging.getLogger(__name__)


class AstronomicalPlotlyBridge:
    """Bridge for creating astronomical visualizations with Plotly."""

    def __init__(self):
        self.default_config = {
            "template": "plotly_dark",
            "width": 800,
            "height": 600,
            "margin": dict(l=50, r=50, t=50, b=50),
        }

    def create_visualization(
        self, tensordict: Any, plot_type: str = "auto", **kwargs
    ) -> go.Figure:
        """
        Create Plotly visualization from TensorDict.

        Args:
            tensordict: AstroLab TensorDict
            plot_type: Type of plot (
                'auto', 'scatter_3d', 'hr_diagram', 'sky_map', etc.
            )
            **kwargs: Plot-specific parameters

        Returns:
            Plotly Figure
        """
        if plot_type == "auto":
            plot_type = self._determine_optimal_plot_type(tensordict)

        if isinstance(tensordict, SpatialTensorDict):
            return self._create_spatial_plot(tensordict, plot_type, **kwargs)
        elif isinstance(tensordict, PhotometricTensorDict):
            return self._create_photometric_plot(tensordict, plot_type, **kwargs)
        elif isinstance(tensordict, AnalysisTensorDict):
            return self._create_analysis_plot(tensordict, plot_type, **kwargs)
        else:
            # Generic coordinate plot
            return self._create_coordinate_plot(tensordict, plot_type, **kwargs)

    def _determine_optimal_plot_type(self, tensordict: Any) -> str:
        """Determine optimal plot type for TensorDict."""
        if isinstance(tensordict, PhotometricTensorDict):
            if tensordict.n_bands >= 2:
                return "hr_diagram"
            else:
                return "magnitude_histogram"

        elif isinstance(tensordict, SpatialTensorDict):
            coordinate_system = tensordict.meta.get("coordinate_system", "unknown")
            if coordinate_system in ["icrs", "galactic"]:
                return "sky_map"
            else:
                return "scatter_3d"

        elif isinstance(tensordict, AnalysisTensorDict):
            return "cluster_analysis"

        else:
            return "scatter_3d"

    def _create_spatial_plot(
        self, spatial_tensor: SpatialTensorDict, plot_type: str, **kwargs
    ) -> go.Figure:
        """Create spatial data visualization."""
        coords = spatial_tensor["coordinates"].cpu().numpy()
        coordinate_system = spatial_tensor.meta.get("coordinate_system", "cartesian")

        if plot_type == "sky_map":
            return self._create_sky_map(coords, coordinate_system, **kwargs)
        elif plot_type == "scatter_3d":
            return self._create_3d_scatter(coords, **kwargs)
        elif plot_type == "cosmic_web":
            return self._create_cosmic_web_plot(spatial_tensor, **kwargs)
        else:
            return self._create_3d_scatter(coords, **kwargs)

    def _create_photometric_plot(
        self, photometric_tensor: PhotometricTensorDict, plot_type: str, **kwargs
    ) -> go.Figure:
        """Create photometric data visualization."""
        magnitudes = photometric_tensor["magnitudes"].cpu().numpy()
        bands = photometric_tensor.bands

        if plot_type == "hr_diagram":
            return self._create_hr_diagram(magnitudes, bands, **kwargs)
        elif plot_type == "color_magnitude":
            return self._create_color_magnitude_diagram(magnitudes, bands, **kwargs)
        elif plot_type == "magnitude_histogram":
            return self._create_magnitude_histogram(magnitudes, bands, **kwargs)
        else:
            return self._create_hr_diagram(magnitudes, bands, **kwargs)

    def _create_analysis_plot(
        self, analysis_tensor: AnalysisTensorDict, plot_type: str, **kwargs
    ) -> go.Figure:
        """Create analysis results visualization."""

        # Extract clustering results
        clustering_results = analysis_tensor.get("clustering_results", {})

        if plot_type == "cluster_analysis":
            return self._create_cluster_analysis_plot(clustering_results, **kwargs)
        elif plot_type == "multi_scale":
            return self._create_multi_scale_plot(clustering_results, **kwargs)
        else:
            # Try to extract spatial data for 3D plot
            base_tensors = getattr(analysis_tensor, "base_tensors", {})
            if "spatial" in base_tensors:
                return self._create_spatial_plot(
                    base_tensors["spatial"], "scatter_3d", **kwargs
                )
            else:
                return self._create_cluster_analysis_plot(clustering_results, **kwargs)

    def _create_sky_map(
        self, coords: np.ndarray, coordinate_system: str, **kwargs
    ) -> go.Figure:
        """Create sky map projection."""

        if coordinate_system == "icrs":
            # RA/Dec coordinates - convert to Mollweide projection
            if coords.shape[1] >= 3:
                # Extract RA, Dec from Cartesian
                ra = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
                dec = np.degrees(
                    np.arcsin(coords[:, 2] / np.linalg.norm(coords, axis=1))
                )
            else:
                ra, dec = coords[:, 0], coords[:, 1]
        else:
            # Assume Cartesian - project to sky
            ra = np.degrees(np.arctan2(coords[:, 1], coords[:, 0]))
            dec = np.degrees(np.arcsin(coords[:, 2] / np.linalg.norm(coords, axis=1)))

        # Mollweide projection
        ra_wrapped = np.where(ra > 180, ra - 360, ra)

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=ra_wrapped,
                y=dec,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 2),
                    color=kwargs.get("point_color", "gold"),
                    opacity=kwargs.get("opacity", 0.7),
                ),
                name=kwargs.get("name", "Objects"),
            )
        )

        fig.update_layout(
            title="Sky Map",
            xaxis_title="RA (degrees)",
            yaxis_title="Dec (degrees)",
            **self.default_config,
        )

        return fig

    def _create_3d_scatter(self, coords: np.ndarray, **kwargs) -> go.Figure:
        """Create 3D scatter plot."""

        # Handle 2D coordinates
        if coords.shape[1] == 2:
            coords = np.column_stack([coords, np.zeros(len(coords))])

        # Color by distance if not specified
        colors = kwargs.get("colors")
        if colors is None:
            distances = np.linalg.norm(coords, axis=1)
            colors = distances
            colorscale = "Viridis"
        else:
            colorscale = kwargs.get("colorscale", "Viridis")

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 3),
                    color=colors,
                    colorscale=colorscale,
                    opacity=kwargs.get("opacity", 0.8),
                    colorbar=dict(title="Distance") if colors is not None else None,
                ),
                name=kwargs.get("name", "Objects"),
            )
        )

        fig.update_layout(
            title="3D Scatter Plot",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.2, y=1.2, z=0.6)),
            ),
            **self.default_config,
        )

        return fig

    def _create_hr_diagram(
        self, magnitudes: np.ndarray, bands: List[str], **kwargs
    ) -> go.Figure:
        """Create Hertzsprung-Russell diagram."""

        if len(bands) < 2:
            raise ValueError("HR diagram requires at least 2 bands")

        # Use first two bands for color and magnitude
        color_band1, color_band2 = bands[0], bands[1]
        mag_band = bands[0]  # Use first band for y-axis

        mag1 = magnitudes[:, 0]
        mag2 = magnitudes[:, 1]
        color = mag1 - mag2

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=color,
                y=mag1,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 3),
                    color=kwargs.get("point_color", "blue"),
                    opacity=kwargs.get("opacity", 0.7),
                ),
                name="Stars",
            )
        )

        fig.update_layout(
            title="Hertzsprung-Russell Diagram",
            xaxis_title=f"{color_band1} - {color_band2} (mag)",
            yaxis_title=f"{mag_band} (mag)",
            yaxis=dict(autorange="reversed"),  # Fainter stars at bottom
            **self.default_config,
        )

        return fig

    def _create_color_magnitude_diagram(
        self, magnitudes: np.ndarray, bands: List[str], **kwargs
    ) -> go.Figure:
        """Create color-magnitude diagram."""

        if len(bands) < 2:
            return self._create_magnitude_histogram(magnitudes, bands, **kwargs)

        mag1 = magnitudes[:, 0]
        mag2 = magnitudes[:, 1]
        color = mag1 - mag2

        fig = go.Figure()

        fig.add_trace(
            go.Scattergl(
                x=color,
                y=mag1,
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 3), opacity=kwargs.get("opacity", 0.6)
                ),
                name="Objects",
            )
        )

        fig.update_layout(
            title="Color-Magnitude Diagram",
            xaxis_title=f"{bands[0]} - {bands[1]} (mag)",
            yaxis_title=f"{bands[0]} (mag)",
            yaxis=dict(autorange="reversed"),
            **self.default_config,
        )

        return fig

    def _create_magnitude_histogram(
        self, magnitudes: np.ndarray, bands: List[str], **kwargs
    ) -> go.Figure:
        """Create magnitude histogram."""

        fig = make_subplots(
            rows=1,
            cols=len(bands),
            subplot_titles=[f"{band} band" for band in bands],
            shared_yaxes=True,
        )

        for i, band in enumerate(bands):
            fig.add_trace(
                go.Histogram(x=magnitudes[:, i], name=f"{band} band", opacity=0.7),
                row=1,
                col=i + 1,
            )

        fig.update_layout(
            title="Magnitude Distribution",
            xaxis_title="Magnitude",
            yaxis_title="Count",
            **self.default_config,
        )

        return fig

    def _create_cosmic_web_plot(
        self, spatial_tensor: SpatialTensorDict, **kwargs
    ) -> go.Figure:
        """Create cosmic web visualization."""

        coords = spatial_tensor["coordinates"].cpu().numpy()

        # Try to get clustering results
        cluster_labels = kwargs.get("cluster_labels")
        if cluster_labels is not None:
            if isinstance(cluster_labels, torch.Tensor):
                cluster_labels = cluster_labels.cpu().numpy()

            # Color by cluster
            colors = cluster_labels
            colorscale = "Set1"
        else:
            # Color by density or distance
            distances = np.linalg.norm(coords, axis=1)
            colors = distances
            colorscale = "Viridis"

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(
                    size=kwargs.get("point_size", 2),
                    color=colors,
                    colorscale=colorscale,
                    opacity=kwargs.get("opacity", 0.8),
                ),
                name="Cosmic Web",
            )
        )

        # Add connections if edges provided
        edges = kwargs.get("edges")
        if edges is not None:
            for edge in edges[:1000]:  # Limit edges for performance
                i, j = edge[0], edge[1]
                fig.add_trace(
                    go.Scatter3d(
                        x=[coords[i, 0], coords[j, 0], None],
                        y=[coords[i, 1], coords[j, 1], None],
                        z=[coords[i, 2], coords[j, 2], None],
                        mode="lines",
                        line=dict(color="rgba(100,100,100,0.3)", width=1),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Cosmic Web Structure",
            scene=dict(
                xaxis_title="X (pc)", yaxis_title="Y (pc)", zaxis_title="Z (pc)"
            ),
            **self.default_config,
        )

        return fig

    def _create_cluster_analysis_plot(
        self, clustering_results: Dict, **kwargs
    ) -> go.Figure:
        """Create cluster analysis visualization."""

        scales = list(clustering_results.keys())
        n_clusters = [clustering_results[scale]["n_clusters"] for scale in scales]
        grouped_fractions = [
            clustering_results[scale]["grouped_fraction"] for scale in scales
        ]

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Number of Clusters vs Scale", "Grouped Fraction vs Scale"],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Number of clusters
        fig.add_trace(
            go.Scatter(
                x=[float(s.replace("pc", "").replace("Mpc", "")) for s in scales],
                y=n_clusters,
                mode="lines+markers",
                name="Clusters",
                line=dict(color="blue"),
            ),
            row=1,
            col=1,
        )

        # Grouped fraction
        fig.add_trace(
            go.Scatter(
                x=[float(s.replace("pc", "").replace("Mpc", "")) for s in scales],
                y=grouped_fractions,
                mode="lines+markers",
                name="Grouped Fraction",
                line=dict(color="red"),
            ),
            row=1,
            col=2,
        )

        fig.update_layout(title="Clustering Analysis Results", **self.default_config)

        return fig

    def _create_multi_scale_plot(self, clustering_results: Dict, **kwargs) -> go.Figure:
        """Create multi-scale analysis plot."""

        scales = list(clustering_results.keys())

        fig = go.Figure()

        for scale in scales:
            results = clustering_results[scale]

            fig.add_trace(
                go.Bar(
                    name=f"Scale {scale}",
                    x=["Clusters", "Noise", "Grouped"],
                    y=[results["n_clusters"], results["n_noise"], results["n_grouped"]],
                    opacity=0.8,
                )
            )

        fig.update_layout(
            title="Multi-Scale Clustering Results",
            xaxis_title="Category",
            yaxis_title="Count",
            barmode="group",
            **self.default_config,
        )

        return fig

    def _create_coordinate_plot(self, data: Any, plot_type: str, **kwargs) -> go.Figure:
        """Create plot from generic coordinate data."""

        if isinstance(data, torch.Tensor):
            coords = data.cpu().numpy()
        elif isinstance(data, np.ndarray):
            coords = data
        else:
            coords = np.array(data)

        return self._create_3d_scatter(coords, **kwargs)


def create_plotly_visualization(tensordict: Any, **kwargs) -> go.Figure:
    """Convenience function to create Plotly visualization."""
    bridge = AstronomicalPlotlyBridge()
    return bridge.create_visualization(tensordict, **kwargs)


def create_survey_comparison_plot(tensordicts: Dict[str, Any], **kwargs) -> go.Figure:
    """Create comparison plot of multiple surveys."""
    fig = go.Figure()

    colors = ["gold", "blue", "red", "green", "purple", "orange"]

    for i, (survey, tensordict) in enumerate(tensordicts.items()):
        color = colors[i % len(colors)]

        if isinstance(tensordict, SpatialTensorDict):
            coords = tensordict["coordinates"].cpu().numpy()

            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
                    mode="markers",
                    marker=dict(
                        size=kwargs.get("point_size", 3),
                        color=color,
                        opacity=kwargs.get("opacity", 0.7),
                    ),
                    name=survey.upper(),
                )
            )

    fig.update_layout(
        title="Survey Comparison",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        template="plotly_dark",
    )

    return fig


__all__ = [
    "AstronomicalPlotlyBridge",
    "create_plotly_visualization",
    "create_survey_comparison_plot",
]
