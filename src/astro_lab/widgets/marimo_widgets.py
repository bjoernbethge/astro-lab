"""
Marimo Widget Integration for AstroLab
=====================================

Native Marimo widgets for astronomical data visualization.
"""

from typing import Any, Dict, Optional, Union

import marimo as mo
import torch
from astro_lab.tensors import (
    SpatialTensorDict,
    PhotometricTensorDict,
    LightcurveTensorDict,
)


class AstroLabWidget(mo.ui.anywidget):
    """Base class for AstroLab Marimo widgets."""
    
    def __init__(self, tensordict: Any, backend: str = "auto", **kwargs):
        self.tensordict = tensordict
        self.backend = backend
        self.kwargs = kwargs
        
        # Create visualization
        from . import visualize
        self.viz = visualize(tensordict, backend=backend, **kwargs)
        
        super().__init__(self._create_widget_spec())
    
    def _create_widget_spec(self) -> Dict[str, Any]:
        """Create widget specification for Marimo."""
        raise NotImplementedError


class CosmicWebWidget(AstroLabWidget):
    """Interactive cosmic web visualization widget."""
    
    def __init__(
        self, 
        spatial_tensor: SpatialTensorDict,
        clustering_results: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ):
        kwargs.setdefault("backend", "cosmograph")
        kwargs["show_connections"] = True
        kwargs["cluster_colors"] = True
        
        if clustering_results:
            kwargs["clustering_results"] = clustering_results
            
        super().__init__(spatial_tensor, **kwargs)
    
    def _create_widget_spec(self) -> Dict[str, Any]:
        """Create Cosmograph widget spec."""
        return {
            "component": "cosmograph",
            "props": {
                "nodes": self.viz.nodes,
                "links": self.viz.links,
                "config": self.viz.config,
            }
        }


class HRDiagramWidget(mo.ui.element):
    """Hertzsprung-Russell diagram widget."""
    
    def __init__(
        self,
        photometric_tensor: PhotometricTensorDict,
        color_bands: tuple = ("B", "V"),
        magnitude_band: str = "V",
        **kwargs
    ):
        self.photometric = photometric_tensor
        self.color_bands = color_bands
        self.magnitude_band = magnitude_band
        
        # Compute colors
        colors = photometric_tensor.compute_colors([color_bands])
        color_key = f"{color_bands[0]}-{color_bands[1]}"
        
        # Create Plotly figure
        from .plotly import plot_hr_diagram
        self.fig = plot_hr_diagram(
            color=colors[color_key],
            magnitude=photometric_tensor.get_band(magnitude_band),
            **kwargs
        )
        
        super().__init__(
            component="plotly",
            props={"figure": self.fig.to_dict()}
        )


class SurveyComparisonWidget(mo.ui.tabs):
    """Multi-survey comparison widget with tabs."""
    
    def __init__(
        self,
        surveys: Dict[str, Any],
        comparison_type: str = "spatial",
        **kwargs
    ):
        # Create tabs for each survey
        tabs = {}
        
        for survey_name, tensordict in surveys.items():
            if comparison_type == "spatial":
                widget = CosmicWebWidget(tensordict, **kwargs)
            elif comparison_type == "photometric":
                widget = HRDiagramWidget(tensordict, **kwargs)
            else:
                widget = AstroLabWidget(tensordict, **kwargs)
                
            tabs[survey_name] = widget
        
        super().__init__(tabs)


class TimeSeriesWidget(mo.ui.element):
    """Interactive time series visualization for lightcurves."""
    
    def __init__(
        self,
        lightcurve_tensor: LightcurveTensorDict,
        object_index: int = 0,
        show_periodogram: bool = True,
        **kwargs
    ):
        self.lightcurve = lightcurve_tensor
        self.object_index = object_index
        
        # Create visualization
        from .plotly import plot_lightcurve
        self.fig = plot_lightcurve(
            times=lightcurve_tensor.time[object_index],
            flux=lightcurve_tensor.flux[object_index],
            show_periodogram=show_periodogram,
            **kwargs
        )
        
        super().__init__(
            component="plotly",
            props={"figure": self.fig.to_dict()}
        )


class AstroLabDashboard(mo.ui.vstack):
    """Complete astronomical analysis dashboard."""
    
    def __init__(self, analysis_results: Dict[str, Any]):
        components = []
        
        # Add cosmic web visualization if available
        if "spatial" in analysis_results:
            components.append(
                mo.md("## Cosmic Web Structure"),
                CosmicWebWidget(
                    analysis_results["spatial"],
                    clustering_results=analysis_results.get("clustering")
                )
            )
        
        # Add HR diagram if photometric data available
        if "photometric" in analysis_results:
            components.append(
                mo.md("## Color-Magnitude Diagram"),
                HRDiagramWidget(analysis_results["photometric"])
            )
        
        # Add statistics table
        if "statistics" in analysis_results:
            components.append(
                mo.md("## Dataset Statistics"),
                mo.ui.table(analysis_results["statistics"])
            )
        
        super().__init__(components)


# Factory functions
def create_cosmic_web_widget(
    data: Union[SpatialTensorDict, Dict[str, Any]],
    **kwargs
) -> CosmicWebWidget:
    """Create cosmic web visualization widget."""
    if isinstance(data, dict) and "spatial" in data:
        return CosmicWebWidget(data["spatial"], **kwargs)
    return CosmicWebWidget(data, **kwargs)


def create_hr_diagram_widget(
    data: Union[PhotometricTensorDict, Dict[str, Any]],
    **kwargs
) -> HRDiagramWidget:
    """Create HR diagram widget."""
    if isinstance(data, dict) and "photometric" in data:
        return HRDiagramWidget(data["photometric"], **kwargs)
    return HRDiagramWidget(data, **kwargs)


def create_survey_comparison_widget(
    surveys: Dict[str, Any],
    **kwargs
) -> SurveyComparisonWidget:
    """Create multi-survey comparison widget."""
    return SurveyComparisonWidget(surveys, **kwargs)


def create_astrolab_dashboard(
    analysis_results: Dict[str, Any]
) -> AstroLabDashboard:
    """Create complete analysis dashboard."""
    return AstroLabDashboard(analysis_results)


__all__ = [
    # Widget classes
    "AstroLabWidget",
    "CosmicWebWidget", 
    "HRDiagramWidget",
    "SurveyComparisonWidget",
    "TimeSeriesWidget",
    "AstroLabDashboard",
    # Factory functions
    "create_cosmic_web_widget",
    "create_hr_diagram_widget",
    "create_survey_comparison_widget",
    "create_astrolab_dashboard",
]
