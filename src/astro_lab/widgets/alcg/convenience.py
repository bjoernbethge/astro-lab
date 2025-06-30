"""
Convenience Functions for AstroLab Cosmograph Integration
========================================================

High-level convenience functions for easy TensorDict to Cosmograph visualization.
"""

import logging
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from cosmograph import cosmo
from sklearn.neighbors import NearestNeighbors

from astro_lab.tensors import (
    AnalysisTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)

from .bridge import CosmographBridge, CosmographConfig

logger = logging.getLogger(__name__)


def create_cosmograph_from_tensordict(
    tensordict: Any,
    survey: str = "unknown",
    config: Optional[CosmographConfig] = None,
    build_graph: bool = True,
    interactive: bool = True,
    **kwargs,
) -> Any:
    """
    Create Cosmograph visualization from any TensorDict with automatic type detection.

    Args:
        tensordict: Any AstroLab TensorDict (Spatial, Photometric, Analysis, etc.)
        survey: Survey name for styling ("gaia", "sdss", "nsa", "tng50", etc.)
        config: Optional Cosmograph configuration
        build_graph: Whether to build connectivity graph
        interactive: Whether to enable interactive features
        **kwargs: Additional parameters

    Returns:
        Cosmograph widget or data structure for visualization
    """
    # Create bridge and convert
    bridge = CosmographBridge()
    result = bridge.tensordict_to_cosmograph(
        tensordict, survey=survey, config=config, build_graph=build_graph, **kwargs
    )

    # Create pandas DataFrames for cosmograph
    nodes_df = pd.DataFrame(result["nodes"])
    links_df = pd.DataFrame(result["links"]) if result["links"] else pd.DataFrame()

    # Create cosmograph widget
    try:
        if len(links_df) > 0:
            color_value = nodes_df.get(
                "color", result["config"].get("node_color", "#ffffff")
            )
            if not isinstance(color_value, str):
                color_value = "#ffffff"
            widget = cosmo(
                points=nodes_df,
                links=links_df,
                point_id_by="id",
                point_x_by="x",
                point_y_by="y",
                point_color=color_value,
                point_size_range=result["config"].get("node_size_range", [2, 8]),
                link_source_by="source",
                link_target_by="target",
                link_color=result["config"].get("link_color", "#333333"),
                **{
                    k: v
                    for k, v in result["config"].items()
                    if k not in ["node_color", "node_size_range", "link_color"]
                },
            )
        else:
            color_value = nodes_df.get(
                "color", result["config"].get("node_color", "#ffffff")
            )
            if not isinstance(color_value, str):
                color_value = "#ffffff"
            widget = cosmo(
                points=nodes_df,
                point_id_by="id",
                point_x_by="x",
                point_y_by="y",
                point_color=color_value,
                point_size_range=result["config"].get("node_size_range", [2, 8]),
                disable_simulation=True,  # No simulation without links
                **{
                    k: v
                    for k, v in result["config"].items()
                    if k not in ["node_color", "node_size_range", "render_links"]
                },
            )

        return widget

    except Exception as e:
        logger.error(f"Failed to create cosmograph widget: {e}")
        return result


def visualize_spatial_tensordict(
    spatial_tensordict: SpatialTensorDict,
    survey: str = "unknown",
    radius: Optional[float] = None,
    k_neighbors: int = 8,
    color_by: str = "survey",
    size_by: str = "uniform",
    **kwargs,
) -> Any:
    """
    Visualize SpatialTensorDict with automatic graph construction and styling.

    Args:
        spatial_tensordict: SpatialTensorDict with coordinates
        survey: Survey name for styling
        radius: Radius for graph construction (auto-determined if None)
        k_neighbors: Number of neighbors for graph
        color_by: Coloring strategy
        size_by: Sizing strategy
        **kwargs: Additional parameters

    Returns:
        Cosmograph visualization
    """
    # Auto-determine radius if not provided
    if radius is None:
        coords = spatial_tensordict["coordinates"].cpu().numpy()
        coord_system = spatial_tensordict.meta.get("coordinate_system", "unknown")
        radius = _auto_determine_radius(coords, coord_system)

    # Create configuration
    config = CosmographConfig()
    config.survey_type = survey

    # Adjust for dataset size
    n_objects = spatial_tensordict.n_objects
    if n_objects > 50000:
        config.simulation_repulsion = 0.3
        config.space_size = 8192
        k_neighbors = min(k_neighbors, 5)  # Fewer neighbors for large datasets

    return create_cosmograph_from_tensordict(
        spatial_tensordict,
        survey=survey,
        config=config,
        k_neighbors=k_neighbors,
        **kwargs,
    )


def visualize_analysis_results(
    analysis_tensordict: AnalysisTensorDict,
    show_clusters: bool = True,
    show_filaments: bool = False,
    interactive_selection: bool = True,
    **kwargs,
) -> Any:
    """
    Visualize AnalysisTensorDict with clustering and structure overlays.

    Args:
        analysis_tensordict: AnalysisTensorDict with analysis results
        show_clusters: Whether to show cluster coloring
        show_filaments: Whether to show filament networks
        interactive_selection: Whether to enable interactive cluster selection
        **kwargs: Additional parameters

    Returns:
        Cosmograph visualization with analysis overlays
    """
    # Use the enhanced bridge directly
    bridge = CosmographBridge()

    return bridge.analysis_tensordict_to_cosmograph(
        analysis_tensordict,
        show_clusters=show_clusters,
        show_filaments=show_filaments,
        interactive_selection=interactive_selection,
        **kwargs,
    )


def create_cosmic_web_cosmograph(
    spatial_tensordict: SpatialTensorDict,
    clustering_scales: Optional[List[float]] = None,
    filament_detection: bool = True,
    multi_scale_view: bool = True,
    survey: str = "unknown",
    **kwargs,
) -> Any:
    """
    Create comprehensive cosmic web visualization with clustering and filaments.

    Args:
        spatial_tensordict: SpatialTensorDict with coordinates
        clustering_scales: Scales for clustering analysis (auto-determined if None)
        filament_detection: Whether to detect and show filaments
        multi_scale_view: Whether to enable multi-scale analysis
        survey: Survey name
        **kwargs: Additional parameters

    Returns:
        Cosmic web Cosmograph visualization
    """
    # Auto-determine clustering scales if not provided
    if clustering_scales is None:
        clustering_scales = _get_default_clustering_scales(
            spatial_tensordict.meta.get("coordinate_system", "unknown")
        )

    # Use the enhanced bridge directly
    bridge = CosmographBridge()

    return bridge.spatial_tensordict_to_cosmograph(
        spatial_tensordict,
        survey=survey,
        clustering_scales=clustering_scales,
        filament_detection=filament_detection,
        multi_scale_view=multi_scale_view,
        **kwargs,
    )


def create_multimodal_cosmograph(
    spatial_tensordict: SpatialTensorDict,
    photometric_tensordict: Optional[PhotometricTensorDict] = None,
    analysis_tensordict: Optional[AnalysisTensorDict] = None,
    color_by: str = "survey",
    size_by: str = "uniform",
    survey: str = "unknown",
    **kwargs,
) -> Any:
    """
    Create multi-modal Cosmograph visualization combining multiple data types.

    Args:
        spatial_tensordict: Primary spatial coordinates
        photometric_tensordict: Optional photometric data for color/size
        analysis_tensordict: Optional analysis results for clustering
        color_by: Coloring strategy ("survey", "magnitude", "color", "cluster")
        size_by: Sizing strategy ("uniform", "magnitude", "distance")
        survey: Survey name for styling
        **kwargs: Additional parameters

    Returns:
        Multi-modal Cosmograph visualization
    """
    bridge = CosmographBridge()

    result = bridge.multimodal_tensordict_to_cosmograph(
        spatial_tensordict=spatial_tensordict,
        photometric_tensordict=photometric_tensordict,
        analysis_tensordict=analysis_tensordict,
        survey=survey,
        color_by=color_by,
        size_by=size_by,
        **kwargs,
    )

    # Try to create widget
    try:
        import pandas as pd
        from cosmograph import cosmo

        nodes_df = pd.DataFrame(result["nodes"])
        links_df = pd.DataFrame(result["links"]) if result["links"] else pd.DataFrame()

        if len(links_df) > 0:
            widget = cosmo(
                points=nodes_df,
                links=links_df,
                point_id_by="id",
                point_x_by="x",
                point_y_by="y",
                point_color_by="color" if "color" in nodes_df.columns else "",
                point_size_by="size" if "size" in nodes_df.columns else "",
                link_source_by="source",
                link_target_by="target",
                **result["config"],
            )
        else:
            widget = cosmo(
                points=nodes_df,
                point_id_by="id",
                point_x_by="x",
                point_y_by="y",
                point_color_by="color" if "color" in nodes_df.columns else "",
                point_size_by="size" if "size" in nodes_df.columns else "",
                disable_simulation=True,
                **result["config"],
            )

        return widget

    except ImportError:
        logger.warning("cosmograph package not available")
        return result
    except Exception as e:
        logger.error(f"Failed to create multimodal cosmograph: {e}")
        return result


def visualize(data: Any, survey: str = "unknown", **kwargs) -> Any:
    """
    Quick visualization function that auto-detects data type and creates \
    appropriate visualization.

    Args:
        data: Any supported data (TensorDict, numpy array, coordinates)
        survey: Survey name for styling
        **kwargs: Additional parameters

    Returns:
        Cosmograph visualization
    """
    # Handle different input types
    if hasattr(data, "coordinates"):
        # TensorDict with coordinates
        return create_cosmograph_from_tensordict(data, survey=survey, **kwargs)

    elif hasattr(data, "shape") and len(data.shape) == 2 and data.shape[1] >= 2:
        # Coordinate array
        import torch

        from astro_lab.tensors import SpatialTensorDict

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=torch.float32)

        spatial = SpatialTensorDict(
            coordinates=data, coordinate_system="unknown", unit="pc"
        )
        return create_cosmograph_from_tensordict(spatial, survey=survey, **kwargs)

    else:
        raise ValueError(f"Cannot visualize data type: {type(data)}")


# Helper functions


def _auto_determine_radius(coords, coordinate_system: str) -> float:
    """Auto-determine optimal radius for graph construction."""

    # Sample for efficiency
    n_sample = min(1000, len(coords))
    if n_sample < len(coords):
        # Use first n_sample indices instead of random sampling
        indices = list(range(n_sample))
        sample_coords = coords[indices]
    else:
        sample_coords = coords

    # Compute nearest neighbor distances
    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    nn.fit(sample_coords)
    distances, _ = nn.kneighbors(sample_coords)

    # Use median of 3rd nearest neighbor distances
    median_distance = np.median(distances[:, 3])

    # Scale based on coordinate system
    if coordinate_system in ["galactocentric", "icrs"]:
        radius = median_distance * 2.5
    elif "galaxy" in coordinate_system.lower():
        radius = median_distance * 4.0
    else:
        radius = median_distance * 3.0

    return float(radius)


def _get_default_clustering_scales(coordinate_system: str) -> List[float]:
    """Get default clustering scales based on coordinate system."""

    if coordinate_system in ["galactocentric", "icrs"]:
        # Stellar surveys (parsecs)
        return [5.0, 10.0, 25.0, 50.0]
    elif "galaxy" in coordinate_system.lower():
        # Galaxy surveys (Mpc)
        return [2.0, 5.0, 10.0, 20.0]
    elif "simulation" in coordinate_system.lower():
        # Simulations (kpc)
        return [50.0, 100.0, 200.0, 500.0]
    else:
        # Default (parsecs)
        return [5.0, 10.0, 25.0, 50.0]


# Export all convenience functions
__all__ = [
    "create_cosmograph_from_tensordict",
    "visualize_spatial_tensordict",
    "visualize_analysis_results",
    "create_cosmic_web_cosmograph",
    "create_multimodal_cosmograph",
    "visualize",
]
