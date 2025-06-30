"""
AstroLab Data Module
===================

Data loading, preprocessing, and graph construction for astronomical surveys.
"""

from astro_lab.data.datamodule import AstroDataModule, create_datamodule
from astro_lab.data.datasets import SurveyGraphDataset, get_supported_surveys
from astro_lab.data.preprocessors import get_preprocessor
from astro_lab.data.converters import (
    dataframe_to_tensordict,
    tensordict_to_dataframe,
    graph_to_tensordict,
    tensordict_to_graph,
    create_spatial_tensor_from_survey,
)
from astro_lab.data.cross_match import CrossMatcher
from astro_lab.data.analysis import (
    CosmicWebAnalyzer,
    analyze_cosmic_web,
    SpatialClustering,
    FilamentDetector,
    StructureAnalyzer,
)
from astro_lab.data.graphs import (
    AstronomicalGraphBuilder,
    PointCloudGraphBuilder,
    create_pointcloud_graph,
    create_multiscale_pointcloud_graph,
)

__all__ = [
    # Lightning DataModule
    "AstroDataModule",
    "create_datamodule",
    # PyG Datasets
    "SurveyGraphDataset",
    "get_supported_surveys",
    # Data Pipeline
    "get_preprocessor",
    # Converters
    "dataframe_to_tensordict",
    "tensordict_to_dataframe",
    "graph_to_tensordict",
    "tensordict_to_graph",
    "create_spatial_tensor_from_survey",
    # Cross-matching
    "CrossMatcher",
    # Analysis
    "CosmicWebAnalyzer",
    "analyze_cosmic_web",
    "SpatialClustering",
    "FilamentDetector",
    "StructureAnalyzer",
    # Graph Construction
    "AstronomicalGraphBuilder",
    "PointCloudGraphBuilder",
    "create_pointcloud_graph",
    "create_multiscale_pointcloud_graph",
]
