"""AstroLab Data Module - Unified astronomical data pipeline.

Exports all core preprocessors, datasets, samplers, transforms, collectors,
datamodules, and analysis tools for astronomical ML workflows.
"""

# Analysis
from .analysis import (
    CosmicWebAnalyzer,
    FilamentDetector,
    ScalableCosmicWebAnalyzer,
    SpatialClustering,
    StructureAnalyzer,
    analyze_cosmic_web_50m,
)

# Collectors
from .collectors import (
    BaseSurveyCollector,
    DESCollector,
    EuclidCollector,
    ExoplanetCollector,
    GaiaCollector,
    LinearCollector,
    NSACollector,
    PanSTARRSCollector,
    RRLyraeCollector,
    SDSSCollector,
    TNG50Collector,
    TwoMASSCollector,
    WISECollector,
)

# Preprocessors
from .preprocessors import (
    AstroLabDataPreprocessor,
    AstronomicalPreprocessorMixin,
    DESPreprocessor,
    ExoplanetPreprocessor,
    GaiaPreprocessor,
    LINEARPreprocessor,
    NSAPreprocessor,
    RRLyraePreprocessor,
    SDSSPreprocessor,
    StatisticalPreprocessorMixin,
    TNG50Preprocessor,
    TwoMASSPreprocessor,
    WISEPreprocessor,
)

# Samplers
from .samplers import (
    AdaptiveGraphSAINTSampler,
    AstroLabSampler,
    AstronomicalSamplerMixin,
    ClusterSampler,
    ClusterSamplerMixin,
    DBSCANClusterSampler,
    GraphSAINTEdgeSamplerWrapper,
    GraphSAINTNodeSamplerWrapper,
    GraphSAINTRandomWalkSamplerWrapper,
    HierarchicalClusterSampler,
    KNNSampler,
    SpatialSamplerMixin,
)
from .samplers.AdaptiveRadiusSampler import AdaptiveRadiusSampler
from .samplers.NeighborSubgraphSampler import NeighborSubgraphSampler
from .samplers.RadiusSampler import RadiusSampler

# Transforms
from .transforms import (
    AstronomicalFeatures,
    CosmicWebClassification,
    CrossMatchObjects,
    DensityFieldEstimation,
    ExtinctionCorrection,
    FilamentDetection,
    GalacticCoordinateTransform,
    HaloIdentification,
    MultiScaleSampling,
    MultiSurveyMerger,
    ProperMotionCorrection,
    VoidDetection,
)

__all__ = [
    # Analysis
    "CosmicWebAnalyzer",
    "FilamentDetector",
    "ScalableCosmicWebAnalyzer",
    "SpatialClustering",
    "StructureAnalyzer",
    "analyze_cosmic_web_50m",
    # Collectors
    "BaseSurveyCollector",
    "DESCollector",
    "EuclidCollector",
    "ExoplanetCollector",
    "GaiaCollector",
    "LinearCollector",
    "NSACollector",
    "PanSTARRSCollector",
    "RRLyraeCollector",
    "SDSSCollector",
    "TNG50Collector",
    "TwoMASSCollector",
    "WISECollector",
    # Preprocessors
    "AstroLabDataPreprocessor",
    "AstronomicalPreprocessorMixin",
    "DESPreprocessor",
    "ExoplanetPreprocessor",
    "GaiaPreprocessor",
    "LINEARPreprocessor",
    "NSAPreprocessor",
    "RRLyraePreprocessor",
    "SDSSPreprocessor",
    "StatisticalPreprocessorMixin",
    "TNG50Preprocessor",
    "TwoMASSPreprocessor",
    "WISEPreprocessor",
    # Samplers
    "AdaptiveGraphSAINTSampler",
    "AdaptiveRadiusSampler",
    "AstroLabSampler",
    "AstronomicalSamplerMixin",
    "ClusterSampler",
    "ClusterSamplerMixin",
    "DBSCANClusterSampler",
    "GraphSAINTEdgeSamplerWrapper",
    "GraphSAINTNodeSamplerWrapper",
    "GraphSAINTRandomWalkSamplerWrapper",
    "HierarchicalClusterSampler",
    "KNNSampler",
    "NeighborSubgraphSampler",
    "RadiusSampler",
    "SpatialSamplerMixin",
    # Transforms
    "AstronomicalFeatures",
    "CosmicWebClassification",
    "CrossMatchObjects",
    "DensityFieldEstimation",
    "ExtinctionCorrection",
    "FilamentDetection",
    "GalacticCoordinateTransform",
    "HaloIdentification",
    "MultiScaleSampling",
    "MultiSurveyMerger",
    "ProperMotionCorrection",
    "VoidDetection",
]
