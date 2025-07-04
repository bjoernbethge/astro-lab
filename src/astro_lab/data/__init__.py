"""AstroLab Data Module - Unified astronomical data pipeline.

Exports all core preprocessors, datasets, samplers, transforms, collectors, datamodules, and analysis tools for astronomical ML workflows.
"""

# Analysis
from .analysis import (
    BaseAutoencoder,
    CosmicWebAnalyzer,
    FilamentDetector,
    PointCloudAutoencoder,
    ScalableCosmicWebAnalyzer,
    SpatialClustering,
    StructureAnalyzer,
    analyze_cosmic_web_50m,
    analyze_with_autoencoder,
)

# Collectors
from .collectors import (
    COLLECTOR_REGISTRY,
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
    get_available_collectors,
    get_collector,
    register_collector,
)

# Dataset
from .dataset import AstroLabDataModule, AstroLabInMemoryDataset

# Preprocessors
from .preprocessors import (
    PREPROCESSOR_REGISTRY,
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
    get_preprocessor,
    list_available_surveys,
)

# Samplers
from .samplers import (
    SAMPLER_REGISTRY,
    AdaptiveGraphSAINTSampler,
    AdaptiveRadiusSampler,
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
    NeighborSubgraphSampler,
    RadiusSampler,
    SpatialSamplerMixin,
    get_sampler,
    list_available_samplers,
)

# Transforms
from .transforms import (
    AstronomicalFeatures,
    Center,
    Compose,
    CosmicWebClassification,
    CrossMatchObjects,
    Delaunay,
    DensityFieldEstimation,
    ExtinctionCorrection,
    FilamentDetection,
    GalacticCoordinateTransform,
    KNNGraph,
    LinearTransformation,
    MultiSurveyMerger,
    NormalizeFeatures,
    ProperMotionCorrection,
    RadiusGraph,
    RandomJitter,
    RandomRotate,
    RandomTranslate,
    ToDevice,
)
