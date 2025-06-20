"""
AstroLab Models Package

Modern Graph Neural Network models für astronomische Daten mit nativer
AstroLab tensor integration und PyTorch Geometric 2.6+ support.

Available Models:
- AstroSurveyGNN: Main survey-specific GNN
- ALCDEFTemporalGNN: Temporal model for lightcurve analysis
- AstroPhotGNN: Galaxy modeling with AstroPhot integration
- TemporalGCN: Base temporal graph networks

Factory Functions:
- create_gaia_classifier()
- create_sdss_galaxy_classifier()
- create_lsst_transient_detector()
- create_multi_survey_model()
"""

# Base classes
# Main models
from .astro import (
    AstroSurveyGNN,
)
from .astrophot_models import AstroPhotGNN, NSAGalaxyModeler
from .base_gnn import BaseAstroGNN, BaseTemporalGNN, BaseTNGModel, FeatureFusion

# Encoders
from .encoders import (
    AstrometryEncoder,
    BaseEncoder,
    LightcurveEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
)

# Factory and registry
from .factory import (
    ModelFactory,
    ModelRegistry,
    compile_astro_model,
    create_asteroid_period_detector,
    create_gaia_classifier,
    create_galactic_structure_model,
    create_lightcurve_classifier,
    create_lsst_transient_detector,
    create_sdss_galaxy_model,
    create_stellar_cluster_analyzer,
    get_model_info,
    list_available_models,
)

# Output heads
from .output_heads import (
    ClassificationHead,
    CosmologicalHead,
    MultiTaskHead,
    OutputHeadRegistry,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
    create_output_head,
)

# Point cloud models
from .point_cloud_models import (
    GalacticStructureGNN,
    HierarchicalStellarGNN,
    StellarClusterGNN,
    StellarPointCloudGNN,
    create_stellar_point_cloud_model,
)

# Temporal models
from .tgnn import (
    ALCDEFTemporalGNN,
    ClassificationHead,
    PeriodDetectionHead,
    ShapeModelingHead,
    TemporalGATCNN,
    TemporalGCN,
)

# TNG-specific models
from .tng_models import (
    CosmicEvolutionGNN,
    EnvironmentalQuenchingGNN,
    GalaxyFormationGNN,
    HaloMergerGNN,
)

# Model utilities
from .utils import AttentionPooling, get_activation, get_pooling, initialize_weights

__version__ = "0.3.0"

__all__ = [
    # Base Classes
    "BaseAstroGNN",
    "BaseTemporalGNN",
    "BaseTNGModel",
    "FeatureFusion",
    # Main Models
    "AstroSurveyGNN",
    "AstroPhotGNN",
    "ALCDEFTemporalGNN",
    "TemporalGCN",
    # Point Cloud Models
    "StellarPointCloudGNN",
    "HierarchicalStellarGNN",
    "StellarClusterGNN",
    "GalacticStructureGNN",
    # Encoders
    "BaseEncoder",
    "PhotometryEncoder",
    "AstrometryEncoder",
    "SpectroscopyEncoder",
    "LightcurveEncoder",
    # Output Heads
    "OutputHeadRegistry",
    "RegressionHead",
    "ClassificationHead",
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "MultiTaskHead",
    "CosmologicalHead",
    "create_output_head",
    # Factory Functions
    "ModelFactory",
    "ModelRegistry",
    "create_gaia_classifier",
    "create_sdss_galaxy_model",
    "create_lsst_transient_detector",
    "create_asteroid_period_detector",
    "create_lightcurve_classifier",
    "create_stellar_cluster_analyzer",
    "create_galactic_structure_model",
    "create_stellar_point_cloud_model",
    "compile_astro_model",
    "get_model_info",
    "list_available_models",
    # Utilities
    "AttentionPooling",
    "get_activation",
    "get_pooling",
    "initialize_weights",
    # Galaxy Modeling
    "NSAGalaxyModeler",
    # Base temporal models
    "TemporalGCN",
    "TemporalGATCNN",
    # ALCDEF temporal models
    "ALCDEFTemporalGNN",
    # TNG-specific models
    "CosmicEvolutionGNN",
    "GalaxyFormationGNN",
    "HaloMergerGNN",
    "EnvironmentalQuenchingGNN",
]

# Model capabilities summary
CAPABILITIES = {
    "surveys": ["gaia", "sdss", "lsst", "euclid", "des", "ps1", "2mass", "wise"],
    "tasks": [
        "stellar_classification",
        "galaxy_property_prediction",
        "transient_detection",
        "period_detection",
        "shape_modeling",
        "asteroid_classification",
    ],
    "tensor_types": [
        "SurveyTensor",
        "PhotometricTensor",
        "SpectralTensor",
        "Spatial3DTensor",
        "LightcurveTensor",
    ],
    "frameworks": ["pytorch_geometric", "torch_compile", "astrolab_tensors"],
}
