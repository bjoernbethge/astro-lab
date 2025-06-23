"""
AstroLab Models - Modern Graph Neural Networks for Astronomical Data

Modern Graph Neural Network models for astronomical data with native
PyTorch Geometric integration and specialized astronomical layers.

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

Configuration Management:
- ModelConfig: Pydantic-based model configuration
- LayerFactory: Centralized layer creation
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)

# Configuration management
from .config import (
    ModelConfig,
    EncoderConfig,
    GraphConfig,
    OutputConfig,
    TrainingConfig,
    get_predefined_config,
    list_predefined_configs,
    PREDEFINED_CONFIGS,
)

# Layer factory and components
from .layers import (
    LayerFactory,
    AttentionPooling,
    ResidualBlock,
    FeatureFusion,
    LayerRegistry,
)

# Base classes
from .base_gnn import BaseAstroGNN, BaseTemporalGNN, BaseTNGModel

# Main models
from .astro import (
    AstroSurveyGNN,
)
from .astrophot_models import AstroPhotGNN, NSAGalaxyModeler

# Encoders
from .encoders import (
    AstrometryEncoder,
    BaseEncoder,
    LightcurveEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
    create_encoder,
    EncoderRegistry,
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

__version__ = "0.4.0"

__all__ = [
    # Configuration Management
    "ModelConfig",
    "EncoderConfig", 
    "GraphConfig",
    "OutputConfig",
    "TrainingConfig",
    "get_predefined_config",
    "list_predefined_configs",
    "PREDEFINED_CONFIGS",
    
    # Layer Factory
    "LayerFactory",
    "AttentionPooling",
    "ResidualBlock", 
    "FeatureFusion",
    "LayerRegistry",
    
    # Base Classes
    "BaseAstroGNN",
    "BaseTemporalGNN",
    "BaseTNGModel",
    
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
    "create_encoder",
    "EncoderRegistry",
    
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
    "config_management": ["pydantic", "type_safety", "validation"],
    "layer_factory": ["centralized", "consistent", "extensible"],
}
