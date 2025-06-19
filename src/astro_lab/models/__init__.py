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

# Main models
from .astro import (
    AstroSurveyGNN,
)
from .astrophot_models import AstroPhotGNN, NSAGalaxyModeler

# Encoders
from .encoders import (
    AstrometryEncoder,
    LightcurveEncoder,
    PhotometryEncoder,
    SpectroscopyEncoder,
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
    # Main Models
    "AstroSurveyGNN",
    "AstroPhotGNN",
    "ALCDEFTemporalGNN",
    "TemporalGCN",
    # Encoders
    "PhotometryEncoder",
    "AstrometryEncoder",
    "SpectroscopyEncoder",
    # Output Heads
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "ClassificationHead",
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
    "PeriodDetectionHead",
    "ShapeModelingHead",
    "ClassificationHead",
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
