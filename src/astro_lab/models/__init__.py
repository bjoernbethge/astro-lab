"""
AstroLab Models
==============

Neural network models for astronomical data processing.
"""

# Core models
from .astro_model import AstroModel

# Autoencoders
from .autoencoders.pointcloud_autoencoder import PointCloudAutoencoder
from .base_model import AstroBaseModel

# Encoders
from .encoders import (
    PhotometricEncoderModule,
)

# Layers - Other
from .layers import (
    AdaptivePointCloudLayer,
    AdaptivePooling,
    AstronomicalGraphConv,
    # Point cloud layers
    AstroPointCloudLayer,
    # Pooling layers
    AttentivePooling,
    BaseGraphLayer,
    # Base layers
    BaseLayer,
    BasePoolingLayer,
    BatchNorm,
    # Convolution layers
    FlexibleGraphConv,
    # Normalization layers
    GraphNorm,
    # Graph layers
    GraphPooling,
    # Heterogeneous layers
    HeteroGNNLayer,
    HierarchicalPooling,
    InstanceNorm,
    LambdaPooling,
    LayerNorm,
    MultiScalePointCloudEncoder,
    MultiScalePooling,
    StatisticalPooling,
)

# Layers - Decoders
from .layers.decoders import (
    AdvancedTemporalDecoder,
    MLPDecoder,
    ModernGraphDecoder,
    PointNetDecoder,
)

# Layers - Encoders
from .layers.encoders import (
    AdvancedTemporalEncoder,
    MLPEncoder,
    ModernGraphEncoder,
    PointNetEncoder,
)

# Layers - Heads
from .layers.heads import (
    ClassificationHead,
    PeriodDetectionHead,
    RegressionHead,
    ShapeModelingHead,
)
from .mixins.analysis import ModelAnalysisMixin

# Mixins
from .mixins.explainability import ExplainabilityMixin

__all__ = [
    "AstroModel",
    "AstroBaseModel",
    "ExplainabilityMixin",
    "ModelAnalysisMixin",
]

# For layers, encoders, decoders, heads: import from their respective submodules.
