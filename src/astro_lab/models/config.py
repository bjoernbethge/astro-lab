"""
Model Configuration Management
=============================

Pydantic-based configuration management for AstroLab models.
Provides type safety, validation, and serialization for model configurations.
"""

from typing import Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field, validator
import torch

# Type aliases
ConvType = Literal["gcn", "gat", "sage", "transformer"]
TaskType = Literal["node_classification", "node_regression", "graph_classification", "stellar_classification", "galaxy_property_prediction", "transient_detection", "period_detection", "shape_modeling"]
PoolingType = Literal["mean", "max", "add", "attention"]
ActivationType = Literal["relu", "gelu", "swish", "mish", "leaky_relu"]

class EncoderConfig(BaseModel):
    """Configuration for feature encoders."""
    
    use_photometry: bool = True
    use_astrometry: bool = True
    use_spectroscopy: bool = False
    use_lightcurve: bool = False
    use_spatial_3d: bool = False
    
    # Encoder-specific settings
    photometry_dim: int = Field(default=64, ge=1)
    astrometry_dim: int = Field(default=64, ge=1)
    spectroscopy_dim: int = Field(default=64, ge=1)
    lightcurve_dim: int = Field(default=64, ge=1)
    spatial_3d_dim: int = Field(default=64, ge=1)
    
    @validator('photometry_dim', 'astrometry_dim', 'spectroscopy_dim', 'lightcurve_dim', 'spatial_3d_dim')
    def validate_dimensions(cls, v):
        if v <= 0:
            raise ValueError("Dimension must be positive")
        return v

class GraphConfig(BaseModel):
    """Configuration for graph neural network layers."""
    
    conv_type: ConvType = "gcn"
    hidden_dim: int = Field(default=128, ge=1)
    num_layers: int = Field(default=3, ge=1, le=10)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    activation: ActivationType = "relu"
    use_residual: bool = True
    use_layer_norm: bool = True
    
    # GAT/Transformer specific
    num_heads: int = Field(default=8, ge=1)
    
    @validator('num_heads')
    def validate_heads(cls, v, values):
        if 'hidden_dim' in values and v > values['hidden_dim']:
            raise ValueError("num_heads cannot be greater than hidden_dim")
        return v

class OutputConfig(BaseModel):
    """Configuration for output heads."""
    
    task: TaskType = "node_classification"
    output_dim: Optional[int] = Field(default=None, ge=1)  # None = auto-detect from data
    pooling: PoolingType = "mean"
    use_attention: bool = False
    attention_dim: Optional[int] = None
    
    @validator('attention_dim')
    def validate_attention_dim(cls, v, values):
        if values.get('use_attention') and v is None:
            return values.get('output_dim', 1)
        return v

class TrainingConfig(BaseModel):
    """Configuration for training settings."""
    
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=32, ge=1)
    num_epochs: int = Field(default=100, ge=1)
    early_stopping_patience: int = Field(default=10, ge=1)
    gradient_clip_val: Optional[float] = Field(default=1.0, gt=0.0)
    
    # Loss function configuration
    loss_function: str = "cross_entropy"  # or "mse", "mae", etc.
    class_weights: Optional[List[float]] = None
    
    @validator('class_weights')
    def validate_class_weights(cls, v):
        if v is not None:
            if any(w <= 0 for w in v):
                raise ValueError("Class weights must be positive")
        return v

class ModelConfig(BaseModel):
    """Complete model configuration."""
    
    name: str = Field(..., min_length=1)
    version: str = "1.0.0"
    description: Optional[str] = None
    
    # Component configurations
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # Survey-specific overrides
    survey_overrides: Dict[str, Dict[str, Union[str, int, float, bool]]] = Field(default_factory=dict)
    
    class Config:
        extra = "forbid"  # Prevent additional fields
    
    def get_survey_config(self, survey: str) -> 'ModelConfig':
        """Get configuration with survey-specific overrides."""
        if survey not in self.survey_overrides:
            return self
        
        # Create a copy with overrides
        config_dict = self.dict()
        overrides = self.survey_overrides[survey]
        
        # Apply overrides recursively
        for key, value in overrides.items():
            keys = key.split('.')
            current = config_dict
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value
        
        return ModelConfig(**config_dict)
    
    def validate_device_compatibility(self, device: torch.device) -> bool:
        """Validate that the model configuration is compatible with the device."""
        # Check if hidden dimensions are reasonable for the device
        if device.type == 'cuda':
            total_params = self._estimate_parameters()
            if total_params > 100_000_000:  # 100M parameters
                return False
        return True
    
    def _estimate_parameters(self) -> int:
        """Estimate the number of parameters in the model."""
        # Rough estimation based on configuration
        encoder_params = sum([
            self.encoder.photometry_dim * self.graph.hidden_dim,
            self.encoder.astrometry_dim * self.graph.hidden_dim,
            self.encoder.spectroscopy_dim * self.graph.hidden_dim,
        ])
        
        graph_params = self.graph.num_layers * self.graph.hidden_dim * self.graph.hidden_dim
        output_params = self.graph.hidden_dim * self.output.output_dim
        
        return encoder_params + graph_params + output_params

# Predefined configurations
PREDEFINED_CONFIGS = {
    "gaia_stellar_classifier": ModelConfig(
        name="gaia_stellar_classifier",
        description="Stellar classification model for Gaia data",
        encoder=EncoderConfig(
            use_photometry=True,
            use_astrometry=True,
            use_spectroscopy=False,
        ),
        graph=GraphConfig(
            conv_type="gat",
            hidden_dim=128,
            num_layers=3,
        ),
        output=OutputConfig(
            task="stellar_classification",
            output_dim=7,
        ),
    ),
    
    "sdss_galaxy_modeler": ModelConfig(
        name="sdss_galaxy_modeler",
        description="Galaxy property prediction for SDSS data",
        encoder=EncoderConfig(
            use_photometry=True,
            use_spectroscopy=True,
            use_astrometry=False,
        ),
        graph=GraphConfig(
            conv_type="transformer",
            hidden_dim=256,
            num_layers=4,
        ),
        output=OutputConfig(
            task="galaxy_property_prediction",
            output_dim=5,
            pooling="attention",
        ),
    ),
    
    "lsst_transient_detector": ModelConfig(
        name="lsst_transient_detector",
        description="Transient detection for LSST data",
        encoder=EncoderConfig(
            use_photometry=True,
            use_astrometry=True,
            use_lightcurve=True,
        ),
        graph=GraphConfig(
            conv_type="sage",
            hidden_dim=192,
            num_layers=3,
        ),
        output=OutputConfig(
            task="transient_detection",
            output_dim=2,
            pooling="max",
        ),
    ),
}

def get_predefined_config(name: str) -> ModelConfig:
    """Get a predefined model configuration."""
    if name not in PREDEFINED_CONFIGS:
        available = list(PREDEFINED_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    
    return PREDEFINED_CONFIGS[name]

def list_predefined_configs() -> List[str]:
    """List all available predefined configurations."""
    return list(PREDEFINED_CONFIGS.keys()) 