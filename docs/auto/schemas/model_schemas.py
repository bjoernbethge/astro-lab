"""
Pydantic schemas for model configurations.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ModelConfigSchema(BaseModel):
    """Base configuration schema for all models."""
    
    model_type: str = Field(
        ...,
        description="Type of model (gnn, transformer, cnn, etc.)"
    )
    input_dim: int = Field(
        ...,
        ge=1,
        description="Input feature dimension"
    )
    hidden_dim: int = Field(
        default=64,
        ge=1,
        le=4096,
        description="Hidden layer dimension"
    )
    output_dim: int = Field(
        ...,
        ge=1,
        description="Output dimension"
    )
    dropout: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Dropout probability"
    )


class GNNConfigSchema(ModelConfigSchema):
    """Configuration schema for Graph Neural Networks."""
    
    model_type: str = Field(
        default="gcn",
        description="GNN type (gcn, gat, sage, gin, etc.)"
    )
    num_layers: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of GNN layers"
    )
    heads: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of attention heads (for GAT)"
    )
    edge_dim: Optional[int] = Field(
        default=None,
        description="Edge feature dimension"
    )
    aggr: str = Field(
        default="mean",
        description="Aggregation method (mean, max, add)"
    )
    residual: bool = Field(
        default=True,
        description="Use residual connections"
    )
    batch_norm: bool = Field(
        default=True,
        description="Use batch normalization"
    )


class TransformerConfigSchema(ModelConfigSchema):
    """Configuration schema for Transformer models."""
    
    model_type: str = Field(
        default="transformer",
        description="Transformer variant"
    )
    num_layers: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Number of transformer layers"
    )
    num_heads: int = Field(
        default=8,
        ge=1,
        le=32,
        description="Number of attention heads"
    )
    ff_dim: int = Field(
        default=256,
        ge=1,
        le=8192,
        description="Feed-forward layer dimension"
    )
    max_seq_length: int = Field(
        default=1024,
        ge=1,
        description="Maximum sequence length"
    )


class CNNConfigSchema(ModelConfigSchema):
    """Configuration schema for Convolutional Neural Networks."""
    
    model_type: str = Field(
        default="cnn",
        description="CNN architecture type"
    )
    num_layers: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of convolutional layers"
    )
    kernel_sizes: List[int] = Field(
        default_factory=lambda: [3, 3, 3, 3],
        description="Kernel sizes for each layer"
    )
    channels: List[int] = Field(
        default_factory=lambda: [32, 64, 128, 256],
        description="Number of channels for each layer"
    )
    pool_sizes: List[int] = Field(
        default_factory=lambda: [2, 2, 2, 2],
        description="Pooling sizes for each layer"
    )


class AstroPhotModelConfigSchema(ModelConfigSchema):
    """Configuration schema for AstroPhot models."""
    
    model_type: str = Field(
        default="astrophot",
        description="AstroPhot model variant"
    )
    cutout_size: int = Field(
        default=128,
        ge=32,
        le=512,
        description="Size of image cutouts"
    )
    pixel_scale: float = Field(
        default=0.262,
        gt=0.0,
        description="Pixel scale in arcsec/pixel"
    )
    magnitude_range: tuple = Field(
        default=(10.0, 18.0),
        description="Magnitude range for training"
    )


class TNG50ModelConfigSchema(ModelConfigSchema):
    """Configuration schema for TNG50 simulation models."""
    
    model_type: str = Field(
        default="tng50",
        description="TNG50 model type"
    )
    particle_types: List[str] = Field(
        default_factory=lambda: ["PartType0", "PartType1", "PartType4"],
        description="Particle types to include"
    )
    max_particles: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Maximum particles per sample"
    )
    environment_types: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Number of environment types"
    ) 