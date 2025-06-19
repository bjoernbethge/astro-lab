"""
Pydantic schemas for data module configurations.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DatasetConfigSchema(BaseModel):
    """Base configuration schema for all datasets."""
    
    root: Optional[str] = Field(
        default=None,
        description="Root directory for dataset files"
    )
    transform: Optional[str] = Field(
        default=None,
        description="Transform to apply to each sample"
    )
    pre_transform: Optional[str] = Field(
        default=None,
        description="Transform to apply before saving"
    )
    pre_filter: Optional[str] = Field(
        default=None,
        description="Filter to apply before saving"
    )


class GaiaDatasetConfigSchema(DatasetConfigSchema):
    """Configuration schema for GaiaGraphDataset."""
    
    magnitude_limit: float = Field(
        default=12.0,
        ge=5.0,
        le=20.0,
        description="Magnitude limit for star selection"
    )
    k_neighbors: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of nearest neighbors for graph construction"
    )
    max_distance: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum distance for connections (kpc)"
    )


class NSADatasetConfigSchema(DatasetConfigSchema):
    """Configuration schema for NSAGraphDataset."""
    
    max_galaxies: int = Field(
        default=10000,
        ge=10,
        le=1000000,
        description="Maximum number of galaxies to include"
    )
    k_neighbors: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of nearest neighbors for graph construction"
    )
    distance_threshold: float = Field(
        default=50.0,
        gt=0.0,
        description="Distance threshold for connections (Mpc)"
    )


class ExoplanetDatasetConfigSchema(DatasetConfigSchema):
    """Configuration schema for ExoplanetGraphDataset."""
    
    k_neighbors: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of nearest neighbors for graph construction"
    )
    max_distance: float = Field(
        default=100.0,
        gt=0.0,
        description="Maximum distance for connections (parsecs)"
    )


class DataLoaderConfigSchema(BaseModel):
    """Configuration schema for PyTorch Geometric DataLoaders."""
    
    batch_size: int = Field(
        default=32,
        ge=1,
        le=10000,
        description="Batch size for data loading"
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle the data"
    )
    num_workers: int = Field(
        default=0,
        ge=0,
        le=16,
        description="Number of worker processes for data loading"
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to pin memory for GPU transfer"
    )
    use_gpu_optimization: bool = Field(
        default=True,
        description="Whether to use GPU optimization if available"
    )


class ProcessingConfigSchema(BaseModel):
    """Configuration schema for data processing."""
    
    device: str = Field(
        default="auto",
        description="Device for tensor operations (auto, cpu, cuda, mps)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=10000,
        description="Batch size for processing"
    )
    max_samples: Optional[Dict[str, int]] = Field(
        default=None,
        description="Maximum samples per survey"
    )
    surveys: Optional[List[str]] = Field(
        default=None,
        description="List of surveys to process"
    ) 