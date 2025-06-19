"""
Pydantic schemas for training configurations.
"""

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field


class TrainingConfigSchema(BaseModel):
    """Configuration schema for training parameters."""
    
    max_epochs: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=10000,
        description="Training batch size"
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="Weight decay for regularization"
    )
    patience: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Early stopping patience"
    )
    min_delta: float = Field(
        default=1e-4,
        ge=0.0,
        description="Minimum change for early stopping"
    )


class OptimizerConfigSchema(BaseModel):
    """Configuration schema for optimizers."""
    
    optimizer_type: str = Field(
        default="adam",
        description="Optimizer type (adam, sgd, adamw, etc.)"
    )
    learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        le=1.0,
        description="Learning rate"
    )
    weight_decay: float = Field(
        default=1e-4,
        ge=0.0,
        le=1.0,
        description="Weight decay"
    )
    momentum: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Momentum (for SGD)"
    )
    betas: Optional[tuple] = Field(
        default=(0.9, 0.999),
        description="Beta parameters (for Adam)"
    )


class SchedulerConfigSchema(BaseModel):
    """Configuration schema for learning rate schedulers."""
    
    scheduler_type: str = Field(
        default="plateau",
        description="Scheduler type (plateau, cosine, step, etc.)"
    )
    factor: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Factor to reduce learning rate"
    )
    patience: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Patience for plateau scheduler"
    )
    min_lr: float = Field(
        default=1e-6,
        gt=0.0,
        description="Minimum learning rate"
    )


class OptunaConfigSchema(BaseModel):
    """Configuration schema for Optuna hyperparameter optimization."""
    
    n_trials: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Number of optimization trials"
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        description="Timeout in seconds"
    )
    study_name: Optional[str] = Field(
        default=None,
        description="Name of the study"
    )
    storage: Optional[str] = Field(
        default=None,
        description="Storage backend URL"
    )
    sampler: str = Field(
        default="tpe",
        description="Sampling algorithm (tpe, random, cmaes)"
    )
    pruner: str = Field(
        default="median",
        description="Pruning algorithm (median, hyperband, none)"
    )


class MLflowConfigSchema(BaseModel):
    """Configuration schema for MLflow logging."""
    
    experiment_name: str = Field(
        ...,
        description="MLflow experiment name"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="MLflow run name"
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI"
    )
    log_model: bool = Field(
        default=True,
        description="Whether to log the model"
    )
    log_artifacts: bool = Field(
        default=True,
        description="Whether to log artifacts"
    )
    tags: Optional[Dict[str, str]] = Field(
        default=None,
        description="Tags for the run"
    )


class LightningConfigSchema(BaseModel):
    """Configuration schema for PyTorch Lightning."""
    
    accelerator: str = Field(
        default="auto",
        description="Accelerator type (auto, cpu, gpu, tpu)"
    )
    devices: Union[int, str, List[int]] = Field(
        default="auto",
        description="Number or list of devices to use"
    )
    precision: Union[int, str] = Field(
        default=32,
        description="Training precision (16, 32, 64, bf16)"
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Training strategy (ddp, fsdp, etc.)"
    )
    max_epochs: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum number of epochs"
    )
    gradient_clip_val: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Gradient clipping value"
    )
    accumulate_grad_batches: int = Field(
        default=1,
        ge=1,
        description="Number of batches to accumulate gradients"
    ) 