"""
Training Configuration Management
================================

Pydantic-based configuration management for AstroLab training.
Integrates with model configurations and provides training-specific settings.
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import torch
from pydantic import BaseModel, Field, validator

from astro_lab.models.config import ModelConfig


class SchedulerConfig(BaseModel):
    """Configuration for learning rate schedulers."""

    scheduler_type: Literal["cosine", "onecycle", "plateau", "step", "exponential"] = (
        "cosine"
    )
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)

    # Scheduler-specific parameters
    warmup_epochs: int = Field(default=5, ge=0)
    max_epochs: int = Field(default=100, ge=1)

    # Cosine annealing
    eta_min: float = Field(default=1e-6, gt=0.0)

    # OneCycle
    div_factor: float = Field(default=25.0, gt=0.0)
    final_div_factor: float = Field(default=1e4, gt=0.0)

    # Plateau
    patience: int = Field(default=10, ge=1)
    factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    min_lr: float = Field(default=1e-6, gt=0.0)

    # Step
    step_size: int = Field(default=30, ge=1)
    gamma: float = Field(default=0.1, gt=0.0)

    # Exponential
    decay_rate: float = Field(default=0.95, gt=0.0, lt=1.0)

    @validator("learning_rate", "eta_min", "min_lr")
    def validate_learning_rates(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v


class CallbackConfig(BaseModel):
    """Configuration for training callbacks."""

    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = Field(default=15, ge=1)
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: Literal["min", "max"] = "min"

    # Model checkpointing
    model_checkpoint: bool = True
    save_top_k: int = Field(default=3, ge=1)
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"

    # Learning rate monitoring
    lr_monitor: bool = True
    lr_monitor_logging_interval: Literal["step", "epoch"] = "epoch"

    # Stochastic weight averaging
    swa: bool = False
    swa_epoch_start: float = Field(default=0.8, ge=0.0, le=1.0)
    swa_lrs: Optional[List[float]] = None

    # Rich progress bar
    rich_progress: bool = True

    # Custom callbacks
    custom_callbacks: List[str] = Field(default_factory=list)


class HardwareConfig(BaseModel):
    """Configuration for hardware and performance settings."""

    # Device configuration
    accelerator: Literal["auto", "cpu", "gpu", "tpu"] = "auto"
    devices: Union[int, List[int], Literal["auto"]] = "auto"

    # Precision
    precision: Literal["32", "16", "16-mixed", "bf16", "bf16-mixed"] = "16-mixed"

    # Performance optimizations
    gradient_clip_val: Optional[float] = Field(default=1.0, gt=0.0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    deterministic: bool = False

    # Memory optimization
    enable_progress_bar: bool = True
    enable_model_summary: bool = True
    enable_checkpointing: bool = True
    num_sanity_val_steps: int = Field(default=0, ge=0)

    @validator("devices")
    def validate_devices(cls, v):
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("Devices list cannot be empty")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging and experiment tracking."""

    # MLflow configuration
    use_mlflow: bool = True
    experiment_name: str = "astro_experiment"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None

    # Logging frequency
    log_every_n_steps: int = Field(default=1, ge=1)
    val_check_interval: float = Field(default=1.0, gt=0.0)

    # Metrics to log
    log_metrics: List[str] = Field(
        default_factory=lambda: [
            "train_loss",
            "val_loss",
            "train_acc",
            "val_acc",
            "learning_rate",
        ]
    )

    # Custom logging
    custom_loggers: List[str] = Field(default_factory=list)


class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""

    # Data loading
    batch_size: int = Field(default=32, ge=1)
    num_workers: int = Field(default=4, ge=0)
    pin_memory: bool = True
    persistent_workers: bool = True

    # Data augmentation
    use_augmentation: bool = False
    augmentation_config: Dict[str, Any] = Field(default_factory=dict)

    # Validation split
    val_split: float = Field(default=0.2, gt=0.0, lt=1.0)
    test_split: float = Field(default=0.1, gt=0.0, lt=1.0)

    # Data paths
    data_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    @validator("val_split", "test_split")
    def validate_splits(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("Split must be between 0 and 1")
        return v


class TrainingConfig(BaseModel):
    """Complete training configuration integrating model and training settings."""

    # Basic training info
    name: str = Field(..., min_length=1)
    version: str = "1.0.0"
    description: Optional[str] = None

    # Model configuration
    model: ModelConfig

    # Training components
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    callbacks: CallbackConfig = Field(default_factory=CallbackConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Override model training config with training-specific settings
    training: Optional[Dict[str, Any]] = None

    # Survey-specific overrides
    survey_overrides: Dict[str, Dict[str, Union[str, int, float, bool]]] = Field(
        default_factory=dict
    )

    class Config:
        extra = "forbid"

    def __init__(self, **data):
        super().__init__(**data)

        # Sync training config if provided
        if self.training is not None:
            self.scheduler.learning_rate = self.training.get(
                "learning_rate", self.scheduler.learning_rate
            )
            self.scheduler.weight_decay = self.training.get(
                "weight_decay", self.scheduler.weight_decay
            )
            self.scheduler.max_epochs = self.training.get(
                "num_epochs", self.scheduler.max_epochs
            )
            self.data.batch_size = self.training.get("batch_size", self.data.batch_size)
            self.hardware.gradient_clip_val = self.training.get(
                "gradient_clip_val", self.hardware.gradient_clip_val
            )

    def get_survey_config(self, survey: str) -> "TrainingConfig":
        """Get configuration with survey-specific overrides."""
        if survey not in self.survey_overrides:
            return self

        # Create a copy with overrides
        config_dict = self.dict()
        overrides = self.survey_overrides[survey]

        # Apply overrides recursively
        for key, value in overrides.items():
            keys = key.split(".")
            current = config_dict
            for k in keys[:-1]:
                current = current[k]
            current[keys[-1]] = value

        return TrainingConfig(**config_dict)

    def validate_hardware_compatibility(self) -> bool:
        """Validate that the configuration is compatible with available hardware."""
        # Check CUDA availability
        if self.hardware.accelerator == "gpu" and not torch.cuda.is_available():
            return False

        # Check precision compatibility
        if (
            self.hardware.precision in ["16", "16-mixed"]
            and not torch.cuda.is_available()
        ):
            return False

            # Model compatibility check not needed with simplified config
        return True

    def get_optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {
            "lr": self.scheduler.learning_rate,
            "weight_decay": self.scheduler.weight_decay,
        }

    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        base_config = {
            "scheduler_type": self.scheduler.scheduler_type,
            "max_epochs": self.scheduler.max_epochs,
            "warmup_epochs": self.scheduler.warmup_epochs,
        }

        if self.scheduler.scheduler_type == "cosine":
            base_config.update(
                {
                    "eta_min": self.scheduler.eta_min,
                }
            )
        elif self.scheduler.scheduler_type == "onecycle":
            base_config.update(
                {
                    "div_factor": self.scheduler.div_factor,
                    "final_div_factor": self.scheduler.final_div_factor,
                }
            )
        elif self.scheduler.scheduler_type == "plateau":
            base_config.update(
                {
                    "patience": self.scheduler.patience,
                    "factor": self.scheduler.factor,
                    "min_lr": self.scheduler.min_lr,
                }
            )
        elif self.scheduler.scheduler_type == "step":
            base_config.update(
                {
                    "step_size": self.scheduler.step_size,
                    "gamma": self.scheduler.gamma,
                }
            )
        elif self.scheduler.scheduler_type == "exponential":
            base_config.update(
                {
                    "decay_rate": self.scheduler.decay_rate,
                }
            )

        return base_config


# Predefined training configurations
PREDEFINED_TRAINING_CONFIGS = {
    "gaia_stellar_training": TrainingConfig(
        name="gaia_stellar_training",
        description="Training configuration for Gaia stellar classification",
        model=ModelConfig(
            name="gaia_stellar_classifier",
            hidden_dim=128,
            num_layers=3,
            conv_type="gat",
            task="classification",
            output_dim=7,
        ),
        scheduler=SchedulerConfig(
            scheduler_type="cosine",
            learning_rate=1e-3,
            warmup_epochs=5,
            max_epochs=100,
        ),
        hardware=HardwareConfig(
            precision="16-mixed",
            gradient_clip_val=1.0,
        ),
        data=DataConfig(
            batch_size=64,
            num_workers=4,
        ),
    ),
    "sdss_galaxy_training": TrainingConfig(
        name="sdss_galaxy_training",
        description="Training configuration for SDSS galaxy property prediction",
        model=ModelConfig(
            name="sdss_galaxy_modeler",
            hidden_dim=256,
            num_layers=4,
            conv_type="transformer",
            task="regression",
            output_dim=5,
            pooling="mean",
        ),
        scheduler=SchedulerConfig(
            scheduler_type="onecycle",
            learning_rate=5e-4,
            max_epochs=150,
        ),
        hardware=HardwareConfig(
            precision="16-mixed",
            gradient_clip_val=0.5,
        ),
        data=DataConfig(
            batch_size=32,
            num_workers=6,
        ),
    ),
    "lsst_transient_training": TrainingConfig(
        name="lsst_transient_training",
        description="Training configuration for LSST transient detection",
        model=ModelConfig(
            name="lsst_transient_detector",
            hidden_dim=192,
            num_layers=3,
            conv_type="sage",
            task="classification",
            output_dim=2,
            pooling="max",
        ),
        scheduler=SchedulerConfig(
            scheduler_type="plateau",
            learning_rate=1e-3,
            max_epochs=200,
        ),
        hardware=HardwareConfig(
            precision="16-mixed",
            gradient_clip_val=1.0,
        ),
        data=DataConfig(
            batch_size=128,
            num_workers=8,
        ),
    ),
}


def get_predefined_training_config(name: str) -> TrainingConfig:
    """Get a predefined training configuration."""
    if name not in PREDEFINED_TRAINING_CONFIGS:
        available = list(PREDEFINED_TRAINING_CONFIGS.keys())
        raise ValueError(f"Unknown training config '{name}'. Available: {available}")

    return PREDEFINED_TRAINING_CONFIGS[name]


def list_predefined_training_configs() -> List[str]:
    """List all available predefined training configurations."""
    return list(PREDEFINED_TRAINING_CONFIGS.keys())
