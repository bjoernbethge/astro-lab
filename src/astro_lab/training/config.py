"""
Training Configuration for AstroLab (2025 Optimized)
===================================================

Configuration classes for training with automatic optimization based on hardware.
"""

import logging
import multiprocessing
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class DataLoaderConfig:
    """Optimized DataLoader configuration."""
    num_workers: int = 0
    pin_memory: bool = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    drop_last: bool = True
    
    @classmethod
    def auto_configure(cls, use_gpu: bool = None, batch_size: int = 32) -> "DataLoaderConfig":
        """Auto-configure based on hardware."""
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
            
        if use_gpu:
            # GPU-optimized settings (2025 best practices)
            cpu_count = multiprocessing.cpu_count()
            
            # Optimal workers based on system
            if torch.cuda.get_device_properties(0).total_memory > 16e9:  # >16GB VRAM
                optimal_workers = min(16, cpu_count)
            else:
                optimal_workers = min(8, cpu_count)
                
            prefetch = 4 if optimal_workers >= 4 else 2
            if batch_size > 128:
                prefetch = 2  # Reduce for large batches
            elif batch_size < 32:
                prefetch = 8  # Increase for small batches
                
            return cls(
                num_workers=optimal_workers,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=prefetch,
                drop_last=True,
            )
        else:
            # CPU settings
            return cls(
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
                prefetch_factor=None,
                drop_last=False,
            )


@dataclass
class TrainingConfig:
    """Configuration for model training with 2025 optimizations."""

    # Basic settings
    dataset: str = "gaia"
    model_type: str = "node"  # node, graph, temporal, point
    model_name: Optional[str] = None
    task: Optional[str] = None  # Will be auto-assigned if not provided
    preset: Optional[str] = None  # Use preset configuration

    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)
    num_features: Optional[int] = None  # Auto-detected from data
    num_classes: Optional[int] = None  # Auto-detected from data
    hidden_dim: Optional[int] = None  # Defaults to num_features

    # Training settings
    max_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Advanced training
    gradient_clip_val: float = 1.0
    accumulate_grad_batches: int = 1
    early_stopping_patience: int = 10
    val_check_interval: float = 1.0
    check_val_every_n_epoch: int = 1

    # Hardware settings
    devices: int = 1
    accelerator: str = "auto"
    precision: str = "16-mixed"
    deterministic: bool = True
    benchmark: bool = True
    gpu_memory_fraction: float = 0.95

    # DataLoader settings
    dataloader_config: Optional[DataLoaderConfig] = None
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # MLflow settings
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None

    # Data settings
    data_path: Union[str, Path] = "./data"
    max_samples: Optional[int] = None
    k_neighbors: int = 8
    subgraph_size: int = 100

    def __post_init__(self):
        """Post-initialization processing with auto-configuration."""
        self.data_path = Path(self.data_path)
        
        # Auto-configure model name
        if self.model_name is None:
            if self.preset:
                self.model_name = f"{self.dataset}_{self.preset}"
            else:
                self.model_name = f"{self.dataset}_{self.model_type}"
        
        # Auto-configure experiment name
        if self.experiment_name is None:
            self.experiment_name = f"{self.dataset}_{self.model_type}"
            
        # Auto-configure DataLoader if not provided
        if self.dataloader_config is None:
            use_gpu = self.accelerator in ["gpu", "auto"] and torch.cuda.is_available()
            self.dataloader_config = DataLoaderConfig.auto_configure(
                use_gpu=use_gpu, 
                batch_size=self.batch_size
            )
            
        # Auto-adjust precision based on hardware
        if self.accelerator == "auto":
            if torch.cuda.is_available():
                self.accelerator = "gpu"
                # Use mixed precision for modern GPUs
                gpu_capability = torch.cuda.get_device_capability(0)
                if gpu_capability[0] >= 7:  # Volta or newer
                    self.precision = "16-mixed"
                else:
                    self.precision = "32"
            else:
                self.accelerator = "cpu"
                self.precision = "32"
                
        # Auto-adjust batch size based on model type and GPU memory
        if torch.cuda.is_available() and self.batch_size == 32:  # Default value
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.batch_size = self._get_optimal_batch_size(gpu_memory_gb)
            
        logger.info(f"📋 Training config initialized:")
        logger.info(f"   Model: {self.model_type} ({self.model_name})")
        logger.info(f"   Dataset: {self.dataset}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Accelerator: {self.accelerator} ({self.precision})")
        logger.info(f"   DataLoader: {self.dataloader_config}")
    
    def _get_optimal_batch_size(self, gpu_memory_gb: float) -> int:
        """Get optimal batch size based on model type and GPU memory."""
        # Base batch sizes for different model types
        base_sizes = {
            "node": 1024,      # Node models can handle larger batches
            "graph": 32,       # Graph models need smaller batches
            "temporal": 64,    # Temporal models moderate
            "point": 128,      # Point cloud models
        }
        
        base_size = base_sizes.get(self.model_type, 64)
        
        # Adjust based on GPU memory
        memory_factor = gpu_memory_gb / 8.0  # Normalized to 8GB baseline
        
        # Calculate optimal size
        optimal_size = int(base_size * memory_factor)
        
        # Clamp to reasonable ranges
        min_sizes = {"node": 32, "graph": 8, "temporal": 16, "point": 16}
        max_sizes = {"node": 4096, "graph": 128, "temporal": 256, "point": 512}
        
        min_size = min_sizes.get(self.model_type, 16)
        max_size = max_sizes.get(self.model_type, 256)
        
        optimal_size = max(min_size, min(optimal_size, max_size))
        
        # Round to nearest power of 2
        optimal_size = 2 ** round(torch.log2(torch.tensor(float(optimal_size))).item())
        
        return int(optimal_size)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for trainer."""
        config_dict = {
            "dataset": self.dataset,
            "model_type": self.model_type,
            "model_name": self.model_name,
            "task": self.task,
            "preset": self.preset,
            "num_features": self.num_features,
            "num_classes": self.num_classes,
            "hidden_dim": self.hidden_dim,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "warmup_epochs": self.warmup_epochs,
            "min_lr": self.min_lr,
            "gradient_clip_val": self.gradient_clip_val,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "early_stopping_patience": self.early_stopping_patience,
            "val_check_interval": self.val_check_interval,
            "check_val_every_n_epoch": self.check_val_every_n_epoch,
            "devices": self.devices,
            "accelerator": self.accelerator,
            "precision": self.precision,
            "deterministic": self.deterministic,
            "benchmark": self.benchmark,
            "gpu_memory_fraction": self.gpu_memory_fraction,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "tracking_uri": self.tracking_uri,
            "data_path": str(self.data_path),
            "max_samples": self.max_samples,
            "k_neighbors": self.k_neighbors,
            "subgraph_size": self.subgraph_size,
        }
        
        # Add DataLoader config
        if self.dataloader_config:
            config_dict.update({
                "num_workers": self.dataloader_config.num_workers,
                "pin_memory": self.dataloader_config.pin_memory,
                "persistent_workers": self.dataloader_config.persistent_workers,
                "prefetch_factor": self.dataloader_config.prefetch_factor,
                "drop_last": self.dataloader_config.drop_last,
            })
            
        # Add model config
        config_dict.update(self.model_config)
        
        return config_dict
