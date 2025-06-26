"""
Training Configuration for AstroLab
===================================

Configuration classes for training and optimization.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Basic settings
    dataset: str = "gaia"
    model_name: str = "gaia_classifier"

    # Model configuration
    model_config: Dict[str, Any] = field(default_factory=dict)

    # Training settings
    max_epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.01

    # Hardware settings
    devices: int = 1
    accelerator: str = "auto"
    precision: str = "16-mixed"

    # MLflow settings
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None

    # Data settings
    data_path: Union[str, Path] = "./data"
    max_samples: Optional[int] = None

    def __post_init__(self):
        """Post-initialization processing."""
        self.data_path = Path(self.data_path)
        if self.experiment_name is None:
            self.experiment_name = f"{self.dataset}_{self.model_name}"
