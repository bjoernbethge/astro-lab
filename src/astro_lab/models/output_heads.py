"""
Output Head Registry for AstroLab Models

Standardized output heads for different astronomical tasks:
- Classification: Multi-class galaxy/star classification
- Regression: Redshift, mass, distance prediction
- Period Detection: Specialized heads for period analysis
- Multi-Task: Combined heads for multiple tasks
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, TypedDict

from astro_lab.models.utils import get_activation

# Configure logging
logger = logging.getLogger(__name__)

class ModelOutput(TypedDict):
    """Type definition for model outputs."""

    predictions: torch.Tensor
    embeddings: torch.Tensor

class OutputHeadRegistry:
    """Registry for output heads."""

    _heads: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str):
        """Decorator to register output heads."""

        def decorator(head_class):
            cls._heads[name] = head_class
            return head_class

        return decorator

    @classmethod
    def create(cls, name: str, hidden_dim: int, output_dim: int, **kwargs) -> nn.Module:
        """Create output head by name."""
        if name not in cls._heads:
            available = list(cls._heads.keys())
            raise ValueError(f"Unknown output head: {name}. Available: {available}")
        return cls._heads[name](hidden_dim, output_dim, **kwargs)

    @classmethod
    def list_available(cls) -> List[str]:
        """List all available output heads."""
        return list(cls._heads.keys())

@OutputHeadRegistry.register("regression")
class RegressionHead(nn.Module):
    """Standard regression head for continuous predictions."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu",
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        current_dim = hidden_dim

        for i in range(num_layers - 1):
            next_dim = hidden_dim // (2 ** (i + 1))
            layers.extend(
                [
                    nn.Linear(current_dim, next_dim),
                    get_activation(activation),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = next_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, output_dim))

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)

@OutputHeadRegistry.register("classification")
class ClassificationHead(nn.Module):
    """Standard classification head with optional class weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            get_activation(activation),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim // 2))

        layers.extend(
            [
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            ]
        )

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.head(x)
        return (
            F.log_softmax(logits, dim=-1)
            if self.num_classes > 1
            else torch.sigmoid(logits)
        )

@OutputHeadRegistry.register("period_detection")
class PeriodDetectionHead(nn.Module):
    """Specialized head for period detection in lightcurves."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 1,
        period_range: tuple = (0.1, 100.0),  # days
        confidence_output: bool = True,
    ):
        super().__init__()
        self.period_range = period_range
        self.confidence_output = confidence_output

        # Period prediction branch
        self.period_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Confidence/quality branch
        if confidence_output:
            self.confidence_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Predict log-period for better numerical stability
        log_period = self.period_head(x)

        # Convert to actual period within range
        min_log, max_log = torch.log10(torch.tensor(self.period_range))
        period = 10 ** (min_log + torch.sigmoid(log_period) * (max_log - min_log))

        result = {"period": period}

        if self.confidence_output:
            confidence = self.confidence_head(x)
            result["confidence"] = confidence

        return result

@OutputHeadRegistry.register("shape_modeling")
class ShapeModelingHead(nn.Module):
    """Head for asteroid/object shape modeling."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        shape_params: int = 6,  # Standard ellipsoid parameters
    ):
        super().__init__()
        self.shape_params = shape_params

        # Shape parameters (a, b, c axes + orientation)
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, shape_params),
        )

        # Spin state parameters
        self.spin_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Spin vector
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shape = self.shape_head(x)
        spin = self.spin_head(x)

        return {"shape_parameters": shape, "spin_vector": spin}

@OutputHeadRegistry.register("multi_task")
class MultiTaskHead(nn.Module):
    """Multi-task head for combined predictions."""

    def __init__(
        self, hidden_dim: int, task_configs: Dict[str, Dict], shared_layers: int = 1
    ):
        super().__init__()
        self.task_configs = task_configs

        # Shared feature extraction
        shared_dim = hidden_dim // 2
        if shared_layers > 0:
            shared = []
            current_dim = hidden_dim
            for _ in range(shared_layers):
                shared.extend(
                    [nn.Linear(current_dim, shared_dim), nn.ReLU(), nn.Dropout(0.1)]
                )
                current_dim = shared_dim
            self.shared = nn.Sequential(*shared)
        else:
            self.shared = nn.Identity()
            shared_dim = hidden_dim

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, config in task_configs.items():
            head_type = config.get("type", "regression")
            output_dim = config["output_dim"]

            if head_type == "regression":
                head = RegressionHead(
                    shared_dim, output_dim, **config.get("kwargs", {})
                )
            elif head_type == "classification":
                head = ClassificationHead(
                    shared_dim, output_dim, **config.get("kwargs", {})
                )
            else:
                # Use registry for other head types
                head = OutputHeadRegistry.create(
                    head_type, shared_dim, output_dim, **config.get("kwargs", {})
                )

            self.task_heads[task_name] = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.shared(x)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)

        return outputs

@OutputHeadRegistry.register("cosmological")
class CosmologicalHead(nn.Module):
    """Head for cosmological parameter prediction."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int = 6,  # Standard cosmological parameters
        parameter_names: Optional[List[str]] = None,
    ):
        super().__init__()

        if parameter_names is None:
            self.parameter_names = ["Omega_m", "Omega_L", "h", "sigma_8", "n_s", "w"]
        else:
            self.parameter_names = parameter_names

        # Parameter-specific heads with appropriate constraints
        self.param_heads = nn.ModuleDict()

        for param in self.parameter_names:
            if param in ["Omega_m", "Omega_L"]:
                # Density parameters: sigmoid to [0, 1]
                head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
            elif param == "h":
                # Hubble parameter: scaled sigmoid to [0.5, 1.0]
                head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )
            else:
                # Other parameters: standard regression
                head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 4, 1),
                )

            self.param_heads[param] = head

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = {}

        for param, head in self.param_heads.items():
            raw_output = head(x)

            # Apply parameter-specific scaling
            if param == "h":
                # Scale to [0.5, 1.0] range
                outputs[param] = 0.5 + 0.5 * raw_output
            elif param == "w":
                # Dark energy equation of state: scale to reasonable range
                outputs[param] = -2.0 + 2.0 * torch.sigmoid(raw_output)
            else:
                outputs[param] = raw_output

        return outputs

# Convenience function for creating heads
def create_output_head(
    task: str, hidden_dim: int, output_dim: int, **kwargs
) -> nn.Module:
    """Create output head for common tasks."""

    # Map common task names to head types
    task_mapping = {
        "stellar_classification": "classification",
        "galaxy_property_prediction": "regression",
        "transient_detection": "classification",
        "period_detection": "period_detection",
        "shape_modeling": "shape_modeling",
        "asteroid_classification": "classification",
        "cosmological_inference": "cosmological",
    }

    head_type = task_mapping.get(task, task)
    return OutputHeadRegistry.create(head_type, hidden_dim, output_dim, **kwargs)
