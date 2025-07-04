"""
Model Ensemble Utilities
=======================

Utility functions for creating and managing model ensembles.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


def create_model_ensemble(
    model_class: type,
    model_configs: List[Dict[str, Any]],
    ensemble_method: str = "average",
    weights: Optional[List[float]] = None,
) -> "ModelEnsemble":
    """
    Create an ensemble of models.

    Args:
        model_class: Class of models to create
        model_configs: List of configuration dictionaries
        ensemble_method: Method for combining predictions ('average', 'voting', 'stacking')
        weights: Optional weights for weighted averaging

    Returns:
        ModelEnsemble instance
    """
    models = []
    for i, config in enumerate(model_configs):
        try:
            model = model_class(**config)
            models.append(model)
            logger.info(f"Created ensemble model {i + 1}/{len(model_configs)}")
        except Exception as e:
            logger.error(f"Failed to create model {i + 1}: {e}")
            raise

    return ModelEnsemble(
        models=models,
        ensemble_method=ensemble_method,
        weights=weights,
    )


class ModelEnsemble(nn.Module):
    """
    Ensemble of multiple models with TensorDict support.
    """

    def __init__(
        self,
        models: List[Union[nn.Module, TensorDictModule]],
        ensemble_method: str = "average",
        weights: Optional[List[float]] = None,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.weights = weights

        if weights and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        if weights:
            # Normalize weights
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        logger.info(
            f"Created ensemble with {len(models)} models using {ensemble_method}"
        )

    def forward(
        self, inputs: Union[torch.Tensor, TensorDict]
    ) -> Union[torch.Tensor, TensorDict]:
        """
        Forward pass through ensemble.

        Args:
            inputs: Input data

        Returns:
            Ensemble predictions
        """
        # Get predictions from all models
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)

        # Combine predictions based on method
        if self.ensemble_method == "average":
            return self._weighted_average(predictions)
        elif self.ensemble_method == "voting":
            return self._voting(predictions)
        elif self.ensemble_method == "stacking":
            return self._stacking(predictions, inputs)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

    def _weighted_average(
        self, predictions: List[Union[torch.Tensor, TensorDict]]
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute weighted average of predictions."""
        if isinstance(predictions[0], TensorDict):
            # Handle TensorDict predictions
            result = TensorDict({}, batch_size=predictions[0].batch_size)

            for key in predictions[0].keys():
                if isinstance(predictions[0][key], torch.Tensor):
                    weighted_sum = torch.zeros_like(predictions[0][key])

                    for i, pred in enumerate(predictions):
                        weight = (
                            self.weights[i] if self.weights else 1.0 / len(predictions)
                        )
                        weighted_sum += weight * pred[key]

                    result[key] = weighted_sum

            return result
        else:
            # Handle tensor predictions
            weighted_sum = torch.zeros_like(predictions[0])

            for i, pred in enumerate(predictions):
                weight = self.weights[i] if self.weights else 1.0 / len(predictions)
                weighted_sum += weight * pred

            return weighted_sum

    def _voting(
        self, predictions: List[Union[torch.Tensor, TensorDict]]
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute voting-based ensemble prediction."""
        if isinstance(predictions[0], TensorDict):
            # For TensorDict, use voting on classification outputs
            result = TensorDict({}, batch_size=predictions[0].batch_size)

            for key in predictions[0].keys():
                if isinstance(predictions[0][key], torch.Tensor):
                    # Assume classification if tensor has multiple classes
                    if (
                        predictions[0][key].dim() > 1
                        and predictions[0][key].shape[-1] > 1
                    ):
                        # Voting for classification
                        votes = torch.stack(
                            [pred[key].argmax(dim=-1) for pred in predictions]
                        )
                        result[key] = torch.mode(votes, dim=0)[0]
                    else:
                        # Average for regression
                        result[key] = torch.stack(
                            [pred[key] for pred in predictions]
                        ).mean(dim=0)

            return result
        else:
            # For tensors, assume classification if multiple classes
            if predictions[0].dim() > 1 and predictions[0].shape[-1] > 1:
                votes = torch.stack([pred.argmax(dim=-1) for pred in predictions])
                return torch.mode(votes, dim=0)[0]
            else:
                return torch.stack(predictions).mean(dim=0)

    def _stacking(
        self,
        predictions: List[Union[torch.Tensor, TensorDict]],
        inputs: Union[torch.Tensor, TensorDict],
    ) -> Union[torch.Tensor, TensorDict]:
        """Compute stacking-based ensemble prediction."""
        # Simple stacking: concatenate predictions and use a simple average
        # In practice, you might want to train a meta-learner here
        logger.warning("Stacking method not fully implemented, using average instead")
        return self._weighted_average(predictions)

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get information about the ensemble."""
        return {
            "num_models": len(self.models),
            "ensemble_method": self.ensemble_method,
            "weights": self.weights,
            "model_types": [type(model).__name__ for model in self.models],
        }
