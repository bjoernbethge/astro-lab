"""
ExplainabilityMixin for AstroLab Models
======================================

Provides model explainability using PyTorch Geometric's explain module.

"""

import logging

import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.explain import (
    AttentionExplainer,
    Explainer,
    GNNExplainer,
    PGExplainer,
)
from torch_geometric.explain.config import ExplanationType, ModelMode, ModelTaskLevel

logger = logging.getLogger(__name__)


class ModelWrapper(nn.Module):
    """Wrapper to make models compatible with PyG's Explainer."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, **kwargs):
        batch = Data(x=x, edge_index=edge_index, **kwargs)
        return self.model(batch)


class ExplainabilityMixin:
    """Mixin for model explainability using PyTorch Geometric's explain module."""

    def setup_explainer(self, algorithm: str = "GNNExplainer", **kwargs) -> None:
        try:
            task_level = ModelTaskLevel.node
            if hasattr(self, "task"):
                if "graph" in self.task:
                    task_level = ModelTaskLevel.graph
                elif "node" in self.task:
                    task_level = ModelTaskLevel.node
            model_mode = ModelMode.regression
            if hasattr(self, "task") and "classification" in self.task:
                if hasattr(self, "num_classes") and self.num_classes == 2:
                    model_mode = ModelMode.binary_classification
                else:
                    model_mode = ModelMode.multiclass_classification
            if algorithm == "GNNExplainer":
                algorithm_instance = GNNExplainer(epochs=200, **kwargs)
            elif algorithm == "PGExplainer":
                algorithm_instance = PGExplainer(epochs=30, **kwargs)
            elif algorithm == "AttentionExplainer":
                algorithm_instance = AttentionExplainer()
            else:
                raise ValueError(
                    f"Unknown algorithm: {algorithm}. Available: GNNExplainer, PGExplainer, AttentionExplainer"
                )
            wrapped_model = ModelWrapper(self)
            self.explainer = Explainer(
                model=wrapped_model,
                algorithm=algorithm_instance,
                explanation_type=ExplanationType.model,
                model_config=dict(
                    mode=model_mode,
                    task_level=task_level,
                    return_type="raw",
                ),
                node_mask_type="attributes",
                edge_mask_type="object",
            )
            logger.info(f"Explainer setup complete with {algorithm} algorithm")
        except Exception as e:
            logger.error(f"Failed to setup explainer: {e}")
            raise

    def explain(self, x, edge_index, **kwargs):
        if not hasattr(self, "explainer"):
            self.setup_explainer()
        try:
            return self.explainer(x=x, edge_index=edge_index, **kwargs)
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise
