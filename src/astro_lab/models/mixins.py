"""Lightning Mixins for AstroLab Models - Only essential functionality not provided by Lightning/PyG."""

import logging

import torch
from torch_geometric.explain import (
    AttentionExplainer,
    Explainer,
    GNNExplainer,
    PGExplainer,
)
from torch_geometric.explain.config import ExplanationType, ModelTaskLevel
from torch_geometric.explain.explanation import Explanation

logger = logging.getLogger(__name__)


class ExplainabilityMixin:
    """Mixin for model explainability using PyTorch Geometric's explain module."""

    def setup_explainer(self, algorithm: str = "GNNExplainer", **kwargs) -> None:
        """Setup the explainer for the model using PyG's official API.

        Args:
            algorithm: Algorithm to use ('GNNExplainer', 'PGExplainer', 'AttentionExplainer')
            **kwargs: Additional arguments passed to the algorithm
        """
        # Determine task level from model's task attribute
        if hasattr(self, "task"):
            if "node" in self.task:
                task_level = ModelTaskLevel.node
            elif "graph" in self.task:
                task_level = ModelTaskLevel.graph
            else:
                task_level = ModelTaskLevel.node
        else:
            task_level = ModelTaskLevel.node

        # Select algorithm
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

        # Create explainer using PyG's official API
        self.explainer = Explainer(
            model=self,
            algorithm=algorithm_instance,
            explanation_type=ExplanationType.model,
            model_config=dict(
                mode="classification"
                if "classification" in getattr(self, "task", "")
                else "regression",
                task_level=task_level,
                return_type="raw",
            ),
            node_mask_type="attributes",
            edge_mask_type="object",
        )

    def explain(
        self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs
    ) -> "Explanation":
        """Generate explanation using PyG's Explainer.

        Args:
            x: Node features
            edge_index: Edge connectivity
            **kwargs: Additional arguments (index for node-level, batch for graph-level, etc.)

        Returns:
            PyG Explanation object
        """
        if not hasattr(self, "explainer"):
            self.setup_explainer()

        return self.explainer(x=x, edge_index=edge_index, **kwargs)
