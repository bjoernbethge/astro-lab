"""
ModelAnalysisMixin for AstroLab Models
=====================================

Provides model parameter counting, summary, and info as a mixin for AstroBaseModel and subclasses.
Originally from utils/analysis.py.
"""

from typing import Optional, Union

import torch
from tensordict import TensorDict


class ModelAnalysisMixin:
    """Mixin for model analysis, summary, and info methods."""

    def count_parameters(self) -> dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "total_mb": total_params * 4 / (1024 * 1024),
            "trainable_mb": trainable_params * 4 / (1024 * 1024),
        }

    def get_model_summary(
        self,
        input_example: Optional[Union[torch.Tensor, TensorDict]] = None,
        max_depth: int = 3,
    ) -> str:
        param_info = self.count_parameters()
        summary_lines = [
            f"Model Summary - {type(self).__name__}",
            "=" * 50,
            f"Total parameters: {param_info['total_parameters']:,}",
            f"Trainable parameters: {param_info['trainable_parameters']:,}",
            f"Model size: {param_info['total_mb']:.2f} MB",
            "",
        ]
        summary_lines.append("Available Features:")
        summary_lines.append("-" * 20)
        if (
            hasattr(self, "setup_explainer")
            and callable(getattr(self, "setup_explainer", None))
            and hasattr(self, "explain")
            and callable(getattr(self, "explain", None))
        ):
            summary_lines.append("✅ Explainability (ExplainabilityMixin)")
            summary_lines.append("  - setup_explainer(algorithm='GNNExplainer')")
            summary_lines.append("  - explain(x, edge_index)")
            summary_lines.append("")
        if (
            hasattr(self, "training_step")
            and callable(getattr(self, "training_step", None))
            and hasattr(self, "validation_step")
            and callable(getattr(self, "validation_step", None))
        ):
            summary_lines.append("✅ Lightning Training")
            summary_lines.append("  - training_step(), validation_step(), test_step()")
            summary_lines.append("  - configure_optimizers()")
            summary_lines.append("")
        if hasattr(self, "task"):
            summary_lines.append(f"📋 Task: {self.task}")
        if hasattr(self, "num_classes"):
            summary_lines.append(f"📋 Classes: {self.num_classes}")
        if hasattr(self, "num_features"):
            summary_lines.append(f"📋 Features: {self.num_features}")
        if hasattr(self, "hidden_dim"):
            summary_lines.append(f"📋 Hidden dim: {self.hidden_dim}")
        if hasattr(self, "conv_type"):
            summary_lines.append(f"🏗️ Convolution: {self.conv_type}")
        if hasattr(self, "heads"):
            summary_lines.append(f"🏗️ Heads: {self.heads}")
        if hasattr(self, "pooling"):
            summary_lines.append(f"🏗️ Pooling: {self.pooling}")
        if hasattr(self, "num_layers"):
            summary_lines.append(f"🏗️ Layers: {self.num_layers}")
        return "\n".join(summary_lines)

    def get_model_info(self) -> dict:
        param_info = self.count_parameters()
        info = {
            "model_type": type(self).__name__,
            "parameters": param_info,
            "has_explainability": hasattr(self, "setup_explainer")
            and callable(getattr(self, "setup_explainer", None))
            and hasattr(self, "explain")
            and callable(getattr(self, "explain", None)),
            "has_lightning_training": hasattr(self, "training_step")
            and callable(getattr(self, "training_step", None))
            and hasattr(self, "validation_step")
            and callable(getattr(self, "validation_step", None)),
        }
        if hasattr(self, "task"):
            info["task"] = self.task
        if hasattr(self, "num_classes"):
            info["num_classes"] = self.num_classes
        if hasattr(self, "num_features"):
            info["num_features"] = self.num_features
        return info
