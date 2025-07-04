"""
Model Analysis Utilities
=======================

Utility functions for analyzing and summarizing neural network models.
"""

import logging
from typing import Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


def count_parameters(model: Union[nn.Module, TensorDictModule]) -> dict:
    """
    Count parameters in a model with TensorDict support.

    Args:
        model: PyTorch model or TensorDict model

    Returns:
        Dictionary with parameter counts
    """
    if isinstance(model, TensorDictModule):
        # Extract the wrapped module
        base_model = model.module
    else:
        base_model = model

    total_params = sum(p.numel() for p in base_model.parameters())
    trainable_params = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )

    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "total_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
        "trainable_mb": trainable_params * 4 / (1024 * 1024),
    }


def get_model_summary(
    model: Union[nn.Module, TensorDictModule],
    input_example: Optional[Union[torch.Tensor, TensorDict]] = None,
    max_depth: int = 3,
) -> str:
    """
    Generate a detailed model summary with TensorDict support.

    Args:
        model: Model to analyze
        input_example: Example input for shape inference
        max_depth: Maximum depth for nested modules

    Returns:
        Formatted model summary string
    """

    def _get_module_info(module, name="", depth=0):
        """Recursively get module information."""
        info = []

        # Current module info
        param_count = sum(p.numel() for p in module.parameters(recurse=False))
        module_type = type(module).__name__

        indent = "  " * depth
        info.append(f"{indent}{name}: {module_type}")

        if param_count > 0:
            info.append(f"{indent}  Parameters: {param_count:,}")

        # Recurse into children if not at max depth
        if depth < max_depth:
            for child_name, child_module in module.named_children():
                child_info = _get_module_info(child_module, child_name, depth + 1)
                info.extend(child_info)

        return info

    # Get base model
    if isinstance(model, TensorDictModule):
        base_model = model.module
        model_type = "TensorDictModule"
        in_keys = getattr(model, "in_keys", [])
        out_keys = getattr(model, "out_keys", [])
    else:
        base_model = model
        model_type = "PyTorch Module"
        in_keys = []
        out_keys = []

    # Build summary
    summary_lines = [f"Model Summary - {model_type}", "=" * 50]

    # TensorDict specific info
    if isinstance(model, TensorDictModule):
        summary_lines.extend([f"Input keys: {in_keys}", f"Output keys: {out_keys}", ""])

    # Parameter counts
    param_info = count_parameters(model)
    summary_lines.extend(
        [
            f"Total parameters: {param_info['total_parameters']:,}",
            f"Trainable parameters: {param_info['trainable_parameters']:,}",
            f"Model size: {param_info['total_mb']:.2f} MB",
            "",
        ]
    )

    # Module structure
    summary_lines.append("Module Structure:")
    summary_lines.append("-" * 20)
    module_info = _get_module_info(base_model)
    summary_lines.extend(module_info)

    # Input/output shapes if example provided
    if input_example is not None:
        summary_lines.append("")
        summary_lines.append("Input/Output Shapes:")
        summary_lines.append("-" * 20)

        try:
            model.eval()
            with torch.no_grad():
                if isinstance(input_example, TensorDict):
                    output = model(input_example)
                    summary_lines.append("TensorDict Input/Output:")
                    for key, tensor in input_example.items():
                        if isinstance(tensor, torch.Tensor):
                            summary_lines.append(
                                f"  Input[{key}]: {tuple(tensor.shape)}"
                            )

                    if isinstance(output, TensorDict):
                        for key, tensor in output.items():
                            if isinstance(tensor, torch.Tensor):
                                summary_lines.append(
                                    f"  Output[{key}]: {tuple(tensor.shape)}"
                                )
                    else:
                        summary_lines.append(f"  Output: {tuple(output.shape)}")
                else:
                    output = model(input_example)
                    summary_lines.append(f"Input shape: {tuple(input_example.shape)}")
                    summary_lines.append(f"Output shape: {tuple(output.shape)}")

        except Exception as e:
            summary_lines.append(f"Could not infer shapes: {e}")

    return "\n".join(summary_lines)
