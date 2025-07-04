"""
TensorDict Utilities
===================

Utility functions for TensorDict model compatibility and conversion.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


def validate_tensordict_compatibility(
    model: nn.Module,
    tensordict_example: TensorDict,
    required_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate TensorDict compatibility for a model.

    Args:
        model: Model to validate
        tensordict_example: Example TensorDict
        required_keys: Keys that must be present

    Returns:
        Validation results dictionary
    """
    results = {
        "is_compatible": True,
        "errors": [],
        "warnings": [],
        "key_analysis": {},
        "shape_analysis": {},
    }

    # Check required keys
    if required_keys:
        missing_keys = set(required_keys) - set(tensordict_example.keys())
        if missing_keys:
            results["errors"].append(f"Missing required keys: {missing_keys}")
            results["is_compatible"] = False

    # Analyze each key
    for key, tensor in tensordict_example.items():
        key_info = {
            "type": type(tensor).__name__,
            "is_tensor": isinstance(tensor, torch.Tensor),
        }

        if isinstance(tensor, torch.Tensor):
            key_info.update(
                {
                    "dtype": str(tensor.dtype),
                    "shape": tuple(tensor.shape),
                    "device": str(tensor.device),
                    "requires_grad": tensor.requires_grad,
                }
            )

        results["key_analysis"][key] = key_info

    # Test forward pass
    try:
        model.eval()
        with torch.no_grad():
            output = model(tensordict_example)
            results["forward_pass_success"] = True

            if isinstance(output, TensorDict):
                results["output_keys"] = list(output.keys())
                for key, tensor in output.items():
                    if isinstance(tensor, torch.Tensor):
                        results["shape_analysis"][f"output_{key}"] = tuple(tensor.shape)
            else:
                results["output_type"] = type(output).__name__
                if isinstance(output, torch.Tensor):
                    results["shape_analysis"]["output"] = tuple(output.shape)

    except Exception as e:
        results["forward_pass_success"] = False
        results["errors"].append(f"Forward pass failed: {e}")

    return results


def convert_model_to_tensordict(
    model: nn.Module,
    in_keys: List[str],
    out_keys: List[str],
    coordinate_system: Optional[str] = None,
    validate_astronomy: bool = True,
) -> TensorDictModule:
    """
    Convert a PyTorch model to TensorDict format.

    Args:
        model: PyTorch model to convert
        in_keys: Input keys for TensorDict
        out_keys: Output keys for TensorDict
        coordinate_system: Optional coordinate system specification
        validate_astronomy: Whether to validate astronomical data

    Returns:
        TensorDictModule wrapper
    """
    # Create TensorDict wrapper
    tensordict_model = TensorDictModule(
        module=model,
        in_keys=in_keys,
        out_keys=out_keys,
    )

    # Add metadata
    tensordict_model.coordinate_system = coordinate_system
    tensordict_model.validate_astronomy = validate_astronomy

    logger.info(
        f"Converted model to TensorDict format with keys: {in_keys} -> {out_keys}"
    )

    return tensordict_model


def tensor_model_info(
    model: Union[nn.Module, TensorDictModule], include_weights: bool = False
) -> Dict[str, Any]:
    """
    Get detailed information about a TensorDict model.

    Args:
        model: Model to analyze
        include_weights: Whether to include weight statistics

    Returns:
        Model information dictionary
    """
    info = {
        "model_type": type(model).__name__,
        "is_tensordict": isinstance(model, TensorDictModule),
    }

    if isinstance(model, TensorDictModule):
        info.update(
            {
                "in_keys": getattr(model, "in_keys", []),
                "out_keys": getattr(model, "out_keys", []),
                "coordinate_system": getattr(model, "coordinate_system", None),
                "validate_astronomy": getattr(model, "validate_astronomy", False),
            }
        )

    # Parameter information
    if include_weights:
        weight_info = {}
        for name, param in model.named_parameters():
            weight_info[name] = {
                "shape": tuple(param.shape),
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
                "mean": param.data.mean().item(),
                "std": param.data.std().item(),
            }
        info["weights"] = weight_info

    return info
