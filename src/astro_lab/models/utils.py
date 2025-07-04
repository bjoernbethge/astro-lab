"""
Model Utilities with TensorDict Integration
==========================================

Utility functions for astronomical neural networks with native TensorDict support.
"""

import logging
from typing import Any, Dict, List, Optional, Union

# Import for model analysis
import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


def count_parameters(model: Union[nn.Module, TensorDictModule]) -> Dict[str, int]:
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
                    "shape": tuple(tensor.shape),
                    "dtype": tensor.dtype,
                    "device": tensor.device,
                    "requires_grad": tensor.requires_grad,
                }
            )

            # Check for common issues
            if tensor.numel() == 0:
                results["warnings"].append(f"Key '{key}' has empty tensor")

            if torch.any(torch.isnan(tensor)):
                results["warnings"].append(f"Key '{key}' contains NaN values")

            if torch.any(torch.isinf(tensor)):
                results["warnings"].append(f"Key '{key}' contains infinite values")

        results["key_analysis"][key] = key_info

    # Try forward pass if model supports TensorDict
    if isinstance(model, TensorDictModule):
        try:
            model.eval()
            with torch.no_grad():
                output = model(tensordict_example)
                results["forward_pass"] = "success"

                if isinstance(output, TensorDict):
                    results["output_keys"] = list(output.keys())
                    results["shape_analysis"]["output"] = {
                        key: tuple(tensor.shape)
                        if isinstance(tensor, torch.Tensor)
                        else "non-tensor"
                        for key, tensor in output.items()
                    }
                else:
                    results["output_keys"] = ["tensor_output"]
                    results["shape_analysis"]["output"] = {
                        "tensor_output": tuple(output.shape)
                    }

        except Exception as e:
            results["errors"].append(f"Forward pass failed: {e}")
            results["is_compatible"] = False
            results["forward_pass"] = "failed"

    return results


def tensor_model_info(
    model: Union[nn.Module, TensorDictModule], include_weights: bool = False
) -> Dict[str, Any]:
    """
    Get comprehensive information about a tensor model.

    Args:
        model: Model to analyze
        include_weights: Whether to include weight statistics

    Returns:
        Model information dictionary
    """
    info = {
        "model_type": type(model).__name__,
        "is_tensordict_model": isinstance(model, TensorDictModule),
        "parameter_info": count_parameters(model),
        "module_count": len(list(model.modules())),
        "training_mode": model.training,
    }

    # TensorDict specific info
    if isinstance(model, TensorDictModule):
        info.update(
            {
                "in_keys": getattr(model, "in_keys", []),
                "out_keys": getattr(model, "out_keys", []),
                "wrapped_module_type": type(model.module).__name__,
            }
        )

    # Device information
    devices = set()
    for param in model.parameters():
        devices.add(str(param.device))
    info["devices"] = list(devices)
    info["primary_device"] = list(devices)[0] if devices else "unknown"

    # Weight statistics
    if include_weights:
        weight_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_stats[name] = {
                    "shape": tuple(param.shape),
                    "mean": param.data.mean().item(),
                    "std": param.data.std().item(),
                    "min": param.data.min().item(),
                    "max": param.data.max().item(),
                    "norm": param.data.norm().item(),
                }
        info["weight_statistics"] = weight_stats

    return info


def convert_model_to_tensordict(
    model: nn.Module,
    in_keys: List[str],
    out_keys: List[str],
    coordinate_system: Optional[str] = None,
    validate_astronomy: bool = True,
) -> TensorDictModule:
    """
    Convert a regular PyTorch model to TensorDict format.

    Args:
        model: PyTorch model to convert
        in_keys: Input keys for TensorDict
        out_keys: Output keys for TensorDict
        coordinate_system: Astronomical coordinate system
        validate_astronomy: Whether to validate astronomical inputs

    Returns:
        TensorDict-wrapped model
    """
    from .components.tensordict_modules import AstroTensorDictModule

    return AstroTensorDictModule(
        module=model,
        in_keys=in_keys,
        out_keys=out_keys,
        coordinate_system=coordinate_system,
        validate_astronomy=validate_astronomy,
    )


def create_model_ensemble(
    models: List[Union[nn.Module, TensorDictModule]],
    ensemble_method: str = "average",
    weights: Optional[List[float]] = None,
) -> TensorDictModule:
    """
    Create an ensemble of models with TensorDict support.

    Args:
        models: List of models to ensemble
        ensemble_method: Ensemble method ("average", "weighted", "voting")
        weights: Weights for weighted ensemble

    Returns:
        TensorDict ensemble model
    """

    class ModelEnsemble(nn.Module):
        def __init__(self, models, method, weights):
            super().__init__()
            self.models = nn.ModuleList(models)
            self.method = method
            self.weights = weights

            if weights and len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")

        def forward(self, x):
            outputs = [model(x) for model in self.models]

            if self.method == "average":
                return torch.stack(outputs).mean(dim=0)
            elif self.method == "weighted" and self.weights:
                weighted_outputs = [w * out for w, out in zip(self.weights, outputs)]
                return torch.stack(weighted_outputs).sum(dim=0)
            elif self.method == "voting":
                # majority voting for classification
                predictions = [torch.argmax(out, dim=-1) for out in outputs]
                stacked_preds = torch.stack(predictions)
                return torch.mode(stacked_preds, dim=0)[0]
            else:
                return torch.stack(outputs).mean(dim=0)

    # Extract base models if they're TensorDict modules
    base_models = []
    sample_tensordict_model = None

    for model in models:
        if isinstance(model, TensorDictModule):
            base_models.append(model.module)
            if sample_tensordict_model is None:
                sample_tensordict_model = model
        else:
            base_models.append(model)

    # Create ensemble
    ensemble = ModelEnsemble(base_models, ensemble_method, weights)

    # Wrap in TensorDict if original models were TensorDict
    if sample_tensordict_model:
        return TensorDictModule(
            module=ensemble,
            in_keys=sample_tensordict_model.in_keys,
            out_keys=sample_tensordict_model.out_keys,
        )
    else:
        # Return as regular module
        return ensemble


def benchmark_model_performance(
    model: Union[nn.Module, TensorDictModule],
    input_data: Union[torch.Tensor, TensorDict],
    num_iterations: int = 100,
    warmup_iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model performance.

    Args:
        model: Model to benchmark
        input_data: Input data for benchmarking
        num_iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        Performance metrics
    """

    model.eval()
    device = next(model.parameters()).device

    # Move input to same device
    if isinstance(input_data, TensorDict):
        input_data = input_data.to(device)
    else:
        input_data = input_data.to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_iterations):
            _ = model(input_data)

    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Benchmark
    import time

    times = []

    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(input_data)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()
            times.append(end_time - start_time)

    # Calculate statistics
    times = np.array(times)

    return {
        "mean_time_ms": times.mean() * 1000,
        "std_time_ms": times.std() * 1000,
        "min_time_ms": times.min() * 1000,
        "max_time_ms": times.max() * 1000,
        "median_time_ms": np.median(times) * 1000,
        "throughput_samples_per_sec": 1.0 / times.mean(),
        "total_iterations": num_iterations,
    }
