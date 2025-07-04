"""
Model Benchmarking Utilities
===========================

Utility functions for benchmarking model performance.
"""

import logging
import time
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

logger = logging.getLogger(__name__)


def benchmark_model_performance(
    model: Union[nn.Module, TensorDictModule],
    input_data: Union[torch.Tensor, TensorDict],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None,
    memory_tracking: bool = True,
) -> Dict[str, float]:
    """
    Benchmark model performance including speed and memory usage.

    Args:
        model: Model to benchmark
        input_data: Input data for benchmarking
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        device: Device to run on
        memory_tracking: Whether to track memory usage

    Returns:
        Benchmark results dictionary
    """
    if device is None:
        device = next(model.parameters()).device

    # Move model and data to device
    model = model.to(device)
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.to(device)
    elif isinstance(input_data, TensorDict):
        input_data = input_data.to(device)

    model.eval()
    results = {}

    # Warmup runs
    logger.info(f"Running {warmup_runs} warmup runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_data)

    # Memory tracking setup
    if memory_tracking:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = (
            torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        )

    # Benchmark runs
    logger.info(f"Running {num_runs} benchmark runs...")
    times = []

    with torch.no_grad():
        for i in range(num_runs):
            start_time = time.perf_counter()
            model(input_data)
            end_time = time.perf_counter()

            times.append(end_time - start_time)

            if (i + 1) % 20 == 0:
                logger.info(f"Completed {i + 1}/{num_runs} runs")

    # Calculate statistics
    times = torch.tensor(times)
    results.update(
        {
            "mean_time_ms": times.mean().item() * 1000,
            "std_time_ms": times.std().item() * 1000,
            "min_time_ms": times.min().item() * 1000,
            "max_time_ms": times.max().item() * 1000,
            "median_time_ms": times.median().item() * 1000,
            "throughput_fps": 1.0 / times.mean().item(),
        }
    )

    # Memory usage
    if memory_tracking and torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        results.update(
            {
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "memory_increase_mb": (peak_memory - initial_memory) / (1024 * 1024),
            }
        )

    # Model size
    param_count = sum(p.numel() for p in model.parameters())
    results["model_size_mb"] = param_count * 4 / (1024 * 1024)  # Assuming float32

    logger.info("Benchmark completed successfully")
    return results


def compare_models_performance(
    models: List[Union[nn.Module, TensorDictModule]],
    model_names: List[str],
    input_data: Union[torch.Tensor, TensorDict],
    num_runs: int = 50,
    device: Optional[torch.device] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compare performance of multiple models.

    Args:
        models: List of models to compare
        model_names: Names for the models
        input_data: Input data for benchmarking
        num_runs: Number of benchmark runs per model
        device: Device to run on

    Returns:
        Dictionary with benchmark results for each model
    """
    if len(models) != len(model_names):
        raise ValueError("Number of models must match number of names")

    results = {}

    for model, name in zip(models, model_names):
        logger.info(f"Benchmarking model: {name}")
        try:
            model_results = benchmark_model_performance(
                model=model,
                input_data=input_data,
                num_runs=num_runs,
                device=device,
            )
            results[name] = model_results
        except Exception as e:
            logger.error(f"Failed to benchmark {name}: {e}")
            results[name] = {"error": str(e)}

    return results


def profile_model_memory(
    model: Union[nn.Module, TensorDictModule],
    input_data: Union[torch.Tensor, TensorDict],
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Profile memory usage of a model.

    Args:
        model: Model to profile
        input_data: Input data
        device: Device to run on

    Returns:
        Memory usage statistics
    """
    if device is None:
        device = next(model.parameters()).device

    if not torch.cuda.is_available():
        logger.warning("CUDA not available, memory profiling limited")
        return {"error": "CUDA not available"}

    # Move to device
    model = model.to(device)
    if isinstance(input_data, torch.Tensor):
        input_data = input_data.to(device)
    elif isinstance(input_data, TensorDict):
        input_data = input_data.to(device)

    model.eval()

    # Clear cache and get initial memory
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    # Forward pass
    with torch.no_grad():
        model(input_data)

    # Get peak memory
    peak_memory = torch.cuda.max_memory_allocated()

    # Get memory after forward pass
    current_memory = torch.cuda.memory_allocated()

    return {
        "initial_memory_mb": initial_memory / (1024 * 1024),
        "peak_memory_mb": peak_memory / (1024 * 1024),
        "current_memory_mb": current_memory / (1024 * 1024),
        "memory_increase_mb": (peak_memory - initial_memory) / (1024 * 1024),
        "memory_retained_mb": (current_memory - initial_memory) / (1024 * 1024),
    }


def get_model_efficiency_metrics(
    model: Union[nn.Module, TensorDictModule],
    input_data: Union[torch.Tensor, TensorDict],
    target_metric: str = "accuracy",
    target_value: float = 0.95,
) -> Dict[str, float]:
    """
    Calculate efficiency metrics for a model.

    Args:
        model: Model to analyze
        input_data: Input data
        target_metric: Target metric name
        target_value: Target value for the metric

    Returns:
        Efficiency metrics
    """
    # Get basic performance metrics
    perf_results = benchmark_model_performance(
        model=model,
        input_data=input_data,
        num_runs=10,  # Fewer runs for efficiency calculation
    )

    # Calculate efficiency metrics
    param_count = sum(p.numel() for p in model.parameters())

    efficiency_metrics = {
        "parameters_per_second": param_count / perf_results["mean_time_ms"] * 1000,
        "memory_efficiency_mb_per_param": perf_results.get("peak_memory_mb", 0)
        / param_count,
        "throughput_per_param": perf_results["throughput_fps"] / param_count,
    }

    # Add target-based efficiency if target metric is provided
    if target_metric and target_value:
        # This would need actual metric calculation in practice
        efficiency_metrics[f"efficiency_vs_{target_metric}"] = (
            perf_results["throughput_fps"] / target_value
        )

    return efficiency_metrics
