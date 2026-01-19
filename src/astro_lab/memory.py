"""
AstroLab Memory Management
==========================

Simplified memory management for AstroLab applications.
Focuses on essential CUDA cache management and basic monitoring.
"""

import logging
from contextlib import contextmanager
from typing import Any, Dict

import torch

from astro_lab.utils.device import is_cuda_available

logger = logging.getLogger(__name__)


def clear_cuda_cache():
    """Clear CUDA cache - alias for clear_gpu_memory for compatibility."""
    clear_gpu_memory()


@contextmanager
def memory_management():
    """Context manager for memory-efficient operations."""
    # Pre-cleanup
    if is_cuda_available():
        torch.cuda.empty_cache()

    try:
        yield
    finally:
        # Post-cleanup
        if is_cuda_available():
            torch.cuda.empty_cache()


def get_memory_info() -> Dict[str, Any]:
    """Get comprehensive memory information."""
    info = {
        "device": "cuda",
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device(),
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "memory_free_gb": (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved()
        )
        / 1024**3,
        "memory_total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
    }

    return info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage in GB."""
    return {
        "allocated": torch.cuda.memory_allocated() / 1024**3,
        "reserved": torch.cuda.memory_reserved() / 1024**3,
        "free": (
            torch.cuda.get_device_properties(0).total_memory
            - torch.cuda.memory_reserved()
        )
        / 1024**3,
    }
