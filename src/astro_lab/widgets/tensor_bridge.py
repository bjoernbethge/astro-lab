"""
Simplified Astronomical Tensor Bridge
=====================================

Clean, efficient bridge between AstroLab TensorDicts and visualization backends.
Pure routing to the main widgets API.
"""

import logging
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


class AstronomicalTensorBridge:
    """Simplified bridge that routes to the main widgets API."""

    def to_visualization(self, tensordict: Any, backend: str = "auto", **kwargs) -> Any:
        """Route to main visualization API."""
        from . import visualize

        return visualize(tensordict, backend=backend, **kwargs)

    def validate_coordinates(self, tensor: Any) -> torch.Tensor:
        """Validate and prepare coordinate tensor."""
        if isinstance(tensor, torch.Tensor):
            coords = tensor
        elif hasattr(tensor, "cpu"):
            coords = tensor.cpu()
        else:
            coords = torch.tensor(tensor, dtype=torch.float32)

        # Ensure 2D
        if coords.dim() != 2:
            raise ValueError(f"Expected 2D coordinates, got {coords.dim()}D")

        # Ensure minimum dimensions
        if coords.shape[1] < 2:
            raise ValueError(
                f"Need at least 2 coordinate dimensions, got {coords.shape[1]}"
            )

        # Make contiguous for zero-copy
        if not coords.is_contiguous():
            coords = coords.contiguous()

        return coords

    def convert_units(
        self, tensor: torch.Tensor, from_unit: str, to_unit: str
    ) -> torch.Tensor:
        """Convert between astronomical units."""
        if from_unit == to_unit:
            return tensor

        # Conversion factors to parsecs
        to_pc = {
            "pc": 1.0,
            "kpc": 1000.0,
            "Mpc": 1000000.0,
            "ly": 3.26156,
            "au": 4.84814e-6,
            "m": 3.24078e-17,
        }

        if from_unit in to_pc and to_unit in to_pc:
            factor = to_pc[from_unit] / to_pc[to_unit]
            return tensor * factor
        else:
            logger.warning(f"Unknown unit conversion: {from_unit} -> {to_unit}")
            return tensor


@contextmanager
def tensor_bridge_context():
    """Context manager for tensor bridge operations."""
    bridge = AstronomicalTensorBridge()
    try:
        yield bridge
    finally:
        pass  # No cleanup needed


def convert_to_backend(tensordict: Any, backend: str, **kwargs) -> Any:
    """Convert TensorDict to specific backend."""
    return bridge.to_visualization(tensordict, backend=backend, **kwargs)


# Backend convenience functions
def to_pyvista(tensordict: Any, **kwargs):
    """Convert to PyVista."""
    return convert_to_backend(tensordict, "pyvista", **kwargs)


def to_open3d(tensordict: Any, **kwargs):
    """Convert to Open3D."""
    return convert_to_backend(tensordict, "open3d", **kwargs)


def to_blender(tensordict: Any, **kwargs):
    """Convert to Blender."""
    return convert_to_backend(tensordict, "blender", **kwargs)


def to_plotly(tensordict: Any, **kwargs):
    """Convert to Plotly."""
    return convert_to_backend(tensordict, "plotly", **kwargs)


def to_cosmograph(tensordict: Any, **kwargs):
    """Convert to Cosmograph."""
    return convert_to_backend(tensordict, "cosmograph", **kwargs)


# Legacy compatibility
def transfer_astronomical_tensor(
    tensor: torch.Tensor, backend: str = "pyvista", **kwargs
) -> Any:
    """Legacy tensor transfer function."""
    return convert_to_backend(tensor, backend=backend, **kwargs)


def astronomical_tensor_zero_copy_context():
    """Legacy context manager."""
    return tensor_bridge_context()


# Legacy aliases
AstronomicalTensorZeroCopyBridge = AstronomicalTensorBridge


__all__ = [
    # Main classes
    "AstronomicalTensorBridge",
    # Main functions
    "tensor_bridge_context",
    # Backend converters
    "to_pyvista",
    "to_open3d",
    "to_blender",
    "to_plotly",
    "to_cosmograph",
    # Legacy compatibility
    "transfer_astronomical_tensor",
    "astronomical_tensor_zero_copy_context",
    "AstronomicalTensorZeroCopyBridge",
]
