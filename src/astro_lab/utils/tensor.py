"""Tensor utility functions for AstroLab.

This module provides helper functions for working with tensors and TensorDicts.
"""

from __future__ import annotations

from typing import Any, Union

import numpy as np
import torch


def extract_coordinates(
    coordinates: Union[torch.Tensor, "SpatialTensorDict"],
) -> torch.Tensor:
    """Extract coordinate tensor from various input types.

    This is a common pattern used throughout the codebase to handle
    both raw tensors and SpatialTensorDict inputs.

    Args:
        coordinates: Either a torch.Tensor or SpatialTensorDict containing coordinates

    Returns:
        Coordinate tensor of shape [N, 3]

    Examples:
        >>> coords = torch.randn(100, 3)
        >>> result = extract_coordinates(coords)
        >>> assert result.shape == (100, 3)

        >>> from astro_lab.tensors import SpatialTensorDict
        >>> spatial = SpatialTensorDict(coords)
        >>> result = extract_coordinates(spatial)
        >>> assert result.shape == (100, 3)
    """
    # Handle SpatialTensorDict
    if hasattr(coordinates, "coordinates"):
        return coordinates.coordinates

    # Handle dict-like TensorDict with "coordinates" key
    if hasattr(coordinates, "__getitem__"):
        try:
            return coordinates["coordinates"]
        except (KeyError, TypeError):
            pass

    # Assume it's already a tensor
    return coordinates


def numpy_to_float32_tensor(arr: Any) -> torch.Tensor:
    """Convert arrays or tensors to ``float32``; support structured NumPy (e.g. FITS/NSA rows).

    Plain ``torch.tensor(numpy_structured, dtype=float32)`` fails with an unsafe-cast error
    when the dtype has named fields (``numpy.record``). This helper picks common coordinate
    triplets or falls back to a contiguous float array.
    """
    if isinstance(arr, torch.Tensor):
        return arr.float() if arr.dtype != torch.float32 else arr

    a = np.asanyarray(arr)

    if a.dtype.names is not None:
        names = a.dtype.names
        lower = {str(n).lower(): n for n in names}
        triplets = (
            ("x", "y", "z"),
            ("ra", "dec", "zdist"),
            ("ra", "dec", "z"),
            ("ra", "dec", "distance_pc"),
        )
        for t in triplets:
            keys = [lower.get(k) for k in t]
            if not all(keys):
                continue
            parts: list[np.ndarray] = []
            for fld in keys:
                col = np.asarray(a[fld], dtype=np.float64).reshape(-1)
                parts.append(col)
            n = min(len(p) for p in parts)
            stacked = np.column_stack([p[:n] for p in parts])
            return torch.from_numpy(np.ascontiguousarray(stacked)).to(torch.float32)

        raise TypeError(
            "Structured numpy array cannot be converted to a float tensor without known "
            f"coordinate fields (x,y,z or ra,dec,z|zdist|distance_pc). Got fields: {names}"
        )

    a64 = np.ascontiguousarray(a.astype(np.float64, copy=False))
    return torch.from_numpy(a64).to(torch.float32)
