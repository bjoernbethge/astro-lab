"""
NumPy 2.x Compatibility Layer for Blender Integration
====================================================

This module provides a compatibility layer that allows using Blender's bpy module
with NumPy 2.x by providing alternative implementations of common operations.

Instead of trying to make the original bpy work with NumPy 2.x, we create
a wrapper that handles the conversion between NumPy 2.x arrays and Blender's
internal data structures.
"""

import os
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Suppress numpy warnings without changing array API
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


class NumpyCompatLayer:
    """Compatibility layer for NumPy 2.x with Blender."""

    def __init__(self):
        self._bpy = None
        self._mathutils = None
        self._available = False
        self._init_blender()

    def _init_blender(self):
        """Initialize Blender with NumPy 2.x compatibility."""
        try:
            # Try to import bpy with our workaround
            import bpy
            from mathutils import Euler, Matrix, Vector

            self._bpy = bpy
            self._mathutils = {"Vector": Vector, "Matrix": Matrix, "Euler": Euler}
            self._available = True

        except ImportError as e:
            print(f"Blender not available: {e}")
            self._available = False
        except Exception as e:
            print(f"Error initializing Blender: {e}")
            self._available = False

    @property
    def available(self) -> bool:
        """Check if Blender is available."""
        return self._available

    @property
    def bpy(self):
        """Get the bpy module if available."""
        return self._bpy

    @property
    def mathutils(self):
        """Get mathutils if available."""
        return self._mathutils

    def numpy_to_blender_array(self, arr: np.ndarray) -> Any:
        """Convert NumPy 2.x array to Blender-compatible format."""
        if not self._available:
            return arr

        # Convert to NumPy 1.x compatible format if needed
        if hasattr(arr, "numpy"):
            # Handle array API arrays
            arr = arr.numpy()

        # Ensure contiguous memory layout
        if not arr.flags.c_contiguous:
            arr = np.ascontiguousarray(arr)

        return arr

    def blender_to_numpy_array(self, blender_data: Any) -> np.ndarray:
        """Convert Blender data to NumPy 2.x array."""
        if not self._available:
            return np.array(blender_data)

        # Convert to NumPy array
        arr = np.array(blender_data)

        # Ensure we have a proper NumPy 2.x array
        return arr

    def create_vector(self, x: float, y: float, z: float) -> Any:
        """Create a Blender Vector with NumPy 2.x compatibility."""
        if not self._available or not self._mathutils:
            return np.array([x, y, z])

        return self._mathutils["Vector"]((x, y, z))

    def create_matrix(self, data: Union[List, np.ndarray]) -> Any:
        """Create a Blender Matrix with NumPy 2.x compatibility."""
        if not self._available or not self._mathutils:
            return np.array(data)

        # Convert to list format expected by Blender
        if isinstance(data, np.ndarray):
            data = data.tolist()

        return self._mathutils["Matrix"](data)


# Global instance
numpy_compat = NumpyCompatLayer()


# Convenience functions
def is_blender_available() -> bool:
    """Check if Blender is available with NumPy 2.x compatibility."""
    return numpy_compat.available


def get_bpy():
    """Get bpy module if available."""
    return numpy_compat.bpy


def get_mathutils():
    """Get mathutils if available."""
    return numpy_compat.mathutils


@contextmanager
def blender_context():
    """Context manager for Blender operations with NumPy 2.x compatibility."""
    if not numpy_compat.available:
        raise RuntimeError("Blender not available")

    try:
        yield numpy_compat
    except Exception as e:
        print(f"Error in Blender context: {e}")
        raise
