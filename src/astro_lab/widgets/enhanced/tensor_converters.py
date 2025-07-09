"""Zero-copy tensor conversion utilities for visualization backends.

This module provides efficient conversion between different tensor formats
without copying data when possible, using memory views and buffer protocols.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

# Set tensordict behavior globally for this module
os.environ["LIST_TO_STACK"] = "1"

import tensordict

tensordict.set_list_to_stack(True)


class ZeroCopyTensorConverter:
    """Efficient tensor converter using zero-copy operations where possible."""

    def to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert tensor to numpy array with zero-copy when possible.

        Args:
            tensor: Input tensor or numpy array

        Returns:
            Numpy array view (zero-copy) or copy if necessary
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, torch.Tensor):
            # Zero-copy for CPU tensors
            if tensor.is_cpu:
                return tensor.detach().numpy()
            else:
                # Must copy from GPU
                return tensor.detach().cpu().numpy()
        else:
            # Try to convert through array protocol
            return np.asarray(tensor)

    def extract_coordinates(
        self, tensor: Union[np.ndarray, torch.Tensor, Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """Extract 3D coordinates from various tensor formats, using canonical TensorDict API if available.

        Args:
            tensor: Input data

        Returns:
            Numpy array of shape (N, 3) or None
        """
        from tensordict import TensorDict

        # Only use .coordinates if this is a pure TensorDict (not also a dict, torch.Tensor, or np.ndarray)
        if (
            isinstance(tensor, TensorDict)
            and not isinstance(tensor, (dict, torch.Tensor, np.ndarray))
            and hasattr(tensor, "coordinates")
        ):
            coords = tensor.coordinates
            return self.to_numpy(coords)
        # Fallback: dict keys
        if isinstance(tensor, dict):
            if "coordinates" in tensor:
                return self.to_numpy(tensor["coordinates"])
            elif "positions" in tensor:
                return self.to_numpy(tensor["positions"])
            elif "pos" in tensor:
                return self.to_numpy(tensor["pos"])
            elif "ra" in tensor and "dec" in tensor:
                ra = self.to_numpy(tensor["ra"])
                dec = self.to_numpy(tensor["dec"])
                distance = self.to_numpy(tensor.get("distance", np.ones_like(ra)))
                ra_rad = np.deg2rad(ra)
                dec_rad = np.deg2rad(dec)
                x = distance * np.cos(dec_rad) * np.cos(ra_rad)
                y = distance * np.cos(dec_rad) * np.sin(ra_rad)
                z = distance * np.sin(dec_rad)
                return np.stack([x, y, z], axis=-1)
        # Fallback: direct array/tensor
        if isinstance(tensor, (np.ndarray, torch.Tensor)):
            arr = self.to_numpy(tensor)
            if arr.ndim == 2 and arr.shape[1] == 3:
                return arr
            elif arr.ndim == 3 and arr.shape[-1] == 3:
                return arr.reshape(-1, 3)
        return None

    def create_memory_view(
        self, array: np.ndarray, dtype: Optional[np.dtype] = None
    ) -> memoryview:
        """Create a memory view for zero-copy access.

        Args:
            array: Input numpy array
            dtype: Optional dtype for reinterpretation

        Returns:
            Memory view of the array
        """
        if dtype is not None and array.dtype != dtype:
            # This creates a view with different dtype (zero-copy)
            array = array.view(dtype)

        return memoryview(array)

    def share_memory(self, array: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Create shared memory representation of array.

        Args:
            array: Input array

        Returns:
            Tuple of (array view, metadata for reconstruction)
        """
        # Ensure C-contiguous for efficient access
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        metadata = {
            "shape": array.shape,
            "dtype": str(array.dtype),
            "strides": array.strides,
            "offset": 0,
        }

        return array, metadata

    def from_shared_memory(self, buffer: Any, metadata: Dict[str, Any]) -> np.ndarray:
        """Reconstruct array from shared memory.

        Args:
            buffer: Memory buffer
            metadata: Array metadata

        Returns:
            Reconstructed array view
        """
        dtype = np.dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])

        # Create array from buffer without copying
        array = np.frombuffer(buffer, dtype=dtype, offset=metadata["offset"])
        array = array.reshape(shape)

        return array

    def convert_units(
        self, array: np.ndarray, from_unit: str, to_unit: str
    ) -> np.ndarray:
        """Convert units in-place when possible."""
        # Simple unit conversion factors
        conversions = {
            ("pc", "kpc"): 0.001,
            ("kpc", "pc"): 1000.0,
            ("pc", "Mpc"): 1e-6,
            ("Mpc", "pc"): 1e6,
            ("kpc", "Mpc"): 0.001,
            ("Mpc", "kpc"): 1000.0,
        }

        factor = conversions.get((from_unit, to_unit), 1.0)
        if factor != 1.0:
            # In-place multiplication for zero-copy
            array *= factor

        return array

    def extract_features(self, data: Any) -> Dict[str, np.ndarray]:
        """Extract all visualization-relevant features using canonical TensorDict API if available.

        Args:
            data: Input data (TensorDict, dict, etc.)

        Returns:
            Dictionary of numpy arrays for all features (coordinates, magnitudes, cluster_labels, ...)
        """
        # Canonical: AstroTensorDict/SpatialTensorDict method
        if hasattr(data, "extract_features"):
            features = data.extract_features()
            # Convert all tensors to numpy
            return {k: self.to_numpy(v) for k, v in features.items()}
        # Fallback: dict logic
        features = {}
        coords = self.extract_coordinates(data)
        if coords is not None:
            features["coordinates"] = coords
        for key in [
            "magnitudes",
            "colors",
            "velocities",
            "masses",
            "temperatures",
            "luminosities",
            "redshifts",
            "cluster_labels",
            "densities",
            "distances",
        ]:
            if isinstance(data, dict) and key in data:
                features[key] = self.to_numpy(data[key])
        return features

    def prepare_for_backend(
        self, data: Dict[str, Any], backend: str, **kwargs
    ) -> Dict[str, Any]:
        """Prepare data for specific visualization backend.

        Args:
            data: Input data
            backend: Target backend ('pyvista', 'open3d', 'plotly', etc.)
            **kwargs: Backend-specific options

        Returns:
            Prepared data dictionary
        """
        features = self.extract_features(data)

        if backend == "pyvista":
            # PyVista expects specific format
            prepared = {
                "points": features.get("coordinates", np.zeros((0, 3))),
                "scalars": {},
            }
            for key, value in features.items():
                if (
                    key != "coordinates"
                    and value.shape[0] == prepared["points"].shape[0]
                ):
                    prepared["scalars"][key] = value

        elif backend == "open3d":
            # Open3D expects specific format
            prepared = {
                "points": features.get("coordinates", np.zeros((0, 3))),
                "colors": features.get("colors"),
                "normals": features.get("normals"),
            }

        elif backend == "plotly":
            # Plotly expects flat arrays
            prepared = features

        else:
            prepared = features

        return prepared


# Global converter instance
converter = ZeroCopyTensorConverter()


# Convenience functions
def to_numpy_zero_copy(tensor: Any) -> np.ndarray:
    """Convert tensor to numpy with zero-copy when possible."""
    return converter.to_numpy(tensor)


def extract_coordinates_zero_copy(data: Any) -> Optional[np.ndarray]:
    """Extract coordinates with zero-copy."""
    return converter.extract_coordinates(data)


def prepare_for_visualization(data: Any, backend: str, **kwargs) -> Dict[str, Any]:
    """Prepare data for visualization backend with zero-copy."""
    return converter.prepare_for_backend(data, backend, **kwargs)
