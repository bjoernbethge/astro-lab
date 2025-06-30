"""
Base TensorDict for AstroLab
============================

Base class for all astronomical TensorDicts with common functionality.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict

logger = logging.getLogger(__name__)


class AstroTensorDict(TensorDict):
    """
    Base class for all astronomical TensorDicts.

    Extends TensorDict with astronomical-specific functionality
    while maintaining native PyTorch performance.
    """

    def __init__(self, data: Dict[str, Any], **kwargs):
        """
        Initialize AstroTensorDict.

        Args:
            data: Dictionary of tensors and metadata
            **kwargs: Additional TensorDict arguments
        """
        # Handle metadata separately to avoid batch dimension issues
        metadata = data.pop("meta", {}) if "meta" in data else {}
        super().__init__(data, **kwargs)

        self._metadata = metadata if metadata else {}
        if "tensor_type" not in self._metadata:
            self._metadata["tensor_type"] = self.__class__.__name__
        if "creation_time" not in self._metadata:
            import time

            self._metadata["creation_time"] = time.time()

        logger.debug(f"Created {self.__class__.__name__} with {self.n_objects} objects")

    @property
    def meta(self) -> Dict[str, Any]:
        """Access to metadata dictionary."""
        return self._metadata

    @property
    def n_objects(self) -> int:
        """Number of astronomical objects in this tensor."""
        return self.batch_size[0] if self.batch_size else 0

    def add_history(self, operation: str, **details) -> "AstroTensorDict":
        """
        Add operation to processing history.

        Args:
            operation: Name of the operation
            **details: Additional details about the operation
        """
        if "history" not in self._metadata:
            self._metadata["history"] = []

        import time

        self._metadata["history"].append(
            {"operation": operation, "timestamp": time.time(), "details": details}
        )
        return self

    def memory_info(self) -> Dict[str, Any]:
        """Get memory usage information."""
        info = {
            "total_bytes": 0,
            "n_tensors": 0,
            "batch_size": self.batch_size,
            "tensor_shapes": {},
        }

        devices = set()
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                info["total_bytes"] += value.element_size() * value.nelement()
                info["n_tensors"] += 1
                info["tensor_shapes"][key] = tuple(value.shape)
                devices.add(str(value.device))

        info["total_mb"] = info["total_bytes"] / (1024 * 1024)
        info["devices"] = list(devices) if devices else ["cpu"]
        info["primary_device"] = list(devices)[0] if devices else "cpu"

        return info

    def validate(self) -> bool:
        """
        Basic validation of tensor structure.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check that all tensors have compatible batch dimensions
            batch_size = self.batch_size
            for key, value in self.items():
                if isinstance(value, torch.Tensor):
                    if value.shape[: len(batch_size)] != batch_size:
                        logger.warning(f"Tensor {key} has incompatible batch size")
                        return False
            return True
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert all tensors to numpy arrays."""
        return {
            key: value.detach().cpu().numpy() if torch.is_tensor(value) else value
            for key, value in self.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to regular dictionary."""
        return dict(self)

    def summary(self) -> str:
        """Generate a summary string of the tensor contents."""
        lines = [
            f"{self.__class__.__name__} Summary",
            "=" * 40,
            f"Objects: {self.n_objects}",
            f"Batch size: {self.batch_size}",
            f"Keys: {list(self.keys())}",
        ]

        # Memory info
        mem_info = self.memory_info()
        lines.append(f"Memory: {mem_info['total_mb']:.1f} MB")
        lines.append(f"Device: {mem_info['primary_device']}")

        # Tensor shapes
        for key, shape in mem_info["tensor_shapes"].items():
            lines.append(f"  {key}: {shape}")

        # History (last few operations)
        if "history" in self._metadata and self._metadata["history"]:
            lines.append("\nRecent Operations:")
            for op in self._metadata["history"][-3:]:  # Last 3 operations
                lines.append(f"  - {op['operation']}")

        return "\n".join(lines)

    def clear_temp_tensors(self):
        """Remove temporary tensors (keys starting with '_temp_')."""
        temp_keys = [key for key in self.keys() if key.startswith("_temp_")]
        for key in temp_keys:
            del self[key]

    def optimize_memory(self):
        """Optimize memory usage."""
        self.clear_temp_tensors()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        """Clean up resources."""
        try:
            self.clear_temp_tensors()
            # Clear metadata tensors that might hold references
            if hasattr(self, "_metadata"):
                for key in list(self._metadata.keys()):
                    if isinstance(self._metadata[key], (torch.Tensor, np.ndarray)):
                        del self._metadata[key]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def __del__(self):
        """Destructor with safe cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Silent cleanup failure in destructor
            pass

    def clone(self) -> "AstroTensorDict":
        """Create a deep copy of the tensor dict."""
        cloned = super().clone()
        # Clone metadata
        cloned._metadata = self._metadata.copy()
        return cloned

    def select_subset(
        self, indices: Union[torch.Tensor, List[int], slice]
    ) -> "AstroTensorDict":
        """
        Select a subset of objects.

        Args:
            indices: Indices to select

        Returns:
            New AstroTensorDict with selected objects
        """
        if isinstance(indices, (list, slice)):
            indices = torch.tensor(indices) if isinstance(indices, list) else indices

        # Select data
        subset_data = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) == len(self.batch_size):
                    # Same dimensionality as batch
                    subset_data[key] = value[indices]
                else:
                    # Higher dimensionality, select along first dimension
                    subset_data[key] = value[indices]
            else:
                subset_data[key] = value

        # Copy metadata
        subset_data["meta"] = self._metadata.copy()

        # Create new instance
        result = type(self)(subset_data)
        result.add_history(
            "select_subset",
            n_selected=len(indices) if hasattr(indices, "__len__") else 1,
        )

        return result

    def filter_by_condition(self, condition: torch.Tensor) -> "AstroTensorDict":
        """
        Filter objects by boolean condition.

        Args:
            condition: Boolean tensor [N] indicating which objects to keep

        Returns:
            Filtered AstroTensorDict
        """
        if not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool:
            raise ValueError("Condition must be a boolean tensor")

        indices = torch.where(condition)[0]
        result = self.select_subset(indices)
        result.add_history(
            "filter_by_condition", n_kept=len(indices), n_total=len(condition)
        )

        return result

    def split_by_key(
        self, split_key: str, values: Optional[List] = None
    ) -> Dict[str, "AstroTensorDict"]:
        """
        Split tensor dict by values in a specific key.

        Args:
            split_key: Key to split by
            values: Specific values to split by (if None, use unique values)

        Returns:
            Dictionary mapping values to AstroTensorDicts
        """
        if split_key not in self:
            raise ValueError(f"Split key '{split_key}' not found")

        split_tensor = self[split_key]
        if values is None:
            values = torch.unique(split_tensor).tolist()

        result = {}
        for value in values:
            mask = split_tensor == value
            subset = self.filter_by_condition(mask)
            result[str(value)] = subset

        return result

    def concatenate(self, other: "AstroTensorDict", dim: int = 0) -> "AstroTensorDict":
        """
        Concatenate with another AstroTensorDict.

        Args:
            other: Other tensor dict to concatenate
            dim: Dimension to concatenate along

        Returns:
            Concatenated AstroTensorDict
        """
        if not isinstance(other, AstroTensorDict):
            raise ValueError("Can only concatenate with another AstroTensorDict")

        # Find common keys
        common_keys = set(self.keys()) & set(other.keys())

        concat_data = {}
        for key in common_keys:
            if isinstance(self[key], torch.Tensor) and isinstance(
                other[key], torch.Tensor
            ):
                concat_data[key] = torch.cat([self[key], other[key]], dim=dim)
            else:
                # Keep the first value for non-tensor data
                concat_data[key] = self[key]

        # Merge metadata
        merged_meta = self._metadata.copy()
        merged_meta.update(other._metadata)
        concat_data["meta"] = merged_meta

        result = type(self)(concat_data)
        result.add_history(
            "concatenate", n_self=self.n_objects, n_other=other.n_objects
        )

        return result

    def to(self, device: Union[str, torch.device]) -> "AstroTensorDict":
        """Move tensor to device."""
        device = torch.device(device) if isinstance(device, str) else device

        # Move all tensors to device
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(device)

        return self

    def cpu(self) -> "AstroTensorDict":
        """Move tensor to CPU."""
        return self.to("cpu")

    def cuda(self) -> "AstroTensorDict":
        """Move tensor to CUDA."""
        return self.to("cuda")
