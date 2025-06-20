"""
Simplified Base Class for Astronomical Tensors
=============================================

Refactored version with simplified memory management, following the guide to:
- Remove complex global registry
- Use Python's garbage collection
- Eliminate unnecessary cleanup callbacks
- Improve type annotations
- Add validation mixin
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel, ConfigDict

from .constants import ASTRO  # Import centralized constants
from .tensor_types import TensorProtocol

logger = logging.getLogger(__name__)


class ValidationMixin:
    """Mixin for common validation patterns."""

    def validate_shape(
        self, expected_dims: int, last_dim_size: Optional[int] = None
    ) -> None:
        """Validate tensor dimensions."""
        if not hasattr(self, "_data"):
            raise ValueError("Tensor data not initialized")

        data = getattr(self, "_data")
        if data.dim() < expected_dims:
            raise ValueError(
                f"{self.__class__.__name__} requires at least {expected_dims}D data, "
                f"got {data.dim()}D"
            )

        if last_dim_size is not None and data.shape[-1] != last_dim_size:
            raise ValueError(
                f"Last dimension must be {last_dim_size}, got {data.shape[-1]}"
            )

    def validate_non_empty(self) -> None:
        """Validate tensor is non-empty."""
        if not hasattr(self, "_data"):
            raise ValueError(f"{self.__class__.__name__} cannot be empty")

        data = getattr(self, "_data")
        if data.numel() == 0:
            raise ValueError(f"{self.__class__.__name__} cannot be empty")

    def validate_finite_values(self) -> None:
        """Validate tensor contains finite values."""
        if not hasattr(self, "_data"):
            return

        data = getattr(self, "_data")
        if not torch.isfinite(data).all():
            raise ValueError(f"{self.__class__.__name__} contains non-finite values")


class AstroTensorBase(BaseModel, ValidationMixin):
    """
    Simplified base class for astronomical tensor types.

    Key improvements over the original:
    - No global registry (relies on Python GC)
    - Simplified memory management
    - Better type annotations
    - Consistent metadata access
    - Common validation patterns
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        data: Union[torch.Tensor, List, Any],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        **metadata,
    ):
        """
        Initialize astronomical tensor with data and metadata.

        Args:
            data: Input tensor data
            dtype: Desired data type
            device: Target device
            **metadata: Astronomical metadata
        """
        # Convert data to tensor if needed
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype, device=device)
        elif dtype is not None or device is not None:
            data = data.to(dtype=dtype, device=device)

        # Store data and clean metadata
        self._data = data
        self._metadata = self._clean_metadata(metadata)

        # Validate the tensor
        self._validate()

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to prevent autograd memory leaks.
        Simplified version - just detach tensors.
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                # Detach to prevent autograd leaks
                cleaned[key] = value.detach() if value.requires_grad else value
            elif isinstance(value, (list, tuple)) and value:
                # Handle tensor sequences
                if isinstance(value[0], torch.Tensor):
                    cleaned[key] = [
                        v.detach() if v.requires_grad else v
                        for v in value
                        if isinstance(v, torch.Tensor)
                    ]
                else:
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    def _validate(self) -> None:
        """
        Validate tensor data - default implementation.
        Subclasses should override with specific validation.
        """
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("Data must be a torch.Tensor")

        # Basic validations
        self.validate_non_empty()
        self.validate_finite_values()

    # =========================================================================
    # Core Properties with consistent access patterns
    # =========================================================================

    @property
    def data(self) -> torch.Tensor:
        """Get the underlying tensor data."""
        return self._data

    @property
    def shape(self) -> torch.Size:
        """Get tensor shape."""
        return self._data.shape

    @property
    def device(self) -> torch.device:
        """Get tensor device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Get tensor data type."""
        return self._data.dtype

    @property
    def ndim(self) -> int:
        """Get number of dimensions."""
        return self._data.dim()

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        """Get size of tensor or specific dimension."""
        if dim is None:
            return self._data.size()
        return self._data.size(dim)

    # =========================================================================
    # Consistent metadata access
    # =========================================================================

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value with consistent default handling."""
        return self._metadata.get(key, default)

    def update_metadata(self, **kwargs) -> None:
        """Update metadata with new values."""
        cleaned_kwargs = self._clean_metadata(kwargs)
        self._metadata.update(cleaned_kwargs)

    def has_metadata(self, key: str) -> bool:
        """Check if metadata key exists."""
        return key in self._metadata

    # =========================================================================
    # Tensor operations
    # =========================================================================

    def clone(self) -> AstroTensorBase:
        """Create a deep copy of the tensor."""
        return self.__class__(data=self._data.clone(), **self._metadata)

    def detach(self) -> AstroTensorBase:
        """Detach from computation graph."""
        return self.__class__(data=self._data.detach(), **self._metadata)

    def to(self, *args, **kwargs) -> AstroTensorBase:
        """Move tensor to device/dtype."""
        return self.__class__(data=self._data.to(*args, **kwargs), **self._metadata)

    def cpu(self) -> AstroTensorBase:
        """Move tensor to CPU."""
        return self.to(device=torch.device("cpu"))

    def cuda(
        self, device: Optional[Union[int, str, torch.device]] = None
    ) -> AstroTensorBase:
        """Move tensor to CUDA device."""
        if device is None:
            device = torch.device("cuda")
        return self.to(device=device)

    def numpy(self):
        """Convert to numpy array (CPU only)."""
        return self._data.detach().cpu().numpy()

    # =========================================================================
    # Memory management - simplified
    # =========================================================================

    def memory_efficient_context(self):
        """
        Simple context manager for memory-efficient operations.
        Just clears CUDA cache if available.
        """

        @contextmanager
        def _context():
            try:
                yield self
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return _context()

    def memory_info(self) -> Dict[str, Any]:
        """Get memory information for this tensor."""
        info = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "numel": self._data.numel(),
            "element_size": self._data.element_size(),
            "memory_bytes": self._data.numel() * self._data.element_size(),
            "memory_mb": (self._data.numel() * self._data.element_size()) / 1024**2,
            "requires_grad": self._data.requires_grad,
            "is_contiguous": self._data.is_contiguous(),
            "data_ptr": self._data.data_ptr(),
            "storage_size": self._data.storage().size(),
        }

        if self._data.is_cuda:
            info.update(
                {
                    "cuda_device": self.device.index,
                    "is_pinned": False,  # Can't check pinned memory for CUDA tensors
                }
            )
        else:
            info["is_pinned"] = (
                self._data.is_pinned() if hasattr(self._data, "is_pinned") else False
            )

        return info

    # =========================================================================
    # Visualization integration (simplified)
    # =========================================================================

    def to_pyvista(self, scalars: Optional[torch.Tensor] = None, **kwargs):
        """
        Convert tensor to PyVista mesh.
        Simplified - requires tensor utils to be available.
        """
        try:
            from ..utils.tensor import transfer_to_framework

            return transfer_to_framework(
                self._data, "pyvista", scalars=scalars, **kwargs
            )
        except ImportError:
            logger.warning("PyVista conversion requires tensor utils")
            return None

    def to_blender(self, name: str = "astro_object", collection_name: str = "AstroLab"):
        """
        Convert tensor to Blender object.
        Simplified - requires tensor utils to be available.
        """
        try:
            from ..utils.tensor import transfer_to_framework

            return transfer_to_framework(
                self._data, "blender", name=name, collection_name=collection_name
            )
        except ImportError:
            logger.warning("Blender conversion requires tensor utils")
            return None

    # =========================================================================
    # Utility methods
    # =========================================================================

    def apply_mask(self, mask: torch.Tensor) -> AstroTensorBase:
        """Apply boolean mask to tensor."""
        if mask.dtype != torch.bool:
            raise ValueError("Mask must be boolean tensor")

        return self.__class__(data=self._data[mask], **self._metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tensor to dictionary representation for serialization."""
        return {
            "tensor_type": self.__class__.__name__,
            "data": self._data.cpu().numpy().tolist(),  # Convert to serializable format
            "shape": list(self._data.shape),
            "dtype": str(self._data.dtype),
            "device": str(self._data.device),
            "metadata": self._metadata,
        }

    def model_dump(self) -> Dict[str, Any]:
        """Pydantic-compatible model serialization."""
        return self.to_dict()

    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        return {
            "data": self._data,
            "metadata": self._metadata,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self._data = state["data"]
        self._metadata = state["metadata"]

    # =========================================================================
    # Standard methods
    # =========================================================================

    def __len__(self) -> int:
        """Get length of first dimension."""
        return self._data.shape[0] if self._data.numel() > 0 else 0

    def __repr__(self) -> str:
        """String representation with consistent format."""
        tensor_type = self.__class__.__name__
        shape_str = "x".join(map(str, self.shape))
        device_str = f", device='{self.device}'" if self.device.type != "cpu" else ""

        return f"{tensor_type}(shape=[{shape_str}], dtype={self.dtype}{device_str})"


# Simplified memory profiling
@contextmanager
def memory_profiling_context(description: str = "Operation"):
    """Simple memory profiling context manager."""
    initial_memory = None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()

    try:
        yield
    finally:
        if torch.cuda.is_available() and initial_memory is not None:
            final_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()

            logger.info(
                f"Memory Profile [{description}]: "
                f"Initial: {initial_memory / 1024**2:.1f}MB, "
                f"Final: {final_memory / 1024**2:.1f}MB, "
                f"Peak: {peak_memory / 1024**2:.1f}MB"
            )


# Direct transfer function (simplified)
def transfer_direct(source, target, attribute: str = "position", device: str = "cpu"):
    """
    Direct tensor transfer between objects.
    Simplified version without complex memory management.
    """
    if hasattr(source, attribute) and hasattr(target, attribute):
        source_tensor = getattr(source, attribute)
        if isinstance(source_tensor, torch.Tensor):
            target_tensor = source_tensor.to(device=device)
            setattr(target, attribute, target_tensor)
            return True
    return False
