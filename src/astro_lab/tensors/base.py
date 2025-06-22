"""
Base Tensor Classes - Core Tensor Infrastructure
===============================================

Provides base classes and interfaces for all tensor types in the AstroLab framework.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import polars as pl
import torch

if TYPE_CHECKING:
    from typing import Self

# Removed memory.py - using simple gc instead
import gc
import logging
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .constants import ASTRO  # Import centralized constants
from .tensor_types import TensorProtocol

logger = logging.getLogger(__name__)

@contextmanager
def comprehensive_cleanup_context(description: str):
    """Minimal no-op context manager."""
    yield

@contextmanager
def pytorch_memory_context(description: str):
    """Simple PyTorch memory context manager."""
    yield

@contextmanager
def memory_tracking_context(description: str):
    """Simple memory tracking context manager."""
    yield

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
    Enhanced base class for astronomical tensors with comprehensive memory management.

    Provides common functionality for all astronomical tensor types including
    automatic memory optimization, device management, and resource cleanup.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, data: Union[torch.Tensor, np.ndarray], **kwargs):
        """Initialize tensor with memory management."""
        # Convert input data to tensor if needed
        if isinstance(data, np.ndarray):
            tensor_data = torch.from_numpy(data).float()
        elif isinstance(data, torch.Tensor):
            tensor_data = data
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        # Extract metadata from kwargs
        metadata = kwargs.pop("metadata", {})
        metadata.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

        # Initialize BaseModel first
        super().__init__()

        # Set attributes directly to bypass Pydantic validation
        object.__setattr__(self, "_data", tensor_data)
        object.__setattr__(self, "_metadata", metadata)

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

    def numel(self) -> int:
        """Get total number of elements."""
        return self._data.numel()

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

    def clone(self) -> "Self":
        """Create a deep copy of the tensor."""
        return self.__class__(data=self._data.clone(), **self._metadata)

    def copy(self) -> "Self":
        """Create a deep copy of the tensor (alias for clone)."""
        return self.clone()

    def detach(self) -> "Self":
        """Detach from computation graph."""
        return self.__class__(data=self._data.detach(), **self._metadata)

    def to(self, *args, **kwargs) -> "Self":
        """Move tensor to device/dtype."""
        return self.__class__(data=self._data.to(*args, **kwargs), **self._metadata)

    def cpu(self) -> "Self":
        """Move tensor to CPU."""
        return self.to(device=torch.device("cpu"))

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "Self":
        """Move tensor to CUDA device."""
        if device is None:
            device = torch.device("cuda")
        return self.to(device=device)

    def numpy(self):
        """Convert to numpy array (CPU only)."""
        return self._data.detach().cpu().numpy()

    # =========================================================================
    # Enhanced Memory Management
    # =========================================================================

    @contextmanager
    def memory_efficient_context(self, operation_name: str = "Tensor operation"):
        """
        Enhanced memory-efficient context manager for tensor operations.

        Args:
            operation_name: Name of the operation for tracking

        Yields:
            self: The tensor instance for chaining operations
        """
        # Ensure tensor is in optimal state
        if not self._data.is_contiguous():
            self._data = self._data.contiguous()

        yield self

    @contextmanager
    def device_transfer_context(self, target_device: Union[str, torch.device]):
        """
        Context manager for efficient device transfers.

        Args:
            target_device: Target device for tensor

        Yields:
            self: Tensor on target device
        """
        original_device = self.device
        target_device = torch.device(target_device)

        # Transfer to target device
        if original_device != target_device:
            self._data = self._data.to(target_device)

        yield self

    @contextmanager
    def batch_processing_context(self, batch_size: int = 1000):
        """
        Context manager for memory-efficient batch processing.

        Args:
            batch_size: Size of each batch

        Yields:
            Generator of tensor batches
        """
        total_size = self.shape[0]

        def batch_generator():
            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)
                yield self._data[i:end_idx]

        yield batch_generator()

    def optimize_memory_layout(self) -> "Self":
        """Optimize tensor memory layout for better performance."""
        with pytorch_memory_context("Memory layout optimization"):
            # Ensure contiguous memory layout
            if not self._data.is_contiguous():
                self._data = self._data.contiguous()

            # Optimize data type if possible
            if self._data.dtype == torch.float64:
                # Convert to float32 for better memory efficiency
                self._data = self._data.float()

            return self

    def memory_info(self) -> Dict[str, Any]:
        """Get memory information (alias for get_memory_info)."""
        return self.get_memory_info()

    def get_memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information for the tensor."""
        storage_size = (
            self._data.untyped_storage().size()
            if hasattr(self._data, "untyped_storage")
            else 0
        )

        info = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "numel": self.numel(),
            "element_size": self._data.element_size(),
            "storage_size": storage_size,
            "data_ptr": self._data.data_ptr(),
            "is_contiguous": self._data.is_contiguous(),
            "requires_grad": self._data.requires_grad,
            "memory_bytes": self.numel() * self._data.element_size(),
            "memory_mb": (self.numel() * self._data.element_size()) / 1024**2,
            "is_pinned": self._data.is_pinned()
            if hasattr(self._data, "is_pinned")
            else False,
        }

        # Add CUDA-specific info
        if self._data.is_cuda:
            info.update(
                {
                    "cuda_device": self._data.device.index,
                    "cuda_allocated": torch.cuda.memory_allocated(self._data.device)
                    / 1024**2,
                    "cuda_reserved": torch.cuda.memory_reserved(self._data.device)
                    / 1024**2,
                }
            )

        return info

    # =========================================================================
    # Enhanced Device Management
    # =========================================================================

    def to_optimal_device(self) -> "Self":
        """Move tensor to optimal device based on availability."""
        with pytorch_memory_context("Optimal device selection"):
            if torch.cuda.is_available() and not self._data.is_cuda:
                optimal_device = torch.device("cuda")
                self._data = self._data.to(optimal_device)
            elif torch.backends.mps.is_available() and not self._data.is_mps:
                optimal_device = torch.device("mps")
                self._data = self._data.to(optimal_device)
            # Otherwise stay on CPU

            return self

    def pin_memory(self) -> "Self":
        """Pin tensor memory for faster CPU-GPU transfers."""
        if not self._data.is_cuda and hasattr(self._data, "pin_memory"):
            with pytorch_memory_context("Memory pinning"):
                self._data = self._data.pin_memory()
        return self

    # =========================================================================
    # Enhanced Operations with Memory Management
    # =========================================================================

    def apply_transform(self, transform_func: callable, **kwargs) -> "Self":
        """
        Apply transformation function with memory management.

        Args:
            transform_func: Function to apply to tensor data
            **kwargs: Additional arguments for transform function

        Returns:
            Transformed tensor
        """
        with self.memory_efficient_context("Transform application"):
            transformed_data = transform_func(self._data, **kwargs)
            return self.__class__(transformed_data, **self._get_init_kwargs())

    def split_batches(self, batch_size: int) -> List["Self"]:
        """
        Split tensor into batches with memory management.

        Args:
            batch_size: Size of each batch

        Returns:
            List of tensor batches
        """
        batches = []
        total_size = self.shape[0]

        with memory_tracking_context(
            f"Tensor splitting: {total_size} -> batches of {batch_size}"
        ):
            for i in range(0, total_size, batch_size):
                end_idx = min(i + batch_size, total_size)
                with pytorch_memory_context(f"Batch creation {i // batch_size + 1}"):
                    batch_data = self._data[i:end_idx]
                    batch_tensor = self.__class__(batch_data, **self._get_init_kwargs())
                    batches.append(batch_tensor)

        return batches

    def merge_batches(self, batches: List["Self"]) -> "Self":
        """
        Merge multiple tensor batches with memory management.

        Args:
            batches: List of tensor batches to merge

        Returns:
            Merged tensor
        """
        with memory_tracking_context(f"Tensor merging: {len(batches)} batches"):
            batch_data = []
            for batch in batches:
                batch_data.append(batch._data)

            with pytorch_memory_context("Tensor concatenation"):
                merged_data = torch.cat(batch_data, dim=0)
                return self.__class__(merged_data, **self._get_init_kwargs())

    def _get_init_kwargs(self) -> Dict[str, Any]:
        """Get initialization kwargs for creating new instances."""
        # Override in subclasses to include specific parameters
        return {}

    # =========================================================================
    # Enhanced Serialization with Memory Management
    # =========================================================================

    def save_optimized(self, path: Union[str, Path], compress: bool = True) -> None:
        """
        Save tensor with memory optimization.

        Args:
            path: Path to save tensor
            compress: Whether to compress the saved data
        """
        path = Path(path)

        with comprehensive_cleanup_context("Tensor saving"):
            # Optimize before saving
            with pytorch_memory_context("Pre-save optimization"):
                optimized_tensor = self.optimize_memory_layout()

            # Save with compression if requested
            with pytorch_memory_context("File writing"):
                if compress:
                    torch.save(optimized_tensor._data, path, pickle_protocol=4)
                else:
                    torch.save(optimized_tensor._data, path)

    @classmethod
    def load_optimized(cls, path: Union[str, Path], **kwargs) -> "Self":
        """
        Load tensor with memory optimization.

        Args:
            path: Path to load tensor from
            **kwargs: Additional initialization arguments

        Returns:
            Loaded tensor instance
        """
        path = Path(path)

        with comprehensive_cleanup_context("Tensor loading"):
            with pytorch_memory_context("File reading"):
                data = torch.load(path, map_location="cpu")

            with pytorch_memory_context("Tensor initialization"):
                return cls(data, **kwargs)

    # =========================================================================
    # Utility methods
    # =========================================================================

    def apply_mask(self, mask: torch.Tensor) -> "Self":
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
