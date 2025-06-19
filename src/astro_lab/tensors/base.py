"""
Base class for astronomical tensors with common functionality.
"""

import gc
import logging
import weakref
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager

import torch
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Import zero-copy data bridges from utils
try:
    from ..utils.tensor import transfer_to_framework, get_tensor_memory_info as utils_memory_info
    TENSOR_UTILS_AVAILABLE = True
except ImportError:
    TENSOR_UTILS_AVAILABLE = False
    transfer_to_framework = None
    utils_memory_info = None


# Global tensor registry for memory tracking - use WeakValueDictionary instead of WeakSet
_tensor_registry = weakref.WeakValueDictionary()
_tensor_counter = 0


class AstroTensorBase(BaseModel):
    """
    Base class for all astronomical tensor types with integrated visualization and 
    state-of-the-art memory management (2025).

    This class wraps a PyTorch tensor and provides:
    - Common astronomical functionality 
    - Direct PyVista mesh conversion
    - Direct Blender object conversion
    - Memory-efficient data exchange with automatic cleanup
    - Zero-copy operations where possible
    - Context manager support for automatic resource management
    - Advanced memory leak prevention
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

        # Store data and metadata as private attributes
        self._data = data
        
        # Clean metadata from tensor references to prevent autograd leaks
        self._metadata = self._clean_metadata(metadata.copy())

        # Memory management
        self._cleanup_callbacks = []
        self._is_cleaned = False
        
        # Register for memory tracking
        global _tensor_counter
        _tensor_registry[_tensor_counter] = self
        self._tensor_id = _tensor_counter
        _tensor_counter += 1

        # Validate the tensor
        self._validate()

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata to prevent autograd memory leaks by detaching tensors.
        Following 2025 best practices for PyTorch memory management.
        """
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                # Detach tensors to prevent autograd leaks
                if value.requires_grad:
                    cleaned[key] = value.detach().clone()
                else:
                    cleaned[key] = value.clone()
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], torch.Tensor):
                # Clean tensor lists/tuples
                cleaned[key] = [v.detach().clone() if v.requires_grad else v.clone() 
                               for v in value if isinstance(v, torch.Tensor)]
            else:
                cleaned[key] = value
        return cleaned

    def _validate(self) -> None:
        """Validate tensor data - default implementation."""
        if not isinstance(self._data, torch.Tensor):
            raise ValueError("Data must be a torch.Tensor")

    def __enter__(self) -> "AstroTensorBase":
        """Context manager entry - 2025 memory management standard."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with automatic cleanup."""
        self.cleanup()
        return False

    def __del__(self) -> None:
        """Destructor with memory cleanup."""
        try:
            self.cleanup()
        except Exception:
            # Avoid exceptions in destructor
            pass

    def cleanup(self) -> None:
        """
        Explicit memory cleanup following 2025 PyTorch best practices.
        Prevents memory leaks and autograd graph retention.
        """
        if self._is_cleaned:
            return
            
        try:
            # Execute cleanup callbacks
            for callback in self._cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            # Clear metadata tensors safely
            if hasattr(self, '_metadata'):
                metadata_copy = self._metadata.copy()
                for key, value in metadata_copy.items():
                    if isinstance(value, torch.Tensor):
                        del self._metadata[key]
                    elif isinstance(value, (list, tuple)):
                        filtered = []
                        for item in value:
                            if not isinstance(item, torch.Tensor):
                                filtered.append(item)
                        self._metadata[key] = filtered
            
            # Clear main data reference
            if hasattr(self, '_data') and self._data is not None:
                self._data = None
            
            # Clear GPU cache if on CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            self._is_cleaned = True
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")

    def add_cleanup_callback(self, callback: callable) -> None:
        """Add callback to be executed during cleanup."""
        self._cleanup_callbacks.append(callback)

    @contextmanager
    def memory_efficient_context(self):
        """
        Context manager for memory-efficient operations.
        Automatically manages GPU memory and prevents leaks.
        """
        initial_memory = None
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()
        
        try:
            yield self
        finally:
            # Cleanup and memory reporting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                if initial_memory is not None:
                    memory_diff = final_memory - initial_memory
                    if memory_diff > 1024**2:  # More than 1MB difference
                        logger.warning(f"Memory increase detected: {memory_diff / 1024**2:.2f} MB")

    # =========================================================================
    # Core tensor operations with memory optimization
    # =========================================================================

    @property
    def data(self) -> torch.Tensor:
        """Access underlying tensor data."""
        if self._is_cleaned:
            raise RuntimeError("Tensor has been cleaned up")
        return self._data

    @property
    def shape(self) -> torch.Size:
        """Tensor shape."""
        return self._data.shape

    @property
    def device(self) -> torch.device:
        """Tensor device."""
        return self._data.device

    @property
    def dtype(self) -> torch.dtype:
        """Tensor data type."""
        return self._data.dtype

    def dim(self) -> int:
        """Number of dimensions."""
        return self._data.dim()

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        """Size of tensor."""
        return self._data.size(dim)

    def unsqueeze(self, dim: int) -> "AstroTensorBase":
        """Unsqueeze tensor preserving metadata."""
        new_data = self._data.unsqueeze(dim)
        return self.__class__(new_data, **self._metadata)

    def squeeze(self, dim: Optional[int] = None) -> "AstroTensorBase":
        """Squeeze tensor preserving metadata."""
        new_data = self._data.squeeze(dim)
        return self.__class__(new_data, **self._metadata)

    # =========================================================================
    # Metadata operations with memory safety
    # =========================================================================

    def update_metadata(self, **kwargs) -> None:
        """Update metadata with automatic tensor cleaning."""
        cleaned_kwargs = self._clean_metadata(kwargs)
        self._metadata.update(cleaned_kwargs)

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value."""
        return self._metadata.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Use detached tensor data to prevent autograd issues
        tensor_data = self._data.detach().cpu()
        return {
            "data": tensor_data.tolist(),
            "metadata": {k: v for k, v in self._metadata.items() 
                        if not isinstance(v, torch.Tensor)},  # Exclude tensor metadata
            "shape": list(self._data.shape),
            "dtype": str(self._data.dtype),
            "device": str(self._data.device),
        }

    # =========================================================================
    # Tensor operations preserving metadata with memory optimization
    # =========================================================================

    def clone(self) -> "AstroTensorBase":
        """Clone tensor preserving all metadata."""
        # Use detach to prevent autograd leaks
        cloned_data = self._data.detach().clone()
        return self.__class__(cloned_data, **self._metadata)

    def detach(self) -> "AstroTensorBase":
        """Detach tensor preserving all metadata."""
        return self.__class__(self._data.detach(), **self._metadata)

    def to(self, *args, **kwargs) -> "AstroTensorBase":
        """Move tensor to device/dtype preserving metadata."""
        return self.__class__(self._data.to(*args, **kwargs), **self._metadata)

    def cpu(self) -> "AstroTensorBase":
        """Move tensor to CPU preserving metadata."""
        return self.__class__(self._data.cpu(), **self._metadata)

    def cuda(self, device: Optional[Union[int, str, torch.device]] = None) -> "AstroTensorBase":
        """Move tensor to CUDA preserving metadata."""
        return self.__class__(self._data.cuda(device), **self._metadata)

    def numpy(self):
        """Convert to numpy array (data only) with memory optimization."""
        # Always detach to prevent autograd issues
        tensor_data = self._data.detach()
        if tensor_data.is_cuda:
            return tensor_data.cpu().numpy()
        return tensor_data.numpy()

    # =========================================================================
    # Memory management and monitoring
    # =========================================================================

    def data_ptr(self) -> int:
        """Get memory pointer for zero-copy operations (PyTorch equivalent)."""
        return self._data.data_ptr()

    def memory_info(self) -> Dict[str, Any]:
        """Get comprehensive memory information for debugging."""
        # Use modern storage access method
        try:
            storage_size = self._data.untyped_storage().size()
        except AttributeError:
            # Fallback for older PyTorch versions
            storage_size = self._data.storage().size()
        
        info = {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "shape": list(self.shape),
            "numel": self._data.numel(),
            "element_size": self._data.element_size(),
            "storage_size": storage_size,
            "data_ptr": self.data_ptr(),
            "is_contiguous": self._data.is_contiguous(),
            "requires_grad": self._data.requires_grad,
            "memory_bytes": self._data.numel() * self._data.element_size(),
        }
        
        # Add CUDA memory info if available
        if torch.cuda.is_available() and self._data.is_cuda:
            info.update({
                "cuda_allocated": torch.cuda.memory_allocated(self.device),
                "cuda_reserved": torch.cuda.memory_reserved(self.device),
                "cuda_max_allocated": torch.cuda.max_memory_allocated(self.device),
            })
        
        return info

    @classmethod
    def get_global_memory_stats(cls) -> Dict[str, Any]:
        """Get global memory statistics for all tensor instances."""
        active_tensors = len(_tensor_registry)
        total_elements = sum(t._data.numel() for t in _tensor_registry.values() if not t._is_cleaned)
        
        stats = {
            "active_tensors": active_tensors,
            "total_elements": total_elements,
        }
        
        if torch.cuda.is_available():
            stats.update({
                "cuda_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "cuda_reserved_mb": torch.cuda.memory_reserved() / 1024**2,
            })
            
        return stats

    # =========================================================================
    # PyVista integration with memory management
    # =========================================================================

    def to_pyvista(
        self,
        scalars: Optional[torch.Tensor] = None,
        **kwargs
    ) -> "Any":
        """
        Convert tensor to PyVista mesh using zero-copy data bridge.
        """
        if not TENSOR_UTILS_AVAILABLE or not transfer_to_framework:
            raise ImportError("Tensor utils module not available for PyVista conversion")

        # Use memory-efficient context
        with self.memory_efficient_context():
            return transfer_to_framework(
                self._data,
                'pyvista',
                scalars=scalars,
                **kwargs
            )

    @classmethod  
    def from_pyvista(cls, mesh: "Any", **kwargs) -> "AstroTensorBase":
        """
        Create tensor from PyVista mesh using zero-copy bridge.
        """
        from ..utils.tensor import NumpyZeroCopyBridge
        
        # Extract points
        if hasattr(mesh, 'points') and mesh.points is not None:
            points_tensor = NumpyZeroCopyBridge.from_numpy(mesh.points.copy())
        else:
            raise ValueError("Mesh has no points data")

        # Extract metadata from field data
        metadata = {}
        if hasattr(mesh, 'field_data'):
            for key, value in mesh.field_data.items():
                if isinstance(value, (int, float, str, bool)):
                    metadata[key] = value

        metadata.update(kwargs)
        return cls(points_tensor, **metadata)

    # =========================================================================
    # Blender integration with memory management
    # =========================================================================

    def to_blender(
        self,
        name: str = "astro_object",
        collection_name: str = "AstroLab",
    ) -> Optional[Any]:
        """
        Convert tensor to Blender object using zero-copy data bridge.
        """
        if not TENSOR_UTILS_AVAILABLE or not transfer_to_framework:
            raise ImportError("Tensor utils module not available for Blender conversion")

        # Use memory-efficient context
        with self.memory_efficient_context():
            obj = transfer_to_framework(
                self._data,
                'blender',
                name=name,
                collection_name=collection_name
            )
            
            # Add cleanup callback to remove object when tensor is destroyed
            if obj:
                def cleanup_blender_object():
                    try:
                        # Import bpy here to avoid dependency at module level
                        import bpy
                        if obj.name in bpy.data.objects:
                            bpy.data.objects.remove(obj, do_unlink=True)
                        if hasattr(obj, 'data') and obj.data.name in bpy.data.meshes:
                            bpy.data.meshes.remove(obj.data, do_unlink=True)
                    except Exception:
                        pass
                
                self.add_cleanup_callback(cleanup_blender_object)
            
            return obj

    # =========================================================================
    # Utility methods with memory safety
    # =========================================================================

    def has_uncertainties(self) -> bool:
        """Check if tensor has associated uncertainties."""
        return 'uncertainties' in self._metadata

    def apply_mask(self, mask: torch.Tensor) -> "AstroTensorBase":
        """Apply a boolean mask to the tensor with memory optimization."""
        # Detach mask to prevent autograd issues
        mask_detached = mask.detach() if mask.requires_grad else mask
        masked_data = self._data[mask_detached]
        return self.__class__(masked_data, **self._metadata)

    def __len__(self) -> int:
        """Length of tensor (first dimension)."""
        return self._data.shape[0]

    def __repr__(self) -> str:
        """String representation with memory info."""
        tensor_type = self._metadata.get("tensor_type", "base")
        memory_mb = (self._data.numel() * self._data.element_size()) / 1024**2
        return (f"{self.__class__.__name__}(shape={list(self.shape)}, "
                f"dtype={self.dtype}, type={tensor_type}, "
                f"memory={memory_mb:.2f}MB, cleaned={self._is_cleaned})")


# =========================================================================
# Global memory management utilities
# =========================================================================

def cleanup_all_tensors() -> None:
    """Clean up all registered tensor instances."""
    for tensor in list(_tensor_registry.values()):
        try:
            tensor.cleanup()
        except Exception as e:
            logger.warning(f"Failed to cleanup tensor: {e}")
    
    # Force garbage collection and GPU cache cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def memory_profiling_context(description: str = "Operation"):
    """Context manager for memory profiling with automatic reporting."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        
        yield
        
        peak_memory = torch.cuda.max_memory_allocated()
        final_memory = torch.cuda.memory_allocated()
        
        logger.info(f"Memory Profile [{description}]:")
        logger.info(f"  Initial: {initial_memory / 1024**2:.2f} MB")
        logger.info(f"  Peak: {peak_memory / 1024**2:.2f} MB")
        logger.info(f"  Final: {final_memory / 1024**2:.2f} MB")
        logger.info(f"  Increase: {(final_memory - initial_memory) / 1024**2:.2f} MB")
    else:
        yield


# Convenience functions for direct operations with memory management
def transfer_direct(source, target, attribute="position", device="cpu"):
    """
    Universal function for direct memory transfer with automatic cleanup.
    """
    if hasattr(source, 'transfer_memory_direct'):
        return source.transfer_memory_direct(target, attribute, device)
    else:
        raise ValueError(f"Source {type(source)} does not support direct transfer")


__all__ = [
    "AstroTensorBase",
    "transfer_direct",
    "cleanup_all_tensors",
    "memory_profiling_context",
]
