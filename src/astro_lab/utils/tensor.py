"""
Tensor Data Bridges and Zero-Copy Utilities
==========================================

High-performance zero-copy data exchange between PyTorch tensors and
visualization frameworks (PyVista, Blender, NumPy) following 2025 best practices.

This module provides:
- Zero-copy memory mapping where possible
- Automatic memory layout optimization
- Device-aware data transfer (CPU/CUDA)
- Memory profiling and leak detection
- Context managers for safe operations
"""

import ctypes
import gc
import logging
import warnings
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union, Protocol

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Optional visualization dependencies
try:
    import pyvista as pv
    import vtk
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    vtk = None

# Use centralized Blender lazy loading
from .blender_lazy import get_blender_modules, is_blender_available


class TensorProtocol(Protocol):
    """Protocol for tensor-like objects."""
    
    def data_ptr(self) -> int: ...
    def is_contiguous(self) -> bool: ...
    def contiguous(self) -> torch.Tensor: ...
    def cpu(self) -> torch.Tensor: ...
    def detach(self) -> torch.Tensor: ...
    def numpy(self) -> np.ndarray: ...
    
    @property
    def shape(self) -> torch.Size: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    @property
    def is_cuda(self) -> bool: ...


# =========================================================================
# Memory profiling and optimization utilities
# =========================================================================

@contextmanager
def zero_copy_context(description: str = "Zero-copy operation"):
    """
    Context manager for zero-copy operations with memory profiling.
    
    Args:
        description: Description for profiling logs
        
    Yields:
        dict: Memory statistics during operation
    """
    initial_stats = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        initial_stats = {
            'allocated': torch.cuda.memory_allocated(),
            'reserved': torch.cuda.memory_reserved(),
        }
    
    try:
        yield initial_stats
    finally:
        # Cleanup and profiling
        if torch.cuda.is_available():
            final_stats = {
                'allocated': torch.cuda.memory_allocated(),
                'reserved': torch.cuda.memory_reserved(),
                'peak': torch.cuda.max_memory_allocated(),
            }
            
            memory_diff = final_stats['allocated'] - initial_stats.get('allocated', 0)
            if memory_diff > 1024**2:  # More than 1MB increase
                logger.warning(f"Zero-copy {description}: Memory increase {memory_diff / 1024**2:.2f} MB")
            else:
                logger.debug(f"Zero-copy {description}: Memory change {memory_diff / 1024**2:.2f} MB")


def optimize_tensor_layout(tensor: torch.Tensor) -> torch.Tensor:
    """
    Optimize tensor memory layout for zero-copy operations.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Optimized tensor (contiguous and detached)
    """
    # Always detach to prevent autograd issues
    optimized = tensor.detach()
    
    # Ensure contiguous layout for zero-copy
    if not optimized.is_contiguous():
        optimized = optimized.contiguous()
    
    return optimized


def get_tensor_memory_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get comprehensive memory information for a tensor.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Dictionary with memory statistics
    """
    try:
        storage_size = tensor.untyped_storage().size()
    except AttributeError:
        # Fallback for older PyTorch versions
        storage_size = tensor.storage().size()
    
    info = {
        'device': str(tensor.device),
        'dtype': str(tensor.dtype),
        'shape': list(tensor.shape),
        'numel': tensor.numel(),
        'element_size': tensor.element_size(),
        'storage_size': storage_size,
        'data_ptr': tensor.data_ptr(),
        'is_contiguous': tensor.is_contiguous(),
        'requires_grad': tensor.requires_grad,
        'memory_bytes': tensor.numel() * tensor.element_size(),
        'memory_mb': (tensor.numel() * tensor.element_size()) / 1024**2,
        'is_pinned': tensor.is_pinned() if hasattr(tensor, 'is_pinned') else False,
    }
    
    # Add CUDA-specific info
    if tensor.is_cuda:
        info.update({
            'cuda_device': tensor.device.index,
            'cuda_allocated': torch.cuda.memory_allocated(tensor.device),
            'cuda_reserved': torch.cuda.memory_reserved(tensor.device),
        })
    
    return info


# =========================================================================
# Zero-Copy Data Bridges
# =========================================================================

class ZeroCopyBridge:
    """Base class for zero-copy data bridges."""
    
    @staticmethod
    def ensure_cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on CPU and contiguous for zero-copy."""
        optimized = optimize_tensor_layout(tensor)
        if optimized.is_cuda:
            optimized = optimized.cpu()
        return optimized
    
    @staticmethod
    def validate_3d_coordinates(tensor: torch.Tensor) -> torch.Tensor:
        """Validate and reshape tensor for 3D coordinates."""
        if tensor.shape[-1] != 3:
            if tensor.shape == torch.Size([3]):
                # Single point [3] -> [1, 3]
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(f"Expected [..., 3] tensor for coordinates, got {tensor.shape}")
        return tensor


class PyVistaZeroCopyBridge(ZeroCopyBridge):
    """High-performance zero-copy bridge to PyVista meshes."""
    
    @staticmethod
    def to_pyvista(
        tensor: torch.Tensor,
        scalars: Optional[torch.Tensor] = None,
        **kwargs
    ) -> "pv.PolyData":
        """Convert tensor to PyVista mesh with zero-copy optimization."""
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista not available")
        
        with zero_copy_context("PyVista conversion"):
            # Validate and optimize
            points_tensor = PyVistaZeroCopyBridge.validate_3d_coordinates(tensor)
            points_optimized = PyVistaZeroCopyBridge.ensure_cpu_contiguous(points_tensor)
            
            # Zero-copy numpy conversion
            points_np = points_optimized.numpy()
            mesh = pv.PolyData(points_np)
            
            # Add scalar data efficiently
            if scalars is not None:
                scalars_optimized = PyVistaZeroCopyBridge.ensure_cpu_contiguous(scalars)
                mesh.point_data['scalars'] = scalars_optimized.numpy()
            
            return mesh


class BlenderZeroCopyBridge(ZeroCopyBridge):
    """Ultra-high-performance zero-copy bridge to Blender."""
    
    @staticmethod
    def to_blender(
        tensor: torch.Tensor,
        name: str = "astro_object",
        collection_name: str = "AstroLab",
    ) -> Optional[Any]:
        """Convert tensor to Blender object with maximum performance."""
        bpy_module, mathutils_module = get_blender_modules()
        if bpy_module is None:
            raise ImportError("Blender (bpy) not available")
        
        with zero_copy_context("Blender conversion"):
            # Validate and optimize
            points_tensor = BlenderZeroCopyBridge.validate_3d_coordinates(tensor)
            data_optimized = BlenderZeroCopyBridge.ensure_cpu_contiguous(points_tensor)
            
            # Create mesh
            mesh = bpy_module.data.meshes.new(name)
            num_verts = data_optimized.shape[0]
            mesh.vertices.add(num_verts)
            
            # Ultra-fast foreach_set transfer
            coords_flat = data_optimized.view(-1).numpy()
            mesh.vertices.foreach_set("co", coords_flat)
            mesh.update()
            
            # Create object and add to collection
            obj = bpy_module.data.objects.new(name, mesh)
            
            try:
                collection = bpy_module.data.collections[collection_name]
            except KeyError:
                collection = bpy_module.data.collections.new(collection_name)
                bpy_module.context.scene.collection.children.link(collection)
            
            collection.objects.link(obj)
            return obj


class NumpyZeroCopyBridge(ZeroCopyBridge):
    """Zero-copy bridge between PyTorch tensors and NumPy arrays."""
    
    @staticmethod
    def to_numpy(tensor: torch.Tensor, force_copy: bool = False) -> np.ndarray:
        """Convert tensor to NumPy with zero-copy when possible."""
        with zero_copy_context("Tensor to NumPy"):
            optimized = NumpyZeroCopyBridge.ensure_cpu_contiguous(tensor)
            return optimized.numpy().copy() if force_copy else optimized.numpy()
    
    @staticmethod
    def from_numpy(
        array: np.ndarray,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Convert NumPy array to tensor with zero-copy when possible."""
        with zero_copy_context("NumPy to tensor"):
            tensor = torch.from_numpy(array)
            if device is not None or dtype is not None:
                tensor = tensor.to(device=device, dtype=dtype)
            return tensor


# =========================================================================
# High-level API
# =========================================================================

def transfer_to_framework(
    tensor: torch.Tensor,
    framework: str,
    **kwargs
) -> Any:
    """
    Transfer tensor data to visualization framework with zero-copy.
    
    Args:
        tensor: Source tensor
        framework: Target framework ('pyvista', 'blender', 'numpy')
        **kwargs: Framework-specific arguments
        
    Returns:
        Converted object in target framework
    """
    framework_lower = framework.lower()
    
    if framework_lower == 'pyvista':
        return PyVistaZeroCopyBridge.to_pyvista(tensor, **kwargs)
    elif framework_lower == 'blender':
        return BlenderZeroCopyBridge.to_blender(tensor, **kwargs)
    elif framework_lower == 'numpy':
        return NumpyZeroCopyBridge.to_numpy(tensor, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


@contextmanager
def pinned_memory_context(size_mb: int = 100):
    """Context manager for pinned memory optimization."""
    if not torch.cuda.is_available():
        yield
        return
    
    try:
        # Pre-allocate pinned memory for efficient CPU-GPU transfers
        pool_size = size_mb * 1024 * 1024 // 4
        pool_tensor = torch.empty(pool_size, dtype=torch.float32).pin_memory()
        yield
    finally:
        if 'pool_tensor' in locals():
            del pool_tensor
        gc.collect()


__all__ = [
    # Bridge classes
    'PyVistaZeroCopyBridge',
    'BlenderZeroCopyBridge',
    'NumpyZeroCopyBridge',
    'ZeroCopyBridge',
    
    # High-level API
    'transfer_to_framework',
    
    # Utilities
    'optimize_tensor_layout',
    'get_tensor_memory_info',
    
    # Context managers
    'zero_copy_context',
    'pinned_memory_context',
    
    # Protocol
    'TensorProtocol',
] 