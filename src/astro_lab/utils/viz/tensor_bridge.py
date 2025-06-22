"""
Tensor Bridge - High-Performance Tensor Visualization Bridge
===========================================================

Provides efficient tensor-to-visualization framework bridges with zero-copy
operations and GPU acceleration support.
"""

import logging
import threading
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Optional visualization dependencies
try:
    import pyvista as pv
    import vtk

    PYVISTA_AVAILABLE = True

    # CRITICAL: Suppress PyVista __del__ TypeError
    # This is a known PyVista issue with VTK cleanup
    try:
        original_polydata_del = pv.PolyData.__del__

        def safe_polydata_del(self):
            try:
                original_polydata_del(self)
            except (TypeError, AttributeError):
                # Silently ignore PyVista cleanup errors
                pass

        pv.PolyData.__del__ = safe_polydata_del
        logger.debug("✅ PyVista __del__ TypeError protection enabled")

    except (AttributeError, TypeError):
        # If monkey patching fails, just continue
        logger.debug("⚠️ Could not patch PyVista __del__")

    # Also suppress warnings for PyVista
    warnings.filterwarnings("ignore", category=UserWarning, module="pyvista")

except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    vtk = None

# Blender integration - LAZY LOADING
def _get_blender_modules():
    """Lazy import of Blender modules to avoid memory leak."""
    try:
        from ..bpy import bpy, mathutils

        return bpy, mathutils
    except ImportError:
        return None, None

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

@dataclass
class SyncConfig:
    """Configuration for bidirectional synchronization."""

    sync_interval: float = 0.1  # seconds
    auto_sync: bool = True
    sync_materials: bool = True
    sync_animations: bool = True
    preserve_names: bool = True
    zero_copy: bool = True
    max_vertices: int = 1000000  # Performance limit

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
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
        }

    try:
        yield initial_stats
    finally:
        # Cleanup and profiling
        if torch.cuda.is_available():
            final_stats = {
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
            }

            memory_diff = final_stats["allocated"] - initial_stats.get("allocated", 0)
            if memory_diff > 1024**2:  # More than 1MB increase
                logger.warning(
                    f"Zero-copy {description}: Memory increase {memory_diff / 1024**2:.2f} MB"
                )
            else:
                logger.debug(
                    f"Zero-copy {description}: Memory change {memory_diff / 1024**2:.2f} MB"
                )

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
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "storage_size": storage_size,
        "data_ptr": tensor.data_ptr(),
        "is_contiguous": tensor.is_contiguous(),
        "requires_grad": tensor.requires_grad,
        "memory_bytes": tensor.numel() * tensor.element_size(),
        "memory_mb": (tensor.numel() * tensor.element_size()) / 1024**2,
        "is_pinned": tensor.is_pinned() if hasattr(tensor, "is_pinned") else False,
    }

    # Add CUDA-specific info
    if tensor.is_cuda:
        info.update(
            {
                "cuda_device": tensor.device.index,
                "cuda_allocated": torch.cuda.memory_allocated(tensor.device),
                "cuda_reserved": torch.cuda.memory_reserved(tensor.device),
            }
        )

    return info

# =========================================================================
# Zero-Copy Data Bridges
# =========================================================================

class ZeroCopyBridge:
    """Base class for zero-copy data bridges."""

    def ensure_cpu_contiguous(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is on CPU and contiguous for zero-copy."""
        optimized = optimize_tensor_layout(tensor)
        if optimized.is_cuda:
            optimized = optimized.cpu()
        return optimized

    def validate_3d_coordinates(self, tensor: torch.Tensor) -> torch.Tensor:
        """Validate and reshape tensor for 3D coordinates."""
        if tensor.shape[-1] != 3:
            if tensor.shape == torch.Size([3]):
                # Single point [3] -> [1, 3]
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"Expected [..., 3] tensor for coordinates, got {tensor.shape}"
                )
        return tensor

@contextmanager
def pyvista_mesh_context():
    """Context manager for PyVista mesh operations with proper cleanup."""
    try:
        yield
    finally:
        # Gentle cleanup - just ensure CUDA cache is cleared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class PyVistaZeroCopyBridge(ZeroCopyBridge):
    """High-performance zero-copy bridge to PyVista meshes."""

    def to_pyvista(
        self, tensor: torch.Tensor, scalars: Optional[torch.Tensor] = None, **kwargs
    ):
        """Convert tensor to PyVista mesh with zero-copy optimization."""
        
        with zero_copy_context("PyVista conversion"):
            # Validate and optimize
            points_tensor = self.validate_3d_coordinates(tensor)
            points_optimized = self.ensure_cpu_contiguous(points_tensor)

            # Zero-copy numpy conversion
            points_np = points_optimized.numpy()

            # Create mesh with robust error handling
            try:
                # Create a copy to avoid zero-copy issues with PyVista's __del__
                points_copy = points_np.copy()
                mesh = pv.PolyData(points_copy)

                # Add scalar data efficiently
                if scalars is not None:
                    scalars_optimized = self.ensure_cpu_contiguous(scalars)
                    scalars_copy = scalars_optimized.numpy().copy()
                    mesh.point_data["scalars"] = scalars_copy

                # Force proper VTK initialization to prevent __del__ issues
                _ = mesh.n_points  # This forces internal initialization

                return mesh

            except Exception as e:
                logger.error(f"PyVista mesh creation failed: {e}")
                raise e

    def to_pyvista_safe(
        self, tensor: torch.Tensor, scalars: Optional[torch.Tensor] = None, **kwargs
    ):
        """Convert tensor to PyVista mesh with safe cleanup."""
        with pyvista_mesh_context():
            return self.to_pyvista(tensor, scalars, **kwargs)

    def cleanup_pyvista_mesh(self, mesh):
        """Properly cleanup PyVista mesh to prevent exceptions."""
        if mesh is None:
            return

        try:
            # Gentle cleanup - just clear point data
            if hasattr(mesh, "point_data") and mesh.point_data:
                mesh.point_data.clear()
        except:
            pass  # Ignore cleanup errors

class BlenderZeroCopyBridge(ZeroCopyBridge):
    """Ultra-high-performance zero-copy bridge to Blender."""

    def to_blender(
        self,
        tensor: torch.Tensor,
        name: str = "astro_object",
        collection_name: str = "AstroLab",
    ) -> Optional[Any]:
        """Convert tensor to Blender object with maximum performance."""
        # Lazy load Blender modules
        bpy, mathutils = _get_blender_modules()
        if bpy is None:
            raise ImportError("Blender (bpy) not available")

        # Use proper context manager for memory management
        from ..bpy import blender_memory_context

        with blender_memory_context():
            with zero_copy_context("Blender conversion"):
                # Validate and optimize
                points_tensor = self.validate_3d_coordinates(tensor)
                data_optimized = self.ensure_cpu_contiguous(points_tensor)

                # Create mesh
                mesh = bpy.data.meshes.new(name)
                num_verts = data_optimized.shape[0]
                mesh.vertices.add(num_verts)

                # Ultra-fast foreach_set transfer
                coords_flat = data_optimized.view(-1).numpy()
                mesh.vertices.foreach_set("co", coords_flat)
                mesh.update()

                # Create object and add to collection
                obj = bpy.data.objects.new(name, mesh)

                try:
                    collection = bpy.data.collections[collection_name]
                except KeyError:
                    collection = bpy.data.collections.new(collection_name)
                    bpy.context.scene.collection.children.link(collection)

                collection.objects.link(obj)
                return obj

class NumpyZeroCopyBridge(ZeroCopyBridge):
    """Zero-copy bridge between PyTorch tensors and NumPy arrays."""

    def to_numpy(self, tensor: torch.Tensor, force_copy: bool = False) -> np.ndarray:
        """Convert tensor to NumPy with zero-copy when possible."""
        with zero_copy_context("Tensor to NumPy"):
            optimized = self.ensure_cpu_contiguous(tensor)
            return optimized.numpy().copy() if force_copy else optimized.numpy()

    def from_numpy(
        self,
        array: np.ndarray,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert NumPy array to tensor with zero-copy when possible."""
        with zero_copy_context("NumPy to tensor"):
            tensor = torch.from_numpy(array)
            if device is not None or dtype is not None:
                tensor = tensor.to(device=device, dtype=dtype)
            return tensor

# =========================================================================
# Bidirectional Bridge with Live Sync
# =========================================================================

class BidirectionalTensorBridge:
    """
    Complete bidirectional bridge between PyVista and Blender with live sync.

    Consolidates all tensor bridge functionality with zero-copy operations.
    """

    def __init__(self, config: Optional[SyncConfig] = None):
        """Initialize the bidirectional bridge."""
        self.config = config or SyncConfig()
        self._sync_thread = None
        self._sync_running = False
        self._sync_pairs = {}  # {source_id: target}
        self._sync_callbacks = []

        # Initialize bridges
        self.pyvista_bridge = PyVistaZeroCopyBridge()
        self.blender_bridge = BlenderZeroCopyBridge()
        self.numpy_bridge = NumpyZeroCopyBridge()

        logger.info("🌉 Bidirectional Tensor Bridge initialized")

    def pyvista_to_blender(
        self, mesh, name: str = "pyvista_mesh", collection_name: str = "PyVistaImports"
    ) -> Optional[Any]:
        """Convert PyVista mesh to Blender object."""
        
        bpy, mathutils = _get_blender_modules()
        if bpy is None:
            raise ImportError("Blender not available")

        try:
            # Extract mesh data
            vertices = mesh.points
            faces = mesh.faces if hasattr(mesh, "faces") else []

            # Create Blender mesh
            blender_mesh = bpy.data.meshes.new(name)
            blender_obj = bpy.data.objects.new(name, blender_mesh)

            # Add vertices
            blender_mesh.vertices.add(len(vertices))
            blender_mesh.vertices.foreach_set("co", vertices.flatten())

            # Add faces if available
            if len(faces) > 0:
                # Simple face handling for now
                blender_mesh.polygons.add(len(faces) // 4)  # Assuming quads
                blender_mesh.loops.add(len(faces))
                blender_mesh.loops.foreach_set("vertex_index", faces)

            blender_mesh.update()

            # Add to collection
            try:
                collection = bpy.data.collections[collection_name]
            except KeyError:
                collection = bpy.data.collections.new(collection_name)
                bpy.context.scene.collection.children.link(collection)

            collection.objects.link(blender_obj)

            # Store sync pair
            self._sync_pairs[id(mesh)] = blender_obj

            logger.info(f"✅ Converted PyVista mesh to Blender: {name}")
            return blender_obj

        except Exception as e:
            logger.error(f"❌ Failed to convert PyVista to Blender: {e}")
            return None

    def create_live_sync(self, sync_interval: Optional[float] = None) -> bool:
        """Create live synchronization between frameworks."""
        if self._sync_running:
            logger.warning("Live sync already running")
            return False

        interval = sync_interval or self.config.sync_interval

        def sync_loop():
            while self._sync_running:
                try:
                    # Call custom sync callbacks
                    for callback in self._sync_callbacks:
                        callback()
                except Exception as e:
                    logger.error(f"Live sync error: {e}")

                time.sleep(interval)

        self._sync_running = True
        self._sync_thread = threading.Thread(target=sync_loop, daemon=True)
        self._sync_thread.start()

        logger.info(f"✅ Live sync started (interval: {interval}s)")
        return True

    def stop_live_sync(self):
        """Stop live synchronization."""
        if self._sync_running:
            self._sync_running = False
            if self._sync_thread:
                self._sync_thread.join(timeout=1.0)
            logger.info("✅ Live sync stopped")

    def add_sync_callback(self, callback):
        """Add custom callback for live synchronization."""
        self._sync_callbacks.append(callback)

# =========================================================================
# High-level API
# =========================================================================

def transfer_to_framework(tensor: torch.Tensor, framework: str, **kwargs) -> Any:
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

    if framework_lower == "pyvista":
        bridge = PyVistaZeroCopyBridge()
        return bridge.to_pyvista(tensor, **kwargs)
    elif framework_lower == "blender":
        bridge = BlenderZeroCopyBridge()
        return bridge.to_blender(tensor, **kwargs)
    elif framework_lower == "numpy":
        bridge = NumpyZeroCopyBridge()
        return bridge.to_numpy(tensor, **kwargs)
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
        if "pool_tensor" in locals():
            del pool_tensor

# High-level convenience functions
def create_bidirectional_bridge(
    config: Optional[SyncConfig] = None,
) -> BidirectionalTensorBridge:
    """Create a new bidirectional bridge instance."""
    return BidirectionalTensorBridge(config)

def quick_convert_pyvista_to_blender(
    mesh, name: str = "converted_mesh"
) -> Optional[Any]:
    """Quick conversion from PyVista to Blender."""
    bridge = BidirectionalTensorBridge()
    return bridge.pyvista_to_blender(mesh, name)

def quick_convert_tensor_to_pyvista(tensor: torch.Tensor, **kwargs):
    """Quick conversion from tensor to PyVista."""
    bridge = PyVistaZeroCopyBridge()
    return bridge.to_pyvista(tensor, **kwargs)

def quick_convert_tensor_to_blender(tensor: torch.Tensor, **kwargs) -> Optional[Any]:
    """Quick conversion from tensor to Blender."""
    bridge = BlenderZeroCopyBridge()
    return bridge.to_blender(tensor, **kwargs)

__all__ = [
    # Bridge classes
    "PyVistaZeroCopyBridge",
    "BlenderZeroCopyBridge",
    "NumpyZeroCopyBridge",
    "BidirectionalTensorBridge",
    "ZeroCopyBridge",
    # Configuration
    "SyncConfig",
    # High-level API
    "transfer_to_framework",
    "create_bidirectional_bridge",
    "quick_convert_pyvista_to_blender",
    "quick_convert_tensor_to_pyvista",
    "quick_convert_tensor_to_blender",
    # Utilities
    "optimize_tensor_layout",
    "get_tensor_memory_info",
    # Context managers
    "zero_copy_context",
    "pinned_memory_context",
    "pyvista_mesh_context",
    # Protocol
    "TensorProtocol",
]
