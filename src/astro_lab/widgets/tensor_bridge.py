"""
AstroLab Tensor Bridge - Zero-Copy Data Transfer
===============================================

Provides zero-copy data transfer between PyTorch tensors and various
visualization frameworks (PyVista, Blender, NumPy).
"""

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import numpy as np
import torch

from .albpy import blender_memory_context, bpy_object_context, mathutils
from .albpy import bpy as albpy

logger = logging.getLogger(__name__)

# =========================================================================
# MEMORY MANAGEMENT
# =========================================================================


class PolyDataManager:
    """Manages PyVista PolyData objects to prevent memory leaks."""

    def __init__(self):
        self.active_meshes = set()

    def register_mesh(self, mesh):
        """Register a mesh for cleanup."""
        self.active_meshes.add(mesh)

    def cleanup_all(self):
        """Clean up all registered meshes."""
        for mesh in self.active_meshes:
            self.safe_polydata_del(mesh)
        self.active_meshes.clear()

    def safe_polydata_del(self, mesh):
        """Safely delete PyVista PolyData object."""
        try:
            if hasattr(mesh, "clear_data"):
                mesh.clear_data()
            if hasattr(mesh, "clear_points"):
                mesh.clear_points()
            if hasattr(mesh, "clear_cells"):
                mesh.clear_cells()
            del mesh
        except Exception as e:
            logger.warning(f"Error cleaning up PolyData: {e}")


# Global mesh manager
_mesh_manager = PolyDataManager()


def _get_blender_modules():
    """Get Blender modules."""
    return albpy, mathutils


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
    info = {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "storage_size": tensor.untyped_storage().size()
        if hasattr(tensor, "untyped_storage")
        else tensor.storage().size(),
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

    def validate_3d_coordinates(self, tensor: Any) -> torch.Tensor:
        """Validate and ensure tensor has correct 3D coordinate format."""
        # Check if it's a SurveyTensorDict object
        if hasattr(tensor, "survey_name") and hasattr(tensor, "spatial"):
            logger.info(
                f"✅ SurveyTensorDict detected: {tensor.survey_name}. Extracting spatial coordinates."
            )
            try:
                if "coordinates" in tensor["spatial"]:
                    coords = tensor["spatial"]["coordinates"]
                    logger.info(
                        f"✅ Extracted 3D coordinates from SurveyTensorDict: {coords.shape}"
                    )
                    tensor = coords
                else:
                    raise ValueError("SurveyTensorDict spatial data has no coordinates")
            except Exception as e:
                logger.error(
                    f"❌ Failed to extract spatial coordinates from SurveyTensorDict: {e}"
                )
                raise ValueError(
                    f"Failed to extract spatial coordinates from SurveyTensorDict: {e}"
                )
        elif hasattr(tensor, "survey_name") and hasattr(tensor, "get_spatial_tensor"):
            # Fallback for old API
            logger.info(
                f"✅ Legacy SurveyTensor detected: {tensor.survey_name}. Extracting spatial coordinates."
            )
            try:
                spatial_tensor = tensor.get_spatial_tensor()
                if hasattr(spatial_tensor, "cartesian"):
                    coords = spatial_tensor.cartesian
                    logger.info(
                        f"✅ Extracted 3D coordinates from legacy SurveyTensor: {coords.shape}"
                    )
                    tensor = coords
                else:
                    raise ValueError(
                        "Legacy SurveyTensor spatial_tensor has no cartesian attribute"
                    )
            except Exception as e:
                logger.error(
                    f"❌ Failed to extract spatial coordinates from legacy SurveyTensor: {e}"
                )
                raise ValueError(
                    f"Failed to extract spatial coordinates from legacy SurveyTensor: {e}"
                )

        # Ensure tensor is 2D with shape [N, 3]
        if tensor.dim() == 1:
            if tensor.shape[0] == 3:
                # Single point [3] -> [1, 3]
                tensor = tensor.unsqueeze(0)
            else:
                raise ValueError(
                    f"1D tensor must have 3 elements, got {tensor.shape[0]}"
                )
        elif tensor.dim() == 2:
            if tensor.shape[-1] != 3:
                raise ValueError(
                    f"Expected [..., 3] tensor for coordinates, got {tensor.shape}"
                )
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {tensor.dim()}D tensor")

        return tensor


@contextmanager
def pyvista_mesh_context():
    """Context manager for PyVista mesh operations with cleanup."""
    try:
        yield
    finally:
        _mesh_manager.cleanup_all()


class PyVistaZeroCopyBridge(ZeroCopyBridge):
    """Zero-copy bridge for PyVista."""

    def to_pyvista(
        self, tensor: torch.Tensor, scalars: Optional[torch.Tensor] = None, **kwargs
    ):
        """
        Convert tensor to PyVista PolyData with zero-copy optimization.

        Args:
            tensor: 3D coordinates tensor [N, 3]
            scalars: Optional scalar values for coloring
            **kwargs: Additional PyVista parameters

        Returns:
            PyVista PolyData object
        """
        import pyvista as pv

        with zero_copy_context("PyVista conversion"):
            # Optimize tensor layout
            coords = self.ensure_cpu_contiguous(tensor)
            coords = self.validate_3d_coordinates(coords)

            # Convert to numpy with zero-copy
            coords_np = coords.numpy()

            # Create PyVista points
            points = pv.PolyData(coords_np)

            # Add scalars if provided
            if scalars is not None:
                scalars_opt = self.ensure_cpu_contiguous(scalars)
                points.point_data["scalars"] = scalars_opt.numpy()

            # Register for cleanup
            _mesh_manager.register_mesh(points)

            return points

    def to_pyvista_safe(
        self, tensor: torch.Tensor, scalars: Optional[torch.Tensor] = None, **kwargs
    ):
        """Safe version with automatic cleanup."""
        with pyvista_mesh_context():
            return self.to_pyvista(tensor, scalars, **kwargs)

    def cleanup_pyvista_mesh(self, mesh):
        """Clean up PyVista mesh."""
        _mesh_manager.safe_polydata_del(mesh)
        
    def cosmic_web_to_pyvista(
        self,
        spatial_tensor: Any,
        cluster_labels: Optional[np.ndarray] = None,
        density_field: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Convert cosmic web data to PyVista mesh with clustering visualization.
        
        Args:
            spatial_tensor: SpatialTensorDict with coordinates
            cluster_labels: Cluster assignments from cosmic web analysis
            density_field: Local density values
            **kwargs: Additional parameters
            
        Returns:
            PyVista PolyData with cosmic web visualization
        """
        import pyvista as pv
        
        with zero_copy_context("Cosmic Web PyVista conversion"):
            # Extract coordinates
            if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
                coords = spatial_tensor["coordinates"]
            else:
                coords = spatial_tensor
                
            coords = self.ensure_cpu_contiguous(coords)
            coords = self.validate_3d_coordinates(coords)
            coords_np = coords.numpy()
            
            # Create point cloud
            mesh = pv.PolyData(coords_np)
            
            # Add cluster labels if provided
            if cluster_labels is not None:
                mesh.point_data["cluster"] = cluster_labels
                
            # Add density field if provided
            if density_field is not None:
                density_opt = self.ensure_cpu_contiguous(density_field)
                mesh.point_data["density"] = density_opt.numpy()
                
            # Add point indices
            mesh.point_data["index"] = np.arange(len(coords_np))
            
            # Register for cleanup
            _mesh_manager.register_mesh(mesh)
            
            return mesh


class BlenderZeroCopyBridge(ZeroCopyBridge):
    """Zero-copy bridge for Blender."""

    def to_blender(
        self,
        tensor: torch.Tensor,
        name: str = "astro_object",
        collection_name: str = "AstroLab",
    ) -> Optional[Any]:
        """
        Convert tensor to Blender mesh with zero-copy optimization.

        Args:
            tensor: 3D coordinates tensor [N, 3]
            name: Name for the Blender object
            collection_name: Name for the collection

        Returns:
            Blender mesh object or None if Blender not available
        """
        if albpy is None:
            logger.warning("Blender (albpy) not available")
            return None

        with zero_copy_context("Blender conversion"):
            with blender_memory_context():
                # Optimize tensor layout
                coords = self.ensure_cpu_contiguous(tensor)
                coords = self.validate_3d_coordinates(coords)

                # Convert to numpy with zero-copy
                coords_np = coords.numpy()

                # Create Blender mesh
                mesh = albpy.data.meshes.new(name)
                obj = albpy.data.objects.new(name, mesh)

                # Add to collection
                collection = albpy.data.collections.get(collection_name)
                if collection is None:
                    collection = albpy.data.collections.new(collection_name)
                    albpy.context.scene.collection.children.link(collection)

                collection.objects.link(obj)

                # Set mesh data
                mesh.from_pydata(coords_np.tolist(), [], [])
                mesh.update()

                return obj
                
    def cosmic_web_to_blender(
        self,
        spatial_tensor: Any,
        cluster_labels: Optional[np.ndarray] = None,
        name: str = "cosmic_web",
        collection_name: str = "CosmicWeb",
        point_size: float = 0.1,
        **kwargs,
    ) -> Optional[Any]:
        """
        Convert cosmic web data to Blender with cluster visualization.
        
        Args:
            spatial_tensor: SpatialTensorDict with coordinates
            cluster_labels: Cluster assignments
            name: Base name for objects
            collection_name: Collection name
            point_size: Size of points
            
        Returns:
            List of Blender objects or None
        """
        if albpy is None:
            logger.warning("Blender (albpy) not available")
            return None
            
        with zero_copy_context("Cosmic Web Blender conversion"):
            with blender_memory_context():
                # Extract coordinates
                if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
                    coords = spatial_tensor["coordinates"]
                else:
                    coords = spatial_tensor
                    
                coords = self.ensure_cpu_contiguous(coords)
                coords = self.validate_3d_coordinates(coords)
                coords_np = coords.numpy()
                
                # Create collection
                collection = albpy.data.collections.get(collection_name)
                if collection is None:
                    collection = albpy.data.collections.new(collection_name)
                    albpy.context.scene.collection.children.link(collection)
                
                objects = []
                
                if cluster_labels is not None:
                    # Create separate objects for each cluster
                    unique_labels = np.unique(cluster_labels)
                    
                    # Create materials for clusters
                    import colorsys
                    n_clusters = len(unique_labels[unique_labels >= 0])
                    
                    for i, label in enumerate(unique_labels):
                        mask = cluster_labels == label
                        cluster_coords = coords_np[mask]
                        
                        if len(cluster_coords) == 0:
                            continue
                            
                        # Create mesh for cluster
                        cluster_name = f"{name}_cluster_{label}" if label >= 0 else f"{name}_noise"
                        mesh = albpy.data.meshes.new(cluster_name)
                        obj = albpy.data.objects.new(cluster_name, mesh)
                        
                        # Set mesh data as vertices
                        mesh.from_pydata(cluster_coords.tolist(), [], [])
                        mesh.update()
                        
                        # Create material
                        mat = albpy.data.materials.new(name=f"mat_{cluster_name}")
                        mat.use_nodes = True
                        
                        if label == -1:
                            # Gray for noise
                            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (0.5, 0.5, 0.5, 1.0)
                        else:
                            # Color from HSV
                            hue = i / max(n_clusters, 1)
                            rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                            mat.node_tree.nodes["Principled BSDF"].inputs[0].default_value = (*rgb, 1.0)
                            
                        obj.data.materials.append(mat)
                        
                        # Add to collection
                        collection.objects.link(obj)
                        objects.append(obj)
                else:
                    # Single object for all points
                    mesh = albpy.data.meshes.new(name)
                    obj = albpy.data.objects.new(name, mesh)
                    mesh.from_pydata(coords_np.tolist(), [], [])
                    mesh.update()
                    collection.objects.link(obj)
                    objects.append(obj)
                    
                return objects


class NumpyZeroCopyBridge(ZeroCopyBridge):
    """Zero-copy bridge for NumPy."""

    def to_numpy(self, tensor: torch.Tensor, force_copy: bool = False) -> np.ndarray:
        """Convert tensor to numpy array with zero-copy when possible."""
        if force_copy or not tensor.is_contiguous():
            return tensor.detach().cpu().numpy()
        return tensor.detach().cpu().numpy()

    def from_numpy(
        self,
        array: np.ndarray,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Convert numpy array to tensor with zero-copy when possible."""
        if device is None:
            device = torch.device("cuda")
        if dtype is None:
            dtype = torch.float32

        return torch.from_numpy(array).to(device=device, dtype=dtype)


class BidirectionalTensorBridge:
    """Bidirectional bridge between different frameworks."""

    def __init__(self, config: Optional[SyncConfig] = None):
        """Initialize bridge with configuration."""
        self.config = config or SyncConfig()
        self.pyvista_bridge = PyVistaZeroCopyBridge()
        self.blender_bridge = BlenderZeroCopyBridge()
        self.numpy_bridge = NumpyZeroCopyBridge()
        self.sync_callbacks = []
        self.sync_active = False

    def pyvista_to_blender(
        self, mesh, name: str = "pyvista_mesh", collection_name: str = "PyVistaImports"
    ) -> Optional[Any]:
        """
        Convert PyVista mesh to Blender object.

        Args:
            mesh: PyVista PolyData object
            name: Name for Blender object
            collection_name: Name for Blender collection

        Returns:
            Blender object or None
        """
        if albpy is None:
            logger.warning("Blender (albpy) not available")
            return None

        try:
            # Extract points from PyVista mesh
            points = mesh.points
            if points is None:
                logger.warning("PyVista mesh has no points")
                return None

            # Convert to tensor
            tensor = torch.tensor(points, dtype=torch.float32)

            # Convert to Blender
            return self.blender_bridge.to_blender(tensor, name, collection_name)

        except Exception as e:
            logger.error(f"Error converting PyVista to Blender: {e}")
            return None

    def create_live_sync(self, sync_interval: Optional[float] = None) -> bool:
        """Create live synchronization between frameworks."""
        if sync_interval is None:
            sync_interval = self.config.sync_interval

        if self.sync_active:
            logger.warning("Live sync already active")
            return False

        self.sync_active = True

        def sync_loop():
            while self.sync_active:
                try:
                    for callback in self.sync_callbacks:
                        callback()
                    time.sleep(sync_interval)
                except Exception as e:
                    logger.error(f"Sync loop error: {e}")
                    break

        import threading

        sync_thread = threading.Thread(target=sync_loop, daemon=True)
        sync_thread.start()

        return True

    def stop_live_sync(self):
        """Stop live synchronization."""
        self.sync_active = False

    def add_sync_callback(self, callback):
        """Add callback for synchronization."""
        self.sync_callbacks.append(callback)


def transfer_to_framework(tensor: torch.Tensor, framework: str, **kwargs) -> Any:
    """
    Transfer tensor to specified framework with zero-copy optimization.

    Args:
        tensor: Input tensor
        framework: Target framework ('pyvista', 'blender', 'numpy')
        **kwargs: Framework-specific parameters

    Returns:
        Framework-specific object
    """
    if framework == "pyvista":
        bridge = PyVistaZeroCopyBridge()
        return bridge.to_pyvista(tensor, **kwargs)
    elif framework == "blender":
        bridge = BlenderZeroCopyBridge()
        return bridge.to_blender(tensor, **kwargs)
    elif framework == "numpy":
        bridge = NumpyZeroCopyBridge()
        return bridge.to_numpy(tensor, **kwargs)
    else:
        raise ValueError(f"Unsupported framework: {framework}")


@contextmanager
def pinned_memory_context(size_mb: int = 100):
    """
    Context manager for pinned memory operations.

    Args:
        size_mb: Size of pinned memory in MB

    Yields:
        Pinned memory context
    """
    if not torch.cuda.is_available():
        yield
        return

    try:
        # Allocate pinned memory
        pinned_tensor = torch.empty(
            size_mb * 1024 * 1024 // 4, dtype=torch.float32, pin_memory=True
        )
        yield pinned_tensor
    finally:
        # Cleanup
        if "pinned_tensor" in locals():
            del pinned_tensor
        torch.cuda.empty_cache()


def create_bidirectional_bridge(
    config: Optional[SyncConfig] = None,
) -> BidirectionalTensorBridge:
    """Create a bidirectional bridge instance."""
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


def visualize_cosmic_web(
    spatial_tensor: Any,
    cluster_labels: Optional[np.ndarray] = None,
    density_field: Optional[torch.Tensor] = None,
    backend: str = "plotly",
    **kwargs,
) -> Any:
    """
    Visualize cosmic web structure with specified backend.
    
    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        cluster_labels: Cluster assignments from cosmic web analysis
        density_field: Local density values
        backend: Visualization backend ('plotly', 'pyvista', 'blender', 'open3d')
        **kwargs: Backend-specific parameters
        
    Returns:
        Visualization object (type depends on backend)
    """
    if backend == "plotly":
        # Use plotly_bridge functions
        from .plotly_bridge import plot_cosmic_web_3d
        return plot_cosmic_web_3d(spatial_tensor, cluster_labels, **kwargs)
        
    elif backend == "pyvista":
        # Use PyVista bridge
        bridge = PyVistaZeroCopyBridge()
        mesh = bridge.cosmic_web_to_pyvista(
            spatial_tensor, cluster_labels, density_field, **kwargs
        )
        
        # Optional: auto-render if requested
        if kwargs.get("show", False):
            import pyvista as pv
            plotter = pv.Plotter()
            
            if cluster_labels is not None:
                plotter.add_mesh(
                    mesh,
                    scalars="cluster",
                    point_size=kwargs.get("point_size", 5),
                    cmap=kwargs.get("cmap", "tab20"),
                )
            elif density_field is not None:
                plotter.add_mesh(
                    mesh,
                    scalars="density",
                    point_size=kwargs.get("point_size", 5),
                    cmap=kwargs.get("cmap", "hot"),
                )
            else:
                plotter.add_mesh(
                    mesh,
                    point_size=kwargs.get("point_size", 5),
                    color=kwargs.get("color", "white"),
                )
                
            plotter.show()
            
        return mesh
        
    elif backend == "blender":
        # Use Blender bridge
        bridge = BlenderZeroCopyBridge()
        return bridge.cosmic_web_to_blender(spatial_tensor, cluster_labels, **kwargs)
        
    elif backend == "open3d":
        # Use Open3D (if available)
        try:
            import open3d as o3d
            
            # Extract coordinates
            if hasattr(spatial_tensor, "__getitem__") and "coordinates" in spatial_tensor:
                coords = spatial_tensor["coordinates"]
            else:
                coords = spatial_tensor
                
            coords = coords.cpu().numpy()
            
            # Create point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(coords)
            
            # Color by clusters if available
            if cluster_labels is not None:
                # Create color map
                unique_labels = np.unique(cluster_labels)
                n_clusters = len(unique_labels[unique_labels >= 0])
                
                colors = np.zeros((len(cluster_labels), 3))
                for i, label in enumerate(unique_labels):
                    mask = cluster_labels == label
                    if label == -1:
                        colors[mask] = [0.5, 0.5, 0.5]  # Gray for noise
                    else:
                        # Generate color
                        hue = i / max(n_clusters, 1)
                        import colorsys
                        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                        colors[mask] = rgb
                        
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                # Default coloring
                pcd.paint_uniform_color([0.7, 0.7, 0.7])
                
            # Show if requested
            if kwargs.get("show", False):
                o3d.visualization.draw_geometries(
                    [pcd],
                    window_name=kwargs.get("window_name", "Cosmic Web - Open3D"),
                    width=kwargs.get("width", 800),
                    height=kwargs.get("height", 600),
                    point_show_normal=False,
                )
                
            return pcd
            
        except ImportError:
            raise ImportError("Open3D not available. Install with: pip install open3d")
            
    else:
        raise ValueError(f"Unsupported backend: {backend}. Choose from: plotly, pyvista, blender, open3d")
