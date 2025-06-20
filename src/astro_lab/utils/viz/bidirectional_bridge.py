"""
Bidirectional PyVista-Blender Bridge
====================================

Provides real-time bidirectional synchronization between PyVista and Blender
for astronomical data visualization.
"""

import logging
import numpy as np
import torch
import pyvista as pv
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

from ..blender import bpy, mathutils

import time
import threading
from dataclasses import dataclass
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import vtk
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    vtk = None

# Blender integration
try:
    from ..blender import bpy, mathutils
except ImportError:
    bpy = None
    mathutils = None


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


class BidirectionalPyVistaBlenderBridge:
    """
    Complete bidirectional bridge between PyVista and Blender.
    
    Provides seamless data exchange and live synchronization between
    the two visualization frameworks.
    """
    
    def __init__(self, config: Optional[SyncConfig] = None):
        """
        Initialize the bidirectional bridge.
        
        Args:
            config: Synchronization configuration
        """
        self.config = config or SyncConfig()
        self._sync_thread = None
        self._sync_running = False
        self._sync_pairs = {}  # {pyvista_id: blender_id}
        self._sync_callbacks = []
        
        # Validate dependencies
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista not available")
        if bpy is None:
            raise ImportError("Blender not available")
            
        logger.info("🌉 Bidirectional PyVista-Blender Bridge initialized")
    
    def pyvista_to_blender(
        self, 
        mesh: "pv.PolyData", 
        name: str = "pyvista_mesh",
        collection_name: str = "PyVistaImports"
    ) -> Optional[Any]:
        """
        Convert PyVista mesh to Blender object.
        
        Args:
            mesh: PyVista PolyData mesh
            name: Name for the Blender object
            collection_name: Collection to add the object to
            
        Returns:
            Blender object or None if conversion failed
        """
        try:
            # Extract mesh data
            vertices = mesh.points
            faces = mesh.faces
            
            # Create Blender mesh
            blender_mesh = bpy.data.meshes.new(name)
            blender_obj = bpy.data.objects.new(name, blender_mesh)
            
            # Add vertices
            blender_mesh.vertices.add(len(vertices))
            blender_mesh.vertices.foreach_set("co", vertices.flatten())
            
            # Add faces (robust conversion)
            if len(faces) > 0:
                # PyVista stores faces as a flat array: [n, v1, v2, v3, n, v1, v2, v3, ...]
                # We need to reshape and extract only triangles/quads
                face_sizes = []
                i = 0
                while i < len(faces):
                    n = faces[i]
                    face_sizes.append(n)
                    i += n + 1
                num_faces = len(face_sizes)
                blender_mesh.loops.add(sum(face_sizes))
                blender_mesh.polygons.add(num_faces)
                # Build the loop indices and polygon sizes
                loop_starts = []
                loop_total = 0
                for n in face_sizes:
                    loop_starts.append(loop_total)
                    loop_total += n
                # Set polygons
                blender_mesh.polygons.foreach_set("loop_start", loop_starts)
                blender_mesh.polygons.foreach_set("loop_total", face_sizes)
                # Set loop vertex indices
                loop_vertex_indices = []
                i = 0
                while i < len(faces):
                    n = faces[i]
                    loop_vertex_indices.extend(faces[i+1:i+1+n])
                    i += n + 1
                blender_mesh.loops.foreach_set("vertex_index", loop_vertex_indices)
            
            # Add scalar data as vertex colors
            if mesh.point_data:
                for key, data in mesh.point_data.items():
                    if len(data) == len(vertices):
                        # Create vertex color layer
                        color_layer = blender_mesh.vertex_colors.new(name=key)
                        
                        # Normalize data to 0-1 range
                        if data.dtype.kind in 'fc':  # float or complex
                            data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
                        else:
                            data_normalized = data.astype(float) / 255.0
                        
                        # Set colors (RGBA)
                        colors = np.column_stack([data_normalized] * 3 + [np.ones_like(data_normalized)])
                        color_layer.data.foreach_set("color", colors.flatten())
            
            # Update mesh
            blender_mesh.update()
            
            # Add to collection
            try:
                collection = bpy.data.collections[collection_name]
            except KeyError:
                collection = bpy.data.collections.new(collection_name)
                bpy.context.scene.collection.children.link(collection)
            
            collection.objects.link(blender_obj)
            
            # Store sync pair
            mesh_id = id(mesh)
            self._sync_pairs[mesh_id] = blender_obj
            
            logger.info(f"✅ Converted PyVista mesh to Blender: {name} ({len(vertices)} vertices)")
            return blender_obj
            
        except Exception as e:
            logger.error(f"❌ Failed to convert PyVista to Blender: {e}")
            return None
    
    def blender_to_pyvista(
        self, 
        obj: Any, 
        include_materials: bool = True
    ) -> Optional["pv.PolyData"]:
        """
        Convert Blender object to PyVista mesh.
        
        Args:
            obj: Blender object
            include_materials: Whether to include material data
            
        Returns:
            PyVista PolyData mesh or None if conversion failed
        """
        try:
            if obj.type != 'MESH':
                logger.warning(f"Object {obj.name} is not a mesh")
                return None
            
            mesh_data = obj.data
            
            # Extract vertices
            vertices = np.array([v.co for v in mesh_data.vertices])
            
            # Extract faces
            faces = []
            for poly in mesh_data.polygons:
                face_vertices = [v for v in poly.vertices]
                faces.append([len(face_vertices)] + face_vertices)
            
            # Create PyVista mesh
            if len(faces) > 0:
                mesh = pv.PolyData(vertices, faces)
            else:
                mesh = pv.PolyData(vertices)
            
            # Add material data
            if include_materials and mesh_data.materials:
                material = mesh_data.materials[0]
                if material and material.use_nodes:
                    # Extract base color
                    principled = material.node_tree.nodes.get("Principled BSDF")
                    if principled:
                        base_color = principled.inputs["Base Color"].default_value
                        mesh.point_data["material_color"] = np.tile(base_color[:3], (len(vertices), 1))
            
            # Add vertex colors
            if mesh_data.vertex_colors:
                for color_layer in mesh_data.vertex_colors:
                    colors = np.array([c.color for c in color_layer.data])
                    mesh.point_data[f"vertex_color_{color_layer.name}"] = colors
            
            # Store sync pair
            obj_id = id(obj)
            self._sync_pairs[obj_id] = mesh
            
            logger.info(f"✅ Converted Blender object to PyVista: {obj.name} ({len(vertices)} vertices)")
            return mesh
            
        except Exception as e:
            logger.error(f"❌ Failed to convert Blender to PyVista: {e}")
            return None
    
    def sync_mesh(
        self, 
        source: Union["pv.PolyData", Any], 
        target: Union["pv.PolyData", Any],
        sync_vertices: bool = True,
        sync_faces: bool = True,
        sync_materials: bool = True
    ):
        """
        Synchronize mesh data between PyVista and Blender objects.
        
        Args:
            source: Source mesh (PyVista or Blender)
            target: Target mesh (PyVista or Blender)
            sync_vertices: Whether to sync vertex positions
            sync_faces: Whether to sync face topology
            sync_materials: Whether to sync material data
        """
        try:
            # Determine source and target types
            is_pyvista_source = isinstance(source, pv.PolyData)
            is_pyvista_target = isinstance(target, pv.PolyData)
            
            if is_pyvista_source and not is_pyvista_target:
                # PyVista → Blender
                self._sync_pyvista_to_blender(source, target, sync_vertices, sync_faces, sync_materials)
            elif not is_pyvista_source and is_pyvista_target:
                # Blender → PyVista
                self._sync_blender_to_pyvista(source, target, sync_vertices, sync_faces, sync_materials)
            else:
                logger.warning("Both source and target are of the same type")
                
        except Exception as e:
            logger.error(f"❌ Mesh synchronization failed: {e}")
    
    def _sync_pyvista_to_blender(
        self, 
        pyvista_mesh: "pv.PolyData", 
        blender_obj: Any,
        sync_vertices: bool,
        sync_faces: bool,
        sync_materials: bool
    ):
        """Synchronize PyVista mesh to Blender object."""
        mesh_data = blender_obj.data
        
        if sync_vertices:
            vertices = pyvista_mesh.points
            if len(vertices) != len(mesh_data.vertices):
                # Recreate mesh if vertex count changed
                mesh_data.clear_geometry()
                mesh_data.vertices.add(len(vertices))
            
            mesh_data.vertices.foreach_set("co", vertices.flatten())
        
        if sync_faces:
            faces = pyvista_mesh.faces
            if len(faces) > 0:
                mesh_data.polygons.clear()
                blender_faces = []
                for face in faces:
                    if len(face) >= 3:
                        blender_faces.extend(face[1:])
                
                mesh_data.polygons.add(len(faces))
                mesh_data.polygons.foreach_set("vertices", blender_faces)
        
        mesh_data.update()
    
    def _sync_blender_to_pyvista(
        self, 
        blender_obj: Any, 
        pyvista_mesh: "pv.PolyData",
        sync_vertices: bool,
        sync_faces: bool,
        sync_materials: bool
    ):
        """Synchronize Blender object to PyVista mesh."""
        mesh_data = blender_obj.data
        
        if sync_vertices:
            vertices = np.array([v.co for v in mesh_data.vertices])
            pyvista_mesh.points = vertices
        
        if sync_faces:
            faces = []
            for poly in mesh_data.polygons:
                face_vertices = [v for v in poly.vertices]
                faces.append([len(face_vertices)] + face_vertices)
            
            if len(faces) > 0:
                pyvista_mesh.faces = faces
    
    def create_live_sync(
        self, 
        pyvista_plotter: "pv.Plotter",
        blender_scene: Any,
        sync_interval: Optional[float] = None
    ) -> bool:
        """
        Create live synchronization between PyVista and Blender.
        
        Args:
            pyvista_plotter: PyVista plotter
            blender_scene: Blender scene
            sync_interval: Sync interval in seconds
            
        Returns:
            True if live sync started successfully
        """
        if self._sync_running:
            logger.warning("Live sync already running")
            return False
        
        interval = sync_interval or self.config.sync_interval
        
        def sync_loop():
            while self._sync_running:
                try:
                    # Sync all registered pairs
                    for source_id, target in self._sync_pairs.items():
                        if hasattr(target, 'points'):  # PyVista mesh
                            # Find corresponding Blender object
                            blender_obj = self._find_blender_object_by_id(source_id)
                            if blender_obj:
                                self.sync_mesh(target, blender_obj)
                        else:  # Blender object
                            # Find corresponding PyVista mesh
                            pyvista_mesh = self._find_pyvista_mesh_by_id(source_id)
                            if pyvista_mesh:
                                self.sync_mesh(target, pyvista_mesh)
                    
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
    
    def add_sync_callback(self, callback: Callable):
        """Add custom callback for live synchronization."""
        self._sync_callbacks.append(callback)
    
    def _find_blender_object_by_id(self, mesh_id: int) -> Optional[Any]:
        """Find Blender object by PyVista mesh ID."""
        for obj in bpy.data.objects:
            if obj.type == 'MESH' and hasattr(obj, '_pyvista_mesh_id'):
                if obj._pyvista_mesh_id == mesh_id:
                    return obj
        return None
    
    def _find_pyvista_mesh_by_id(self, obj_id: int) -> Optional["pv.PolyData"]:
        """Find PyVista mesh by Blender object ID."""
        for mesh_id, target in self._sync_pairs.items():
            if hasattr(target, 'points') and hasattr(target, '_blender_obj_id'):
                if target._blender_obj_id == obj_id:
                    return target
        return None
    
    @contextmanager
    def sync_context(self, auto_stop: bool = True):
        """Context manager for live synchronization."""
        try:
            yield self
        finally:
            if auto_stop:
                self.stop_live_sync()


class MaterialBridge:
    """Bridge for material and texture transfer between PyVista and Blender."""
    
    @staticmethod
    def pyvista_to_blender_material(
        pyvista_mesh: "pv.PolyData",
        material_name: str = "PyVistaMaterial"
    ) -> Optional[Any]:
        """Convert PyVista material data to Blender material."""
        try:
            material = bpy.data.materials.new(material_name)
            material.use_nodes = True
            
            # Get base color from point data
            if "material_color" in pyvista_mesh.point_data:
                colors = pyvista_mesh.point_data["material_color"]
                avg_color = np.mean(colors, axis=0)
                principled = material.node_tree.nodes["Principled BSDF"]
                principled.inputs["Base Color"].default_value = (*avg_color, 1.0)
            
            return material
            
        except Exception as e:
            logger.error(f"❌ Material conversion failed: {e}")
            return None
    
    @staticmethod
    def blender_to_pyvista_material(
        blender_material: Any,
        num_vertices: int
    ) -> Dict[str, np.ndarray]:
        """Convert Blender material to PyVista material data."""
        try:
            material_data = {}
            
            if blender_material and blender_material.use_nodes:
                principled = blender_material.node_tree.nodes.get("Principled BSDF")
                if principled:
                    base_color = principled.inputs["Base Color"].default_value
                    material_data["material_color"] = np.tile(base_color[:3], (num_vertices, 1))
            
            return material_data
            
        except Exception as e:
            logger.error(f"❌ Material conversion failed: {e}")
            return {}


# High-level convenience functions
def create_bidirectional_bridge(config: Optional[SyncConfig] = None) -> BidirectionalPyVistaBlenderBridge:
    """Create a new bidirectional bridge instance."""
    return BidirectionalPyVistaBlenderBridge(config)


def quick_convert_pyvista_to_blender(mesh: "pv.PolyData", name: str = "converted_mesh") -> Optional[Any]:
    """Quick conversion from PyVista to Blender."""
    bridge = BidirectionalPyVistaBlenderBridge()
    return bridge.pyvista_to_blender(mesh, name)


def quick_convert_blender_to_pyvista(obj: Any) -> Optional["pv.PolyData"]:
    """Quick conversion from Blender to PyVista."""
    bridge = BidirectionalPyVistaBlenderBridge()
    return bridge.blender_to_pyvista(obj)


__all__ = [
    "BidirectionalPyVistaBlenderBridge",
    "MaterialBridge", 
    "SyncConfig",
    "create_bidirectional_bridge",
    "quick_convert_pyvista_to_blender",
    "quick_convert_blender_to_pyvista",
] 