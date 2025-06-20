"""
TNG50 Visualization Utilities
============================

Direct visualization tools for TNG50 graph data (.pt files).
No redundant preprocessing - uses existing processed data directly.

Features:
- Load TNG50 .pt files efficiently
- Convert to Blender meshes via DataBridge
- Convert to PyVista meshes for 3D viz
- Handle multiple particle types
- Extract features for color/size mapping

Typical workflow:
1. Load .pt file → get positions, features, edges
2. Convert to visualization format (Blender/PyVista)
3. Apply styling based on particle features
4. Render or export
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# Use centralized Blender lazy loading
from ..blender.lazy import is_blender_available

try:
    from astro_lab.utils.data_bridge import transfer_direct
    DATA_BRIDGE_AVAILABLE = True
except ImportError:
    DATA_BRIDGE_AVAILABLE = False


class TNG50Visualizer:
    """
    Direct TNG50 visualization from processed .pt files.
    
    No redundant data loading or processing - uses existing graph data.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize TNG50 visualizer.
        
        Args:
            data_dir: Directory containing processed TNG50 graphs
        """
        self.data_dir = data_dir or Path("data/processed/tng50_graphs")
        
        logger.info("🌌 TNG50Visualizer initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info(f"   PyVista: {'✅' if PYVISTA_AVAILABLE else '❌'}")
        logger.info(f"   Blender: {'✅' if is_blender_available() else '❌'}")
        logger.info(f"   DataBridge: {'✅' if DATA_BRIDGE_AVAILABLE else '❌'}")
    
    def list_available_graphs(self) -> Dict[str, List[str]]:
        """
        List all available TNG50 graph files.
        
        Returns:
            Dictionary with particle types and their available files
        """
        available = {}
        
        if not self.data_dir.exists():
            logger.warning(f"Data directory not found: {self.data_dir}")
            return available
        
        for particle_dir in self.data_dir.iterdir():
            if particle_dir.is_dir():
                processed_dir = particle_dir / "processed"
                if processed_dir.exists():
                    pt_files = list(processed_dir.glob("*.pt"))
                    # Filter out metadata files
                    graph_files = [f.name for f in pt_files if "graph" in f.name]
                    if graph_files:
                        available[particle_dir.name] = graph_files
        
        return available
    
    def load_tng50_graph(
        self, 
        particle_type: str = "gas",
        snapshot: str = "snap_099.0",
        radius: float = 1.0,
        max_particles: int = 1000
    ) -> Dict[str, Any]:
        """
        Load TNG50 graph data from processed .pt file.
        
        Args:
            particle_type: Particle type ("gas", "stars", "black_holes")
            snapshot: Snapshot identifier  
            radius: Graph radius used during processing
            max_particles: Max particles used during processing
            
        Returns:
            Dictionary with graph data
        """
        # Construct filename
        filename = f"tng50_graph_{snapshot}_parttype0_r{radius:.1f}_n{max_particles}.pt"
        pt_file = self.data_dir / particle_type / "processed" / filename
        
        if not pt_file.exists():
            # Try to find any matching file
            available = self.list_available_graphs()
            if particle_type in available:
                logger.warning(f"Exact file not found, using: {available[particle_type][0]}")
                pt_file = self.data_dir / particle_type / "processed" / available[particle_type][0]
            else:
                raise FileNotFoundError(f"No TNG50 graph data found for {particle_type}")
        
        logger.info(f"📂 Loading TNG50 graph: {pt_file.name}")
        
        # Load PyG InMemoryDataset format
        loaded_data = torch.load(pt_file, map_location='cpu', weights_only=False)
        
        if isinstance(loaded_data, tuple) and len(loaded_data) >= 1:
            data_dict = loaded_data[0]
            
            # Extract key data
            result = {
                'positions': data_dict['pos'].numpy(),  # [N, 3]
                'features': data_dict['x'].numpy(),     # [N, feature_dim]
                'edge_index': data_dict['edge_index'].numpy(),  # [2, num_edges]
                'edge_weights': data_dict['edge_attr'].numpy(), # [num_edges, 1]
                'num_particles': data_dict['num_particles'],
                'particle_type': data_dict['particle_type'],
                'feature_names': data_dict['feature_names'],
                'snapshot_file': data_dict['snapshot_file'],
                'file_path': str(pt_file)
            }
            
            logger.info(f"✅ Loaded {result['num_particles']:,} particles")
            logger.info(f"   Features: {result['feature_names']}")
            logger.info(f"   Edges: {result['edge_index'].shape[1]:,}")
            
            return result
        
        else:
            raise ValueError(f"Unexpected data format in {pt_file}")
    
    def to_pyvista_mesh(
        self, 
        graph_data: Dict[str, Any],
        point_size: float = 5.0,
        color_by: str = "mass",
        include_edges: bool = False
    ) -> "pv.PolyData":
        """
        Convert TNG50 graph to PyVista mesh for 3D visualization.
        
        Args:
            graph_data: Graph data from load_tng50_graph()
            point_size: Point size for rendering
            color_by: Feature to use for coloring
            include_edges: Whether to include graph edges
            
        Returns:
            PyVista PolyData mesh
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista not available")
        
        positions = graph_data['positions']
        features = graph_data['features']
        feature_names = graph_data['feature_names']
        
        logger.info(f"🔧 Converting to PyVista mesh: {len(positions):,} points")
        
        # Create point cloud
        mesh = pv.PolyData(positions)
        
        # Add features as point data
        for i, name in enumerate(feature_names):
            mesh.point_data[name] = features[:, i]
        
        # Add particle indices
        mesh.point_data['particle_id'] = np.arange(len(positions))
        
        # Set default coloring
        if color_by in feature_names:
            mesh.set_active_scalars(color_by)
            logger.info(f"   Coloring by: {color_by}")
        
        # Add edges if requested
        if include_edges:
            edge_index = graph_data['edge_index']
            edge_weights = graph_data['edge_weights']
            
            # Create lines between connected particles
            lines = []
            for i in range(edge_index.shape[1]):
                start_idx = edge_index[0, i]
                end_idx = edge_index[1, i]
                lines.extend([2, start_idx, end_idx])  # PyVista line format
            
            mesh.lines = np.array(lines)
            mesh.line_data['edge_weight'] = edge_weights.flatten()
            
            logger.info(f"   Added {edge_index.shape[1]:,} edges")
        
        return mesh
    
    def to_blender_objects(
        self,
        graph_data: Dict[str, Any],
        object_name: str = "TNG50_particles",
        use_instancing: bool = True
    ) -> List[Any]:
        """
        Convert TNG50 graph to Blender objects.
        
        Args:
            graph_data: Graph data from load_tng50_graph()
            object_name: Base name for Blender objects
            use_instancing: Use instancing for better performance
            
        Returns:
            List of created Blender objects
        """
        if not is_blender_available():
            raise ImportError("Blender not available")
        
        if not DATA_BRIDGE_AVAILABLE:
            raise ImportError("DataBridge not available")
        
        positions = graph_data['positions']
        features = graph_data['features']
        
        logger.info(f"🎨 Converting to Blender: {len(positions):,} particles")
        
        # Create or get base mesh (sphere for particles)
        if use_instancing:
            # Create base sphere
            import bpy  # Import locally after availability check
            bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0, 0, 0))
            base_sphere = bpy.context.active_object
            base_sphere.name = f"{object_name}_base"
            
            # Create collection for instances
            collection_name = f"{object_name}_collection"
            collection = bpy.data.collections.new(collection_name)
            bpy.context.scene.collection.children.link(collection)
            
            objects = [base_sphere]
            
            # Use geometry nodes for instancing (more efficient)
            # For now, create simple instances
            for i, pos in enumerate(positions[:100]):  # Limit for performance
                bpy.ops.object.duplicate()
                instance = bpy.context.active_object
                instance.location = pos
                instance.name = f"{object_name}_particle_{i}"
                
                # Scale by mass if available
                if len(features) > 0:
                    mass = features[i, 0]  # Assume first feature is mass
                    scale = np.log10(mass * 1000) if mass > 0 else 1.0
                    instance.scale = (scale, scale, scale)
                
                collection.objects.link(instance)
                objects.append(instance)
            
            logger.info(f"   Created {len(objects)-1} particle instances")
            
        else:
            # Create single mesh with all particles
            import bmesh
            
            bm = bmesh.new()
            
            for i, pos in enumerate(positions):
                # Add vertex at particle position
                vert = bm.verts.new(pos)
                
                # Could add more sophisticated geometry here
            
            # Create mesh
            mesh = bpy.data.meshes.new(object_name)
            bm.to_mesh(mesh)
            bm.free()
            
            # Create object
            obj = bpy.data.objects.new(object_name, mesh)
            bpy.context.scene.collection.objects.link(obj)
            
            objects = [obj]
            logger.info(f"   Created single mesh object")
        
        return objects
    
    def quick_visualization(
        self,
        particle_type: str = "gas",
        method: str = "pyvista",
        **kwargs
    ) -> Any:
        """
        Quick visualization of TNG50 data.
        
        Args:
            particle_type: Particle type to visualize
            method: Visualization method ("pyvista" or "blender")
            **kwargs: Additional arguments
            
        Returns:
            Visualization object (PyVista mesh or Blender objects)
        """
        logger.info(f"🚀 Quick TNG50 visualization: {particle_type} via {method}")
        
        # Load data
        graph_data = self.load_tng50_graph(particle_type=particle_type)
        
        # Convert to visualization format
        if method == "pyvista":
            result = self.to_pyvista_mesh(graph_data, **kwargs)
            
            # Auto-show if possible
            try:
                result.plot(
                    scalars=graph_data['feature_names'][0],
                    point_size=kwargs.get('point_size', 5.0),
                    title=f"TNG50 {particle_type.title()} Particles"
                )
            except Exception as e:
                logger.warning(f"Could not auto-plot: {e}")
            
            return result
            
        elif method == "blender":
            return self.to_blender_objects(graph_data, **kwargs)
            
        else:
            raise ValueError(f"Unknown method: {method}")


# Convenience functions
def load_tng50_gas(max_particles: int = 1000) -> Dict[str, Any]:
    """Quick load of TNG50 gas particles."""
    viz = TNG50Visualizer()
    return viz.load_tng50_graph("gas", max_particles=max_particles)


def load_tng50_stars(max_particles: int = 1000) -> Dict[str, Any]:
    """Quick load of TNG50 star particles."""
    viz = TNG50Visualizer()
    return viz.load_tng50_graph("stars", max_particles=max_particles)


def quick_pyvista_plot(particle_type: str = "gas", **kwargs):
    """Quick PyVista plot of TNG50 data."""
    viz = TNG50Visualizer()
    return viz.quick_visualization(particle_type, "pyvista", **kwargs)


def quick_blender_import(particle_type: str = "gas", **kwargs):
    """Quick Blender import of TNG50 data.""" 
    viz = TNG50Visualizer()
    return viz.quick_visualization(particle_type, "blender", **kwargs)


def list_available_data() -> Dict[str, List[str]]:
    """List all available TNG50 data files."""
    viz = TNG50Visualizer()
    return viz.list_available_graphs()


__all__ = [
    "TNG50Visualizer",
    "load_tng50_gas",
    "load_tng50_stars", 
    "quick_pyvista_plot",
    "quick_blender_import",
    "list_available_data",
] 