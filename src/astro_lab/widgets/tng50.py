"""
TNG50 Data Visualizer
=====================

Visualization helpers for TNG50 cosmological simulation data (gas, stars, dark matter).

- Load processed ``.pt`` graph files from the AstroLab data layout.
- **PyVista:** via ``astro_lab.widgets.enhanced.to_pyvista``.
- **Blender:** via the PyPI ``bpy`` module (same stack as AlbPy); requires a usable
  ``bpy.context`` / scene where relevant (e.g. ``to_blender_objects``, ``render``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import bpy as _bpy

# Blender's ``bpy`` is a dynamic extension module; treat as ``Any`` for static analysis.
bpy: Any = _bpy

from astro_lab.config import get_data_config
from astro_lab.widgets.enhanced import to_pyvista

logger = logging.getLogger(__name__)


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
        if data_dir is None:
            data_config = get_data_config()
            self.data_dir = Path(data_config["processed_dir"]) / "tng50" / "graphs"
        else:
            self.data_dir = data_dir

        logger.info("🌌 TNG50Visualizer initialized")
        logger.info(f"   Data directory: {self.data_dir}")
        logger.info("   PyVista: ✅ (widgets.enhanced.to_pyvista)")
        logger.info("   Blender: ✅ (PyPI bpy)")

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
        max_particles: int = 1000,
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
                logger.warning(
                    f"Exact file not found, using: {available[particle_type][0]}"
                )
                pt_file = (
                    self.data_dir
                    / particle_type
                    / "processed"
                    / available[particle_type][0]
                )
            else:
                raise FileNotFoundError(
                    f"No TNG50 graph data found for {particle_type}"
                )

        logger.info(f"📂 Loading TNG50 graph: {pt_file.name}")

        # Load PyG InMemoryDataset format
        loaded_data = torch.load(pt_file, map_location="cpu", weights_only=False)

        if isinstance(loaded_data, tuple) and len(loaded_data) >= 1:
            data_dict = loaded_data[0]

            # Extract key data
            result = {
                "positions": data_dict["pos"].numpy(),  # [N, 3]
                "features": data_dict["x"].numpy(),  # [N, feature_dim]
                "edge_index": data_dict["edge_index"].numpy(),  # [2, num_edges]
                "edge_weights": data_dict["edge_attr"].numpy(),  # [num_edges, 1]
                "num_particles": data_dict["num_particles"],
                "particle_type": data_dict["particle_type"],
                "feature_names": data_dict["feature_names"],
                "snapshot_file": data_dict["snapshot_file"],
                "file_path": str(pt_file),
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
        include_edges: bool = False,
    ) -> Any:
        """
        Convert TNG50 graph to PyVista mesh for 3D visualization using Enhanced-API.
        """
        mesh = to_pyvista(graph_data)
        # Optionally set coloring or other attributes here if needed
        return mesh

    def to_blender_objects(
        self,
        graph_data: Dict[str, Any],
        object_name: str = "TNG50_particles",
        use_instancing: bool = True,
    ) -> List[Any]:
        _ = use_instancing  # reserved for instancing / GN workflow
        scene = getattr(bpy.context, "scene", None)
        if scene is None:
            raise RuntimeError("to_blender_objects: no bpy.context.scene")

        positions = graph_data["positions"]

        logger.info(f"🎨 Converting to Blender: {len(positions):,} particles")

        mesh_data = bpy.data.meshes.new(object_name)
        mesh_obj = bpy.data.objects.new(object_name, mesh_data)

        vertices = [tuple(map(float, row)) for row in positions]

        # Create mesh from vertices
        mesh_data.from_pydata(vertices, [], [])
        mesh_data.update()

        scene.collection.objects.link(mesh_obj)

        objects = [mesh_obj]
        logger.info("   Created single mesh object")

        return objects

    def quick_visualization(
        self, particle_type: str = "gas", method: str = "pyvista", **kwargs
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
                    scalars=graph_data["feature_names"][0],
                    point_size=kwargs.get("point_size", 5.0),
                    title=f"TNG50 {particle_type.title()} Particles",
                )
            except Exception as e:
                logger.warning(f"Could not auto-plot: {e}")

            return result

        elif method == "blender":
            return self.to_blender_objects(graph_data, **kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")

    def render(
        self,
        output_path: str = "results/tng50_render.png",
        engine: str = "CYCLES",
        resolution: Tuple[int, int] = (1920, 1080),
        samples: int = 128,
        animation: bool = False,
        **kwargs,
    ) -> bool:
        scene = bpy.context.scene
        if scene is None:
            logger.error("render: no bpy.context.scene")
            return False
        try:
            scene.render.engine = engine
            scene.render.filepath = output_path
            scene.render.resolution_x = resolution[0]
            scene.render.resolution_y = resolution[1]
            cycles = getattr(scene, "cycles", None)
            if cycles is not None:
                cycles.samples = samples
            elif hasattr(scene.render, "samples"):
                scene.render.samples = samples

            # Add camera if not present
            if not any(obj.type == "CAMERA" for obj in scene.objects):
                bpy.ops.object.camera_add(location=[10, -10, 5])
                camera = bpy.context.active_object
                camera.rotation_euler = [1.1, 0, 0.8]
                scene.camera = camera

            # Add light if not present
            if not any(obj.type == "LIGHT" for obj in scene.objects):
                bpy.ops.object.light_add(type="SUN", location=[5, 5, 10])
                light = bpy.context.active_object
                light.data.energy = 5.0

            if animation:
                bpy.ops.render.render(animation=True)
            else:
                bpy.ops.render.render(write_still=True)
            return True
        except Exception as e:
            logger.error("Failed to render TNG50 scene: %s", e)
            return False


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
