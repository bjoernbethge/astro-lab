"""AstroLab Widget System - Advanced visualization for astronomical data.

This module provides a unified interface for multiple visualization backends,
each optimized for specific use cases while maintaining consistency.

Backends:
- PyVista (alpv): Scientific 3D visualization with VTK
- Open3D (alo3d): Real-time point cloud processing
- Blender (albpy): Photorealistic rendering and animation
- Plotly: Interactive web-based visualizations
- Cosmograph (alcg): Large-scale graph visualization

The system uses astropy for all astronomical calculations and unit handling.
"""

import logging
from typing import Any, Dict, List, Optional

# Import all backend modules with their full APIs
from . import (
    albpy,  # Blender backend
    alcg,  # Cosmograph backend
    alo3d,  # Open3D backend
    alpv,  # PyVista backend
    plotly,  # Plotly backend
)

# Import bridges and utilities
# CosmographBridge now in alcg module - import directly from alcg
# Import enhanced utilities
from .enhanced import (
    # Tensor bridge
    AstronomicalTensorBridge,
    # Image processing
    ImageProcessor,
    # Post-processing
    PostProcessor,
    # Texture generation
    TextureGenerator,
    ZeroCopyTensorConverter,
    converter,
    orchestrate_pipeline,
    orchestrate_post_processing,
    orchestrate_texture_pipeline,
    tensor_bridge_context,
)

# Marimo widgets are now in the UI module
# Import TNG50 visualization
from .tng50 import TNG50Visualizer

logger = logging.getLogger(__name__)


class UnifiedVisualizationPipeline:
    """Unified pipeline that orchestrates multiple backends for optimal results.

    This pipeline intelligently combines backends based on the task:
    1. PyVista for scientific analysis and feature extraction
    2. Blender for photorealistic rendering with effects
    3. Open3D for interactive exploration
    4. Plotly for web export and sharing

    All astronomical calculations use astropy units and coordinates.
    """

    def __init__(self):
        """Initialize the unified pipeline."""
        self.tensor_bridge = AstronomicalTensorBridge()
        self.image_processor = ImageProcessor()
        self.post_processor = PostProcessor()
        self.texture_generator = TextureGenerator()

        # Backend availability is checked within each module
        self.backends = {
            "pyvista": alpv,
            "open3d": alo3d,
            "blender": albpy,
            "plotly": plotly,
            "cosmograph": alcg,
        }

    def create_visualization(
        self,
        data: Any,
        visualization_type: str = "auto",
        effects: Optional[List[str]] = None,
        export_formats: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create visualization using optimal backend combination.

        Args:
            data: Input data (TensorDict, numpy array, astropy Table, etc.)
            visualization_type: Type of visualization ('stellar', 'galaxy', 'cosmic_web', etc.)
            effects: Visual effects to apply ('glow', 'volumetric', 'bloom', etc.)
            export_formats: Desired export formats ('png', 'html', 'blend', 'ply', etc.)
            **kwargs: Additional parameters passed to backends

        Returns:
            Dictionary with results from each backend and export paths
        """
        results = {}

        # Step 1: Scientific analysis with PyVista
        logger.info("Step 1: Scientific analysis with PyVista")
        pv_result = self._process_with_pyvista(data, visualization_type, **kwargs)
        results["pyvista"] = pv_result

        # Step 2: Apply effects if requested
        if effects:
            logger.info(f"Step 2: Applying effects: {effects}")

            # For photorealistic effects, use Blender
            if any(
                effect in ["glow", "volumetric", "bloom", "lens_flare"]
                for effect in effects
            ):
                blender_result = self._process_with_blender(
                    data,
                    visualization_type,
                    pyvista_result=pv_result,
                    effects=effects,
                    **kwargs,
                )
                results["blender"] = blender_result

            # For scientific filters, use PyVista
            if any(
                effect in ["contour", "streamlines", "vectors"] for effect in effects
            ):
                pv_result = self.post_processor.apply_pyvista_filters(
                    pv_result["mesh"], filters=effects
                )
                results["pyvista_filtered"] = pv_result

        # Step 3: Interactive visualization if requested
        if kwargs.get("interactive", True):
            logger.info("Step 3: Creating interactive visualization")

            # Use Open3D for point cloud interaction
            o3d_result = self._process_with_open3d(
                data, visualization_type, pyvista_result=pv_result, **kwargs
            )
            results["open3d"] = o3d_result

        # Step 4: Export in requested formats
        if export_formats:
            logger.info(f"Step 4: Exporting in formats: {export_formats}")
            export_paths = self._export_results(results, export_formats, **kwargs)
            results["exports"] = export_paths

        return results

    def _process_with_pyvista(
        self, data: Any, viz_type: str, **kwargs
    ) -> Dict[str, Any]:
        """Process data with PyVista for scientific visualization."""
        # Use PyVista's create_visualization function
        plotter = alpv.create_visualization(data, plot_type=viz_type, **kwargs)

        # Extract mesh for further processing
        if hasattr(plotter, "mesh"):
            mesh = plotter.mesh
        else:
            # Create mesh from plotter
            mesh = None
            for actor in plotter.renderer.actors.values():
                if hasattr(actor, "GetMapper"):
                    mapper = actor.GetMapper()
                    if mapper and hasattr(mapper, "GetInput"):
                        mesh = mapper.GetInput()
                        break

        return {
            "plotter": plotter,
            "mesh": mesh,
            "bounds": mesh.bounds if mesh else None,
            "center": mesh.center if mesh else None,
        }

    def _process_with_blender(
        self,
        data: Any,
        viz_type: str,
        pyvista_result: Optional[Dict] = None,
        effects: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process with Blender for photorealistic rendering."""
        # Clear existing scene
        albpy.operators.AstronomicalOperators.clear_scene()

        # Create visualization
        if viz_type == "stellar":
            objects = albpy.create_stellar_field(data, **kwargs)
        elif viz_type == "galaxy":
            objects = albpy.create_galaxy_cluster_visualization(data, **kwargs)
        elif viz_type == "cosmic_web":
            scene = albpy.setup_cosmic_web_scene(data, **kwargs)
            objects = scene
        else:
            objects = albpy.create_blender_visualization(data, **kwargs)

        # Apply effects using compositing
        if effects:
            self.post_processor.use_blender_compositing_nodes(effects)

        # Render if requested
        if kwargs.get("render", False):
            output_path = kwargs.get("render_path", "astro_render.png")
            albpy.render_astronomical_scene(
                output_path,
                resolution=kwargs.get("resolution", (1920, 1080)),
                samples=kwargs.get("samples", 128),
            )
            return {"objects": objects, "render_path": output_path}

        return {"objects": objects}

    def _process_with_open3d(
        self, data: Any, viz_type: str, pyvista_result: Optional[Dict] = None, **kwargs
    ) -> Dict[str, Any]:
        """Process with Open3D for interactive visualization."""
        # Create Open3D visualization
        if viz_type == "cosmic_web":
            geometries = alo3d.create_cosmic_web_visualization(data, **kwargs)
        else:
            geometry = alo3d.create_visualization(data, plot_type=viz_type, **kwargs)
            geometries = {"main": geometry}

        # Create interactive viewer if requested
        if kwargs.get("show_interactive", True):
            viewer = alo3d.create_interactive_viewer(
                geometries, window_name=f"AstroLab - {viz_type}", **kwargs
            )
            return {"geometries": geometries, "viewer": viewer}

        return {"geometries": geometries}

    def _export_results(
        self, results: Dict[str, Any], formats: List[str], **kwargs
    ) -> Dict[str, str]:
        """Export results in multiple formats."""
        export_paths = {}
        base_path = kwargs.get("output_dir", ".")
        base_name = kwargs.get("output_name", "astro_visualization")

        for fmt in formats:
            if fmt == "png" and "blender" in results:
                # High-quality render from Blender
                path = f"{base_path}/{base_name}.png"
                albpy.render_astronomical_scene(path, **kwargs)
                export_paths["png"] = path

            elif fmt == "html" and "plotly" not in results:
                # Create Plotly visualization for web export
                plotly_fig = plotly.create_plotly_visualization(
                    results.get("pyvista", {}).get("mesh"), **kwargs
                )
                path = f"{base_path}/{base_name}.html"
                plotly_fig.write_html(path)
                export_paths["html"] = path

            elif fmt == "ply" and "open3d" in results:
                # Export point cloud
                path = f"{base_path}/{base_name}.ply"
                alo3d.save_visualization(results["open3d"]["geometries"], path)
                export_paths["ply"] = path

            elif fmt == "blend" and "blender" in results:
                # Save Blender file
                import bpy

                path = f"{base_path}/{base_name}.blend"
                bpy.ops.wm.save_as_mainfile(filepath=path)
                export_paths["blend"] = path

        return export_paths


# Create global pipeline instance
unified_pipeline = UnifiedVisualizationPipeline()


# Main API functions
def create_visualization(
    data: Any, backend: Optional[str] = None, visualization_type: str = "auto", **kwargs
) -> Any:
    """Create visualization with specified or automatic backend selection.

    Args:
        data: Input astronomical data
        backend: Specific backend to use, or None for automatic selection
        visualization_type: Type of visualization
        **kwargs: Backend-specific parameters

    Returns:
        Visualization object(s) from the selected backend(s)
    """
    if backend:
        # Use specific backend
        backend_module = unified_pipeline.backends.get(backend.lower())
        if backend_module:
            return backend_module.create_visualization(
                data, plot_type=visualization_type, **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    else:
        # Use unified pipeline for automatic backend selection
        return unified_pipeline.create_visualization(
            data, visualization_type=visualization_type, **kwargs
        )


def plot_cosmic_web(
    data: Any,
    effects: Optional[List[str]] = None,
    interactive: bool = True,
    photorealistic: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Create cosmic web visualization with multiple backends.

    Args:
        data: Cosmic web data (positions, edges, clusters, etc.)
        effects: Visual effects ('glow', 'volumetric', 'filaments')
        interactive: Create interactive visualization
        photorealistic: Create photorealistic rendering
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from multiple backends
    """
    # Default effects for cosmic web
    if effects is None:
        effects = ["glow"] if photorealistic else []

    return unified_pipeline.create_visualization(
        data,
        visualization_type="cosmic_web",
        effects=effects,
        interactive=interactive,
        render=photorealistic,
        **kwargs,
    )


def plot_stellar_data(
    data: Any, color_by: str = "temperature", size_by: str = "luminosity", **kwargs
) -> Dict[str, Any]:
    """Create stellar visualization with appropriate coloring and sizing.

    Args:
        data: Stellar data with positions and properties
        color_by: Property to use for coloring
        size_by: Property to use for sizing
        **kwargs: Additional parameters

    Returns:
        Visualization results
    """
    return unified_pipeline.create_visualization(
        data,
        visualization_type="stellar",
        color_property=color_by,
        size_property=size_by,
        **kwargs,
    )


def plot_galaxy_data(
    data: Any, morphology: Optional[str] = None, redshift_effects: bool = True, **kwargs
) -> Dict[str, Any]:
    """Create galaxy visualization with morphology and redshift effects.

    Args:
        data: Galaxy data (positions, properties, morphology)
        morphology: Galaxy morphology type
        redshift_effects: Apply redshift-based coloring
        **kwargs: Additional parameters

    Returns:
        Visualization results
    """
    return unified_pipeline.create_visualization(
        data,
        visualization_type="galaxy",
        morphology=morphology,
        apply_redshift=redshift_effects,
        **kwargs,
    )


def create_publication_figure(
    data: Any,
    output_path: str,
    figure_type: str = "panel",
    style: str = "scientific",
    **kwargs,
) -> str:
    """Create publication-ready figure with professional styling.

    Args:
        data: Input data
        output_path: Output file path
        figure_type: Type of figure ('single', 'panel', 'comparison')
        style: Visual style ('scientific', 'presentation', 'poster')
        **kwargs: Additional parameters

    Returns:
        Path to saved figure
    """
    # Style presets
    styles = {
        "scientific": {
            "background_color": "white",
            "colormap": "viridis",
            "font_size": 12,
            "dpi": 300,
        },
        "presentation": {
            "background_color": "black",
            "colormap": "plasma",
            "font_size": 18,
            "effects": ["glow"],
            "dpi": 150,
        },
        "poster": {
            "background_color": "white",
            "colormap": "cividis",
            "font_size": 24,
            "dpi": 200,
        },
    }

    # Apply style
    style_kwargs = styles.get(style, {})
    style_kwargs.update(kwargs)

    # Create visualization
    results = unified_pipeline.create_visualization(
        data,
        render=True,
        render_path=output_path,
        export_formats=["png"],
        **style_kwargs,
    )

    return results["exports"]["png"]


# Export main components
__all__ = [
    # Main pipeline
    "UnifiedVisualizationPipeline",
    "unified_pipeline",
    # Main API functions
    "create_visualization",
    "plot_cosmic_web",
    "plot_stellar_data",
    "plot_galaxy_data",
    "create_publication_figure",
    # Backend modules (with full APIs)
    "alpv",  # PyVista
    "alo3d",  # Open3D
    "albpy",  # Blender
    "plotly",  # Plotly
    "alcg",  # Cosmograph
    # Bridges and utilities
    "AstronomicalTensorBridge",
    "tensor_bridge_context",
    # Processing utilities
    "ImageProcessor",
    "PostProcessor",
    "TextureGenerator",
    "orchestrate_pipeline",
    "orchestrate_post_processing",
    "orchestrate_texture_pipeline",
    # Specialized visualizers
    "TNG50Visualizer",
    "ZeroCopyTensorConverter",
    "converter",
]
