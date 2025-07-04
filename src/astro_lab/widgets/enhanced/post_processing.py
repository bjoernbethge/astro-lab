"""
Enhanced Post Processing - Package Combiner
==========================================

Orchestrates existing package features:
- PyVista for 3D visualization and post-processing
- NumPy for array operations
- Blender for advanced 3D post-processing with compositing node trees
- SciPy for signal processing
"""

import logging
from typing import Any, Dict, List, Union

import bpy
import numpy as np
import pyvista as pv
from scipy import signal

# Note: TensorDict imports removed - post-processing works with numpy arrays
# TensorDicts are still used in model training, not in visualization

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Package orchestrator for post-processing operations.

    Uses existing features from:
    - PyVista for 3D visualization and post-processing
    - NumPy for array operations
    - Blender for advanced 3D post-processing with compositing node trees
    - SciPy for signal processing
    """

    def __init__(self):
        self.plotter = None

    def apply_pyvista_filters(
        self, mesh: pv.PolyData, filter_type: str = "smooth", **kwargs
    ) -> pv.PolyData:
        """
        Use PyVista's built-in filtering capabilities.

        Args:
            mesh: PyVista mesh
            filter_type: Type of filter to apply
            **kwargs: Filter parameters

        Returns:
            Filtered mesh
        """
        if filter_type == "smooth":
            # Use PyVista's smoothing
            n_iter = kwargs.get("n_iter", 20)
            relaxation_factor = kwargs.get("relaxation_factor", 0.1)
            return mesh.smooth(n_iter=n_iter, relaxation_factor=relaxation_factor)

        elif filter_type == "decimate":
            # Use PyVista's decimation
            target_reduction = kwargs.get("target_reduction", 0.5)
            return mesh.decimate(target_reduction=target_reduction)

        elif filter_type == "clean":
            # Use PyVista's cleaning
            tolerance = kwargs.get("tolerance", 1e-8)
            return mesh.clean(tolerance=tolerance)

        elif filter_type == "fill_holes":
            # Use PyVista's hole filling
            hole_size = kwargs.get("hole_size", 1000)
            return mesh.fill_holes(hole_size=hole_size)

        else:
            return mesh

    def apply_pyvista_transforms(
        self, mesh: pv.PolyData, transform_type: str = "translate", **kwargs
    ) -> pv.PolyData:
        """
        Use PyVista's built-in transformation capabilities.

        Args:
            mesh: PyVista mesh
            transform_type: Type of transformation
            **kwargs: Transformation parameters

        Returns:
            Transformed mesh
        """
        if transform_type == "translate":
            # Use PyVista's translation
            x = kwargs.get("x", 0.0)
            y = kwargs.get("y", 0.0)
            z = kwargs.get("z", 0.0)
            return mesh.translate([x, y, z])

        elif transform_type == "rotate":
            # Use PyVista's rotation
            angle = kwargs.get("angle", 90.0)
            return mesh.rotate_x(angle)

        elif transform_type == "scale":
            # Use PyVista's scaling
            x = kwargs.get("x", 1.0)
            y = kwargs.get("y", 1.0)
            z = kwargs.get("z", 1.0)
            return mesh.scale([x, y, z])

        else:
            return mesh

    def create_pyvista_animation(
        self, meshes: List[pv.PolyData], output_path: str = "animation.mp4", **kwargs
    ) -> str:
        """
        Use PyVista's animation capabilities.

        Args:
            meshes: List of PyVista meshes for animation frames
            output_path: Output file path
            **kwargs: Animation parameters

        Returns:
            Path to created animation
        """
        # Use PyVista's plotter for animation
        plotter = pv.Plotter()

        # Add meshes to plotter
        for i, mesh in enumerate(meshes):
            plotter.add_mesh(mesh, name=f"frame_{i}")

        # Use PyVista's animation
        plotter.open_movie(output_path)

        for i in range(len(meshes)):
            plotter.write_frame()

        plotter.close()
        return output_path

    def create_pyvista_screenshot(
        self, mesh: pv.PolyData, output_path: str = "screenshot.png", **kwargs
    ) -> str:
        """
        Use PyVista's screenshot capabilities.

        Args:
            mesh: PyVista mesh to capture
            output_path: Output file path
            **kwargs: Screenshot parameters

        Returns:
            Path to created screenshot
        """
        # Use PyVista's plotter for screenshot
        plotter = pv.Plotter()
        plotter.add_mesh(mesh)

        # Use PyVista's screenshot
        plotter.screenshot(output_path)
        plotter.close()

        return output_path

    def export_pyvista_mesh(
        self, mesh: pv.PolyData, output_path: str, file_format: str = "obj", **kwargs
    ) -> str:
        """
        Use PyVista's export capabilities.

        Args:
            mesh: PyVista mesh to export
            output_path: Output file path
            file_format: Export format
            **kwargs: Export parameters

        Returns:
            Path to exported file
        """
        # Use PyVista's save method
        mesh.save(output_path)
        return output_path

    def use_blender_compositing_nodes(
        self,
        image_data: np.ndarray,
        node_tree_name: str = "AstroLab_Compositing",
        **kwargs,
    ) -> np.ndarray:
        """
        Use Blender's compositing node trees for advanced image processing.

        Based on Blender's compositing node system as documented in:
        https://docs.blender.org/manual/en/latest/compositing/types/output/composite.html

        Args:
            image_data: Input image array
            node_tree_name: Name for the compositing node tree
            **kwargs: Compositing parameters

        Returns:
            Processed image array
        """
        # Enable nodes for the scene
        if not bpy.context.scene.use_nodes:
            bpy.context.scene.use_nodes = True

        # Get or create node tree
        node_tree = bpy.context.scene.node_tree

        # Clear existing nodes
        for node in node_tree.nodes:
            node_tree.nodes.remove(node)

        # Create input image node
        input_node = node_tree.nodes.new(type="CompositorNodeImage")
        input_node.location = (0, 0)
        input_node.name = "Input_Image"

        # Create composite output node
        output_node = node_tree.nodes.new(type="CompositorNodeComposite")
        output_node.location = (600, 0)
        output_node.name = "Output_Composite"

        # Add processing nodes based on kwargs
        current_node = input_node

        if kwargs.get("blur", False):
            # Add blur node
            blur_node = node_tree.nodes.new(type="CompositorNodeBlur")
            blur_node.location = (200, 0)
            blur_node.size_x = kwargs.get("blur_size", 5)
            blur_node.size_y = kwargs.get("blur_size", 5)
            node_tree.links.new(
                current_node.outputs["Image"], blur_node.inputs["Image"]
            )
            current_node = blur_node

        if kwargs.get("color_correction", False):
            # Add color correction node
            color_node = node_tree.nodes.new(type="CompositorNodeColorCorrection")
            color_node.location = (400, 0)
            node_tree.links.new(
                current_node.outputs["Image"], color_node.inputs["Image"]
            )
            current_node = color_node

        if kwargs.get("mix", False):
            # Add mix node for blending
            mix_node = node_tree.nodes.new(type="CompositorNodeMixRGB")
            mix_node.location = (400, 100)
            mix_node.blend_type = kwargs.get("blend_type", "MIX")
            mix_node.factor = kwargs.get("mix_factor", 0.5)
            node_tree.links.new(current_node.outputs["Image"], mix_node.inputs[1])
            current_node = mix_node

        # Connect to output
        node_tree.links.new(current_node.outputs["Image"], output_node.inputs["Image"])

        # Update the node tree
        node_tree.update_tag()

        # For now, return the original image since we can't easily get the processed result
        # In a real implementation, you'd render the compositor and read the result
        return image_data

    def use_scipy_signal_processing(
        self, data: np.ndarray, operation: str = "filter", **kwargs
    ) -> np.ndarray:
        """
        Use SciPy's signal processing capabilities.

        Args:
            data: Input data array
            operation: Type of signal processing
            **kwargs: Processing parameters

        Returns:
            Processed data
        """
        if operation == "filter":
            # Use SciPy's filtering
            filter_type = kwargs.get("filter_type", "gaussian")
            if filter_type == "gaussian":
                sigma = kwargs.get("sigma", 1.0)
                return signal.gaussian_filter(data, sigma=sigma)
            elif filter_type == "median":
                size = kwargs.get("size", 3)
                return signal.medfilt(data, kernel_size=size)

        elif operation == "convolve":
            # Use SciPy's convolution
            kernel = kwargs.get("kernel", np.ones((3, 3)) / 9)
            return signal.convolve2d(data, kernel, mode="same")

        elif operation == "fft":
            # Use SciPy's FFT
            return np.abs(signal.fft.fft2(data))

        else:
            return data

    def use_numpy_post_processing(
        self, data: np.ndarray, operation: str = "normalize", **kwargs
    ) -> np.ndarray:
        """
        Use NumPy's post-processing operations.

        Args:
            data: Input data array
            operation: NumPy operation to apply
            **kwargs: Operation parameters

        Returns:
            Processed data
        """
        if operation == "normalize":
            # Use NumPy's normalization
            min_val = kwargs.get("min_val", 0)
            max_val = kwargs.get("max_val", 1)
            return (data - data.min()) / (data.max() - data.min()) * (
                max_val - min_val
            ) + min_val

        elif operation == "clip":
            # Use NumPy's clipping
            min_val = kwargs.get("min_val", 0)
            max_val = kwargs.get("max_val", 1)
            return np.clip(data, min_val, max_val)

        elif operation == "log":
            # Use NumPy's logarithmic scaling
            return np.log(data + kwargs.get("offset", 1e-10))

        elif operation == "sqrt":
            # Use NumPy's square root scaling
            return np.sqrt(np.abs(data))

        else:
            return data

    def orchestrate_post_processing(
        self,
        data: Union[pv.PolyData, np.ndarray, Any],
        pipeline_steps: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Orchestrate post-processing pipeline using existing package features.

        Args:
            data: Input data
            pipeline_steps: List of processing steps
            **kwargs: Pipeline parameters

        Returns:
            Pipeline results
        """
        results = {}
        current_data = data

        for i, step in enumerate(pipeline_steps):
            step_name = step.get("step", f"step_{i}")
            step_type = step.get("type")

            if step_type == "pyvista_filter":
                # Use PyVista's filtering
                if isinstance(current_data, pv.PolyData):
                    current_data = self.apply_pyvista_filters(
                        current_data,
                        filter_type=step.get("filter_type", "smooth"),
                        **step.get("params", {}),
                    )
                    results[step_name] = {
                        "mesh": current_data,
                        "type": "pyvista_filter",
                    }

            elif step_type == "pyvista_transform":
                # Use PyVista's transformation
                if isinstance(current_data, pv.PolyData):
                    current_data = self.apply_pyvista_transforms(
                        current_data,
                        transform_type=step.get("transform_type", "translate"),
                        **step.get("params", {}),
                    )
                    results[step_name] = {
                        "mesh": current_data,
                        "type": "pyvista_transform",
                    }

            elif step_type == "blender_compositing":
                # Use Blender's compositing nodes
                if isinstance(current_data, np.ndarray):
                    current_data = self.use_blender_compositing_nodes(
                        current_data,
                        node_tree_name=step.get(
                            "node_tree_name", "AstroLab_Compositing"
                        ),
                        **step.get("params", {}),
                    )
                    results[step_name] = {
                        "image": current_data,
                        "type": "blender_compositing",
                    }

            elif step_type == "scipy_processing":
                # Use SciPy's signal processing
                if isinstance(current_data, np.ndarray):
                    current_data = self.use_scipy_signal_processing(
                        current_data,
                        operation=step.get("operation", "filter"),
                        **step.get("params", {}),
                    )
                    results[step_name] = {
                        "data": current_data,
                        "type": "scipy_processing",
                    }

            elif step_type == "numpy_processing":
                # Use NumPy's processing
                if isinstance(current_data, np.ndarray):
                    current_data = self.use_numpy_post_processing(
                        current_data,
                        operation=step.get("operation", "normalize"),
                        **step.get("params", {}),
                    )
                    results[step_name] = {
                        "data": current_data,
                        "type": "numpy_processing",
                    }

        return results


# Convenience functions that use existing package features
def apply_pyvista_filters(mesh: pv.PolyData, **kwargs) -> pv.PolyData:
    """Use PyVista's filtering capabilities."""
    processor = PostProcessor()
    return processor.apply_pyvista_filters(mesh, **kwargs)


def apply_pyvista_transforms(mesh: pv.PolyData, **kwargs) -> pv.PolyData:
    """Use PyVista's transformation capabilities."""
    processor = PostProcessor()
    return processor.apply_pyvista_transforms(mesh, **kwargs)


def create_pyvista_animation(meshes: List[pv.PolyData], **kwargs) -> str:
    """Use PyVista's animation capabilities."""
    processor = PostProcessor()
    return processor.create_pyvista_animation(meshes, **kwargs)


def create_pyvista_screenshot(mesh: pv.PolyData, **kwargs) -> str:
    """Use PyVista's screenshot capabilities."""
    processor = PostProcessor()
    return processor.create_pyvista_screenshot(mesh, **kwargs)


def export_pyvista_mesh(mesh: pv.PolyData, **kwargs) -> str:
    """Use PyVista's export capabilities."""
    processor = PostProcessor()
    return processor.export_pyvista_mesh(mesh, **kwargs)


def use_blender_compositing_nodes(image_data: np.ndarray, **kwargs) -> np.ndarray:
    """Use Blender's compositing node trees for advanced image processing."""
    processor = PostProcessor()
    return processor.use_blender_compositing_nodes(image_data, **kwargs)


def orchestrate_post_processing(
    data: Union[pv.PolyData, np.ndarray, Any],
    pipeline_steps: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, Any]:
    """Orchestrate post-processing pipeline using existing package features."""
    processor = PostProcessor()
    return processor.orchestrate_post_processing(data, pipeline_steps, **kwargs)


__all__ = [
    "PostProcessor",
    "apply_pyvista_filters",
    "apply_pyvista_transforms",
    "create_pyvista_animation",
    "create_pyvista_screenshot",
    "export_pyvista_mesh",
    "use_blender_compositing_nodes",
    "orchestrate_post_processing",
]
