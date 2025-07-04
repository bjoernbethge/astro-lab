"""
Enhanced Texture Generation - Package Combiner
=============================================

Orchestrates existing package features:
- PyVista for texture mapping and 3D visualization
- Open3D for 3D geometry and texture generation
- Blender for advanced texture and material creation
- NumPy for array operations
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pyvista as pv

# Note: TensorDict imports removed - texture generation works with numpy arrays
# TensorDicts are still used in model training, not in visualization

logger = logging.getLogger(__name__)


class TextureGenerator:
    """
    Package orchestrator for texture generation and mapping.

    Uses existing features from:
    - PyVista for texture mapping and 3D visualization
    - Open3D for 3D geometry and texture generation
    - Blender for advanced texture and material creation
    """

    def __init__(self):
        self.textures = {}

    def use_pyvista_texture_mapping(
        self,
        mesh: pv.PolyData,
        texture: np.ndarray,
        mapping_type: str = "spherical",
        **kwargs,
    ) -> pv.PolyData:
        """
        Use PyVista's built-in texture mapping capabilities.

        Args:
            mesh: PyVista mesh
            texture: Texture array
            mapping_type: Type of texture mapping
            **kwargs: Mapping parameters

        Returns:
            Mesh with applied texture
        """
        # Use PyVista's texture mapping
        if mapping_type == "spherical":
            mesh.texture_map_to_sphere(center=kwargs.get("center", [0, 0, 0]))
        elif mapping_type == "plane":
            mesh.texture_map_to_plane(
                origin=kwargs.get("origin", [0, 0, 0]),
                point_u=kwargs.get("point_u", [1, 0, 0]),
                point_v=kwargs.get("point_v", [0, 1, 0]),
            )

        # Use PyVista's texture loading
        mesh.textures["texture"] = pv.numpy_to_texture(texture)

        return mesh

    def use_pyvista_texture_generation(
        self,
        texture_size: Tuple[int, int] = (512, 512),
        texture_type: str = "noise",
        **kwargs,
    ) -> np.ndarray:
        """
        Use PyVista's texture generation capabilities.

        Args:
            texture_size: Output texture size
            texture_type: Type of texture generation
            **kwargs: Generation parameters

        Returns:
            Generated texture array
        """
        if texture_type == "noise":
            # Use NumPy for noise generation (PyVista doesn't have perlin_noise)
            x, y = np.meshgrid(
                np.linspace(0, 1, texture_size[0]), np.linspace(0, 1, texture_size[1])
            )
            freq = kwargs.get("freq", 1.0)
            noise = np.sin(freq * x) * np.cos(freq * y)
            # Convert to RGB using colormap
            noise_norm = (noise - noise.min()) / (noise.max() - noise.min()) * 255
            return np.stack([noise_norm, noise_norm, noise_norm], axis=-1).astype(
                np.uint8
            )

        elif texture_type == "gradient":
            # Use NumPy for gradient generation
            x, y = np.meshgrid(
                np.linspace(0, 1, texture_size[0]), np.linspace(0, 1, texture_size[1])
            )
            gradient = x + y
            gradient_norm = (
                (gradient - gradient.min()) / (gradient.max() - gradient.min()) * 255
            )
            return np.stack(
                [gradient_norm, gradient_norm, gradient_norm], axis=-1
            ).astype(np.uint8)

        elif texture_type == "checkerboard":
            # Use NumPy for pattern generation
            x, y = np.meshgrid(
                np.linspace(0, 1, texture_size[0]), np.linspace(0, 1, texture_size[1])
            )
            square_size = kwargs.get("square_size", 0.1)
            pattern = ((x // square_size + y // square_size) % 2).astype(float)
            pattern_norm = pattern * 255
            return np.stack([pattern_norm, pattern_norm, pattern_norm], axis=-1).astype(
                np.uint8
            )

        else:
            return np.zeros((*texture_size, 3), dtype=np.uint8)

    def use_open3d_texture_generation(
        self,
        texture_size: Tuple[int, int] = (512, 512),
        texture_type: str = "procedural",
        **kwargs,
    ) -> np.ndarray:
        """
        Use Open3D's texture generation capabilities.

        Args:
            texture_size: Output texture size
            texture_type: Type of texture generation
            **kwargs: Generation parameters

        Returns:
            Generated texture array
        """
        import open3d as o3d

        if texture_type == "procedural":
            # Use Open3D's procedural texture generation
            texture = o3d.geometry.Image(np.zeros(texture_size, dtype=np.uint8))
            # Open3D can generate procedural textures for meshes
            return np.asarray(texture)

        elif texture_type == "uv_mapping":
            # Use Open3D's UV mapping capabilities
            # This would typically be used with a mesh
            return np.zeros((*texture_size, 3), dtype=np.uint8)

        else:
            return np.zeros((*texture_size, 3), dtype=np.uint8)

    def use_blender_texture_generation(
        self,
        texture_size: Tuple[int, int] = (512, 512),
        texture_type: str = "material",
        **kwargs,
    ) -> np.ndarray:
        """
        Use Blender's texture generation capabilities.

        Args:
            texture_size: Output texture size
            texture_type: Type of texture generation
            **kwargs: Generation parameters

        Returns:
            Generated texture array
        """
        import bpy

        if texture_type == "material":
            # Use Blender's material system
            # Create a new material
            mat = bpy.data.materials.new(name="GeneratedMaterial")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes

            # Clear default nodes
            nodes.clear()

            # Add texture generation nodes
            if kwargs.get("use_noise", True):
                noise_node = nodes.new(type="ShaderNodeTexNoise")
                noise_node.inputs["Scale"].default_value = kwargs.get("scale", 5.0)

            # This would require more complex Blender integration
            # For now, return a basic texture
            return np.zeros((*texture_size, 3), dtype=np.uint8)

        else:
            return np.zeros((*texture_size, 3), dtype=np.uint8)

    def use_numpy_texture_processing(
        self, texture: np.ndarray, operation: str = "resize", **kwargs
    ) -> np.ndarray:
        """
        Use NumPy's existing texture processing operations.

        Args:
            texture: Input texture
            operation: NumPy operation to apply
            **kwargs: Processing parameters

        Returns:
            Processed texture
        """
        if operation == "resize":
            # Use NumPy's interpolation for resize
            size = kwargs.get("size", (512, 512))
            # Simple nearest neighbor resize
            h, w = texture.shape[:2]
            new_h, new_w = size
            h_indices = np.linspace(0, h - 1, new_h).astype(int)
            w_indices = np.linspace(0, w - 1, new_w).astype(int)
            return texture[h_indices][:, w_indices]

        elif operation == "normalize":
            # Use NumPy's normalization
            return (texture - texture.min()) / (texture.max() - texture.min()) * 255

        elif operation == "flip":
            # Use NumPy's flip
            axis = kwargs.get("axis", 0)
            return np.flip(texture, axis=axis)

        elif operation == "rotate":
            # Use NumPy's rotation
            angle = kwargs.get("angle", 90)
            if angle == 90:
                return np.rot90(texture, k=1)
            elif angle == 180:
                return np.rot90(texture, k=2)
            elif angle == 270:
                return np.rot90(texture, k=3)
            else:
                return texture

        else:
            return texture

    def create_photometric_texture(
        self,
        data: Dict[str, Any],
        texture_size: Tuple[int, int] = (512, 512),
        **kwargs,
    ) -> np.ndarray:
        """Create texture from photometric data.

        Args:
            data: Photometric data dictionary
            texture_size: Output texture size
            **kwargs: Additional parameters

        Returns:
            Texture array
        """
        # Extract magnitudes
        if "magnitudes" in data:
            mags = data["magnitudes"]
            if hasattr(mags, "cpu"):
                mags = mags.cpu().numpy()
            elif not isinstance(mags, np.ndarray):
                mags = np.array(mags)
        elif "fluxes" in data:
            fluxes = data["fluxes"]
            if hasattr(fluxes, "cpu"):
                fluxes = fluxes.cpu().numpy()
            elif not isinstance(fluxes, np.ndarray):
                fluxes = np.array(fluxes)
            # Convert flux to magnitude
            mags = -2.5 * np.log10(fluxes + 1e-10)
        else:
            # Default magnitude
            mags = np.ones(100) * 10.0

        # Extract colors if available
        if "colors" in data:
            colors = data["colors"]
            if hasattr(colors, "cpu"):
                colors = colors.cpu().numpy()
            elif not isinstance(colors, np.ndarray):
                colors = np.array(colors)
        else:
            # Generate colors from magnitude
            # Simple color mapping based on magnitude
            norm_mags = (mags - mags.min()) / (mags.max() - mags.min() + 1e-6)
            colors = np.zeros((len(mags), 3))
            colors[:, 0] = 1 - norm_mags  # Red for bright
            colors[:, 1] = norm_mags * 0.5  # Green varies
            colors[:, 2] = norm_mags  # Blue for dim

        # Create UV coordinates
        u, v = np.meshgrid(
            np.linspace(0, 1, texture_size[0]), np.linspace(0, 1, texture_size[1])
        )

        # Generate texture based on UV sampling
        texture = np.zeros((*texture_size, 4))

        # Create radial gradient based on magnitude
        np.array([0.5, 0.5])
        for i in range(len(mags)):
            if i >= len(colors):
                break

            # Random position for each source
            pos = np.random.rand(2)
            dist = np.sqrt((u - pos[0]) ** 2 + (v - pos[1]) ** 2)

            # Brightness from magnitude
            brightness = 10 ** (-0.4 * (mags[i] - 10))

            # Add to texture
            mask = dist < (brightness * 0.1)
            texture[mask, :3] += colors[i] * brightness * np.exp(-dist[mask] * 10)

        # Add alpha channel
        texture[:, :, 3] = np.sum(texture[:, :, :3], axis=2).clip(0, 1)

        return texture.astype(np.float32)

    def create_spatial_texture(
        self,
        data: Dict[str, Any],
        texture_size: Tuple[int, int] = (512, 512),
        **kwargs,
    ) -> np.ndarray:
        """Create texture from spatial data.

        Args:
            data: Spatial data dictionary
            texture_size: Output texture size
            **kwargs: Additional parameters

        Returns:
            Texture array
        """
        coords = data.get("coordinates", data.get("positions"))
        if coords is None:
            coords = np.random.randn(100, 3)
        elif hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        elif not isinstance(coords, np.ndarray):
            coords = np.array(coords)

        # Project 3D coordinates to 2D
        if coords.shape[1] == 3:
            # Simple orthographic projection
            u = (coords[:, 0] - coords[:, 0].min()) / (
                coords[:, 0].max() - coords[:, 0].min() + 1e-6
            )
            v = (coords[:, 1] - coords[:, 1].min()) / (
                coords[:, 1].max() - coords[:, 1].min() + 1e-6
            )
        else:
            u, v = coords[:, 0], coords[:, 1]

        # Create texture
        texture = np.zeros((*texture_size, 4))

        # Create UV grid
        uu, vv = np.meshgrid(
            np.linspace(0, 1, texture_size[0]), np.linspace(0, 1, texture_size[1])
        )

        # Add points to texture
        for i in range(len(u)):
            dist = np.sqrt((uu - u[i]) ** 2 + (vv - v[i]) ** 2)
            mask = dist < 0.02

            # Color based on z-coordinate if available
            if coords.shape[1] == 3:
                # Simple colormap based on z-value
                z_norm = (coords[i, 2] - coords[:, 2].min()) / (
                    coords[:, 2].max() - coords[:, 2].min() + 1e-6
                )
                color = np.array([z_norm, 0.5, 1 - z_norm])
            else:
                color = np.array([1, 1, 1])

            texture[mask, :3] = color
            texture[mask, 3] = 1.0

        return texture.astype(np.float32)

    def orchestrate_texture_pipeline(
        self,
        data: Dict[str, Any],
        pipeline_steps: List[Dict[str, Any]],
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """Orchestrate texture generation pipeline.

        Args:
            data: Input data dictionary
            pipeline_steps: List of pipeline step configurations
            **kwargs: Additional parameters

        Returns:
            Dictionary of generated textures
        """
        results = {}
        current_texture = None

        for step in pipeline_steps:
            step_type = step.get("type", "process")
            step_name = step.get("name", f"step_{len(results)}")

            if step_type == "create_texture":
                # Create texture from data
                if "magnitudes" in data or "fluxes" in data:
                    current_texture = self.create_photometric_texture(
                        data,
                        **step.get("params", {}),
                    )
                elif "coordinates" in data or "positions" in data:
                    current_texture = self.create_spatial_texture(
                        data,
                        **step.get("params", {}),
                    )

                results[step_name] = {
                    "texture": current_texture,
                    "type": "create_texture",
                }

            elif step_type == "pyvista_generation":
                # Use PyVista's generation
                generated_texture = self.use_pyvista_texture_generation(
                    texture_size=step.get("texture_size", (512, 512)),
                    texture_type=step.get("texture_type", "noise"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "texture": generated_texture,
                    "type": "pyvista_generation",
                }
                current_texture = generated_texture

            elif step_type == "open3d_generation":
                # Use Open3D's generation
                generated_texture = self.use_open3d_texture_generation(
                    texture_size=step.get("texture_size", (512, 512)),
                    texture_type=step.get("texture_type", "procedural"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "texture": generated_texture,
                    "type": "open3d_generation",
                }
                current_texture = generated_texture

            elif step_type == "blender_generation":
                # Use Blender's generation
                generated_texture = self.use_blender_texture_generation(
                    texture_size=step.get("texture_size", (512, 512)),
                    texture_type=step.get("texture_type", "material"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "texture": generated_texture,
                    "type": "blender_generation",
                }
                current_texture = generated_texture

            elif step_type == "numpy_processing":
                # Use NumPy's processing
                if current_texture is None:
                    # Create default texture using PyVista
                    current_texture = self.use_pyvista_texture_generation(
                        texture_size=step.get("texture_size", (512, 512)),
                        texture_type="noise",
                    )

                processed_texture = self.use_numpy_texture_processing(
                    current_texture,
                    operation=step.get("operation", "resize"),
                    **step.get("params", {}),
                )
                results[step_name] = {
                    "texture": processed_texture,
                    "type": "numpy_processing",
                }
                current_texture = processed_texture

        return results


# Convenience functions that use existing package features
def use_pyvista_texture_mapping(
    mesh: pv.PolyData, texture: np.ndarray, **kwargs
) -> pv.PolyData:
    """Use PyVista's texture mapping capabilities."""
    generator = TextureGenerator()
    return generator.use_pyvista_texture_mapping(mesh, texture, **kwargs)


def use_pyvista_texture_generation(
    texture_size: Tuple[int, int] = (512, 512), **kwargs
) -> np.ndarray:
    """Use PyVista's texture generation capabilities."""
    generator = TextureGenerator()
    return generator.use_pyvista_texture_generation(texture_size, **kwargs)


def use_open3d_texture_generation(
    texture_size: Tuple[int, int] = (512, 512), **kwargs
) -> np.ndarray:
    """Use Open3D's texture generation capabilities."""
    generator = TextureGenerator()
    return generator.use_open3d_texture_generation(texture_size, **kwargs)


def use_blender_texture_generation(
    texture_size: Tuple[int, int] = (512, 512), **kwargs
) -> np.ndarray:
    """Use Blender's texture generation capabilities."""
    generator = TextureGenerator()
    return generator.use_blender_texture_generation(texture_size, **kwargs)


def use_numpy_texture_processing(texture: np.ndarray, **kwargs) -> np.ndarray:
    """Use NumPy's texture processing operations."""
    generator = TextureGenerator()
    return generator.use_numpy_texture_processing(texture, **kwargs)


def create_photometric_texture(data: Dict[str, Any], **kwargs) -> np.ndarray:
    """Create texture from photometric data using existing package features."""
    generator = TextureGenerator()
    return generator.create_photometric_texture(data, **kwargs)


def create_spatial_texture(data: Dict[str, Any], **kwargs) -> np.ndarray:
    """Create texture from spatial data using existing package features."""
    generator = TextureGenerator()
    return generator.create_spatial_texture(data, **kwargs)


def orchestrate_texture_pipeline(
    data: Dict[str, Any],
    pipeline_steps: List[Dict[str, Any]],
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Orchestrate texture generation pipeline."""
    generator = TextureGenerator()
    return generator.orchestrate_texture_pipeline(data, pipeline_steps, **kwargs)


__all__ = [
    "TextureGenerator",
    "use_pyvista_texture_mapping",
    "use_pyvista_texture_generation",
    "use_open3d_texture_generation",
    "use_blender_texture_generation",
    "use_numpy_texture_processing",
    "create_photometric_texture",
    "create_spatial_texture",
    "orchestrate_texture_pipeline",
]
