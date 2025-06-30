"""
post-processing for astronomical data visualization.

This module provides post-processing effects for cosmic web structures,
including scientific color mapping, data enhancement, and visualization optimization.
"""

# pyright: reportAttributeAccessIssue=false
# pyright: reportGeneralTypeIssues=false

import os
import warnings


from .. import (
    bpy,
)

# Set environment variable for NumPy 2.x compatibility with bpy
os.environ["NUMPY_EXPERIMENTAL_ARRAY_API"] = "1"

# Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# NumPy and mathutils are already imported above

# Author: Bjoern Bethge


class PostProcessingSuite:
    """post-processing for astronomical visualization"""

    def __init__(self, scene_name: str = "AstroScene"):
        self.scene_name = scene_name
        self.scene = bpy.context.scene

    def setup_compositor(self) -> None:
        """Setup compositor for post-processing."""
        self.scene.use_nodes = True
        tree = self.scene.node_tree

        # Clear existing nodes
        tree.nodes.clear()

        # Add render layers
        render_layers = tree.nodes.new("CompositorNodeRLayers")
        render_layers.location = (0, 0)

        # Add composite output
        composite = tree.nodes.new("CompositorNodeComposite")
        composite.location = (1200, 0)

        # Connect basic render
        tree.links.new(render_layers.outputs["Image"], composite.inputs["Image"])

    def add_lens_flare(
        self, flare_type: str = "stellar", intensity: float = 1.0
    ) -> None:
        """
        Add lens flare effect to the scene.

        Args:
            flare_type: Type of flare ('stellar', 'nebula', 'galactic')
            intensity: Flare intensity
        """
        tree = self.scene.node_tree

        # Get render layers
        render_layers = tree.nodes.get("Render Layers")
        if not render_layers:
            return

        # Add lens distortion for flare
        lens_distortion = tree.nodes.new("CompositorNodeLensDist")
        lens_distortion.location = (200, 0)
        lens_distortion.inputs["Distort"].default_value = 0.02 * intensity

        # Add glow effect
        glow = tree.nodes.new("CompositorNodeGlare")
        glow.location = (400, 0)
        glow.glare_type = "FOG_GLOW"
        glow.quality = "HIGH"
        glow.size = 9
        glow.mix = 0.3 * intensity

        # Add color correction for flare type
        color_correction = tree.nodes.new("CompositorNodeColorCorrection")
        color_correction.location = (600, 0)

        if flare_type == "stellar":
            color_correction.master_saturation = 1.2
            color_correction.master_gain = 1.1
        elif flare_type == "nebula":
            color_correction.master_saturation = 1.5
            color_correction.master_gain = 1.3
        elif flare_type == "galactic":
            color_correction.master_saturation = 0.8
            color_correction.master_gain = 1.0

        # Connect nodes
        tree.links.new(render_layers.outputs["Image"], lens_distortion.inputs["Image"])
        tree.links.new(lens_distortion.outputs["Image"], glow.inputs["Image"])
        tree.links.new(glow.outputs["Image"], color_correction.inputs["Image"])

        # Update composite connection
        composite = tree.nodes.get("Composite")
        if composite:
            tree.links.new(color_correction.outputs["Image"], composite.inputs["Image"])

    def add_vignette(self, intensity: float = 0.3, radius: float = 0.8) -> None:
        """
        Add vignette effect for artistic framing.

        Args:
            intensity: Vignette darkness (0-1)
            radius: Vignette radius (0-1)
        """
        tree = self.scene.node_tree

        # Get current image input
        current_input = self._get_current_image_input()
        if not current_input:
            return

        # Add vignette using radial gradient
        radial = tree.nodes.new("CompositorNodeEllipseMask")
        radial.location = (800, 200)
        radial.width = radius * 2
        radial.height = radius * 2

        # Invert mask for vignette
        invert = tree.nodes.new("CompositorNodeInvert")
        invert.location = (1000, 200)

        # Mix with original image
        mix = tree.nodes.new("CompositorNodeMixRGB")
        mix.location = (1200, 0)
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = intensity

        # Connect nodes
        tree.links.new(radial.outputs["Mask"], invert.inputs["Color"])
        tree.links.new(current_input, mix.inputs["Image1"])
        tree.links.new(invert.outputs["Color"], mix.inputs["Image2"])

        # Update composite connection
        self._update_composite_connection(mix.outputs["Image"])

    def add_color_grading(self, style: str = "cinematic") -> None:
        """
        Add color grading for different moods.

        Args:
            style: Grading style ('cinematic', 'warm', 'cool', 'dramatic', 'dreamy')
        """
        tree = self.scene.node_tree

        # Get current image input
        current_input = self._get_current_image_input()
        if not current_input:
            return

        # Add color correction
        color_correction = tree.nodes.new("CompositorNodeColorCorrection")
        color_correction.location = (1000, 0)

        # Apply style-specific settings
        if style == "cinematic":
            color_correction.master_contrast = 1.2
            color_correction.master_saturation = 0.9
            color_correction.master_gain = 1.1
            color_correction.master_lift = 0.05

        elif style == "warm":
            color_correction.master_saturation = 1.3
            color_correction.master_gain = 1.2
            color_correction.master_lift = 0.1
            color_correction.master_gamma = 0.9

        elif style == "cool":
            color_correction.master_saturation = 1.1
            color_correction.master_gain = 0.9
            color_correction.master_lift = -0.05
            color_correction.master_gamma = 1.1

        elif style == "dramatic":
            color_correction.master_contrast = 1.5
            color_correction.master_saturation = 1.4
            color_correction.master_gain = 1.3
            color_correction.master_lift = 0.15

        elif style == "dreamy":
            color_correction.master_contrast = 0.8
            color_correction.master_saturation = 1.2
            color_correction.master_gain = 1.0
            color_correction.master_lift = 0.05

        # Connect nodes
        tree.links.new(current_input, color_correction.inputs["Image"])

        # Update composite connection
        self._update_composite_connection(color_correction.outputs["Image"])

    def add_motion_blur(self, samples: int = 32, shutter: float = 0.5) -> None:
        """
        Add motion blur for dynamic scenes.

        Args:
            samples: Motion blur samples
            shutter: Shutter time (0-1)
        """
        # Enable motion blur in render settings
        self.scene.render.use_motion_blur = True
        self.scene.render.motion_blur_shutter = shutter

        # Set motion blur samples
        if hasattr(self.scene.render, "motion_blur_samples"):
            self.scene.render.motion_blur_samples = samples

    def add_depth_of_field(
        self, focus_distance: float = 10.0, f_stop: float = 2.8
    ) -> None:
        """
        Add depth of field effect.

        Args:
            focus_distance: Focus distance
            f_stop: Aperture f-stop
        """
        # Get active camera
        camera = self.scene.camera
        if not camera:
            return

        # Enable depth of field
        camera.data.dof.use_dof = True
        camera.data.dof.focus_distance = focus_distance
        camera.data.dof.aperture_fstop = f_stop

    def add_star_glow(self, glow_intensity: float = 1.0, glow_size: int = 9) -> None:
        """
        Add glow effect specifically for stars.

        Args:
            glow_intensity: Glow intensity
            glow_size: Glow size
        """
        tree = self.scene.node_tree

        # Get current image input
        current_input = self._get_current_image_input()
        if not current_input:
            return

        # Add glow effect
        glow = tree.nodes.new("CompositorNodeGlare")
        glow.location = (800, 0)
        glow.glare_type = "STREAKS"
        glow.quality = "HIGH"
        glow.size = glow_size
        glow.mix = 0.4 * glow_intensity
        glow.angle_offset = 0.0

        # Add second glow for cross pattern
        glow2 = tree.nodes.new("CompositorNodeGlare")
        glow2.location = (1000, 0)
        glow2.glare_type = "STREAKS"
        glow2.quality = "HIGH"
        glow2.size = glow_size // 2
        glow2.mix = 0.2 * glow_intensity
        glow2.angle_offset = 1.5708  # 90 degrees

        # Mix glows
        mix = tree.nodes.new("CompositorNodeMixRGB")
        mix.location = (1200, 0)
        mix.blend_type = "ADD"
        mix.inputs["Fac"].default_value = 1.0

        # Connect nodes
        tree.links.new(current_input, glow.inputs["Image"])
        tree.links.new(current_input, glow2.inputs["Image"])
        tree.links.new(glow.outputs["Image"], mix.inputs["Image1"])
        tree.links.new(glow2.outputs["Image"], mix.inputs["Image2"])

        # Update composite connection
        self._update_composite_connection(mix.outputs["Image"])

    def apply_cinematic_preset(self) -> None:
        """Apply cinematic post-processing preset."""
        self.setup_compositor()
        self.add_lens_flare("stellar", 0.8)
        self.add_vignette(0.4, 0.7)
        self.add_color_grading("cinematic")
        self.add_star_glow(1.2, 11)
        self.add_depth_of_field(15.0, 2.0)

    def apply_dramatic_preset(self) -> None:
        """Apply dramatic post-processing preset."""
        self.setup_compositor()
        self.add_lens_flare("nebula", 1.2)
        self.add_vignette(0.6, 0.6)
        self.add_color_grading("dramatic")
        self.add_star_glow(1.5, 13)
        self.add_depth_of_field(10.0, 1.4)

    def apply_dreamy_preset(self) -> None:
        """Apply dreamy post-processing preset."""
        self.setup_compositor()
        self.add_lens_flare("galactic", 0.6)
        self.add_vignette(0.2, 0.8)
        self.add_color_grading("dreamy")
        self.add_star_glow(0.8, 7)
        self.add_depth_of_field(20.0, 4.0)

    def _get_current_image_input(self):
        """Get current image input for compositor."""
        tree = self.scene.node_tree
        composite = tree.nodes.get("Composite")
        if composite and composite.inputs["Image"].links:
            return composite.inputs["Image"].links[0].from_socket
        return None

    def _update_composite_connection(self, output_socket):
        """Update composite node connection."""
        tree = self.scene.node_tree
        composite = tree.nodes.get("Composite")
        if composite:
            # Remove existing connection
            if composite.inputs["Image"].links:
                tree.links.remove(composite.inputs["Image"].links[0])
            # Add new connection
            tree.links.new(output_socket, composite.inputs["Image"])


class ArtisticFilters:
    """Artistic filters for astronomical visualization"""

    @staticmethod  # type: ignore
    def add_film_grain(intensity: float = 0.1) -> None:  # type: ignore
        """Add film grain effect."""
        scene = bpy.context.scene
        tree = scene.node_tree

        # Add noise texture
        noise = tree.nodes.new("CompositorNodeTexNoise")
        noise.location = (600, 200)
        noise.inputs["Scale"].default_value = 100.0
        noise.inputs["Detail"].default_value = 2.0

        # Mix with current image
        mix = tree.nodes.new("CompositorNodeMixRGB")
        mix.location = (800, 0)
        mix.blend_type = "OVERLAY"
        mix.inputs["Fac"].default_value = intensity

        # Connect
        current_input = PostProcessingSuite._get_current_image_input_static(scene)
        if current_input:
            tree.links.new(current_input, mix.inputs["Image1"])
            tree.links.new(noise.outputs["Color"], mix.inputs["Image2"])
            PostProcessingSuite._update_composite_connection_static(
                scene, mix.outputs["Image"]
            )

    @staticmethod  # type: ignore
    def add_chromatic_aberration(intensity: float = 0.02) -> None:  # type: ignore
        """Add chromatic aberration effect."""
        scene = bpy.context.scene
        tree = scene.node_tree

        # Add lens distortion for red channel
        lens_red = tree.nodes.new("CompositorNodeLensDist")
        lens_red.location = (600, 100)
        lens_red.inputs["Distort"].default_value = intensity

        # Add lens distortion for blue channel
        lens_blue = tree.nodes.new("CompositorNodeLensDist")
        lens_blue.location = (600, -100)
        lens_blue.inputs["Distort"].default_value = -intensity

        # Separate RGB
        separate = tree.nodes.new("CompositorNodeSepRGBA")
        separate.location = (400, 0)

        # Combine RGB
        combine = tree.nodes.new("CompositorNodeCombRGBA")
        combine.location = (800, 0)

        # Connect
        current_input = PostProcessingSuite._get_current_image_input_static(scene)
        if current_input:
            tree.links.new(current_input, separate.inputs["Image"])
            tree.links.new(separate.outputs["R"], lens_red.inputs["Image"])
            tree.links.new(separate.outputs["B"], lens_blue.inputs["Image"])
            tree.links.new(separate.outputs["G"], combine.inputs["G"])
            tree.links.new(lens_red.outputs["Image"], combine.inputs["R"])
            tree.links.new(lens_blue.outputs["Image"], combine.inputs["B"])
            PostProcessingSuite._update_composite_connection_static(
                scene, combine.outputs["Image"]
            )

    @staticmethod  # type: ignore
    def _get_current_image_input_static(scene):  # type: ignore
        """Static version of _get_current_image_input."""
        tree = scene.node_tree
        composite = tree.nodes.get("Composite")
        if composite and composite.inputs["Image"].links:
            return composite.inputs["Image"].links[0].from_socket
        return None

    @staticmethod  # type: ignore
    def _update_composite_connection_static(scene, output_socket):  # type: ignore
        """Static version of _update_composite_connection."""
        tree = scene.node_tree
        composite = tree.nodes.get("Composite")
        if composite:
            if composite.inputs["Image"].links:
                tree.links.remove(composite.inputs["Image"].links[0])
            tree.links.new(output_socket, composite.inputs["Image"])
