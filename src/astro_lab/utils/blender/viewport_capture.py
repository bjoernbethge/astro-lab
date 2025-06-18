"""
Direct 3D Viewport pixel data extraction using modern GPU module.

This module provides access to raw viewport framebuffer data for real-time
analysis, computer vision, or data extraction without rendering overhead.
Based on gpu.types.GPUOffScreen.draw_view3d() for efficient pixel access.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import bpy
    import gpu
    import mathutils
    from gpu_extras.batch import batch_for_shader

    BLENDER_AVAILABLE = True

    # Test GPU access
    try:
        gpu.state.active_framebuffer_get()
        GPU_AVAILABLE = True
    except:
        GPU_AVAILABLE = False

except ImportError:
    BLENDER_AVAILABLE = False
    GPU_AVAILABLE = False
    bpy = None
    gpu = None
    mathutils = None


def capture_viewport_pixels(
    width: int = 512, height: int = 512, format: str = "RGBA8"
) -> Optional[np.ndarray]:
    """
    Capture raw pixel data from current 3D viewport using OffScreen buffer.

    This creates an OffScreen buffer and renders the current view to it,
    then extracts the pixel data directly. This is the modern approach
    that replaces deprecated bgl.glReadPixels().

    Args:
        width: Width of capture buffer
        height: Height of capture buffer
        format: GPU texture format ('RGBA8', 'RGBA16F', etc.)

    Returns:
        numpy array with shape (height, width, channels) or None on error
    """
    if not BLENDER_AVAILABLE or not GPU_AVAILABLE:
        print("⚠️ Blender GPU not available")
        return None

    try:
        # Get current context
        context = bpy.context
        scene = context.scene
        view_layer = context.view_layer

        # Find active 3D viewport
        area = None
        space = None
        region = None

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                for region in area.regions:
                    if region.type == "WINDOW":
                        break
                break

        if not (area and space and region):
            print("❌ No active 3D viewport found")
            return None

        # Get view and projection matrices
        region_3d = space.region_3d
        view_matrix = region_3d.view_matrix.copy()
        projection_matrix = region_3d.window_matrix.copy()

        # Create offscreen buffer
        offscreen = gpu.types.GPUOffScreen(width, height, format=format)

        with offscreen.bind():
            # Clear buffer
            offscreen.clear()

            # Draw current view
            offscreen.draw_view3d(
                scene,
                view_layer,
                space,
                region,
                view_matrix,
                projection_matrix,
                do_color_management=False,
                draw_background=True,
            )

            # Read pixels from color attachment
            fb = gpu.state.active_framebuffer_get()
            pixel_data = fb.read_color(0, 0, width, height, 4, 0, "FLOAT")

        # Convert to numpy array
        pixels = np.array(pixel_data, dtype=np.float32)
        pixels = pixels.reshape((height, width, 4))

        # Flip Y axis (OpenGL convention)
        pixels = np.flipud(pixels)

        # Free offscreen buffer
        offscreen.free()

        return pixels

    except Exception as e:
        print(f"❌ Viewport pixel capture failed: {e}")
        return None


def capture_depth_buffer(width: int = 512, height: int = 512) -> Optional[np.ndarray]:
    """
    Capture depth buffer from current 3D viewport.

    Args:
        width: Width of capture buffer
        height: Height of capture buffer

    Returns:
        numpy array with depth values (height, width) or None on error
    """
    if not BLENDER_AVAILABLE or not GPU_AVAILABLE:
        return None

    try:
        # Similar to capture_viewport_pixels but for depth
        context = bpy.context
        scene = context.scene
        view_layer = context.view_layer

        # Find active 3D viewport
        area = None
        space = None
        region = None

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                for region in area.regions:
                    if region.type == "WINDOW":
                        break
                break

        if not (area and space and region):
            return None

        region_3d = space.region_3d
        view_matrix = region_3d.view_matrix.copy()
        projection_matrix = region_3d.window_matrix.copy()

        # Create offscreen with depth
        offscreen = gpu.types.GPUOffScreen(width, height)

        with offscreen.bind():
            offscreen.clear()
            offscreen.draw_view3d(
                scene, view_layer, space, region, view_matrix, projection_matrix
            )

            # Read depth data
            fb = gpu.state.active_framebuffer_get()
            depth_data = fb.read_depth(0, 0, width, height)

        # Convert to numpy
        depth = np.array(depth_data, dtype=np.float32)
        depth = depth.reshape((height, width))
        depth = np.flipud(depth)

        offscreen.free()
        return depth

    except Exception as e:
        print(f"❌ Depth capture failed: {e}")
        return None


def get_viewport_info() -> Dict[str, any]:
    """Get detailed viewport state information."""
    if not BLENDER_AVAILABLE:
        return {}

    try:
        context = bpy.context

        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                region_3d = space.region_3d

                for region in area.regions:
                    if region.type == "WINDOW":
                        return {
                            "viewport_size": (region.width, region.height),
                            "view_location": list(region_3d.view_location),
                            "view_rotation": list(region_3d.view_rotation),
                            "view_distance": region_3d.view_distance,
                            "is_perspective": region_3d.is_perspective,
                            "is_orthographic": region_3d.is_orthographic_side_view,
                            "lens": getattr(space, "lens", None),
                            "shading_type": space.shading.type,
                            "clip_start": getattr(space, "clip_start", 0.01),
                            "clip_end": getattr(space, "clip_end", 1000.0),
                        }

        return {}

    except Exception as e:
        print(f"⚠️ Could not get viewport info: {e}")
        return {}


def analyze_viewport_content(
    pixel_data: np.ndarray, depth_data: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Analyze viewport pixel data for astronomical objects or features.

    Args:
        pixel_data: RGBA pixel array from capture_viewport_pixels()
        depth_data: Optional depth buffer from capture_depth_buffer()

    Returns:
        Analysis results including detected features, statistics, etc.
    """
    if pixel_data is None:
        return {}

    try:
        analysis = {
            "pixel_stats": {
                "mean_color": np.mean(pixel_data[:, :, :3], axis=(0, 1)).tolist(),
                "brightness": np.mean(pixel_data[:, :, :3]),
                "contrast": np.std(pixel_data[:, :, :3]),
                "alpha_coverage": np.mean(pixel_data[:, :, 3]),
            }
        }

        # Convert to grayscale for analysis
        gray = np.mean(pixel_data[:, :, :3], axis=2)

        # Basic feature detection
        analysis["features"] = {
            "bright_spots": int(np.sum(gray > 0.8)),
            "dark_regions": int(np.sum(gray < 0.2)),
            "edge_density": float(
                np.mean(np.abs(np.gradient(gray)[0]) + np.abs(np.gradient(gray)[1]))
            ),
        }

        # Depth analysis if available
        if depth_data is not None:
            analysis["depth_stats"] = {
                "near_plane": float(np.min(depth_data)),
                "far_plane": float(np.max(depth_data)),
                "mean_depth": float(np.mean(depth_data)),
                "depth_variance": float(np.var(depth_data)),
            }

        return analysis

    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")
        return {}


# Legacy compatibility functions
def capture_viewport_screenshot(output_path: Optional[str] = None) -> bool:
    """Legacy function - captures pixels and saves as image."""
    pixels = capture_viewport_pixels()
    if pixels is None:
        return False

    if output_path is None:
        output_path = str(Path(tempfile.gettempdir()) / "viewport_capture.png")

    try:
        # Convert float to uint8
        img_data = (pixels * 255).astype(np.uint8)

        # Use PIL if available, otherwise basic save
        try:
            from PIL import Image

            img = Image.fromarray(img_data, "RGBA")
            img.save(output_path)
        except ImportError:
            print("⚠️ PIL not available, basic save not implemented")
            return False

        print(f"✅ Viewport screenshot saved: {output_path}")
        return True

    except Exception as e:
        print(f"❌ Screenshot save failed: {e}")
        return False


# Convenience functions
def get_live_viewport_stream(
    width: int = 256, height: int = 256, include_depth: bool = False
) -> Dict[str, np.ndarray]:
    """
    Get live viewport data stream for real-time processing.

    This is optimized for repeated calls and minimal overhead.
    """
    result = {}

    pixels = capture_viewport_pixels(width, height)
    if pixels is not None:
        result["color"] = pixels

    if include_depth:
        depth = capture_depth_buffer(width, height)
        if depth is not None:
            result["depth"] = depth

    return result


__all__ = [
    "capture_viewport_pixels",
    "capture_depth_buffer",
    "get_viewport_info",
    "analyze_viewport_content",
    "get_live_viewport_stream",
    "capture_viewport_screenshot",  # Legacy
    "BLENDER_AVAILABLE",
    "GPU_AVAILABLE",
]
