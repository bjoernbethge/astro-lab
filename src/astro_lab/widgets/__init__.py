"""AstroLab Widgets - Interactive astronomical visualization."""

from .astro_lab import AstroLabWidget, AstroPipeline, BlenderUtils

__all__ = [
    "AstroLabWidget", 
    "AstroPipeline", 
    "BlenderUtils"
]

# Blender API directly available on widget:
# - widget.ops: Blender Operations (bpy.ops)
# - widget.data: Blender Data (bpy.data) 
# - widget.context: Blender Context (bpy.context)
# - widget.scene: Current Scene
# - widget.utils: Blender Utilities (BlenderUtils)
