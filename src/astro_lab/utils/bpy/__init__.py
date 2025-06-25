"""
AstroLab BPY (Blender Python API) Integration
============================================

Provides clean, memory-managed access to Blender Python API with ONLY BPY (no bmesh).
"""

import gc
import warnings
from contextlib import contextmanager

# 1. NumPy-Compatibility Patch (muss vor bpy-Import kommen!)
from . import numpy_compat  # noqa: F401

# 2. Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# 3. Import bpy/mathutils direkt mit Memory Management
import bpy  # type: ignore
import mathutils  # type: ignore

# 4. Import operators with memory management
from .operators import AstroLabApi
from .operators import register as al_register
from .operators import unregister as al_unregister

# Force garbage collection after Blender imports
gc.collect()


# 5. Context Managers for proper memory management - ONLY BPY
@contextmanager
def blender_memory_context():
    """
    Context manager for Blender operations with proper memory cleanup.
    ONLY uses BPY - no bmesh imports to avoid memory leaks.
    """
    if bpy is None:
        raise ImportError("Blender (bpy) not available")

    try:
        # Setup: Clear any orphaned data blocks
        try:
            bpy.ops.outliner.orphans_purge(  # type: ignore
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        except:
            pass  # Ignore if purge fails
        yield bpy
    finally:
        # Cleanup: Purge orphaned data and force garbage collection
        try:
            bpy.ops.outliner.orphans_purge(  # type: ignore
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )
        except:
            pass  # Ignore if purge fails
        gc.collect()


def bpy_object_context(mesh_obj):
    """
    Context manager for BPY mesh operations - NO BMESH.
    Only uses pure BPY API for mesh manipulation.
    """

    @contextmanager
    def _context():
        if bpy is None:
            raise ImportError("Blender (bpy) not available")

        # Store original selection state
        original_active = bpy.context.view_layer.objects.active  # type: ignore
        original_selected = [obj for obj in bpy.context.selected_objects]  # type: ignore

        try:
            # Setup: Make target object active and selected
            bpy.ops.object.select_all(action="DESELECT")  # type: ignore
            bpy.context.view_layer.objects.active = mesh_obj  # type: ignore
            mesh_obj.select_set(True)

            yield mesh_obj

        finally:
            # Cleanup: Restore original selection
            try:
                bpy.ops.object.select_all(action="DESELECT")  # type: ignore
                for obj in original_selected:
                    obj.select_set(True)
                if original_active:
                    bpy.context.view_layer.objects.active = original_active  # type: ignore
            except:
                pass  # Ignore if restoration fails
            gc.collect()

    return _context()


# 6. Lazy-import functions for submodules
def get_core():
    """Get core module with memory management."""
    from . import core

    gc.collect()  # Clean up after import
    return core


def get_grease_pencil_2d():
    """Get grease pencil 2D module with memory management."""
    from . import grease_pencil_2d

    gc.collect()  # Clean up after import
    return grease_pencil_2d


def get_grease_pencil_3d():
    """Get grease pencil 3D module with memory management."""
    from . import grease_pencil_3d

    gc.collect()  # Clean up after import
    return grease_pencil_3d


def get_advanced():
    """Get advanced module with memory management."""
    from . import advanced

    gc.collect()  # Clean up after import
    return advanced


__all__ = [
    "bpy",
    "mathutils",
    "AstroLabApi",
    "al_register",
    "al_unregister",
    "blender_memory_context",
    "bpy_object_context",  # Replaced bmesh_context
    # Removed: "bmesh_context", "edit_mode_context"
    "get_core",
    "get_grease_pencil_2d",
    "get_grease_pencil_3d",
    "get_advanced",
]


def register():
    """Register all Blender modules for Astro-Lab."""
    if al_register:
        al_register()
        gc.collect()  # Clean up after registration


def unregister():
    """Unregister all Blender modules for Astro-Lab."""
    if al_unregister:
        al_unregister()
        gc.collect()  # Clean up after unregistration


# Automatisch registrieren, wenn in Blender-Umgebung
if bpy and hasattr(bpy, "context"):
    register()
