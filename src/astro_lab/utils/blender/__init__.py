"""
Blender Integration Module
=========================

Centralized Blender import management.
Ensures NumPy compatibility and prevents memory leaks.
"""

import os
import warnings
import gc

# 1. NumPy-Compatibility Patch (muss vor bpy-Import kommen!)
from . import numpy_compat  # noqa: F401

# 2. Suppress numpy warnings that occur with bpy
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")

# 3. Import bpy/mathutils direkt mit Memory Management
try:
    import bpy
    import mathutils
    
    # Force garbage collection after Blender imports
    gc.collect()
    
except ImportError as e:
    print(f"Blender modules not available: {e}")
    bpy = None
    mathutils = None

# 4. Lazy-Import-Funktionen für Submodule (ohne lazy)
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

def get_live_tensor_bridge():
    """Get live tensor bridge module with memory management."""
    from . import live_tensor_bridge
    gc.collect()  # Clean up after import
    return live_tensor_bridge

def get_advanced():
    """Get advanced module with memory management."""
    from . import advanced
    gc.collect()  # Clean up after import
    return advanced

# 5. Import operators with memory management
try:
    from .operators import AstroLabApi, register as al_register, unregister as al_unregister
    gc.collect()  # Clean up after import
except ImportError as e:
    print(f"Blender operators not available: {e}")
    AstroLabApi = None
    al_register = None
    al_unregister = None

__all__ = [
    "bpy",
    "mathutils",
    "AstroLabApi",
    "al_register",
    "al_unregister",
    "get_core",
    "get_grease_pencil_2d",
    "get_grease_pencil_3d",
    "get_live_tensor_bridge",
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
