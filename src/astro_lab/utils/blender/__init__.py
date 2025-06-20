"""
Blender Integration Module
=========================

Centralized Blender availability check and import management.
Prevents memory leaks and numpy multiarray issues.
"""

import warnings
import sys
from contextlib import contextmanager
from typing import Optional, Any

# Global state to prevent multiple imports
_bpy = None
_mathutils = None
_blender_checked = False
_blender_available = False
_blender_error = None

@contextmanager
def _suppress_all_output():
    """Suppress all output during import."""
    import os
    import sys
    
    # Redirect stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create null devices
    null_fd = os.open(os.devnull, os.O_RDWR)
    old_stdout_fd = os.dup(sys.stdout.fileno())
    old_stderr_fd = os.dup(sys.stderr.fileno())
    
    sys.stdout = os.fdopen(null_fd, 'w')
    sys.stderr = os.fdopen(null_fd, 'w')
    
    try:
        yield
    finally:
        # Restore stdout/stderr
        sys.stdout.close()
        sys.stderr.close()
        os.dup2(old_stdout_fd, sys.stdout.fileno())
        os.dup2(old_stderr_fd, sys.stderr.fileno())
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

def _safe_import_blender() -> tuple[Optional[Any], Optional[Any], bool, Optional[str]]:
    """
    Safely import Blender modules once.
    
    Returns:
        Tuple of (bpy, mathutils, available, error_message)
    """
    global _bpy, _mathutils, _blender_checked, _blender_available, _blender_error
    
    if _blender_checked:
        return _bpy, _mathutils, _blender_available, _blender_error
    
    try:
        # Suppress all output and warnings during import
        with _suppress_all_output():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Import bpy and mathutils
                import bpy as bpy_module
                import mathutils as mathutils_module
                
                # Test basic functionality
                _ = bpy_module.context
                _ = mathutils_module.Vector((0, 0, 0))
                
                _bpy = bpy_module
                _mathutils = mathutils_module
                _blender_available = True
                _blender_error = None
                
    except ImportError as e:
        _bpy = None
        _mathutils = None
        _blender_available = False
        _blender_error = f"Blender modules not available: {e}"
    except Exception as e:
        _bpy = None
        _mathutils = None
        _blender_available = False
        _blender_error = f"Blender import failed: {e}"
    
    _blender_checked = True
    return _bpy, _mathutils, _blender_available, _blender_error

# Initialize Blender modules once
bpy, mathutils, BLENDER_AVAILABLE, BLENDER_ERROR = _safe_import_blender()

# Import other modules only if Blender is available
if BLENDER_AVAILABLE:
    try:
        from . import advanced as b3d_adv
        from . import core
        from . import grease_pencil_2d
        from . import grease_pencil_3d
        from . import lazy
        from .operators import AstroLabApi, register as al_register, unregister as al_unregister
        from . import live_tensor_bridge
    except ImportError as e:
        BLENDER_ERROR = f"Blender submodules not available: {e}"
        BLENDER_AVAILABLE = False
        b3d_adv = None
        core = None
        grease_pencil_2d = None
        grease_pencil_3d = None
        lazy = None
        AstroLabApi = None
        al_register = None
        al_unregister = None
        live_tensor_bridge = None
else:
    # Set all to None if Blender is not available
    b3d_adv = None
    core = None
    grease_pencil_2d = None
    grease_pencil_3d = None
    lazy = None
    AstroLabApi = None
    al_register = None
    al_unregister = None
    live_tensor_bridge = None

# Export availability status
__all__ = [
    "BLENDER_ERROR", 
    "bpy",
    "mathutils",
    "AstroLabApi",
    "al_register",
    "al_unregister"
]

def register():
    """Register all Blender modules for Astro-Lab."""
    if BLENDER_AVAILABLE and al_register:
        al_register()

def unregister():
    """Unregister all Blender modules for Astro-Lab."""
    if BLENDER_AVAILABLE and al_unregister:
        al_unregister()

# Automatically register when the module is loaded inside Blender
# Check for 'bpy.context' to ensure it's not a headless/background run
if BLENDER_AVAILABLE and hasattr(bpy, "context"):
    register()
