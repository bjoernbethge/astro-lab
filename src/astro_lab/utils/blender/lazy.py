"""
Centralized Blender lazy loading utility.
Prevents numpy multiarray issues by deferring bpy imports until actually needed.
"""

import warnings
from typing import Optional, Tuple, Any

# Global state
_bpy = None
_mathutils = None
_blender_checked = False
_blender_available = False
_blender_error = None


def get_blender_modules() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Get Blender modules with lazy loading.
    
    Returns:
        Tuple of (bpy, mathutils) or (None, None) if not available
    """
    global _bpy, _mathutils, _blender_checked, _blender_available, _blender_error
    
    if _blender_checked:
        return _bpy, _mathutils
    
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            import bpy as _bpy_module
            import mathutils as _mathutils_module
            
        _bpy = _bpy_module
        _mathutils = _mathutils_module
        _blender_available = True
        _blender_error = None
        
    except Exception as e:
        _bpy = None
        _mathutils = None
        _blender_available = False
        _blender_error = str(e)
    
    _blender_checked = True
    return _bpy, _mathutils


def is_blender_available() -> bool:
    """Check if Blender is available."""
    bpy, _ = get_blender_modules()
    return bpy is not None


def get_blender_error() -> Optional[str]:
    """Get Blender import error if any."""
    get_blender_modules()  # Ensure check has been performed
    return _blender_error 