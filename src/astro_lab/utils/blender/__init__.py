"""
Blender integration for astronomical visualization.

Handles compatibility issues with NumPy 2.x and provides robust fallbacks.
"""

import contextlib
import io
import os
import sys
import warnings
from typing import Any, Optional

# Suppress NumPy 2.x compatibility warnings globally
warnings.filterwarnings("ignore", message=".*NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*Unable to initialise audio.*")
warnings.filterwarnings("ignore", message=".*unable to initialise audio.*")
warnings.filterwarnings("ignore", message=".*numpy.core.multiarray.*")
warnings.filterwarnings("ignore", message=".*compiled using NumPy 1.x.*")
warnings.filterwarnings("ignore", message=".*cannot be run in NumPy.*")
warnings.filterwarnings(
    "ignore", message=".*A module that was compiled using NumPy 1.x.*"
)
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Additionally suppress all numpy-related warnings during imports
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning:numpy"

# Set numpy to ignore errors if possible
import numpy as np

if hasattr(np, "_NoValue"):
    try:
        np.seterr(all="ignore")
    except:
        pass


@contextlib.contextmanager
def _suppress_all_output():
    """Suppress both stdout and stderr completely."""
    # Save original file descriptors
    stdout_fd = os.dup(sys.stdout.fileno())
    stderr_fd = os.dup(sys.stderr.fileno())

    try:
        # Redirect to null device (complete suppression)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), sys.stdout.fileno())
            os.dup2(devnull.fileno(), sys.stderr.fileno())
        yield
    finally:
        # Restore original file descriptors
        os.dup2(stdout_fd, sys.stdout.fileno())
        os.dup2(stderr_fd, sys.stderr.fileno())
        os.close(stdout_fd)
        os.close(stderr_fd)


# Blender availability check with suppressed output
BLENDER_AVAILABLE = False
BLENDER_ERROR = None

try:
    # Suppress all output during Blender import
    with _suppress_all_output():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Try importing bpy - this can fail with numpy multiarray errors
            import bpy
            import mathutils
            from . import advanced as b3d_adv
            from . import core
            from . import grease_pencil_2d
            from . import grease_pencil_3d
            from . import lazy
            from .operators import AstroLabApi, register as al_register, unregister as al_unregister
            from . import live_tensor_bridge

        # Test basic functionality safely
        try:
            _ = bpy.context
            _ = mathutils.Vector((0, 0, 0))
            BLENDER_AVAILABLE = True
        except (AttributeError, RuntimeError):
            # Blender modules loaded but context not available (headless mode)
            BLENDER_AVAILABLE = True

except ImportError as e:
    BLENDER_ERROR = f"Blender modules not available: {e}"
    BLENDER_AVAILABLE = False
    bpy = None
    mathutils = None
    b3d_adv = None
    core = None
    grease_pencil_2d = None
    grease_pencil_3d = None
    lazy = None
    AstroLabApi = None
    al_register = None
    al_unregister = None
    live_tensor_bridge = None

except Exception as e:
    # Handle numpy multiarray import errors gracefully
    if "numpy.core.multiarray" in str(e):
        BLENDER_ERROR = f"Blender-NumPy compatibility issue (NumPy 2.x): {e}. Blender features disabled."
    else:
        BLENDER_ERROR = f"Blender initialization failed: {e}"
    BLENDER_AVAILABLE = False
    # Make sure all potential imports are None
    bpy = None
    mathutils = None
    b3d_adv = None
    core = None
    grease_pencil_2d = None
    grease_pencil_3d = None
    lazy = None
    AstroLabApi = None
    al_register = None
    al_unregister = None
    live_tensor_bridge = None


# Base exports - always available
__all__ = [
    "bpy",
    "BLENDER_AVAILABLE",
    "BLENDER_ERROR",
    "AstroLabApi",
    "core",
    "advanced",
    "grease_pencil_2d",
    "grease_pencil_3d",
    "lazy",
    "live_tensor_bridge",
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
