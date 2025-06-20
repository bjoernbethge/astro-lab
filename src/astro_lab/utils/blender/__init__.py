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

except Exception as e:
    # Handle numpy multiarray import errors gracefully
    if "numpy.core.multiarray" in str(e):
        BLENDER_ERROR = f"Blender-NumPy compatibility issue (NumPy 2.x): {e}. Blender features disabled."
    else:
        BLENDER_ERROR = f"Blender initialization failed: {e}"
    BLENDER_AVAILABLE = False

# Module imports with fallback handling
_CORE_AVAILABLE = False
_GREASE_PENCIL_AVAILABLE = False
_VIEWPORT_AVAILABLE = False
_ADVANCED_AVAILABLE = False

if BLENDER_AVAILABLE:
    # Core Blender utilities
    try:
        with _suppress_all_output():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from .core import *
        _CORE_AVAILABLE = True
    except (ImportError, AttributeError, RuntimeError):
        _CORE_AVAILABLE = False
    except Exception:
        _CORE_AVAILABLE = False

    # Grease Pencil utilities
    try:
        with _suppress_all_output():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from .grease_pencil_2d import GreasePencil2DPlotter
                from .grease_pencil_3d import GreasePencil3DPlotter
        _GREASE_PENCIL_AVAILABLE = True
    except (ImportError, AttributeError):
        _GREASE_PENCIL_AVAILABLE = False
    except Exception:
        _GREASE_PENCIL_AVAILABLE = False

    # Viewport capture removed - not needed for prototyping
    _VIEWPORT_AVAILABLE = False

    # Advanced Blender features
    try:
        with _suppress_all_output():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from .advanced import ADVANCED_AVAILABLE as _ADV_CHECK

                if _ADV_CHECK:
                    from .advanced import *

                    _ADVANCED_AVAILABLE = True
                else:
                    _ADVANCED_AVAILABLE = False
    except (ImportError, AttributeError):
        _ADVANCED_AVAILABLE = False
    except Exception:
        _ADVANCED_AVAILABLE = False

# Base exports - always available
__all__ = [
    "BLENDER_AVAILABLE",
    "BLENDER_ERROR",
    "get_blender_info",
    "check_blender_compatibility",
]

# Conditional exports based on successful imports
if _CORE_AVAILABLE:
    __all__.extend(
        [
            "AstroPlotter",
            "FuturisticAstroPlotter",
            "GeometryNodesVisualizer",
            "GreasePencilPlotter",
            "reset_scene",
            "normalize_scene",
            "setup_scene",
            "create_material",
            "create_light",
            "setup_lighting_preset",
            "create_camera",
            "animate_camera",
            "create_camera_path",
            "create_astro_object",
            "setup_astronomical_scene",
            "render_scene",
            "setup_render_settings",
        ]
    )

if _GREASE_PENCIL_AVAILABLE:
    __all__.extend(
        [
            "GreasePencil2DPlotter",
            "GreasePencil3DPlotter",
        ]
    )

if _ADVANCED_AVAILABLE:
    __all__.extend(
        [
            "ProceduralAstronomy",
            "AstronomicalMaterials",
            "VolumetricAstronomy",
            "GravitationalSimulation",
        ]
    )


def get_blender_info() -> dict[str, Any]:
    """Get detailed information about Blender availability and capabilities."""
    info = {
        "available": BLENDER_AVAILABLE,
        "error": BLENDER_ERROR,
        "modules": {
            "core": _CORE_AVAILABLE,
            "grease_pencil": _GREASE_PENCIL_AVAILABLE,
            "viewport": _VIEWPORT_AVAILABLE,
            "advanced": _ADVANCED_AVAILABLE,
        },
    }

    if BLENDER_AVAILABLE:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                info["version"] = bpy.app.version_string
                info["build"] = (
                    bpy.app.build_platform.decode()
                    if hasattr(bpy.app, "build_platform")
                    else "unknown"
                )
        except:
            info["version"] = "unknown"
            info["build"] = "unknown"

    return info


def check_blender_compatibility() -> tuple[bool, list[str]]:
    """Check Blender compatibility and return status with any issues."""
    issues = []

    if not BLENDER_AVAILABLE:
        issues.append(f"Blender not available: {BLENDER_ERROR}")
        return False, issues

    # Check core module availability
    if not _CORE_AVAILABLE:
        issues.append("Core Blender utilities not available")

    if not _GREASE_PENCIL_AVAILABLE:
        issues.append("Grease Pencil utilities not available")

    if not _ADVANCED_AVAILABLE:
        issues.append("Advanced Blender features not available")

    # Basic functionality test
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            test_vec = mathutils.Vector((1, 0, 0))
            if len(test_vec) != 3:
                issues.append("mathutils functionality test failed")
    except Exception as e:
        issues.append(f"mathutils test failed: {e}")

    return len(issues) == 0, issues
