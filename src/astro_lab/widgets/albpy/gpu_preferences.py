"""Blender viewport/compositor GPU backend when using the PyPI ``bpy`` module (no ``binary_path``).

Loaded from ``albpy`` package init **before** node groups and operators so preferences apply
early. See ``astro_lab.widgets.albpy`` docstring for import order.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

import bpy

logger = logging.getLogger(__name__)

GpuBackendName = Literal["VULKAN", "OPENGL", "METAL"]


def prefer_gpu_backend(
    backend: GpuBackendName | None = None,
    *,
    preferred_device: str | None = None,
) -> str | None:
    """Set :attr:`bpy.types.PreferencesSystem.gpu_backend` (and optional device).

    Blender API: https://docs.blender.org/api/current/bpy.types.PreferencesSystem.html

    Intended for ``bpy`` installed as a wheel: avoids relying on ``bpy.app.binary_path``.
    On Apple Silicon, ``METAL`` is often appropriate; on Windows/Linux with Vulkan drivers,
    ``VULKAN`` is typical.

    Environment:
        ``ASTROLAB_BPY_GPU_BACKEND``: ``VULKAN`` (default), ``OPENGL``, or ``METAL``.
        ``ASTROLAB_BPY_GPU_DEVICE``: optional string for ``gpu_preferred_device`` if supported.
    """
    ctx = getattr(bpy, "context", None)
    if ctx is None or not hasattr(ctx, "preferences"):
        logger.debug(
            "prefer_gpu_backend: no bpy.context/preferences (stub bpy or non-Blender runtime)"
        )
        return None

    sys_prefs = bpy.context.preferences.system
    if not hasattr(sys_prefs, "gpu_backend"):
        logger.debug("prefer_gpu_backend: no gpu_backend on this Blender build")
        return None

    env_backend = (os.environ.get("ASTROLAB_BPY_GPU_BACKEND") or "").strip().upper()
    resolved: GpuBackendName
    if backend is not None:
        resolved = backend
    elif env_backend in ("VULKAN", "OPENGL", "METAL"):
        resolved = env_backend  # type: ignore[assignment]
    else:
        resolved = "VULKAN"

    try:
        sys_prefs.gpu_backend = resolved
    except (TypeError, ValueError, AttributeError) as exc:
        logger.warning("Could not set gpu_backend=%s: %s", resolved, exc)
        return getattr(sys_prefs, "gpu_backend", None)

    dev = preferred_device or os.environ.get("ASTROLAB_BPY_GPU_DEVICE")
    if dev and hasattr(sys_prefs, "gpu_preferred_device"):
        try:
            sys_prefs.gpu_preferred_device = dev
        except (TypeError, ValueError) as exc:
            logger.debug("gpu_preferred_device ignored: %s", exc)

    out = getattr(sys_prefs, "gpu_backend", resolved)
    logger.info("Blender preferences GPU backend: %s", out)
    return str(out) if out is not None else None
