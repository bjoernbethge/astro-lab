"""Agent toolsets aligned with ``astro_lab.widgets`` subpackages (one toolset each + common)."""

from __future__ import annotations

from pydantic_ai import CombinedToolset

from .alcg import alcg_widgets_toolset
from .albpy_widget import albpy_widgets_toolset
from .alo3d_widget import alo3d_widgets_toolset
from .alpv_widget import alpv_widgets_toolset
from .enhanced_widget import enhanced_widgets_toolset
from .plotly_widget import plotly_widgets_toolset
from .widget_common import widget_common_toolset


def build_widgets_combined_toolset() -> CombinedToolset:
    """All widget-related tools: shared helpers + one group per ``widgets`` subpackage."""
    return CombinedToolset(
        [
            widget_common_toolset,
            alcg_widgets_toolset,
            plotly_widgets_toolset,
            enhanced_widgets_toolset,
            albpy_widgets_toolset,
            alpv_widgets_toolset,
            alo3d_widgets_toolset,
        ]
    )


astro_widgets_toolset = build_widgets_combined_toolset()

__all__ = [
    "alcg_widgets_toolset",
    "albpy_widgets_toolset",
    "alo3d_widgets_toolset",
    "alpv_widgets_toolset",
    "astro_widgets_toolset",
    "build_widgets_combined_toolset",
    "enhanced_widgets_toolset",
    "plotly_widgets_toolset",
    "widget_common_toolset",
]
