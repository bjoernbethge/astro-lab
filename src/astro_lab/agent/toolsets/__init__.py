"""Pydantic AI FunctionToolsets for AstroLab."""

from .analysis import analysis_toolset
from .astro_duck import astro_duck_toolset
from .astrophot_tools import astrophot_toolset
from .data_pipeline import data_pipeline_toolset
from .safety import safety_toolset
from .survey_config import survey_config_toolset
from .training import training_toolset
from .widget_tools import (
    alcg_widgets_toolset,
    albpy_widgets_toolset,
    alo3d_widgets_toolset,
    alpv_widgets_toolset,
    astro_widgets_toolset,
    enhanced_widgets_toolset,
    plotly_widgets_toolset,
    widget_common_toolset,
)

__all__ = [
    "alcg_widgets_toolset",
    "albpy_widgets_toolset",
    "alo3d_widgets_toolset",
    "alpv_widgets_toolset",
    "analysis_toolset",
    "astro_duck_toolset",
    "astrophot_toolset",
    "astro_widgets_toolset",
    "data_pipeline_toolset",
    "enhanced_widgets_toolset",
    "plotly_widgets_toolset",
    "safety_toolset",
    "survey_config_toolset",
    "training_toolset",
    "widget_common_toolset",
]
