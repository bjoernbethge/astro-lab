"""
Pydantic schemas for utility configurations.
"""

from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


class BlenderConfigSchema(BaseModel):
    """Configuration schema for Blender visualizations."""
    
    output_path: str = Field(
        default="output.png",
        description="Output file path"
    )
    resolution: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Output resolution (width, height)"
    )
    samples: int = Field(
        default=128,
        ge=1,
        le=4096,
        description="Number of render samples"
    )
    engine: str = Field(
        default="CYCLES",
        description="Render engine (CYCLES, EEVEE)"
    )
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for rendering"
    )


class ScatterPlotConfigSchema(BaseModel):
    """Configuration schema for 3D scatter plots."""
    
    point_size: float = Field(
        default=0.1,
        gt=0.0,
        le=10.0,
        description="Size of scatter points"
    )
    color_map: str = Field(
        default="viridis",
        description="Color map for points"
    )
    show_axes: bool = Field(
        default=True,
        description="Show coordinate axes"
    )
    background_color: Tuple[float, float, float, float] = Field(
        default=(0.0, 0.0, 0.0, 1.0),
        description="Background color (RGBA)"
    )


class SurfacePlotConfigSchema(BaseModel):
    """Configuration schema for surface plots."""
    
    subdivision_levels: int = Field(
        default=2,
        ge=0,
        le=6,
        description="Surface subdivision levels"
    )
    wireframe: bool = Field(
        default=False,
        description="Show wireframe"
    )
    smooth_shading: bool = Field(
        default=True,
        description="Use smooth shading"
    )
    material_type: str = Field(
        default="principled",
        description="Material type for surface"
    )


class AnimationConfigSchema(BaseModel):
    """Configuration schema for animations."""
    
    frame_start: int = Field(
        default=1,
        ge=1,
        description="Starting frame number"
    )
    frame_end: int = Field(
        default=250,
        ge=1,
        description="Ending frame number"
    )
    fps: int = Field(
        default=24,
        ge=1,
        le=120,
        description="Frames per second"
    )
    output_format: str = Field(
        default="MP4",
        description="Output video format"
    )
    codec: str = Field(
        default="H264",
        description="Video codec"
    )


class LightingConfigSchema(BaseModel):
    """Configuration schema for lighting setup."""
    
    light_type: str = Field(
        default="SUN",
        description="Light type (SUN, POINT, SPOT, AREA)"
    )
    energy: float = Field(
        default=5.0,
        gt=0.0,
        le=1000.0,
        description="Light energy/strength"
    )
    color: Tuple[float, float, float] = Field(
        default=(1.0, 1.0, 1.0),
        description="Light color (RGB)"
    )
    location: Tuple[float, float, float] = Field(
        default=(5.0, 5.0, 5.0),
        description="Light location (x, y, z)"
    )
    rotation: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Light rotation (x, y, z) in radians"
    )


class CameraConfigSchema(BaseModel):
    """Configuration schema for camera setup."""
    
    location: Tuple[float, float, float] = Field(
        default=(10.0, 10.0, 10.0),
        description="Camera location (x, y, z)"
    )
    rotation: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Camera rotation (x, y, z) in radians"
    )
    lens: float = Field(
        default=50.0,
        gt=0.0,
        le=300.0,
        description="Camera lens focal length (mm)"
    )
    sensor_width: float = Field(
        default=36.0,
        gt=0.0,
        description="Camera sensor width (mm)"
    )


class PostProcessingConfigSchema(BaseModel):
    """Configuration schema for post-processing effects."""
    
    bloom: bool = Field(
        default=False,
        description="Enable bloom effect"
    )
    motion_blur: bool = Field(
        default=False,
        description="Enable motion blur"
    )
    depth_of_field: bool = Field(
        default=False,
        description="Enable depth of field"
    )
    color_grading: bool = Field(
        default=False,
        description="Enable color grading"
    )
    contrast: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Contrast adjustment"
    )
    brightness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Brightness adjustment"
    )
    saturation: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Saturation adjustment"
    )


class VisualizationConfigSchema(BaseModel):
    """Configuration schema for general visualizations."""
    
    theme: str = Field(
        default="dark",
        description="Visualization theme (dark, light, custom)"
    )
    dpi: int = Field(
        default=300,
        ge=72,
        le=600,
        description="Output DPI for static images"
    )
    format: str = Field(
        default="png",
        description="Output format (png, jpg, svg, pdf)"
    )
    transparent_background: bool = Field(
        default=False,
        description="Use transparent background"
    )
    show_grid: bool = Field(
        default=True,
        description="Show grid lines"
    )
    show_legend: bool = Field(
        default=True,
        description="Show legend"
    )
    font_size: int = Field(
        default=12,
        ge=6,
        le=72,
        description="Base font size"
    )
    color_palette: str = Field(
        default="viridis",
        description="Color palette name"
    ) 