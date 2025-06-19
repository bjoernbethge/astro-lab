# Utils_Schemas Module

Auto-generated documentation for `schemas.utils_schemas`

## AnimationConfigSchema

Configuration schema for animations.

### Parameters

**`frame_start`** *(integer)* = `1`
  Starting frame number
  *≥1*

**`frame_end`** *(integer)* = `250`
  Ending frame number
  *≥1*

**`fps`** *(integer)* = `24`
  Frames per second
  *≥1, ≤120*

**`output_format`** *(string)* = `MP4`
  Output video format

**`codec`** *(string)* = `H264`
  Video codec

### Usage

```python
from docs.auto.schemas.data_schemas import AnimationConfigSchema

config = AnimationConfigSchema(

    # Optional parameters:
    # frame_start=1
    # frame_end=1
    # fps=1
    # output_format="example"
    # codec="example"
)
```

## BlenderConfigSchema

Configuration schema for Blender visualizations.

### Parameters

**`output_path`** *(string)* = `output.png`
  Output file path

**`resolution`** *(array)* = `[1920, 1080]`
  Output resolution (width, height)

**`samples`** *(integer)* = `128`
  Number of render samples
  *≥1, ≤4096*

**`engine`** *(string)* = `CYCLES`
  Render engine (CYCLES, EEVEE)

**`use_gpu`** *(boolean)* = `True`
  Use GPU for rendering

### Usage

```python
from docs.auto.schemas.data_schemas import BlenderConfigSchema

config = BlenderConfigSchema(

    # Optional parameters:
    # output_path="example"
    # resolution=[]
    # samples=1
    # engine="example"
    # use_gpu=True
)
```

## CameraConfigSchema

Configuration schema for camera setup.

### Parameters

**`location`** *(array)* = `[10.0, 10.0, 10.0]`
  Camera location (x, y, z)

**`rotation`** *(array)* = `[0.0, 0.0, 0.0]`
  Camera rotation (x, y, z) in radians

**`lens`** *(number)* = `50.0`
  Camera lens focal length (mm)
  *≤300.0, >0.0*

**`sensor_width`** *(number)* = `36.0`
  Camera sensor width (mm)
  *>0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import CameraConfigSchema

config = CameraConfigSchema(

    # Optional parameters:
    # location=[]
    # rotation=[]
    # lens=1.0
    # sensor_width=1.0
)
```

## LightingConfigSchema

Configuration schema for lighting setup.

### Parameters

**`light_type`** *(string)* = `SUN`
  Light type (SUN, POINT, SPOT, AREA)

**`energy`** *(number)* = `5.0`
  Light energy/strength
  *≤1000.0, >0.0*

**`color`** *(array)* = `[1.0, 1.0, 1.0]`
  Light color (RGB)

**`location`** *(array)* = `[5.0, 5.0, 5.0]`
  Light location (x, y, z)

**`rotation`** *(array)* = `[0.0, 0.0, 0.0]`
  Light rotation (x, y, z) in radians

### Usage

```python
from docs.auto.schemas.data_schemas import LightingConfigSchema

config = LightingConfigSchema(

    # Optional parameters:
    # light_type="example"
    # energy=1.0
    # color=[]
    # location=[]
    # rotation=[]
)
```

## PostProcessingConfigSchema

Configuration schema for post-processing effects.

### Parameters

**`bloom`** *(boolean)* = `False`
  Enable bloom effect

**`motion_blur`** *(boolean)* = `False`
  Enable motion blur

**`depth_of_field`** *(boolean)* = `False`
  Enable depth of field

**`color_grading`** *(boolean)* = `False`
  Enable color grading

**`contrast`** *(number)* = `1.0`
  Contrast adjustment
  *≥0.0, ≤5.0*

**`brightness`** *(number)* = `0.0`
  Brightness adjustment
  *≥-1.0, ≤1.0*

**`saturation`** *(number)* = `1.0`
  Saturation adjustment
  *≥0.0, ≤5.0*

### Usage

```python
from docs.auto.schemas.data_schemas import PostProcessingConfigSchema

config = PostProcessingConfigSchema(

    # Optional parameters:
    # bloom=False
    # motion_blur=False
    # depth_of_field=False
    # color_grading=False
    # contrast=0.0
    # brightness=-1.0
    # saturation=0.0
)
```

## ScatterPlotConfigSchema

Configuration schema for 3D scatter plots.

### Parameters

**`point_size`** *(number)* = `0.1`
  Size of scatter points
  *≤10.0, >0.0*

**`color_map`** *(string)* = `viridis`
  Color map for points

**`show_axes`** *(boolean)* = `True`
  Show coordinate axes

**`background_color`** *(array)* = `[0.0, 0.0, 0.0, 1.0]`
  Background color (RGBA)

### Usage

```python
from docs.auto.schemas.data_schemas import ScatterPlotConfigSchema

config = ScatterPlotConfigSchema(

    # Optional parameters:
    # point_size=1.0
    # color_map="example"
    # show_axes=True
    # background_color=[]
)
```

## SurfacePlotConfigSchema

Configuration schema for surface plots.

### Parameters

**`subdivision_levels`** *(integer)* = `2`
  Surface subdivision levels
  *≥0, ≤6*

**`wireframe`** *(boolean)* = `False`
  Show wireframe

**`smooth_shading`** *(boolean)* = `True`
  Use smooth shading

**`material_type`** *(string)* = `principled`
  Material type for surface

### Usage

```python
from docs.auto.schemas.data_schemas import SurfacePlotConfigSchema

config = SurfacePlotConfigSchema(

    # Optional parameters:
    # subdivision_levels=0
    # wireframe=False
    # smooth_shading=True
    # material_type="example"
)
```

## VisualizationConfigSchema

Configuration schema for general visualizations.

### Parameters

**`theme`** *(string)* = `dark`
  Visualization theme (dark, light, custom)

**`dpi`** *(integer)* = `300`
  Output DPI for static images
  *≥72, ≤600*

**`format`** *(string)* = `png`
  Output format (png, jpg, svg, pdf)

**`transparent_background`** *(boolean)* = `False`
  Use transparent background

**`show_grid`** *(boolean)* = `True`
  Show grid lines

**`show_legend`** *(boolean)* = `True`
  Show legend

**`font_size`** *(integer)* = `12`
  Base font size
  *≥6, ≤72*

**`color_palette`** *(string)* = `viridis`
  Color palette name

### Usage

```python
from docs.auto.schemas.data_schemas import VisualizationConfigSchema

config = VisualizationConfigSchema(

    # Optional parameters:
    # theme="example"
    # dpi=72
    # format="example"
    # transparent_background=False
    # show_grid=True
    # show_legend=True
    # font_size=6
    # color_palette="example"
)
```
