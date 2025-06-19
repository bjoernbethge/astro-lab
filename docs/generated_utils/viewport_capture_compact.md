# Viewport_Capture Module

Auto-generated documentation for `utils.blender.viewport_capture`

## Functions

### analyze_viewport_content(pixel_data: numpy.ndarray, depth_data: Optional[numpy.ndarray] = None) -> Dict[str, <built-in function any>]

Analyze viewport pixel data for astronomical objects or features.

Args:
    pixel_data: RGBA pixel array from capture_viewport_pixels()
    depth_data: Optional depth buffer from capture_depth_buffer()

Returns:
    Analysis results including detected features, statistics, etc.

### capture_depth_buffer(width: int = 512, height: int = 512) -> Optional[numpy.ndarray]

Capture depth buffer from current 3D viewport.

Args:
    width: Width of capture buffer
    height: Height of capture buffer

Returns:
    numpy array with depth values (height, width) or None on error

### capture_viewport_pixels(width: int = 512, height: int = 512, format: str = 'RGBA8') -> Optional[numpy.ndarray]

Capture raw pixel data from current 3D viewport using OffScreen buffer.

This creates an OffScreen buffer and renders the current view to it,
then extracts the pixel data directly. This is the modern approach
that replaces deprecated bgl.glReadPixels().

Args:
    width: Width of capture buffer
    height: Height of capture buffer
    format: GPU texture format ('RGBA8', 'RGBA16F', etc.)

Returns:
    numpy array with shape (height, width, channels) or None on error

### capture_viewport_screenshot(output_path: Optional[str] = None) -> bool

Legacy function - captures pixels and saves as image.

### get_live_viewport_stream(width: int = 256, height: int = 256, include_depth: bool = False) -> Dict[str, numpy.ndarray]

Get live viewport data stream for real-time processing.

This is optimized for repeated calls and minimal overhead.

### get_viewport_info() -> Dict[str, <built-in function any>]

Get detailed viewport state information.

## Constants

- **BLENDER_AVAILABLE** (bool): `True`
- **GPU_AVAILABLE** (bool): `False`
