# view2d

Part of `bpy.ops`
Module: `bpy.ops.view2d`

## Operators (13)

### `edge_pan`

bpy.ops.view2d.edge_pan(inside_padding=1, outside_padding=0, speed_ramp=1, max_speed=500, delay=1, zoom_influence=0)
Pan the view when the mouse is held at an edge

### `pan`

bpy.ops.view2d.pan(deltax=0, deltay=0)
Pan the view

### `reset`

bpy.ops.view2d.reset()
Reset the view

### `scroll_down`

bpy.ops.view2d.scroll_down(deltax=0, deltay=0, page=False)
Scroll the view down

### `scroll_left`

bpy.ops.view2d.scroll_left(deltax=0, deltay=0)
Scroll the view left

### `scroll_right`

bpy.ops.view2d.scroll_right(deltax=0, deltay=0)
Scroll the view right

### `scroll_up`

bpy.ops.view2d.scroll_up(deltax=0, deltay=0, page=False)
Scroll the view up

### `scroller_activate`

bpy.ops.view2d.scroller_activate()
Scroll view by mouse click and drag

### `smoothview`

bpy.ops.view2d.smoothview(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
(undocumented operator)

### `zoom`

bpy.ops.view2d.zoom(deltax=0, deltay=0, use_cursor_init=True)
Zoom in/out the view

### `zoom_border`

bpy.ops.view2d.zoom_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, zoom_out=False)
Zoom in the view to the nearest item contained in the border

### `zoom_in`

bpy.ops.view2d.zoom_in(zoomfacx=0, zoomfacy=0)
Zoom in the view

### `zoom_out`

bpy.ops.view2d.zoom_out(zoomfacx=0, zoomfacy=0)
Zoom out the view
