# marker

Part of `bpy.ops`
Module: `bpy.ops.marker`

## Operators (11)

### `add`

bpy.ops.marker.add()
Add a new time marker

### `camera_bind`

bpy.ops.marker.camera_bind()
Bind the selected camera to a marker on the current frame

### `delete`

bpy.ops.marker.delete(confirm=True)
Delete selected time marker(s)

### `duplicate`

bpy.ops.marker.duplicate(frames=0)
Duplicate selected time marker(s)

### `make_links_scene`

bpy.ops.marker.make_links_scene(scene='<UNKNOWN ENUM>')
Copy selected markers to another scene

### `move`

bpy.ops.marker.move(frames=0, tweak=False)
Move selected time marker(s)

### `rename`

bpy.ops.marker.rename(name="RenamedMarker")
Rename first selected time marker

### `select`

bpy.ops.marker.select(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, camera=False)
Select time marker(s)

### `select_all`

bpy.ops.marker.select_all(action='TOGGLE')
Change selection of all time markers

### `select_box`

bpy.ops.marker.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET', tweak=False)
Select all time markers using box selection

### `select_leftright`

bpy.ops.marker.select_leftright(mode='LEFT', extend=False)
Select markers on and left/right of the current frame
