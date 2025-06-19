# geometry

Part of `bpy.ops`
Module: `bpy.ops.geometry`

## Operators (10)

### `attribute_add`

bpy.ops.geometry.attribute_add(name="", domain='<UNKNOWN ENUM>', data_type='FLOAT')
Add attribute to geometry

### `attribute_convert`

bpy.ops.geometry.attribute_convert(mode='GENERIC', domain='<UNKNOWN ENUM>', data_type='FLOAT')
Change how the attribute is stored

### `attribute_remove`

bpy.ops.geometry.attribute_remove()
Remove attribute from geometry

### `color_attribute_add`

bpy.ops.geometry.color_attribute_add(name="", domain='POINT', data_type='FLOAT_COLOR', color=(0, 0, 0, 1))
Add color attribute to geometry

### `color_attribute_convert`

bpy.ops.geometry.color_attribute_convert(domain='POINT', data_type='FLOAT_COLOR')
Change how the color attribute is stored

### `color_attribute_duplicate`

bpy.ops.geometry.color_attribute_duplicate()
Duplicate color attribute

### `color_attribute_remove`

bpy.ops.geometry.color_attribute_remove()
Remove color attribute from geometry

### `color_attribute_render_set`

bpy.ops.geometry.color_attribute_render_set(name="Color")
Set default color attribute used for rendering

### `execute_node_group`

bpy.ops.geometry.execute_node_group(asset_library_type='LOCAL', asset_library_identifier="", relative_asset_identifier="", name="", session_uid=0, mouse_position=(0, 0), region_size=(0, 0), cursor_position=(0, 0, 0), cursor_rotation=(0, 0, 0, 0), viewport_projection_matrix=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), viewport_view_matrix=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), viewport_is_perspective=False)
Execute a node group on geometry

### `geometry_randomization`

bpy.ops.geometry.geometry_randomization(value=False)
Toggle geometry randomization for debugging purposes
