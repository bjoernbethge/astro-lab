# scene

Part of `bpy.ops`
Module: `bpy.ops.scene`

## Operators (37)

### `delete`

bpy.ops.scene.delete()
Delete active scene

### `freestyle_add_edge_marks_to_keying_set`

bpy.ops.scene.freestyle_add_edge_marks_to_keying_set()
Add the data paths to the Freestyle Edge Mark property of selected edges to the active keying set

### `freestyle_add_face_marks_to_keying_set`

bpy.ops.scene.freestyle_add_face_marks_to_keying_set()
Add the data paths to the Freestyle Face Mark property of selected polygons to the active keying set

### `freestyle_alpha_modifier_add`

bpy.ops.scene.freestyle_alpha_modifier_add(type='ALONG_STROKE')
Add an alpha transparency modifier to the line style associated with the active lineset

### `freestyle_color_modifier_add`

bpy.ops.scene.freestyle_color_modifier_add(type='ALONG_STROKE')
Add a line color modifier to the line style associated with the active lineset

### `freestyle_fill_range_by_selection`

bpy.ops.scene.freestyle_fill_range_by_selection(type='COLOR', name="")
Fill the Range Min/Max entries by the min/max distance between selected mesh objects and the source object (either a user-specified object or the active camera)

### `freestyle_geometry_modifier_add`

bpy.ops.scene.freestyle_geometry_modifier_add(type='2D_OFFSET')
Add a stroke geometry modifier to the line style associated with the active lineset

### `freestyle_lineset_add`

bpy.ops.scene.freestyle_lineset_add()
Add a line set into the list of line sets

### `freestyle_lineset_copy`

bpy.ops.scene.freestyle_lineset_copy()
Copy the active line set to the internal clipboard

### `freestyle_lineset_move`

bpy.ops.scene.freestyle_lineset_move(direction='UP')
Change the position of the active line set within the list of line sets

### `freestyle_lineset_paste`

bpy.ops.scene.freestyle_lineset_paste()
Paste the internal clipboard content to the active line set

### `freestyle_lineset_remove`

bpy.ops.scene.freestyle_lineset_remove()
Remove the active line set from the list of line sets

### `freestyle_linestyle_new`

bpy.ops.scene.freestyle_linestyle_new()
Create a new line style, reusable by multiple line sets

### `freestyle_modifier_copy`

bpy.ops.scene.freestyle_modifier_copy()
Duplicate the modifier within the list of modifiers

### `freestyle_modifier_move`

bpy.ops.scene.freestyle_modifier_move(direction='UP')
Move the modifier within the list of modifiers

### `freestyle_modifier_remove`

bpy.ops.scene.freestyle_modifier_remove()
Remove the modifier from the list of modifiers

### `freestyle_module_add`

bpy.ops.scene.freestyle_module_add()
Add a style module into the list of modules

### `freestyle_module_move`

bpy.ops.scene.freestyle_module_move(direction='UP')
Change the position of the style module within in the list of style modules

### `freestyle_module_open`

bpy.ops.scene.freestyle_module_open(filepath="", make_internal=True)
Open a style module file

### `freestyle_module_remove`

bpy.ops.scene.freestyle_module_remove()
Remove the style module from the stack

### `freestyle_stroke_material_create`

bpy.ops.scene.freestyle_stroke_material_create()
Create Freestyle stroke material for testing

### `freestyle_thickness_modifier_add`

bpy.ops.scene.freestyle_thickness_modifier_add(type='ALONG_STROKE')
Add a line thickness modifier to the line style associated with the active lineset

### `gltf2_action_filter_refresh`

bpy.ops.scene.gltf2_action_filter_refresh()
Refresh list of actions

### `gpencil_brush_preset_add`

bpy.ops.scene.gpencil_brush_preset_add(name="", remove_name=False, remove_active=False)
Add or remove grease pencil brush preset

### `gpencil_material_preset_add`

bpy.ops.scene.gpencil_material_preset_add(name="", remove_name=False, remove_active=False)
Add or remove Grease Pencil material preset

### `new`

bpy.ops.scene.new(type='NEW')
Add new scene by type

### `new_sequencer`

bpy.ops.scene.new_sequencer(type='NEW')
Add new scene by type in the sequence editor and assign to active strip

### `render_view_add`

bpy.ops.scene.render_view_add()
Add a render view

### `render_view_remove`

bpy.ops.scene.render_view_remove()
Remove the selected render view

### `view_layer_add`

bpy.ops.scene.view_layer_add(type='NEW')
Add a view layer

### `view_layer_add_aov`

bpy.ops.scene.view_layer_add_aov()
Add a Shader AOV

### `view_layer_add_lightgroup`

bpy.ops.scene.view_layer_add_lightgroup(name="")
Add a Light Group

### `view_layer_add_used_lightgroups`

bpy.ops.scene.view_layer_add_used_lightgroups()
Add all used Light Groups

### `view_layer_remove`

bpy.ops.scene.view_layer_remove()
Remove the selected view layer

### `view_layer_remove_aov`

bpy.ops.scene.view_layer_remove_aov()
Remove Active AOV

### `view_layer_remove_lightgroup`

bpy.ops.scene.view_layer_remove_lightgroup()
Remove Active Lightgroup

### `view_layer_remove_unused_lightgroups`

bpy.ops.scene.view_layer_remove_unused_lightgroups()
Remove all unused Light Groups
