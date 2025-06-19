# mask

Part of `bpy.ops`
Module: `bpy.ops.mask`

## Operators (39)

### `add_feather_vertex`

bpy.ops.mask.add_feather_vertex(location=(0, 0))
Add vertex to feather

### `add_feather_vertex_slide`

bpy.ops.mask.add_feather_vertex_slide(MASK_OT_add_feather_vertex={"location":(0, 0)}, MASK_OT_slide_point={"slide_feather":False, "is_new_point":False})
Add new vertex to feather and slide it

### `add_vertex`

bpy.ops.mask.add_vertex(location=(0, 0))
Add vertex to active spline

### `add_vertex_slide`

bpy.ops.mask.add_vertex_slide(MASK_OT_add_vertex={"location":(0, 0)}, MASK_OT_slide_point={"slide_feather":False, "is_new_point":False})
Add new vertex and slide it

### `copy_splines`

bpy.ops.mask.copy_splines()
Copy the selected splines to the internal clipboard

### `cyclic_toggle`

bpy.ops.mask.cyclic_toggle()
Toggle cyclic for selected splines

### `delete`

bpy.ops.mask.delete(confirm=True)
Delete selected control points or splines

### `duplicate`

bpy.ops.mask.duplicate()
Duplicate selected control points and segments between them

### `duplicate_move`

bpy.ops.mask.duplicate_move(MASK_OT_duplicate={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate mask and move

### `feather_weight_clear`

bpy.ops.mask.feather_weight_clear()
Reset the feather weight to zero

### `handle_type_set`

bpy.ops.mask.handle_type_set(type='AUTO')
Set type of handles for selected control points

### `hide_view_clear`

bpy.ops.mask.hide_view_clear(select=True)
Reveal temporarily hidden mask layers

### `hide_view_set`

bpy.ops.mask.hide_view_set(unselected=False)
Temporarily hide mask layers

### `layer_move`

bpy.ops.mask.layer_move(direction='UP')
Move the active layer up/down in the list

### `layer_new`

bpy.ops.mask.layer_new(name="")
Add new mask layer for masking

### `layer_remove`

bpy.ops.mask.layer_remove()
Remove mask layer

### `new`

bpy.ops.mask.new(name="")
Create new mask

### `normals_make_consistent`

bpy.ops.mask.normals_make_consistent()
Recalculate the direction of selected handles

### `parent_clear`

bpy.ops.mask.parent_clear()
Clear the mask's parenting

### `parent_set`

bpy.ops.mask.parent_set()
Set the mask's parenting

### `paste_splines`

bpy.ops.mask.paste_splines()
Paste splines from the internal clipboard

### `primitive_circle_add`

bpy.ops.mask.primitive_circle_add(size=100, location=(0, 0))
Add new circle-shaped spline

### `primitive_square_add`

bpy.ops.mask.primitive_square_add(size=100, location=(0, 0))
Add new square-shaped spline

### `select`

bpy.ops.mask.select(extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, location=(0, 0))
Select spline points

### `select_all`

bpy.ops.mask.select_all(action='TOGGLE')
Change selection of all curve points

### `select_box`

bpy.ops.mask.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Select curve points using box selection

### `select_circle`

bpy.ops.mask.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Select curve points using circle selection

### `select_lasso`

bpy.ops.mask.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select curve points using lasso selection

### `select_less`

bpy.ops.mask.select_less()
Deselect spline points at the boundary of each selection region

### `select_linked`

bpy.ops.mask.select_linked()
Select all curve points linked to already selected ones

### `select_linked_pick`

bpy.ops.mask.select_linked_pick(deselect=False)
(De)select all points linked to the curve under the mouse cursor

### `select_more`

bpy.ops.mask.select_more()
Select more spline points connected to initial selection

### `shape_key_clear`

bpy.ops.mask.shape_key_clear()
Remove mask shape keyframe for active mask layer at the current frame

### `shape_key_feather_reset`

bpy.ops.mask.shape_key_feather_reset()
Reset feather weights on all selected points animation values

### `shape_key_insert`

bpy.ops.mask.shape_key_insert()
Insert mask shape keyframe for active mask layer at the current frame

### `shape_key_rekey`

bpy.ops.mask.shape_key_rekey(location=True, feather=True)
Recalculate animation data on selected points for frames selected in the dopesheet

### `slide_point`

bpy.ops.mask.slide_point(slide_feather=False, is_new_point=False)
Slide control points

### `slide_spline_curvature`

bpy.ops.mask.slide_spline_curvature()
Slide a point on the spline to define its curvature

### `switch_direction`

bpy.ops.mask.switch_direction()
Switch direction of selected splines
