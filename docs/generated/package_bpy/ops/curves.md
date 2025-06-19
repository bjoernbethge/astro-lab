# curves

Part of `bpy.ops`
Module: `bpy.ops.curves`

## Operators (28)

### `add_bezier`

bpy.ops.curves.add_bezier(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add new bezier curve

### `add_circle`

bpy.ops.curves.add_circle(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add new circle curve

### `attribute_set`

bpy.ops.curves.attribute_set(value_float=0, value_float_vector_2d=(0, 0), value_float_vector_3d=(0, 0, 0), value_int=0, value_int_vector_2d=(0, 0), value_color=(1, 1, 1, 1), value_bool=False)
Set values of the active attribute for selected elements

### `convert_from_particle_system`

bpy.ops.curves.convert_from_particle_system()
Add a new curves object based on the current state of the particle system

### `convert_to_particle_system`

bpy.ops.curves.convert_to_particle_system()
Add a new or update an existing hair particle system on the surface object

### `curve_type_set`

bpy.ops.curves.curve_type_set(type='POLY', use_handles=False)
Set type of selected curves

### `cyclic_toggle`

bpy.ops.curves.cyclic_toggle()
Make active curve closed/opened loop

### `delete`

bpy.ops.curves.delete()
Remove selected control points or curves

### `draw`

bpy.ops.curves.draw(error_threshold=0, fit_method='REFIT', corner_angle=1.22173, use_cyclic=True, stroke=[], wait_for_input=True, is_curve_2d=False, bezier_as_nurbs=False)
Draw a freehand curve

### `duplicate`

bpy.ops.curves.duplicate()
Copy selected points or curves

### `duplicate_move`

bpy.ops.curves.duplicate_move(CURVES_OT_duplicate={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Make copies of selected elements and move them

### `extrude`

bpy.ops.curves.extrude()
Extrude selected control point(s)

### `extrude_move`

bpy.ops.curves.extrude_move(CURVES_OT_extrude={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude curve and move result

### `handle_type_set`

bpy.ops.curves.handle_type_set(type='AUTO')
Set the handle type for bezier curves

### `sculptmode_toggle`

bpy.ops.curves.sculptmode_toggle()
Enter/Exit sculpt mode for curves

### `select_all`

bpy.ops.curves.select_all(action='TOGGLE')
(De)select all control points

### `select_ends`

bpy.ops.curves.select_ends(amount_start=0, amount_end=1)
Select end points of curves

### `select_less`

bpy.ops.curves.select_less()
Shrink the selection by one point

### `select_linked`

bpy.ops.curves.select_linked()
Select all points in curves with any point selection

### `select_linked_pick`

bpy.ops.curves.select_linked_pick(deselect=False)
Select all points in the curve under the cursor

### `select_more`

bpy.ops.curves.select_more()
Grow the selection by one point

### `select_random`

bpy.ops.curves.select_random(seed=0, probability=0.5)
Randomizes existing selection or create new random selection

### `set_selection_domain`

bpy.ops.curves.set_selection_domain(domain='POINT')
Change the mode used for selection masking in curves sculpt mode

### `snap_curves_to_surface`

bpy.ops.curves.snap_curves_to_surface(attach_mode='NEAREST')
Move curves so that the first point is exactly on the surface mesh

### `subdivide`

bpy.ops.curves.subdivide(number_cuts=1)
Subdivide selected curve segments

### `surface_set`

bpy.ops.curves.surface_set()
Use the active object as surface for selected curves objects and set it as the parent

### `switch_direction`

bpy.ops.curves.switch_direction()
Reverse the direction of the selected curves

### `tilt_clear`

bpy.ops.curves.tilt_clear()
Clear the tilt of selected control points
