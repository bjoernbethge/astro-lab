# curve

Part of `bpy.ops`
Module: `bpy.ops.curve`

## Operators (51)

### `cyclic_toggle`

bpy.ops.curve.cyclic_toggle(direction='CYCLIC_U')
Make active spline closed/opened loop

### `de_select_first`

bpy.ops.curve.de_select_first()
(De)select first of visible part of each NURBS

### `de_select_last`

bpy.ops.curve.de_select_last()
(De)select last of visible part of each NURBS

### `decimate`

bpy.ops.curve.decimate(ratio=1)
Simplify selected curves

### `delete`

bpy.ops.curve.delete(type='VERT')
Delete selected control points or segments

### `dissolve_verts`

bpy.ops.curve.dissolve_verts()
Delete selected control points, correcting surrounding handles

### `draw`

bpy.ops.curve.draw(error_threshold=0, fit_method='REFIT', corner_angle=1.22173, use_cyclic=True, stroke=[], wait_for_input=True)
Draw a freehand spline

### `duplicate`

bpy.ops.curve.duplicate()
Duplicate selected control points

### `duplicate_move`

bpy.ops.curve.duplicate_move(CURVE_OT_duplicate={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate curve and move

### `extrude`

bpy.ops.curve.extrude(mode='TRANSLATION')
Extrude selected control point(s)

### `extrude_move`

bpy.ops.curve.extrude_move(CURVE_OT_extrude={"mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude curve and move result

### `handle_type_set`

bpy.ops.curve.handle_type_set(type='AUTOMATIC')
Set type of handles for selected control points

### `hide`

bpy.ops.curve.hide(unselected=False)
Hide (un)selected control points

### `make_segment`

bpy.ops.curve.make_segment()
Join two curves by their selected ends

### `match_texture_space`

bpy.ops.curve.match_texture_space()
Match texture space to object's bounding box

### `normals_make_consistent`

bpy.ops.curve.normals_make_consistent(calc_length=False)
Recalculate the direction of selected handles

### `pen`

bpy.ops.curve.pen(extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, extrude_point=False, extrude_handle='VECTOR', delete_point=False, insert_point=False, move_segment=False, select_point=False, move_point=False, close_spline=True, close_spline_method='OFF', toggle_vector=False, cycle_handle_type=False)
Construct and edit splines

### `primitive_bezier_circle_add`

bpy.ops.curve.primitive_bezier_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Bézier Circle

### `primitive_bezier_curve_add`

bpy.ops.curve.primitive_bezier_curve_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Bézier Curve

### `primitive_nurbs_circle_add`

bpy.ops.curve.primitive_nurbs_circle_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Nurbs Circle

### `primitive_nurbs_curve_add`

bpy.ops.curve.primitive_nurbs_curve_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Nurbs Curve

### `primitive_nurbs_path_add`

bpy.ops.curve.primitive_nurbs_path_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Path

### `radius_set`

bpy.ops.curve.radius_set(radius=1)
Set per-point radius which is used for bevel tapering

### `reveal`

bpy.ops.curve.reveal(select=True)
Reveal hidden control points

### `select_all`

bpy.ops.curve.select_all(action='TOGGLE')
(De)select all control points

### `select_less`

bpy.ops.curve.select_less()
Deselect control points at the boundary of each selection region

### `select_linked`

bpy.ops.curve.select_linked()
Select all control points linked to the current selection

### `select_linked_pick`

bpy.ops.curve.select_linked_pick(deselect=False)
Select all control points linked to already selected ones

### `select_more`

bpy.ops.curve.select_more()
Select control points at the boundary of each selection region

### `select_next`

bpy.ops.curve.select_next()
Select control points following already selected ones along the curves

### `select_nth`

bpy.ops.curve.select_nth(skip=1, nth=1, offset=0)
Deselect every Nth point starting from the active one

### `select_previous`

bpy.ops.curve.select_previous()
Select control points preceding already selected ones along the curves

### `select_random`

bpy.ops.curve.select_random(ratio=0.5, seed=0, action='SELECT')
Randomly select some control points

### `select_row`

bpy.ops.curve.select_row()
Select a row of control points including active one. Successive use on the same point switches between U/V directions

### `select_similar`

bpy.ops.curve.select_similar(type='WEIGHT', compare='EQUAL', threshold=0.1)
Select similar curve points by property type

### `separate`

bpy.ops.curve.separate()
Separate selected points from connected unselected points into a new object

### `shade_flat`

bpy.ops.curve.shade_flat()
Set shading to flat

### `shade_smooth`

bpy.ops.curve.shade_smooth()
Set shading to smooth

### `shortest_path_pick`

bpy.ops.curve.shortest_path_pick()
Select shortest path between two selections

### `smooth`

bpy.ops.curve.smooth()
Flatten angles of selected points

### `smooth_radius`

bpy.ops.curve.smooth_radius()
Interpolate radii of selected points

### `smooth_tilt`

bpy.ops.curve.smooth_tilt()
Interpolate tilt of selected points

### `smooth_weight`

bpy.ops.curve.smooth_weight()
Interpolate weight of selected points

### `spin`

bpy.ops.curve.spin(center=(0, 0, 0), axis=(0, 0, 0))
Extrude selected boundary row around pivot point and current view axis

### `spline_type_set`

bpy.ops.curve.spline_type_set(type='POLY', use_handles=False)
Set type of active spline

### `spline_weight_set`

bpy.ops.curve.spline_weight_set(weight=1)
Set softbody goal weight for selected points

### `split`

bpy.ops.curve.split()
Split off selected points from connected unselected points

### `subdivide`

bpy.ops.curve.subdivide(number_cuts=1)
Subdivide selected segments

### `switch_direction`

bpy.ops.curve.switch_direction()
Switch direction of selected splines

### `tilt_clear`

bpy.ops.curve.tilt_clear()
Clear the tilt of selected control points

### `vertex_add`

bpy.ops.curve.vertex_add(location=(0, 0, 0))
Add a new control point (linked to only selected end-curve one, if any)
