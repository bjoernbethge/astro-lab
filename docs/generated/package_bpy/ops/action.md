# action

Part of `bpy.ops`
Module: `bpy.ops.action`

## Operators (38)

### `bake_keys`

bpy.ops.action.bake_keys()
Add keyframes on every frame between the selected keyframes

### `clean`

bpy.ops.action.clean(threshold=0.001, channels=False)
Simplify F-Curves by removing closely spaced keyframes

### `clickselect`

bpy.ops.action.clickselect(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, deselect_all=False, column=False, channel=False)
Select keyframes by clicking on them

### `copy`

bpy.ops.action.copy()
Copy selected keyframes to the internal clipboard

### `delete`

bpy.ops.action.delete(confirm=True)
Remove all selected keyframes

### `duplicate`

bpy.ops.action.duplicate()
Make a copy of all selected keyframes

### `duplicate_move`

bpy.ops.action.duplicate_move(ACTION_OT_duplicate={}, TRANSFORM_OT_transform={"mode":'TRANSLATION', "value":(0, 0, 0, 0), "orient_axis":'Z', "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "center_override":(0, 0, 0), "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Make a copy of all selected keyframes and move them

### `easing_type`

bpy.ops.action.easing_type(type='AUTO')
Set easing type for the F-Curve segments starting from the selected keyframes

### `extrapolation_type`

bpy.ops.action.extrapolation_type(type='CONSTANT')
Set extrapolation mode for selected F-Curves

### `frame_jump`

bpy.ops.action.frame_jump()
Set the current frame to the average frame value of selected keyframes

### `handle_type`

bpy.ops.action.handle_type(type='FREE')
Set type of handle for selected keyframes

### `interpolation_type`

bpy.ops.action.interpolation_type(type='CONSTANT')
Set interpolation mode for the F-Curve segments starting from the selected keyframes

### `keyframe_insert`

bpy.ops.action.keyframe_insert(type='ALL')
Insert keyframes for the specified channels

### `keyframe_type`

bpy.ops.action.keyframe_type(type='KEYFRAME')
Set type of keyframe for the selected keyframes

### `layer_next`

bpy.ops.action.layer_next()
Switch to editing action in animation layer above the current action in the NLA Stack

### `layer_prev`

bpy.ops.action.layer_prev()
Switch to editing action in animation layer below the current action in the NLA Stack

### `markers_make_local`

bpy.ops.action.markers_make_local()
Move selected scene markers to the active Action as local 'pose' markers

### `mirror`

bpy.ops.action.mirror(type='CFRA')
Flip selected keyframes over the selected mirror line

### `new`

bpy.ops.action.new()
Create new action

### `paste`

bpy.ops.action.paste(offset='START', merge='MIX', flipped=False)
Paste keyframes from the internal clipboard for the selected channels, starting on the current frame

### `previewrange_set`

bpy.ops.action.previewrange_set()
Set Preview Range based on extents of selected Keyframes

### `push_down`

bpy.ops.action.push_down()
Push action down on to the NLA stack as a new strip

### `select_all`

bpy.ops.action.select_all(action='TOGGLE')
Toggle selection of all keyframes

### `select_box`

bpy.ops.action.select_box(axis_range=False, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET', tweak=False)
Select all keyframes within the specified region

### `select_circle`

bpy.ops.action.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Select keyframe points using circle selection

### `select_column`

bpy.ops.action.select_column(mode='KEYS')
Select all keyframes on the specified frame(s)

### `select_lasso`

bpy.ops.action.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select keyframe points using lasso selection

### `select_leftright`

bpy.ops.action.select_leftright(mode='CHECK', extend=False)
Select keyframes to the left or the right of the current frame

### `select_less`

bpy.ops.action.select_less()
Deselect keyframes on ends of selection islands

### `select_linked`

bpy.ops.action.select_linked()
Select keyframes occurring in the same F-Curves as selected ones

### `select_more`

bpy.ops.action.select_more()
Select keyframes beside already selected ones

### `snap`

bpy.ops.action.snap(type='CFRA')
Snap selected keyframes to the times specified

### `stash`

bpy.ops.action.stash(create_new=True)
Store this action in the NLA stack as a non-contributing strip for later use

### `stash_and_create`

bpy.ops.action.stash_and_create()
Store this action in the NLA stack as a non-contributing strip for later use, and create a new action

### `unlink`

bpy.ops.action.unlink(force_delete=False)
Unlink this action from the active action slot (and/or exit Tweak Mode)

### `view_all`

bpy.ops.action.view_all()
Reset viewable area to show full keyframe range

### `view_frame`

bpy.ops.action.view_frame()
Move the view to the current frame

### `view_selected`

bpy.ops.action.view_selected()
Reset viewable area to show selected keyframes range
