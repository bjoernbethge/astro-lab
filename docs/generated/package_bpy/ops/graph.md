# graph

Part of `bpy.ops`
Module: `bpy.ops.graph`

## Operators (65)

### `bake_keys`

bpy.ops.graph.bake_keys()
Add keyframes on every frame between the selected keyframes

### `blend_offset`

bpy.ops.graph.blend_offset(factor=0)
Shift selected keys to the value of the neighboring keys as a block

### `blend_to_default`

bpy.ops.graph.blend_to_default(factor=0)
Blend selected keys to their default value from their current position

### `blend_to_ease`

bpy.ops.graph.blend_to_ease(factor=0)
Blends keyframes from current state to an ease-in or ease-out curve

### `blend_to_neighbor`

bpy.ops.graph.blend_to_neighbor(factor=0)
Blend selected keyframes to their left or right neighbor

### `breakdown`

bpy.ops.graph.breakdown(factor=0)
Move selected keyframes to an inbetween position relative to adjacent keys

### `butterworth_smooth`

bpy.ops.graph.butterworth_smooth(cutoff_frequency=3, filter_order=4, samples_per_frame=1, blend=1, blend_in_out=1)
Smooth an F-Curve while maintaining the general shape of the curve

### `clean`

bpy.ops.graph.clean(threshold=0.001, channels=False)
Simplify F-Curves by removing closely spaced keyframes

### `click_insert`

bpy.ops.graph.click_insert(frame=1, value=1, extend=False)
Insert new keyframe at the cursor position for the active F-Curve

### `clickselect`

bpy.ops.graph.clickselect(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, deselect_all=False, column=False, curves=False)
Select keyframes by clicking on them

### `copy`

bpy.ops.graph.copy()
Copy selected keyframes to the internal clipboard

### `cursor_set`

bpy.ops.graph.cursor_set(frame=0, value=0)
Interactively set the current frame and value cursor

### `decimate`

bpy.ops.graph.decimate(mode='RATIO', factor=0.333333, remove_error_margin=0)
Decimate F-Curves by removing keyframes that influence the curve shape the least

### `delete`

bpy.ops.graph.delete(confirm=True)
Remove all selected keyframes

### `driver_delete_invalid`

bpy.ops.graph.driver_delete_invalid()
Delete all visible drivers considered invalid

### `driver_variables_copy`

bpy.ops.graph.driver_variables_copy()
Copy the driver variables of the active driver

### `driver_variables_paste`

bpy.ops.graph.driver_variables_paste(replace=False)
Add copied driver variables to the active driver

### `duplicate`

bpy.ops.graph.duplicate(mode='TRANSLATION')
Make a copy of all selected keyframes

### `duplicate_move`

bpy.ops.graph.duplicate_move(GRAPH_OT_duplicate={"mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Make a copy of all selected keyframes and move them

### `ease`

bpy.ops.graph.ease(factor=0, sharpness=2)
Align keyframes on a ease-in or ease-out curve

### `easing_type`

bpy.ops.graph.easing_type(type='AUTO')
Set easing type for the F-Curve segments starting from the selected keyframes

### `equalize_handles`

bpy.ops.graph.equalize_handles(side='LEFT', handle_length=5, flatten=False)
Ensure selected keyframes' handles have equal length, optionally making them horizontal. Automatic, Automatic Clamped, or Vector handle types will be converted to Aligned

### `euler_filter`

bpy.ops.graph.euler_filter()
Fix large jumps and flips in the selected Euler Rotation F-Curves arising from rotation values being clipped when baking physics

### `extrapolation_type`

bpy.ops.graph.extrapolation_type(type='CONSTANT')
Set extrapolation mode for selected F-Curves

### `fmodifier_add`

bpy.ops.graph.fmodifier_add(type='NULL', only_active=False)
Add F-Modifier to the active/selected F-Curves

### `fmodifier_copy`

bpy.ops.graph.fmodifier_copy()
Copy the F-Modifier(s) of the active F-Curve

### `fmodifier_paste`

bpy.ops.graph.fmodifier_paste(only_active=False, replace=False)
Add copied F-Modifiers to the selected F-Curves

### `frame_jump`

bpy.ops.graph.frame_jump()
Place the cursor on the midpoint of selected keyframes

### `gaussian_smooth`

bpy.ops.graph.gaussian_smooth(factor=1, sigma=0.33, filter_width=6)
Smooth the curve using a Gaussian filter

### `ghost_curves_clear`

bpy.ops.graph.ghost_curves_clear()
Clear F-Curve snapshots (Ghosts) for active Graph Editor

### `ghost_curves_create`

bpy.ops.graph.ghost_curves_create()
Create snapshot (Ghosts) of selected F-Curves as background aid for active Graph Editor

### `handle_type`

bpy.ops.graph.handle_type(type='FREE')
Set type of handle for selected keyframes

### `hide`

bpy.ops.graph.hide(unselected=False)
Hide selected curves from Graph Editor view

### `interpolation_type`

bpy.ops.graph.interpolation_type(type='CONSTANT')
Set interpolation mode for the F-Curve segments starting from the selected keyframes

### `keyframe_insert`

bpy.ops.graph.keyframe_insert(type='ALL')
Insert keyframes for the specified channels

### `keyframe_jump`

bpy.ops.graph.keyframe_jump(next=True)
Jump to previous/next keyframe

### `keys_to_samples`

bpy.ops.graph.keys_to_samples()
Convert selected channels to an uneditable set of samples to save storage space

### `match_slope`

bpy.ops.graph.match_slope(factor=0)
Blend selected keys to the slope of neighboring ones

### `mirror`

bpy.ops.graph.mirror(type='CFRA')
Flip selected keyframes over the selected mirror line

### `paste`

bpy.ops.graph.paste(offset='START', value_offset='NONE', merge='MIX', flipped=False)
Paste keyframes from the internal clipboard for the selected channels, starting on the current frame

### `previewrange_set`

bpy.ops.graph.previewrange_set()
Set Preview Range based on range of selected keyframes

### `push_pull`

bpy.ops.graph.push_pull(factor=1)
Exaggerate or minimize the value of the selected keys

### `reveal`

bpy.ops.graph.reveal(select=True)
Make previously hidden curves visible again in Graph Editor view

### `samples_to_keys`

bpy.ops.graph.samples_to_keys()
Convert selected channels from samples to keyframes

### `scale_average`

bpy.ops.graph.scale_average(factor=1)
Scale selected key values by their combined average

### `scale_from_neighbor`

bpy.ops.graph.scale_from_neighbor(factor=0, anchor='LEFT')
Increase or decrease the value of selected keys in relationship to the neighboring one

### `select_all`

bpy.ops.graph.select_all(action='TOGGLE')
Toggle selection of all keyframes

### `select_box`

bpy.ops.graph.select_box(axis_range=False, include_handles=True, tweak=False, use_curve_selection=True, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Select all keyframes within the specified region

### `select_circle`

bpy.ops.graph.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET', use_curve_selection=True)
Select keyframe points using circle selection

### `select_column`

bpy.ops.graph.select_column(mode='KEYS')
Select all keyframes on the specified frame(s)

### `select_key_handles`

bpy.ops.graph.select_key_handles(left_handle_action='SELECT', right_handle_action='SELECT', key_action='KEEP')
For selected keyframes, select/deselect any combination of the key itself and its handles

### `select_lasso`

bpy.ops.graph.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET', use_curve_selection=True)
Select keyframe points using lasso selection

### `select_leftright`

bpy.ops.graph.select_leftright(mode='CHECK', extend=False)
Select keyframes to the left or the right of the current frame

### `select_less`

bpy.ops.graph.select_less()
Deselect keyframes on ends of selection islands

### `select_linked`

bpy.ops.graph.select_linked()
Select keyframes occurring in the same F-Curves as selected ones

### `select_more`

bpy.ops.graph.select_more()
Select keyframes beside already selected ones

### `shear`

bpy.ops.graph.shear(factor=0, direction='FROM_LEFT')
Affect the value of the keys linearly, keeping the same relationship between them using either the left or the right key as reference

### `smooth`

bpy.ops.graph.smooth()
Apply weighted moving means to make selected F-Curves less bumpy

### `snap`

bpy.ops.graph.snap(type='CFRA')
Snap selected keyframes to the chosen times/values

### `snap_cursor_value`

bpy.ops.graph.snap_cursor_value()
Place the cursor value on the average value of selected keyframes

### `sound_to_samples`

bpy.ops.graph.sound_to_samples(filepath="", check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=True, filter_python=False, filter_font=False, filter_sound=True, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', low=0, high=100000, attack=0.005, release=0.2, threshold=0, use_accumulate=False, use_additive=False, use_square=False, sthreshold=0.1)
Bakes a sound wave to samples on selected channels

### `time_offset`

bpy.ops.graph.time_offset(frame_offset=0)
Shifts the value of selected keys in time

### `view_all`

bpy.ops.graph.view_all(include_handles=True)
Reset viewable area to show full keyframe range

### `view_frame`

bpy.ops.graph.view_frame()
Move the view to the current frame

### `view_selected`

bpy.ops.graph.view_selected(include_handles=True)
Reset viewable area to show selected keyframe range
