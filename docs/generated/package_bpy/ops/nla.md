# nla

Part of `bpy.ops`
Module: `bpy.ops.nla`

## Operators (39)

### `action_pushdown`

bpy.ops.nla.action_pushdown(track_index=-1)
Push action down onto the top of the NLA stack as a new strip

### `action_sync_length`

bpy.ops.nla.action_sync_length(active=True)
Synchronize the length of the referenced Action with the length used in the strip

### `action_unlink`

bpy.ops.nla.action_unlink(force_delete=False)
Unlink this action from the active action slot (and/or exit Tweak Mode)

### `actionclip_add`

bpy.ops.nla.actionclip_add(action='<UNKNOWN ENUM>')
Add an Action-Clip strip (i.e. an NLA Strip referencing an Action) to the active track

### `apply_scale`

bpy.ops.nla.apply_scale()
Apply scaling of selected strips to their referenced Actions

### `bake`

bpy.ops.nla.bake(frame_start=1, frame_end=250, step=1, only_selected=True, visual_keying=False, clear_constraints=False, clear_parents=False, use_current_action=False, clean_curves=False, bake_types={'POSE'}, channel_types={'LOCATION', 'ROTATION', 'SCALE', 'BBONE', 'PROPS'})
Bake all selected objects location/scale/rotation animation to an action

### `channels_click`

bpy.ops.nla.channels_click(extend=False)
Handle clicks to select NLA tracks

### `clear_scale`

bpy.ops.nla.clear_scale()
Reset scaling of selected strips

### `click_select`

bpy.ops.nla.click_select(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, deselect_all=False)
Handle clicks to select NLA Strips

### `delete`

bpy.ops.nla.delete()
Delete selected strips

### `duplicate`

bpy.ops.nla.duplicate(linked=False)
Duplicate selected NLA-Strips, adding the new strips to new track(s)

### `duplicate_linked_move`

bpy.ops.nla.duplicate_linked_move(NLA_OT_duplicate={"linked":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate Linked selected NLA-Strips, adding the new strips to new track(s)

### `duplicate_move`

bpy.ops.nla.duplicate_move(NLA_OT_duplicate={"linked":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate selected NLA-Strips, adding the new strips to new track(s)

### `fmodifier_add`

bpy.ops.nla.fmodifier_add(type='NULL', only_active=True)
Add F-Modifier to the active/selected NLA-Strips

### `fmodifier_copy`

bpy.ops.nla.fmodifier_copy()
Copy the F-Modifier(s) of the active NLA-Strip

### `fmodifier_paste`

bpy.ops.nla.fmodifier_paste(only_active=True, replace=False)
Add copied F-Modifiers to the selected NLA-Strips

### `make_single_user`

bpy.ops.nla.make_single_user(confirm=True)
Make linked action local to each strip

### `meta_add`

bpy.ops.nla.meta_add()
Add new meta-strips incorporating the selected strips

### `meta_remove`

bpy.ops.nla.meta_remove()
Separate out the strips held by the selected meta-strips

### `move_down`

bpy.ops.nla.move_down()
Move selected strips down a track if there's room

### `move_up`

bpy.ops.nla.move_up()
Move selected strips up a track if there's room

### `mute_toggle`

bpy.ops.nla.mute_toggle()
Mute or un-mute selected strips

### `previewrange_set`

bpy.ops.nla.previewrange_set()
Set Preview Range based on extends of selected strips

### `select_all`

bpy.ops.nla.select_all(action='TOGGLE')
Select or deselect all NLA-Strips

### `select_box`

bpy.ops.nla.select_box(axis_range=False, tweak=False, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Use box selection to grab NLA-Strips

### `select_leftright`

bpy.ops.nla.select_leftright(mode='CHECK', extend=False)
Select strips to the left or the right of the current frame

### `selected_objects_add`

bpy.ops.nla.selected_objects_add()
Make selected objects appear in NLA Editor by adding Animation Data

### `snap`

bpy.ops.nla.snap(type='CFRA')
Move start of strips to specified time

### `soundclip_add`

bpy.ops.nla.soundclip_add()
Add a strip for controlling when speaker plays its sound clip

### `split`

bpy.ops.nla.split()
Split selected strips at their midpoints

### `swap`

bpy.ops.nla.swap()
Swap order of selected strips within tracks

### `tracks_add`

bpy.ops.nla.tracks_add(above_selected=False)
Add NLA-Tracks above/after the selected tracks

### `tracks_delete`

bpy.ops.nla.tracks_delete()
Delete selected NLA-Tracks and the strips they contain

### `transition_add`

bpy.ops.nla.transition_add()
Add a transition strip between two adjacent selected strips

### `tweakmode_enter`

bpy.ops.nla.tweakmode_enter(isolate_action=False, use_upper_stack_evaluation=False)
Enter tweaking mode for the action referenced by the active strip to edit its keyframes

### `tweakmode_exit`

bpy.ops.nla.tweakmode_exit(isolate_action=False)
Exit tweaking mode for the action referenced by the active strip

### `view_all`

bpy.ops.nla.view_all()
Reset viewable area to show full strips range

### `view_frame`

bpy.ops.nla.view_frame()
Move the view to the current frame

### `view_selected`

bpy.ops.nla.view_selected()
Reset viewable area to show selected strips range
