# anim

Part of `bpy.ops`
Module: `bpy.ops.anim`

## Operators (61)

### `change_frame`

bpy.ops.anim.change_frame(frame=0, snap=False)
Interactively change the current frame number

### `channel_select_keys`

bpy.ops.anim.channel_select_keys(extend=False)
Select all keyframes of channel under mouse

### `channel_view_pick`

bpy.ops.anim.channel_view_pick(include_handles=True, use_preview_range=True)
Reset viewable area to show the channel under the cursor

### `channels_bake`

bpy.ops.anim.channels_bake(range=(0, 0), step=1, remove_outside_range=False, interpolation_type='BEZIER', bake_modifiers=True)
Create keyframes following the current shape of F-Curves of selected channels

### `channels_clean_empty`

bpy.ops.anim.channels_clean_empty()
Delete all empty animation data containers from visible data-blocks

### `channels_click`

bpy.ops.anim.channels_click(extend=False, extend_range=False, children_only=False)
Handle mouse clicks over animation channels

### `channels_collapse`

bpy.ops.anim.channels_collapse(all=True)
Collapse (close) all selected expandable animation channels

### `channels_delete`

bpy.ops.anim.channels_delete()
Delete all selected animation channels

### `channels_editable_toggle`

bpy.ops.anim.channels_editable_toggle(mode='TOGGLE', type='PROTECT')
Toggle editability of selected channels

### `channels_expand`

bpy.ops.anim.channels_expand(all=True)
Expand (open) all selected expandable animation channels

### `channels_fcurves_enable`

bpy.ops.anim.channels_fcurves_enable()
Clear 'disabled' tag from all F-Curves to get broken F-Curves working again

### `channels_group`

bpy.ops.anim.channels_group(name="New Group")
Add selected F-Curves to a new group

### `channels_move`

bpy.ops.anim.channels_move(direction='DOWN')
Rearrange selected animation channels

### `channels_rename`

bpy.ops.anim.channels_rename()
Rename animation channel under mouse

### `channels_select_all`

bpy.ops.anim.channels_select_all(action='TOGGLE')
Toggle selection of all animation channels

### `channels_select_box`

bpy.ops.anim.channels_select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, deselect=False, extend=True)
Select all animation channels within the specified region

### `channels_select_filter`

bpy.ops.anim.channels_select_filter()
Start entering text which filters the set of channels shown to only include those with matching names

### `channels_setting_disable`

bpy.ops.anim.channels_setting_disable(mode='DISABLE', type='PROTECT')
Disable specified setting on all selected animation channels

### `channels_setting_enable`

bpy.ops.anim.channels_setting_enable(mode='ENABLE', type='PROTECT')
Enable specified setting on all selected animation channels

### `channels_setting_toggle`

bpy.ops.anim.channels_setting_toggle(mode='TOGGLE', type='PROTECT')
Toggle specified setting on all selected animation channels

### `channels_ungroup`

bpy.ops.anim.channels_ungroup()
Remove selected F-Curves from their current groups

### `channels_view_selected`

bpy.ops.anim.channels_view_selected(include_handles=True, use_preview_range=True)
Reset viewable area to show the selected channels

### `clear_useless_actions`

bpy.ops.anim.clear_useless_actions(only_unused=True)
Mark actions with no F-Curves for deletion after save and reload of file preserving "action libraries"

### `convert_legacy_action`

bpy.ops.anim.convert_legacy_action()
Convert a legacy Action to a layered Action on the active object

### `copy_driver_button`

bpy.ops.anim.copy_driver_button()
Copy the driver for the highlighted button

### `driver_button_add`

bpy.ops.anim.driver_button_add()
Add driver for the property under the cursor

### `driver_button_edit`

bpy.ops.anim.driver_button_edit()
Edit the drivers for the connected property represented by the highlighted button

### `driver_button_remove`

bpy.ops.anim.driver_button_remove(all=True)
Remove the driver(s) for the connected property(s) represented by the highlighted button

### `end_frame_set`

bpy.ops.anim.end_frame_set()
Set the current frame as the preview or scene end frame

### `keyframe_clear_button`

bpy.ops.anim.keyframe_clear_button(all=True)
Clear all keyframes on the currently active property

### `keyframe_clear_v3d`

bpy.ops.anim.keyframe_clear_v3d(confirm=True)
Remove all keyframe animation for selected objects

### `keyframe_delete`

bpy.ops.anim.keyframe_delete(type='DEFAULT')
Delete keyframes on the current frame for all properties in the specified Keying Set

### `keyframe_delete_button`

bpy.ops.anim.keyframe_delete_button(all=True)
Delete current keyframe of current UI-active property

### `keyframe_delete_by_name`

bpy.ops.anim.keyframe_delete_by_name(type="")
Alternate access to 'Delete Keyframe' for keymaps to use

### `keyframe_delete_v3d`

bpy.ops.anim.keyframe_delete_v3d(confirm=True)
Remove keyframes on current frame for selected objects and bones

### `keyframe_insert`

bpy.ops.anim.keyframe_insert(type='DEFAULT')
Insert keyframes on the current frame using either the active keying set, or the user preferences if no keying set is active

### `keyframe_insert_button`

bpy.ops.anim.keyframe_insert_button(all=True)
Insert a keyframe for current UI-active property

### `keyframe_insert_by_name`

bpy.ops.anim.keyframe_insert_by_name(type="")
Alternate access to 'Insert Keyframe' for keymaps to use

### `keyframe_insert_menu`

bpy.ops.anim.keyframe_insert_menu(type='DEFAULT', always_prompt=False)
Insert Keyframes for specified Keying Set, with menu of available Keying Sets if undefined

### `keying_set_active_set`

bpy.ops.anim.keying_set_active_set(type='DEFAULT')
Set a new active keying set

### `keying_set_add`

bpy.ops.anim.keying_set_add()
Add a new (empty) keying set to the active Scene

### `keying_set_export`

bpy.ops.anim.keying_set_export(filepath="", filter_folder=True, filter_text=True, filter_python=True)
Export Keying Set to a Python script

### `keying_set_path_add`

bpy.ops.anim.keying_set_path_add()
Add empty path to active keying set

### `keying_set_path_remove`

bpy.ops.anim.keying_set_path_remove()
Remove active Path from active keying set

### `keying_set_remove`

bpy.ops.anim.keying_set_remove()
Remove the active keying set

### `keyingset_button_add`

bpy.ops.anim.keyingset_button_add(all=True)
Add current UI-active property to current keying set

### `keyingset_button_remove`

bpy.ops.anim.keyingset_button_remove()
Remove current UI-active property from current keying set

### `merge_animation`

bpy.ops.anim.merge_animation()
Merge the animation of the selected objects into the action of the active object. Actions are not deleted by this, but might end up with zero users

### `paste_driver_button`

bpy.ops.anim.paste_driver_button()
Paste the driver in the internal clipboard to the highlighted button

### `previewrange_clear`

bpy.ops.anim.previewrange_clear()
Clear preview range

### `previewrange_set`

bpy.ops.anim.previewrange_set(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Interactively define frame range used for playback

### `scene_range_frame`

bpy.ops.anim.scene_range_frame()
Reset the horizontal view to the current scene frame range, taking the preview range into account if it is active

### `separate_slots`

bpy.ops.anim.separate_slots()
Move all slots of the action on the active object into newly created, separate actions. All users of those slots will be reassigned to the new actions. The current action won't be deleted but will be empty and might end up having zero users

### `slot_channels_move_to_new_action`

bpy.ops.anim.slot_channels_move_to_new_action()
Move the selected slots into a newly created action

### `slot_new_for_id`

bpy.ops.anim.slot_new_for_id()
Create a new action slot for this data-block, to hold its animation

### `slot_unassign_from_constraint`

bpy.ops.anim.slot_unassign_from_constraint()
Un-assign the action slot from this constraint

### `slot_unassign_from_id`

bpy.ops.anim.slot_unassign_from_id()
Un-assign the action slot, effectively making this data-block non-animated

### `slot_unassign_from_nla_strip`

bpy.ops.anim.slot_unassign_from_nla_strip()
Un-assign the action slot from this NLA strip, effectively making it non-animated

### `start_frame_set`

bpy.ops.anim.start_frame_set()
Set the current frame as the preview or scene start frame

### `update_animated_transform_constraints`

bpy.ops.anim.update_animated_transform_constraints(use_convert_to_radians=True)
Update f-curves/drivers affecting Transform constraints (use it with files from 2.70 and earlier)

### `view_curve_in_graph_editor`

bpy.ops.anim.view_curve_in_graph_editor(all=False, isolate=False)
Frame the property under the cursor in the Graph Editor
