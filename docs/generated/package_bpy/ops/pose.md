# pose

Part of `bpy.ops`
Module: `bpy.ops.pose`

## Operators (51)

### `armature_apply`

bpy.ops.pose.armature_apply(selected=False)
Apply the current pose as the new rest pose

### `autoside_names`

bpy.ops.pose.autoside_names(axis='XAXIS')
Automatically renames the selected bones according to which side of the target axis they fall on

### `blend_to_neighbor`

bpy.ops.pose.blend_to_neighbor(factor=0.5, prev_frame=0, next_frame=0, channels='ALL', axis_lock='FREE')
Blend from current position to previous or next keyframe

### `blend_with_rest`

bpy.ops.pose.blend_with_rest(factor=0.5, prev_frame=0, next_frame=0, channels='ALL', axis_lock='FREE')
Make the current pose more similar to, or further away from, the rest pose

### `breakdown`

bpy.ops.pose.breakdown(factor=0.5, prev_frame=0, next_frame=0, channels='ALL', axis_lock='FREE')
Create a suitable breakdown pose on the current frame

### `constraint_add`

bpy.ops.pose.constraint_add(type='<UNKNOWN ENUM>')
Add a constraint to the active bone

### `constraint_add_with_targets`

bpy.ops.pose.constraint_add_with_targets(type='<UNKNOWN ENUM>')
Add a constraint to the active bone, with target (where applicable) set to the selected Objects/Bones

### `constraints_clear`

bpy.ops.pose.constraints_clear()
Clear all constraints from the selected bones

### `constraints_copy`

bpy.ops.pose.constraints_copy()
Copy constraints to other selected bones

### `copy`

bpy.ops.pose.copy()
Copy the current pose of the selected bones to the internal clipboard

### `flip_names`

bpy.ops.pose.flip_names(do_strip_numbers=False)
Flips (and corrects) the axis suffixes of the names of selected bones

### `hide`

bpy.ops.pose.hide(unselected=False)
Tag selected bones to not be visible in Pose Mode

### `ik_add`

bpy.ops.pose.ik_add(with_targets=True)
Add IK Constraint to the active Bone

### `ik_clear`

bpy.ops.pose.ik_clear()
Remove all IK Constraints from selected bones

### `loc_clear`

bpy.ops.pose.loc_clear()
Reset locations of selected bones to their default values

### `paste`

bpy.ops.pose.paste(flipped=False, selected_mask=False)
Paste the stored pose on to the current pose

### `paths_calculate`

bpy.ops.pose.paths_calculate(display_type='RANGE', range='SCENE', bake_location='HEADS')
Calculate paths for the selected bones

### `paths_clear`

bpy.ops.pose.paths_clear(only_selected=False)
(undocumented operator)

### `paths_range_update`

bpy.ops.pose.paths_range_update()
Update frame range for motion paths from the Scene's current frame range

### `paths_update`

bpy.ops.pose.paths_update()
Recalculate paths for bones that already have them

### `propagate`

bpy.ops.pose.propagate(mode='NEXT_KEY', end_frame=250)
Copy selected aspects of the current pose to subsequent poses already keyframed

### `push`

bpy.ops.pose.push(factor=0.5, prev_frame=0, next_frame=0, channels='ALL', axis_lock='FREE')
Exaggerate the current pose in regards to the breakdown pose

### `quaternions_flip`

bpy.ops.pose.quaternions_flip()
Flip quaternion values to achieve desired rotations, while maintaining the same orientations

### `relax`

bpy.ops.pose.relax(factor=0.5, prev_frame=0, next_frame=0, channels='ALL', axis_lock='FREE')
Make the current pose more similar to its breakdown pose

### `reveal`

bpy.ops.pose.reveal(select=True)
Reveal all bones hidden in Pose Mode

### `rot_clear`

bpy.ops.pose.rot_clear()
Reset rotations of selected bones to their default values

### `rotation_mode_set`

bpy.ops.pose.rotation_mode_set(type='QUATERNION')
Set the rotation representation used by selected bones

### `scale_clear`

bpy.ops.pose.scale_clear()
Reset scaling of selected bones to their default values

### `select_all`

bpy.ops.pose.select_all(action='TOGGLE')
Toggle selection status of all bones

### `select_constraint_target`

bpy.ops.pose.select_constraint_target()
Select bones used as targets for the currently selected bones

### `select_grouped`

bpy.ops.pose.select_grouped(extend=False, type='COLLECTION')
Select all visible bones grouped by similar properties

### `select_hierarchy`

bpy.ops.pose.select_hierarchy(direction='PARENT', extend=False)
Select immediate parent/children of selected bones

### `select_linked`

bpy.ops.pose.select_linked()
Select all bones linked by parent/child connections to the current selection

### `select_linked_pick`

bpy.ops.pose.select_linked_pick(extend=False)
Select bones linked by parent/child connections under the mouse cursor

### `select_mirror`

bpy.ops.pose.select_mirror(only_active=False, extend=False)
Mirror the bone selection

### `select_parent`

bpy.ops.pose.select_parent()
Select bones that are parents of the currently selected bones

### `selection_set_add`

bpy.ops.pose.selection_set_add()
Create a new empty Selection Set

### `selection_set_add_and_assign`

bpy.ops.pose.selection_set_add_and_assign()
Create a new Selection Set with the currently selected bones

### `selection_set_assign`

bpy.ops.pose.selection_set_assign()
Add selected bones to Selection Set

### `selection_set_copy`

bpy.ops.pose.selection_set_copy()
Copy the selected Selection Set(s) to the clipboard

### `selection_set_delete_all`

bpy.ops.pose.selection_set_delete_all()
Remove all Selection Sets from this Armature

### `selection_set_deselect`

bpy.ops.pose.selection_set_deselect()
Remove Selection Set bones from current selection

### `selection_set_move`

bpy.ops.pose.selection_set_move(direction='UP')
Move the active Selection Set up/down the list of sets

### `selection_set_paste`

bpy.ops.pose.selection_set_paste()
Add new Selection Set(s) from the clipboard

### `selection_set_remove`

bpy.ops.pose.selection_set_remove()
Remove a Selection Set from this Armature

### `selection_set_remove_bones`

bpy.ops.pose.selection_set_remove_bones()
Remove the selected bones from all Selection Sets

### `selection_set_select`

bpy.ops.pose.selection_set_select(selection_set_index=-1)
Select the bones from this Selection Set

### `selection_set_unassign`

bpy.ops.pose.selection_set_unassign()
Remove selected bones from Selection Set

### `transforms_clear`

bpy.ops.pose.transforms_clear()
Reset location, rotation, and scaling of selected bones to their default values

### `user_transforms_clear`

bpy.ops.pose.user_transforms_clear(only_selected=True)
Reset pose bone transforms to keyframed state

### `visual_transform_apply`

bpy.ops.pose.visual_transform_apply()
Apply final constrained position of pose bones to their transform
