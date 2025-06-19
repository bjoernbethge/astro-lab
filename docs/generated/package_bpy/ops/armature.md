# armature

Part of `bpy.ops`
Module: `bpy.ops.armature`

## Operators (48)

### `align`

bpy.ops.armature.align()
Align selected bones to the active bone (or to their parent)

### `assign_to_collection`

bpy.ops.armature.assign_to_collection(collection_index=-1, new_collection_name="")
Assign all selected bones to a collection, or unassign them, depending on whether the active bone is already assigned or not

### `autoside_names`

bpy.ops.armature.autoside_names(type='XAXIS')
Automatically renames the selected bones according to which side of the target axis they fall on

### `bone_primitive_add`

bpy.ops.armature.bone_primitive_add(name="")
Add a new bone located at the 3D cursor

### `calculate_roll`

bpy.ops.armature.calculate_roll(type='POS_X', axis_flip=False, axis_only=False)
Automatically fix alignment of select bones' axes

### `click_extrude`

bpy.ops.armature.click_extrude()
Create a new bone going from the last selected joint to the mouse position

### `collection_add`

bpy.ops.armature.collection_add()
Add a new bone collection

### `collection_assign`

bpy.ops.armature.collection_assign(name="")
Add selected bones to the chosen bone collection

### `collection_create_and_assign`

bpy.ops.armature.collection_create_and_assign(name="")
Create a new bone collection and assign all selected bones

### `collection_deselect`

bpy.ops.armature.collection_deselect()
Deselect bones of active Bone Collection

### `collection_move`

bpy.ops.armature.collection_move(direction='UP')
Change position of active Bone Collection in list of Bone collections

### `collection_remove`

bpy.ops.armature.collection_remove()
Remove the active bone collection

### `collection_remove_unused`

bpy.ops.armature.collection_remove_unused()
Remove all bone collections that have neither bones nor children. This is done recursively, so bone collections that only have unused children are also removed

### `collection_select`

bpy.ops.armature.collection_select()
Select bones in active Bone Collection

### `collection_show_all`

bpy.ops.armature.collection_show_all()
Show all bone collections

### `collection_unassign`

bpy.ops.armature.collection_unassign(name="")
Remove selected bones from the active bone collection

### `collection_unassign_named`

bpy.ops.armature.collection_unassign_named(name="", bone_name="")
Unassign the named bone from this bone collection

### `collection_unsolo_all`

bpy.ops.armature.collection_unsolo_all()
Clear the 'solo' setting on all bone collections

### `copy_bone_color_to_selected`

bpy.ops.armature.copy_bone_color_to_selected(bone_type='EDIT')
Copy the bone color of the active bone to all selected bones

### `delete`

bpy.ops.armature.delete(confirm=True)
Remove selected bones from the armature

### `dissolve`

bpy.ops.armature.dissolve()
Dissolve selected bones from the armature

### `duplicate`

bpy.ops.armature.duplicate(do_flip_names=False)
Make copies of the selected bones within the same armature

### `duplicate_move`

bpy.ops.armature.duplicate_move(ARMATURE_OT_duplicate={"do_flip_names":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Make copies of the selected bones within the same armature and move them

### `extrude`

bpy.ops.armature.extrude(forked=False)
Create new bones from the selected joints

### `extrude_forked`

bpy.ops.armature.extrude_forked(ARMATURE_OT_extrude={"forked":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Create new bones from the selected joints and move them

### `extrude_move`

bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Create new bones from the selected joints and move them

### `fill`

bpy.ops.armature.fill()
Add bone between selected joint(s) and/or 3D cursor

### `flip_names`

bpy.ops.armature.flip_names(do_strip_numbers=False)
Flips (and corrects) the axis suffixes of the names of selected bones

### `hide`

bpy.ops.armature.hide(unselected=False)
Tag selected bones to not be visible in Edit Mode

### `move_to_collection`

bpy.ops.armature.move_to_collection(collection_index=-1, new_collection_name="")
Move bones to a collection

### `parent_clear`

bpy.ops.armature.parent_clear(type='CLEAR')
Remove the parent-child relationship between selected bones and their parents

### `parent_set`

bpy.ops.armature.parent_set(type='CONNECTED')
Set the active bone as the parent of the selected bones

### `reveal`

bpy.ops.armature.reveal(select=True)
Reveal all bones hidden in Edit Mode

### `roll_clear`

bpy.ops.armature.roll_clear(roll=0)
Clear roll for selected bones

### `select_all`

bpy.ops.armature.select_all(action='TOGGLE')
Toggle selection status of all bones

### `select_hierarchy`

bpy.ops.armature.select_hierarchy(direction='PARENT', extend=False)
Select immediate parent/children of selected bones

### `select_less`

bpy.ops.armature.select_less()
Deselect those bones at the boundary of each selection region

### `select_linked`

bpy.ops.armature.select_linked(all_forks=False)
Select all bones linked by parent/child connections to the current selection

### `select_linked_pick`

bpy.ops.armature.select_linked_pick(deselect=False, all_forks=False)
(De)select bones linked by parent/child connections under the mouse cursor

### `select_mirror`

bpy.ops.armature.select_mirror(only_active=False, extend=False)
Mirror the bone selection

### `select_more`

bpy.ops.armature.select_more()
Select those bones connected to the initial selection

### `select_similar`

bpy.ops.armature.select_similar(type='LENGTH', threshold=0.1)
Select similar bones by property types

### `separate`

bpy.ops.armature.separate()
Isolate selected bones into a separate armature

### `shortest_path_pick`

bpy.ops.armature.shortest_path_pick()
Select shortest path between two bones

### `split`

bpy.ops.armature.split()
Split off selected bones from connected unselected bones

### `subdivide`

bpy.ops.armature.subdivide(number_cuts=1)
Break selected bones into chains of smaller bones

### `switch_direction`

bpy.ops.armature.switch_direction()
Change the direction that a chain of bones points in (head and tail swap)

### `symmetrize`

bpy.ops.armature.symmetrize(direction='NEGATIVE_X')
Enforce symmetry, make copies of the selection or use existing
