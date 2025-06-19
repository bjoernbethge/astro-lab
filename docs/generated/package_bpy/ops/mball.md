# mball

Part of `bpy.ops`
Module: `bpy.ops.mball`

## Operators (8)

### `delete_metaelems`

bpy.ops.mball.delete_metaelems(confirm=True)
Delete selected metaball element(s)

### `duplicate_metaelems`

bpy.ops.mball.duplicate_metaelems()
Duplicate selected metaball element(s)

### `duplicate_move`

bpy.ops.mball.duplicate_move(MBALL_OT_duplicate_metaelems={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Make copies of the selected metaball elements and move them

### `hide_metaelems`

bpy.ops.mball.hide_metaelems(unselected=False)
Hide (un)selected metaball element(s)

### `reveal_metaelems`

bpy.ops.mball.reveal_metaelems(select=True)
Reveal all hidden metaball elements

### `select_all`

bpy.ops.mball.select_all(action='TOGGLE')
Change selection of all metaball elements

### `select_random_metaelems`

bpy.ops.mball.select_random_metaelems(ratio=0.5, seed=0, action='SELECT')
Randomly select metaball elements

### `select_similar`

bpy.ops.mball.select_similar(type='TYPE', threshold=0.1)
Select similar metaballs by property types
