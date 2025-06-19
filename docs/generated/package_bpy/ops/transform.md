# transform

Part of `bpy.ops`
Module: `bpy.ops.transform`

## Operators (27)

### `bbone_resize`

bpy.ops.transform.bbone_resize(value=(1, 1, 1), orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, release_confirm=False, use_accurate=False)
Scale selected bendy bones display size

### `bend`

bpy.ops.transform.bend(value=(0, ), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, gpencil_strokes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Bend selected items between the 3D cursor and the mouse

### `create_orientation`

bpy.ops.transform.create_orientation(name="", use_view=False, use=False, overwrite=False)
Create transformation orientation from selection

### `delete_orientation`

bpy.ops.transform.delete_orientation()
Delete transformation orientation

### `edge_bevelweight`

bpy.ops.transform.edge_bevelweight(value=0, snap=False, release_confirm=False, use_accurate=False)
Change the bevel weight of edges

### `edge_crease`

bpy.ops.transform.edge_crease(value=0, snap=False, release_confirm=False, use_accurate=False)
Change the crease of edges

### `edge_slide`

bpy.ops.transform.edge_slide(value=0, single_side=False, use_even=False, flipped=False, use_clamp=True, mirror=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), correct_uv=True, release_confirm=False, use_accurate=False)
Slide an edge loop along a mesh

### `from_gizmo`

bpy.ops.transform.from_gizmo()
Transform selected items by mode type

### `mirror`

bpy.ops.transform.mirror(orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), gpencil_strokes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Mirror selected items around one or more axes

### `push_pull`

bpy.ops.transform.push_pull(value=0, mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Push/Pull selected items

### `resize`

bpy.ops.transform.resize(value=(1, 1, 1), mouse_dir_constraint=(0, 0, 0), orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), gpencil_strokes=False, texture_space=False, remove_on_cancel=False, use_duplicated_keyframes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Scale (resize) selected items

### `rotate`

bpy.ops.transform.rotate(value=0, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), gpencil_strokes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Rotate selected items

### `rotate_normal`

bpy.ops.transform.rotate_normal(value=0, orient_axis='Z', orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, release_confirm=False, use_accurate=False)
Rotate split normal of selected items

### `select_orientation`

bpy.ops.transform.select_orientation(orientation='GLOBAL')
Select transformation orientation

### `seq_slide`

bpy.ops.transform.seq_slide(value=(0, 0), use_restore_handle_selection=False, snap=False, view2d_edge_pan=False, release_confirm=False, use_accurate=False)
Slide a sequence strip in time

### `shear`

bpy.ops.transform.shear(value=0, orient_axis='Z', orient_axis_ortho='X', orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, gpencil_strokes=False, release_confirm=False, use_accurate=False)
Shear selected items along the given axis

### `shrink_fatten`

bpy.ops.transform.shrink_fatten(value=0, use_even_offset=False, mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, release_confirm=False, use_accurate=False)
Shrink/fatten selected vertices along normals

### `skin_resize`

bpy.ops.transform.skin_resize(value=(1, 1, 1), orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), release_confirm=False, use_accurate=False)
Scale selected vertices' skin radii

### `tilt`

bpy.ops.transform.tilt(value=0, mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, release_confirm=False, use_accurate=False)
Tilt selected control vertices of 3D curve

### `tosphere`

bpy.ops.transform.tosphere(value=0, mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, gpencil_strokes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Move selected items outward in a spherical shape around geometric center

### `trackball`

bpy.ops.transform.trackball(value=(0, 0), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, gpencil_strokes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False)
Trackball style rotation of selected items

### `transform`

bpy.ops.transform.transform(mode='TRANSLATION', value=(0, 0, 0, 0), orient_axis='Z', orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), snap_align=False, snap_normal=(0, 0, 0), gpencil_strokes=False, texture_space=False, remove_on_cancel=False, use_duplicated_keyframes=False, center_override=(0, 0, 0), release_confirm=False, use_accurate=False, use_automerge_and_split=False)
Transform selected items by mode type

### `translate`

bpy.ops.transform.translate(value=(0, 0, 0), orient_type='GLOBAL', orient_matrix=((0, 0, 0), (0, 0, 0), (0, 0, 0)), orient_matrix_type='GLOBAL', constraint_axis=(False, False, False), mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), snap_align=False, snap_normal=(0, 0, 0), gpencil_strokes=False, cursor_transform=False, texture_space=False, remove_on_cancel=False, use_duplicated_keyframes=False, view2d_edge_pan=False, release_confirm=False, use_accurate=False, use_automerge_and_split=False)
Move selected items

### `vert_crease`

bpy.ops.transform.vert_crease(value=0, snap=False, release_confirm=False, use_accurate=False)
Change the crease of vertices

### `vert_slide`

bpy.ops.transform.vert_slide(value=0, use_even=False, flipped=False, use_clamp=True, mirror=False, snap=False, snap_elements={'INCREMENT'}, use_snap_project=False, snap_target='CLOSEST', use_snap_self=True, use_snap_edit=True, use_snap_nonedit=True, use_snap_selectable=False, snap_point=(0, 0, 0), correct_uv=True, release_confirm=False, use_accurate=False)
Slide a vertex along a mesh

### `vertex_random`

bpy.ops.transform.vertex_random(offset=0, uniform=0, normal=0, seed=0, wait_for_input=True)
Randomize vertices

### `vertex_warp`

bpy.ops.transform.vertex_warp(warp_angle=6.28319, offset_angle=0, min=-1, max=1, viewmat=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), center=(0, 0, 0))
Warp vertices around the cursor
