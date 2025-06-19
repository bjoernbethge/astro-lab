# uv

Part of `bpy.ops`
Module: `bpy.ops.uv`

## Operators (49)

### `align`

bpy.ops.uv.align(axis='ALIGN_AUTO')
Aligns selected UV vertices on a line

### `align_rotation`

bpy.ops.uv.align_rotation(method='AUTO', axis='X', correct_aspect=False)
Align the UV island's rotation

### `average_islands_scale`

bpy.ops.uv.average_islands_scale(scale_uv=False, shear=False)
Average the size of separate UV islands, based on their area in 3D space

### `copy`

bpy.ops.uv.copy()
Copy selected UV vertices

### `cube_project`

bpy.ops.uv.cube_project(cube_size=1, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
Project the UV vertices of the mesh over the six faces of a cube

### `cursor_set`

bpy.ops.uv.cursor_set(location=(0, 0))
Set 2D cursor location

### `cylinder_project`

bpy.ops.uv.cylinder_project(direction='VIEW_ON_EQUATOR', align='POLAR_ZX', pole='PINCH', seam=False, radius=1, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
Project the UV vertices of the mesh over the curved wall of a cylinder

### `export_layout`

bpy.ops.uv.export_layout(filepath="", export_all=False, export_tiles='NONE', modified=False, mode='PNG', size=(1024, 1024), opacity=0.25, check_existing=True)
Export UV layout to file

### `follow_active_quads`

bpy.ops.uv.follow_active_quads(mode='LENGTH_AVERAGE')
Follow UVs from active quads along continuous face loops

### `hide`

bpy.ops.uv.hide(unselected=False)
Hide (un)selected UV vertices

### `lightmap_pack`

bpy.ops.uv.lightmap_pack(PREF_CONTEXT='SEL_FACES', PREF_PACK_IN_ONE=True, PREF_NEW_UVLAYER=False, PREF_BOX_DIV=12, PREF_MARGIN_DIV=0.1)
Pack each face's UVs into the UV bounds

### `mark_seam`

bpy.ops.uv.mark_seam(clear=False)
Mark selected UV edges as seams

### `minimize_stretch`

bpy.ops.uv.minimize_stretch(fill_holes=True, blend=0, iterations=0)
Reduce UV stretching by relaxing angles

### `pack_islands`

bpy.ops.uv.pack_islands(udim_source='CLOSEST_UDIM', rotate=True, rotate_method='ANY', scale=True, merge_overlap=False, margin_method='SCALED', margin=0.001, pin=False, pin_method='LOCKED', shape_method='CONCAVE')
Transform all islands so that they fill up the UV/UDIM space as much as possible

### `paste`

bpy.ops.uv.paste()
Paste selected UV vertices

### `pin`

bpy.ops.uv.pin(clear=False, invert=False)
Set/clear selected UV vertices as anchored between multiple unwrap operations

### `project_from_view`

bpy.ops.uv.project_from_view(orthographic=False, camera_bounds=True, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
Project the UV vertices of the mesh as seen in current 3D view

### `randomize_uv_transform`

bpy.ops.uv.randomize_uv_transform(random_seed=0, use_loc=True, loc=(0, 0), use_rot=True, rot=0, use_scale=True, scale_even=False, scale=(1, 1))
Randomize the UV island's location, rotation, and scale

### `remove_doubles`

bpy.ops.uv.remove_doubles(threshold=0.02, use_unselected=False, use_shared_vertex=False)
Selected UV vertices that are within a radius of each other are welded together

### `reset`

bpy.ops.uv.reset()
Reset UV projection

### `reveal`

bpy.ops.uv.reveal(select=True)
Reveal all hidden UV vertices

### `rip`

bpy.ops.uv.rip(mirror=False, release_confirm=False, use_accurate=False, location=(0, 0))
Rip selected vertices or a selected region

### `rip_move`

bpy.ops.uv.rip_move(UV_OT_rip={"mirror":False, "release_confirm":False, "use_accurate":False, "location":(0, 0)}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Unstitch UVs and move the result

### `seams_from_islands`

bpy.ops.uv.seams_from_islands(mark_seams=True, mark_sharp=False)
Set mesh seams according to island setup in the UV editor

### `select`

bpy.ops.uv.select(extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, location=(0, 0))
Select UV vertices

### `select_all`

bpy.ops.uv.select_all(action='TOGGLE')
Change selection of all UV vertices

### `select_box`

bpy.ops.uv.select_box(pinned=False, xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Select UV vertices using box selection

### `select_circle`

bpy.ops.uv.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Select UV vertices using circle selection

### `select_edge_ring`

bpy.ops.uv.select_edge_ring(extend=False, location=(0, 0))
Select an edge ring of connected UV vertices

### `select_lasso`

bpy.ops.uv.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select UVs using lasso selection

### `select_less`

bpy.ops.uv.select_less()
Deselect UV vertices at the boundary of each selection region

### `select_linked`

bpy.ops.uv.select_linked()
Select all UV vertices linked to the active UV map

### `select_linked_pick`

bpy.ops.uv.select_linked_pick(extend=False, deselect=False, location=(0, 0))
Select all UV vertices linked under the mouse

### `select_loop`

bpy.ops.uv.select_loop(extend=False, location=(0, 0))
Select a loop of connected UV vertices

### `select_mode`

bpy.ops.uv.select_mode(type='VERTEX')
Change UV selection mode

### `select_more`

bpy.ops.uv.select_more()
Select more UV vertices connected to initial selection

### `select_overlap`

bpy.ops.uv.select_overlap(extend=False)
Select all UV faces which overlap each other

### `select_pinned`

bpy.ops.uv.select_pinned()
Select all pinned UV vertices

### `select_similar`

bpy.ops.uv.select_similar(type='PIN', compare='EQUAL', threshold=0)
Select similar UVs by property types

### `select_split`

bpy.ops.uv.select_split()
Select only entirely selected faces

### `shortest_path_pick`

bpy.ops.uv.shortest_path_pick(use_face_step=False, use_topology_distance=False, use_fill=False, skip=0, nth=1, offset=0, object_index=-1, index=-1)
Select shortest path between two selections

### `shortest_path_select`

bpy.ops.uv.shortest_path_select(use_face_step=False, use_topology_distance=False, use_fill=False, skip=0, nth=1, offset=0)
Selected shortest path between two vertices/edges/faces

### `smart_project`

bpy.ops.uv.smart_project(angle_limit=1.15192, margin_method='SCALED', rotate_method='AXIS_ALIGNED_Y', island_margin=0, area_weight=0, correct_aspect=True, scale_to_bounds=False)
Projection unwraps the selected faces of mesh objects

### `snap_cursor`

bpy.ops.uv.snap_cursor(target='PIXELS')
Snap cursor to target type

### `snap_selected`

bpy.ops.uv.snap_selected(target='PIXELS')
Snap selected UV vertices to target type

### `sphere_project`

bpy.ops.uv.sphere_project(direction='VIEW_ON_EQUATOR', align='POLAR_ZX', pole='PINCH', seam=False, correct_aspect=True, clip_to_bounds=False, scale_to_bounds=False)
Project the UV vertices of the mesh over the curved surface of a sphere

### `stitch`

bpy.ops.uv.stitch(use_limit=False, snap_islands=True, limit=0.01, static_island=0, active_object_index=0, midpoint_snap=False, clear_seams=True, mode='VERTEX', stored_mode='VERTEX', selection=[], objects_selection_count=(0, 0, 0, 0, 0, 0))
Stitch selected UV vertices by proximity

### `unwrap`

bpy.ops.uv.unwrap(method='CONFORMAL', fill_holes=False, correct_aspect=True, use_subsurf_data=False, margin_method='SCALED', margin=0.001, no_flip=False, iterations=10, use_weights=False, weight_group="uv_importance", weight_factor=1)
Unwrap the mesh of the object being edited

### `weld`

bpy.ops.uv.weld()
Weld selected UV vertices together
