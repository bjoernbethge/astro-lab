# mesh

Part of `bpy.ops`
Module: `bpy.ops.mesh`

## Operators (164)

### `attribute_set`

bpy.ops.mesh.attribute_set(value_float=0, value_float_vector_2d=(0, 0), value_float_vector_3d=(0, 0, 0), value_int=0, value_int_vector_2d=(0, 0), value_color=(1, 1, 1, 1), value_bool=False)
Set values of the active attribute for selected elements

### `average_normals`

bpy.ops.mesh.average_normals(average_type='CUSTOM_NORMAL', weight=50, threshold=0.01)
Average custom normals of selected vertices

### `beautify_fill`

bpy.ops.mesh.beautify_fill(angle_limit=3.14159)
Rearrange some faces to try to get less degenerated geometry

### `bevel`

bpy.ops.mesh.bevel(offset_type='OFFSET', offset=0, profile_type='SUPERELLIPSE', offset_pct=0, segments=1, profile=0.5, affect='EDGES', clamp_overlap=False, loop_slide=True, mark_seam=False, mark_sharp=False, material=-1, harden_normals=False, face_strength_mode='NONE', miter_outer='SHARP', miter_inner='SHARP', spread=0.1, vmesh_method='ADJ', release_confirm=False)
Cut into selected items at an angle to create bevel or chamfer

### `bisect`

bpy.ops.mesh.bisect(plane_co=(0, 0, 0), plane_no=(0, 0, 0), use_fill=False, clear_inner=False, clear_outer=False, threshold=0.0001, xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5)
Cut geometry along a plane (click-drag to define plane)

### `blend_from_shape`

bpy.ops.mesh.blend_from_shape(shape='<UNKNOWN ENUM>', blend=1, add=True)
Blend in shape from a shape key

### `bridge_edge_loops`

bpy.ops.mesh.bridge_edge_loops(type='SINGLE', use_merge=False, merge_factor=0.5, twist_offset=0, number_cuts=0, interpolation='PATH', smoothness=1, profile_shape_factor=0, profile_shape='SMOOTH')
Create a bridge of faces between two or more selected edge loops

### `colors_reverse`

bpy.ops.mesh.colors_reverse()
Flip direction of face corner color attribute inside faces

### `colors_rotate`

bpy.ops.mesh.colors_rotate(use_ccw=False)
Rotate face corner color attribute inside faces

### `convex_hull`

bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=True, make_holes=False, join_triangles=True, face_threshold=0.698132, shape_threshold=0.698132, topology_influence=0, uvs=False, vcols=False, seam=False, sharp=False, materials=False, deselect_joined=False)
Enclose selected vertices in a convex polyhedron

### `customdata_custom_splitnormals_add`

bpy.ops.mesh.customdata_custom_splitnormals_add()
Add a custom split normals layer, if none exists yet

### `customdata_custom_splitnormals_clear`

bpy.ops.mesh.customdata_custom_splitnormals_clear()
Remove the custom split normals layer, if it exists

### `customdata_mask_clear`

bpy.ops.mesh.customdata_mask_clear()
Clear vertex sculpt masking data from the mesh

### `customdata_skin_add`

bpy.ops.mesh.customdata_skin_add()
Add a vertex skin layer

### `customdata_skin_clear`

bpy.ops.mesh.customdata_skin_clear()
Clear vertex skin layer

### `decimate`

bpy.ops.mesh.decimate(ratio=1, use_vertex_group=False, vertex_group_factor=1, invert_vertex_group=False, use_symmetry=False, symmetry_axis='Y')
Simplify geometry by collapsing edges

### `delete`

bpy.ops.mesh.delete(type='VERT')
Delete selected vertices, edges or faces

### `delete_edgeloop`

bpy.ops.mesh.delete_edgeloop(use_face_split=True)
Delete an edge loop by merging the faces on each side

### `delete_loose`

bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
Delete loose vertices, edges or faces

### `dissolve_degenerate`

bpy.ops.mesh.dissolve_degenerate(threshold=0.0001)
Dissolve zero area faces and zero length edges

### `dissolve_edges`

bpy.ops.mesh.dissolve_edges(use_verts=True, use_face_split=False)
Dissolve edges, merging faces

### `dissolve_faces`

bpy.ops.mesh.dissolve_faces(use_verts=False)
Dissolve faces

### `dissolve_limited`

bpy.ops.mesh.dissolve_limited(angle_limit=0.0872665, use_dissolve_boundaries=False, delimit={'NORMAL'})
Dissolve selected edges and vertices, limited by the angle of surrounding geometry

### `dissolve_mode`

bpy.ops.mesh.dissolve_mode(use_verts=False, use_face_split=False, use_boundary_tear=False)
Dissolve geometry based on the selection mode

### `dissolve_verts`

bpy.ops.mesh.dissolve_verts(use_face_split=False, use_boundary_tear=False)
Dissolve vertices, merge edges and faces

### `dupli_extrude_cursor`

bpy.ops.mesh.dupli_extrude_cursor(rotate_source=True)
Duplicate and extrude selected vertices, edges or faces towards the mouse cursor

### `duplicate`

bpy.ops.mesh.duplicate(mode=1)
Duplicate selected vertices, edges or faces

### `duplicate_move`

bpy.ops.mesh.duplicate_move(MESH_OT_duplicate={"mode":1}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate mesh and move

### `edge_collapse`

bpy.ops.mesh.edge_collapse()
Collapse isolated edge and face regions, merging data such as UVs and color attributes. This can collapse edge-rings as well as regions of connected faces into vertices

### `edge_face_add`

bpy.ops.mesh.edge_face_add()
Add an edge or face to selected

### `edge_rotate`

bpy.ops.mesh.edge_rotate(use_ccw=False)
Rotate selected edge or adjoining faces

### `edge_split`

bpy.ops.mesh.edge_split(type='EDGE')
Split selected edges so that each neighbor face gets its own copy

### `edgering_select`

bpy.ops.mesh.edgering_select(extend=False, deselect=False, toggle=False, ring=True)
Select an edge ring

### `edges_select_sharp`

bpy.ops.mesh.edges_select_sharp(sharpness=0.523599)
Select all sharp enough edges

### `extrude_context`

bpy.ops.mesh.extrude_context(use_normal_flip=False, use_dissolve_ortho_edges=False, mirror=False)
Extrude selection

### `extrude_context_move`

bpy.ops.mesh.extrude_context_move(MESH_OT_extrude_context={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude region together along the average normal

### `extrude_edges_indiv`

bpy.ops.mesh.extrude_edges_indiv(use_normal_flip=False, mirror=False)
Extrude individual edges only

### `extrude_edges_move`

bpy.ops.mesh.extrude_edges_move(MESH_OT_extrude_edges_indiv={"use_normal_flip":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude edges and move result

### `extrude_faces_indiv`

bpy.ops.mesh.extrude_faces_indiv(mirror=False)
Extrude individual faces only

### `extrude_faces_move`

bpy.ops.mesh.extrude_faces_move(MESH_OT_extrude_faces_indiv={"mirror":False}, TRANSFORM_OT_shrink_fatten={"value":0, "use_even_offset":False, "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "release_confirm":False, "use_accurate":False})
Extrude each individual face separately along local normals

### `extrude_manifold`

bpy.ops.mesh.extrude_manifold(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude, dissolves edges whose faces form a flat surface and intersect new edges

### `extrude_region`

bpy.ops.mesh.extrude_region(use_normal_flip=False, use_dissolve_ortho_edges=False, mirror=False)
Extrude region of faces

### `extrude_region_move`

bpy.ops.mesh.extrude_region_move(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude region and move result

### `extrude_region_shrink_fatten`

bpy.ops.mesh.extrude_region_shrink_fatten(MESH_OT_extrude_region={"use_normal_flip":False, "use_dissolve_ortho_edges":False, "mirror":False}, TRANSFORM_OT_shrink_fatten={"value":0, "use_even_offset":False, "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "release_confirm":False, "use_accurate":False})
Extrude region together along local normals

### `extrude_repeat`

bpy.ops.mesh.extrude_repeat(steps=10, offset=(0, 0, 0), scale_offset=1)
Extrude selected vertices, edges or faces repeatedly

### `extrude_vertices_move`

bpy.ops.mesh.extrude_vertices_move(MESH_OT_extrude_verts_indiv={"mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extrude vertices and move result

### `extrude_verts_indiv`

bpy.ops.mesh.extrude_verts_indiv(mirror=False)
Extrude individual vertices only

### `face_make_planar`

bpy.ops.mesh.face_make_planar(factor=1, repeat=1)
Flatten selected faces

### `face_set_extract`

bpy.ops.mesh.face_set_extract(add_boundary_loop=True, smooth_iterations=4, apply_shrinkwrap=True, add_solidify=True)
Create a new mesh object from the selected Face Set

### `face_split_by_edges`

bpy.ops.mesh.face_split_by_edges()
Weld loose edges into faces (splitting them into new faces)

### `faces_mirror_uv`

bpy.ops.mesh.faces_mirror_uv(direction='POSITIVE', precision=3)
Copy mirror UV coordinates on the X axis based on a mirrored mesh

### `faces_select_linked_flat`

bpy.ops.mesh.faces_select_linked_flat(sharpness=0.0174533)
Select linked faces by angle

### `faces_shade_flat`

bpy.ops.mesh.faces_shade_flat()
Display faces flat

### `faces_shade_smooth`

bpy.ops.mesh.faces_shade_smooth()
Display faces smooth (using vertex normals)

### `fill`

bpy.ops.mesh.fill(use_beauty=True)
Fill a selected edge loop with faces

### `fill_grid`

bpy.ops.mesh.fill_grid(span=1, offset=0, use_interp_simple=False)
Fill grid from two loops

### `fill_holes`

bpy.ops.mesh.fill_holes(sides=4)
Fill in holes (boundary edge loops)

### `flip_normals`

bpy.ops.mesh.flip_normals(only_clnors=False)
Flip the direction of selected faces' normals (and of their vertices)

### `flip_quad_tessellation`

bpy.ops.mesh.flip_quad_tessellation()
Flips the tessellation of selected quads

### `hide`

bpy.ops.mesh.hide(unselected=False)
Hide (un)selected vertices, edges or faces

### `inset`

bpy.ops.mesh.inset(use_boundary=True, use_even_offset=True, use_relative_offset=False, use_edge_rail=False, thickness=0, depth=0, use_outset=False, use_select_inset=False, use_individual=False, use_interpolate=True, release_confirm=False)
Inset new faces into selected faces

### `intersect`

bpy.ops.mesh.intersect(mode='SELECT_UNSELECT', separate_mode='CUT', threshold=1e-06, solver='EXACT')
Cut an intersection into faces

### `intersect_boolean`

bpy.ops.mesh.intersect_boolean(operation='DIFFERENCE', use_swap=False, use_self=False, threshold=1e-06, solver='EXACT')
Cut solid geometry from selected to unselected

### `knife_project`

bpy.ops.mesh.knife_project(cut_through=False)
Use other objects outlines and boundaries to project knife cuts

### `knife_tool`

bpy.ops.mesh.knife_tool(use_occlude_geometry=True, only_selected=False, xray=True, visible_measurements='NONE', angle_snapping='NONE', angle_snapping_increment=0.523599, wait_for_input=True)
Cut new topology

### `loop_multi_select`

bpy.ops.mesh.loop_multi_select(ring=False)
Select a loop of connected edges by connection type

### `loop_select`

bpy.ops.mesh.loop_select(extend=False, deselect=False, toggle=False, ring=False)
Select a loop of connected edges

### `loop_to_region`

bpy.ops.mesh.loop_to_region(select_bigger=False)
Select region of faces inside of a selected loop of edges

### `loopcut`

bpy.ops.mesh.loopcut(number_cuts=1, smoothness=0, falloff='INVERSE_SQUARE', object_index=-1, edge_index=-1, mesh_select_mode_init=(False, False, False))
Add a new loop between existing loops

### `loopcut_slide`

bpy.ops.mesh.loopcut_slide(MESH_OT_loopcut={"number_cuts":1, "smoothness":0, "falloff":'INVERSE_SQUARE', "object_index":-1, "edge_index":-1, "mesh_select_mode_init":(False, False, False)}, TRANSFORM_OT_edge_slide={"value":0, "single_side":False, "use_even":False, "flipped":False, "use_clamp":True, "mirror":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "correct_uv":True, "release_confirm":False, "use_accurate":False})
Cut mesh loop and slide it

### `mark_freestyle_edge`

bpy.ops.mesh.mark_freestyle_edge(clear=False)
(Un)mark selected edges as Freestyle feature edges

### `mark_freestyle_face`

bpy.ops.mesh.mark_freestyle_face(clear=False)
(Un)mark selected faces for exclusion from Freestyle feature edge detection

### `mark_seam`

bpy.ops.mesh.mark_seam(clear=False)
(Un)mark selected edges as a seam

### `mark_sharp`

bpy.ops.mesh.mark_sharp(clear=False, use_verts=False)
(Un)mark selected edges as sharp

### `merge`

bpy.ops.mesh.merge(type='CENTER', uvs=False)
Merge selected vertices

### `merge_normals`

bpy.ops.mesh.merge_normals()
Merge custom normals of selected vertices

### `mod_weighted_strength`

bpy.ops.mesh.mod_weighted_strength(set=False, face_strength='MEDIUM')
Set/Get strength of face (used in Weighted Normal modifier)

### `normals_make_consistent`

bpy.ops.mesh.normals_make_consistent(inside=False)
Make face and vertex normals point either outside or inside the mesh

### `normals_tools`

bpy.ops.mesh.normals_tools(mode='COPY', absolute=False)
Custom normals tools using Normal Vector of UI

### `offset_edge_loops`

bpy.ops.mesh.offset_edge_loops(use_cap_endpoint=False)
Create offset edge loop from the current selection

### `offset_edge_loops_slide`

bpy.ops.mesh.offset_edge_loops_slide(MESH_OT_offset_edge_loops={"use_cap_endpoint":False}, TRANSFORM_OT_edge_slide={"value":0, "single_side":False, "use_even":False, "flipped":False, "use_clamp":True, "mirror":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "correct_uv":True, "release_confirm":False, "use_accurate":False})
Offset edge loop slide

### `paint_mask_extract`

bpy.ops.mesh.paint_mask_extract(mask_threshold=0.5, add_boundary_loop=True, smooth_iterations=4, apply_shrinkwrap=True, add_solidify=True)
Create a new mesh object from the current paint mask

### `paint_mask_slice`

bpy.ops.mesh.paint_mask_slice(mask_threshold=0.5, fill_holes=True, new_object=True)
Slices the paint mask from the mesh

### `point_normals`

bpy.ops.mesh.point_normals(mode='COORDINATES', invert=False, align=False, target_location=(0, 0, 0), spherize=False, spherize_strength=0.1)
Point selected custom normals to specified Target

### `poke`

bpy.ops.mesh.poke(offset=0, use_relative_offset=False, center_mode='MEDIAN_WEIGHTED')
Split a face into a fan

### `polybuild_delete_at_cursor`

bpy.ops.mesh.polybuild_delete_at_cursor(mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False)
(undocumented operator)

### `polybuild_dissolve_at_cursor`

bpy.ops.mesh.polybuild_dissolve_at_cursor()
(undocumented operator)

### `polybuild_extrude_at_cursor_move`

bpy.ops.mesh.polybuild_extrude_at_cursor_move(MESH_OT_polybuild_transform_at_cursor={"mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False}, MESH_OT_extrude_edges_indiv={"use_normal_flip":False, "mirror":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
(undocumented operator)

### `polybuild_face_at_cursor`

bpy.ops.mesh.polybuild_face_at_cursor(create_quads=True, mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False)
(undocumented operator)

### `polybuild_face_at_cursor_move`

bpy.ops.mesh.polybuild_face_at_cursor_move(MESH_OT_polybuild_face_at_cursor={"create_quads":True, "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
(undocumented operator)

### `polybuild_split_at_cursor`

bpy.ops.mesh.polybuild_split_at_cursor(mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False)
(undocumented operator)

### `polybuild_split_at_cursor_move`

bpy.ops.mesh.polybuild_split_at_cursor_move(MESH_OT_polybuild_split_at_cursor={"mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
(undocumented operator)

### `polybuild_transform_at_cursor`

bpy.ops.mesh.polybuild_transform_at_cursor(mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False)
(undocumented operator)

### `polybuild_transform_at_cursor_move`

bpy.ops.mesh.polybuild_transform_at_cursor_move(MESH_OT_polybuild_transform_at_cursor={"mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
(undocumented operator)

### `primitive_circle_add`

bpy.ops.mesh.primitive_circle_add(vertices=32, radius=1, fill_type='NOTHING', calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a circle mesh

### `primitive_cone_add`

bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1, radius2=0, depth=2, end_fill_type='NGON', calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a conic mesh

### `primitive_cube_add`

bpy.ops.mesh.primitive_cube_add(size=2, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a cube mesh that consists of six square faces

### `primitive_cube_add_gizmo`

bpy.ops.mesh.primitive_cube_add_gizmo(calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0), matrix=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)))
Construct a cube mesh

### `primitive_cylinder_add`

bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=1, depth=2, end_fill_type='NGON', calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a cylinder mesh

### `primitive_grid_add`

bpy.ops.mesh.primitive_grid_add(x_subdivisions=10, y_subdivisions=10, size=2, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a subdivided plane mesh

### `primitive_ico_sphere_add`

bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=1, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a spherical mesh that consists of equally sized triangles

### `primitive_monkey_add`

bpy.ops.mesh.primitive_monkey_add(size=2, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a Suzanne mesh

### `primitive_plane_add`

bpy.ops.mesh.primitive_plane_add(size=2, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a filled planar mesh with 4 vertices

### `primitive_torus_add`

bpy.ops.mesh.primitive_torus_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), major_segments=48, minor_segments=12, mode='MAJOR_MINOR', major_radius=1, minor_radius=0.25, abso_major_rad=1.25, abso_minor_rad=0.75, generate_uvs=True)
Construct a torus mesh

### `primitive_uv_sphere_add`

bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=1, calc_uvs=True, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Construct a spherical mesh with quad faces, except for triangle faces at the top and bottom

### `quads_convert_to_tris`

bpy.ops.mesh.quads_convert_to_tris(quad_method='BEAUTY', ngon_method='BEAUTY')
Triangulate selected faces

### `region_to_loop`

bpy.ops.mesh.region_to_loop()
Select boundary edges around the selected faces

### `remove_doubles`

bpy.ops.mesh.remove_doubles(threshold=0.0001, use_unselected=False, use_sharp_edge_from_normals=False)
Merge vertices based on their proximity

### `reveal`

bpy.ops.mesh.reveal(select=True)
Reveal all hidden vertices, edges and faces

### `rip`

bpy.ops.mesh.rip(mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False, use_fill=False)
Disconnect vertex or edges from connected geometry

### `rip_edge`

bpy.ops.mesh.rip_edge(mirror=False, use_proportional_edit=False, proportional_edit_falloff='SMOOTH', proportional_size=1, use_proportional_connected=False, use_proportional_projected=False, release_confirm=False, use_accurate=False)
Extend vertices along the edge closest to the cursor

### `rip_edge_move`

bpy.ops.mesh.rip_edge_move(MESH_OT_rip_edge={"mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Extend vertices and move the result

### `rip_move`

bpy.ops.mesh.rip_move(MESH_OT_rip={"mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "release_confirm":False, "use_accurate":False, "use_fill":False}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Rip polygons and move the result

### `screw`

bpy.ops.mesh.screw(steps=9, turns=1, center=(0, 0, 0), axis=(0, 0, 0))
Extrude selected vertices in screw-shaped rotation around the cursor in indicated viewport

### `select_all`

bpy.ops.mesh.select_all(action='TOGGLE')
(De)select all vertices, edges or faces

### `select_axis`

bpy.ops.mesh.select_axis(orientation='LOCAL', sign='POS', axis='X', threshold=0.0001)
Select all data in the mesh on a single axis

### `select_by_attribute`

bpy.ops.mesh.select_by_attribute()
Select elements based on the active boolean attribute

### `select_by_pole_count`

bpy.ops.mesh.select_by_pole_count(pole_count=4, type='NOTEQUAL', extend=False, exclude_nonmanifold=True)
Select vertices at poles by the number of connected edges. In edge and face mode the geometry connected to the vertices is selected

### `select_face_by_sides`

bpy.ops.mesh.select_face_by_sides(number=4, type='EQUAL', extend=True)
Select vertices or faces by the number of face sides

### `select_interior_faces`

bpy.ops.mesh.select_interior_faces()
Select faces where all edges have more than 2 face users

### `select_less`

bpy.ops.mesh.select_less(use_face_step=True)
Deselect vertices, edges or faces at the boundary of each selection region

### `select_linked`

bpy.ops.mesh.select_linked(delimit={'SEAM'})
Select all vertices connected to the current selection

### `select_linked_pick`

bpy.ops.mesh.select_linked_pick(deselect=False, delimit={'SEAM'}, object_index=-1, index=-1)
(De)select all vertices linked to the edge under the mouse cursor

### `select_loose`

bpy.ops.mesh.select_loose(extend=False)
Select loose geometry based on the selection mode

### `select_mirror`

bpy.ops.mesh.select_mirror(axis={'X'}, extend=False)
Select mesh items at mirrored locations

### `select_mode`

bpy.ops.mesh.select_mode(use_extend=False, use_expand=False, type='VERT', action='TOGGLE')
Change selection mode

### `select_more`

bpy.ops.mesh.select_more(use_face_step=True)
Select more vertices, edges or faces connected to initial selection

### `select_next_item`

bpy.ops.mesh.select_next_item()
Select the next element (using selection order)

### `select_non_manifold`

bpy.ops.mesh.select_non_manifold(extend=True, use_wire=True, use_boundary=True, use_multi_face=True, use_non_contiguous=True, use_verts=True)
Select all non-manifold vertices or edges

### `select_nth`

bpy.ops.mesh.select_nth(skip=1, nth=1, offset=0)
Deselect every Nth element starting from the active vertex, edge or face

### `select_prev_item`

bpy.ops.mesh.select_prev_item()
Select the previous element (using selection order)

### `select_random`

bpy.ops.mesh.select_random(ratio=0.5, seed=0, action='SELECT')
Randomly select vertices

### `select_similar`

bpy.ops.mesh.select_similar(type='VERT_NORMAL', compare='EQUAL', threshold=0)
Select similar vertices, edges or faces by property types

### `select_similar_region`

bpy.ops.mesh.select_similar_region()
Select similar face regions to the current selection

### `select_ungrouped`

bpy.ops.mesh.select_ungrouped(extend=False)
Select vertices without a group

### `separate`

bpy.ops.mesh.separate(type='SELECTED')
Separate selected geometry into a new mesh

### `set_normals_from_faces`

bpy.ops.mesh.set_normals_from_faces(keep_sharp=False)
Set the custom normals from the selected faces ones

### `set_sharpness_by_angle`

bpy.ops.mesh.set_sharpness_by_angle(angle=0.523599, extend=False)
Set edge sharpness based on the angle between neighboring faces

### `shape_propagate_to_all`

bpy.ops.mesh.shape_propagate_to_all()
Apply selected vertex locations to all other shape keys

### `shortest_path_pick`

bpy.ops.mesh.shortest_path_pick(edge_mode='SELECT', use_face_step=False, use_topology_distance=False, use_fill=False, skip=0, nth=1, offset=0, index=-1)
Select shortest path between two selections

### `shortest_path_select`

bpy.ops.mesh.shortest_path_select(edge_mode='SELECT', use_face_step=False, use_topology_distance=False, use_fill=False, skip=0, nth=1, offset=0)
Selected shortest path between two vertices/edges/faces

### `smooth_normals`

bpy.ops.mesh.smooth_normals(factor=0.5)
Smooth custom normals based on adjacent vertex normals

### `solidify`

bpy.ops.mesh.solidify(thickness=0.01)
Create a solid skin by extruding, compensating for sharp angles

### `sort_elements`

bpy.ops.mesh.sort_elements(type='VIEW_ZAXIS', elements={'VERT'}, reverse=False, seed=0)
The order of selected vertices/edges/faces is modified, based on a given method

### `spin`

bpy.ops.mesh.spin(steps=12, dupli=False, angle=1.5708, use_auto_merge=True, use_normal_flip=False, center=(0, 0, 0), axis=(0, 0, 0))
Extrude selected vertices in a circle around the cursor in indicated viewport

### `split`

bpy.ops.mesh.split()
Split off selected geometry from connected unselected geometry

### `split_normals`

bpy.ops.mesh.split_normals()
Split custom normals of selected vertices

### `subdivide`

bpy.ops.mesh.subdivide(number_cuts=1, smoothness=0, ngon=True, quadcorner='STRAIGHT_CUT', fractal=0, fractal_along_normal=0, seed=0)
Subdivide selected edges

### `subdivide_edgering`

bpy.ops.mesh.subdivide_edgering(number_cuts=10, interpolation='PATH', smoothness=1, profile_shape_factor=0, profile_shape='SMOOTH')
Subdivide perpendicular edges to the selected edge-ring

### `symmetrize`

bpy.ops.mesh.symmetrize(direction='NEGATIVE_X', threshold=0.0001)
Enforce symmetry (both form and topological) across an axis

### `symmetry_snap`

bpy.ops.mesh.symmetry_snap(direction='NEGATIVE_X', threshold=0.05, factor=0.5, use_center=True)
Snap vertex pairs to their mirrored locations

### `tris_convert_to_quads`

bpy.ops.mesh.tris_convert_to_quads(face_threshold=0.698132, shape_threshold=0.698132, topology_influence=0, uvs=False, vcols=False, seam=False, sharp=False, materials=False, deselect_joined=False)
Join triangles into quads

### `unsubdivide`

bpy.ops.mesh.unsubdivide(iterations=2)
Un-subdivide selected edges and faces

### `uv_texture_add`

bpy.ops.mesh.uv_texture_add()
Add UV map

### `uv_texture_remove`

bpy.ops.mesh.uv_texture_remove()
Remove UV map

### `uvs_reverse`

bpy.ops.mesh.uvs_reverse()
Flip direction of UV coordinates inside faces

### `uvs_rotate`

bpy.ops.mesh.uvs_rotate(use_ccw=False)
Rotate UV coordinates inside faces

### `vert_connect`

bpy.ops.mesh.vert_connect()
Connect selected vertices of faces, splitting the face

### `vert_connect_concave`

bpy.ops.mesh.vert_connect_concave()
Make all faces convex

### `vert_connect_nonplanar`

bpy.ops.mesh.vert_connect_nonplanar(angle_limit=0.0872665)
Split non-planar faces that exceed the angle threshold

### `vert_connect_path`

bpy.ops.mesh.vert_connect_path()
Connect vertices by their selection order, creating edges, splitting faces

### `vertices_smooth`

bpy.ops.mesh.vertices_smooth(factor=0, repeat=1, xaxis=True, yaxis=True, zaxis=True, wait_for_input=True)
Flatten angles of selected vertices

### `vertices_smooth_laplacian`

bpy.ops.mesh.vertices_smooth_laplacian(repeat=1, lambda_factor=1, lambda_border=5e-05, use_x=True, use_y=True, use_z=True, preserve_volume=True)
Laplacian smooth of selected vertices

### `wireframe`

bpy.ops.mesh.wireframe(use_boundary=True, use_even_offset=True, use_relative_offset=False, use_replace=True, thickness=0.01, offset=0.01, use_crease=False, crease_weight=0.01)
Create a solid wireframe from faces
