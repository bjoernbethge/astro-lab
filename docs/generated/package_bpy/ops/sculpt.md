# sculpt

Part of `bpy.ops`
Module: `bpy.ops.sculpt`

## Operators (37)

### `brush_stroke`

bpy.ops.sculpt.brush_stroke(stroke=[], mode='NORMAL', pen_flip=False, override_location=False, ignore_background_click=False)
Sculpt a stroke into the geometry

### `cloth_filter`

bpy.ops.sculpt.cloth_filter(start_mouse=(0, 0), area_normal_radius=0.25, strength=1, iteration_count=1, event_history=[], type='GRAVITY', force_axis={'X', 'Y', 'Z'}, orientation='LOCAL', cloth_mass=1, cloth_damping=0, use_face_sets=False, use_collisions=False)
Applies a cloth simulation deformation to the entire mesh

### `color_filter`

bpy.ops.sculpt.color_filter(start_mouse=(0, 0), area_normal_radius=0.25, strength=1, iteration_count=1, event_history=[], type='FILL', fill_color=(1, 1, 1))
Applies a filter to modify the active color attribute

### `detail_flood_fill`

bpy.ops.sculpt.detail_flood_fill()
Flood fill the mesh with the selected detail setting

### `dynamic_topology_toggle`

bpy.ops.sculpt.dynamic_topology_toggle()
Dynamic topology alters the mesh topology while sculpting

### `dyntopo_detail_size_edit`

bpy.ops.sculpt.dyntopo_detail_size_edit()
Modify the detail size of dyntopo interactively

### `expand`

bpy.ops.sculpt.expand(target='MASK', falloff_type='GEODESIC', invert=False, use_mask_preserve=False, use_falloff_gradient=False, use_modify_active=False, use_reposition_pivot=True, max_geodesic_move_preview=10000, use_auto_mask=False, normal_falloff_smooth=2)
Generic sculpt expand operator

### `face_set_box_gesture`

bpy.ops.sculpt.face_set_box_gesture(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, use_front_faces_only=False)
Add a face set in a rectangle defined by the cursor

### `face_set_change_visibility`

bpy.ops.sculpt.face_set_change_visibility(mode='TOGGLE')
Change the visibility of the Face Sets of the sculpt

### `face_set_edit`

bpy.ops.sculpt.face_set_edit(active_face_set=1, mode='GROW', strength=1, modify_hidden=False)
Edits the current active Face Set

### `face_set_lasso_gesture`

bpy.ops.sculpt.face_set_lasso_gesture(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, use_front_faces_only=False)
Add a face set in a shape defined by the cursor

### `face_set_line_gesture`

bpy.ops.sculpt.face_set_line_gesture(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5, use_front_faces_only=False, use_limit_to_segment=False)
Add a face set to one side of a line defined by the cursor

### `face_set_polyline_gesture`

bpy.ops.sculpt.face_set_polyline_gesture(path=[], use_front_faces_only=False)
Add a face set in a shape defined by the cursor

### `face_sets_create`

bpy.ops.sculpt.face_sets_create(mode='MASKED')
Create a new Face Set

### `face_sets_init`

bpy.ops.sculpt.face_sets_init(mode='LOOSE_PARTS', threshold=0.5)
Initializes all Face Sets in the mesh

### `face_sets_randomize_colors`

bpy.ops.sculpt.face_sets_randomize_colors()
Generates a new set of random colors to render the Face Sets in the viewport

### `mask_by_color`

bpy.ops.sculpt.mask_by_color(contiguous=False, invert=False, preserve_previous_mask=False, threshold=0.35)
Creates a mask based on the active color attribute

### `mask_filter`

bpy.ops.sculpt.mask_filter(filter_type='SMOOTH', iterations=1, auto_iteration_count=True)
Applies a filter to modify the current mask

### `mask_from_boundary`

bpy.ops.sculpt.mask_from_boundary(mix_mode='MIX', mix_factor=1, settings_source='OPERATOR', boundary_mode='MESH', propagation_steps=1)
Creates a mask based on the boundaries of the surface

### `mask_from_cavity`

bpy.ops.sculpt.mask_from_cavity(mix_mode='MIX', mix_factor=1, settings_source='OPERATOR', factor=0.5, blur_steps=2, use_curve=False, invert=False)
Creates a mask based on the curvature of the surface

### `mask_init`

bpy.ops.sculpt.mask_init(mode='RANDOM_PER_VERTEX')
Creates a new mask for the entire mesh

### `mesh_filter`

bpy.ops.sculpt.mesh_filter(start_mouse=(0, 0), area_normal_radius=0.25, strength=1, iteration_count=1, event_history=[], type='INFLATE', deform_axis={'X', 'Y', 'Z'}, orientation='LOCAL', surface_smooth_shape_preservation=0.5, surface_smooth_current_vertex=0.5, sharpen_smooth_ratio=0.35, sharpen_intensify_detail_strength=0, sharpen_curvature_smooth_iterations=0)
Applies a filter to modify the current mesh

### `optimize`

bpy.ops.sculpt.optimize()
Recalculate the sculpt BVH to improve performance

### `project_line_gesture`

bpy.ops.sculpt.project_line_gesture(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5, use_front_faces_only=False, use_limit_to_segment=False)
Project the geometry onto a plane defined by a line

### `sample_color`

bpy.ops.sculpt.sample_color()
Sample the vertex color of the active vertex

### `sample_detail_size`

bpy.ops.sculpt.sample_detail_size(location=(0, 0), mode='DYNTOPO')
Sample the mesh detail on clicked point

### `sculptmode_toggle`

bpy.ops.sculpt.sculptmode_toggle()
Toggle sculpt mode in 3D view

### `set_persistent_base`

bpy.ops.sculpt.set_persistent_base()
Reset the copy of the mesh that is being sculpted on

### `set_pivot_position`

bpy.ops.sculpt.set_pivot_position(mode='UNMASKED', mouse_x=0, mouse_y=0)
Sets the sculpt transform pivot position

### `symmetrize`

bpy.ops.sculpt.symmetrize(merge_tolerance=0.0005)
Symmetrize the topology modifications

### `trim_box_gesture`

bpy.ops.sculpt.trim_box_gesture(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, use_front_faces_only=False, location=(0, 0), trim_mode='DIFFERENCE', use_cursor_depth=False, trim_orientation='VIEW', trim_extrude_mode='FIXED', trim_solver='FAST')
Execute a boolean operation on the mesh and a rectangle defined by the cursor

### `trim_lasso_gesture`

bpy.ops.sculpt.trim_lasso_gesture(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, use_front_faces_only=False, location=(0, 0), trim_mode='DIFFERENCE', use_cursor_depth=False, trim_orientation='VIEW', trim_extrude_mode='FIXED', trim_solver='FAST')
Execute a boolean operation on the mesh and a shape defined by the cursor

### `trim_line_gesture`

bpy.ops.sculpt.trim_line_gesture(xstart=0, xend=0, ystart=0, yend=0, flip=False, cursor=5, use_front_faces_only=False, use_limit_to_segment=False, location=(0, 0), trim_mode='DIFFERENCE', use_cursor_depth=False, trim_orientation='VIEW', trim_extrude_mode='FIXED', trim_solver='FAST')
Remove a portion of the mesh on one side of a line

### `trim_polyline_gesture`

bpy.ops.sculpt.trim_polyline_gesture(path=[], use_front_faces_only=False, location=(0, 0), trim_mode='DIFFERENCE', use_cursor_depth=False, trim_orientation='VIEW', trim_extrude_mode='FIXED', trim_solver='FAST')
Execute a boolean operation on the mesh and a polygonal shape defined by the cursor

### `uv_sculpt_grab`

bpy.ops.sculpt.uv_sculpt_grab(use_invert=False)
Grab UVs

### `uv_sculpt_pinch`

bpy.ops.sculpt.uv_sculpt_pinch(use_invert=False)
Pinch UVs

### `uv_sculpt_relax`

bpy.ops.sculpt.uv_sculpt_relax(use_invert=False, relax_method='COTAN')
Relax UVs
