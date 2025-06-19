# view3d

Part of `bpy.ops`
Module: `bpy.ops.view3d`

## Operators (63)

### `bone_select_menu`

bpy.ops.view3d.bone_select_menu(name='<UNKNOWN ENUM>', extend=False, deselect=False, toggle=False)
Menu bone selection

### `camera_background_image_add`

bpy.ops.view3d.camera_background_image_add(filepath="", relative_path=True, name="", session_uid=0)
Add a new background image to the active camera

### `camera_background_image_remove`

bpy.ops.view3d.camera_background_image_remove(index=0)
Remove a background image from the camera

### `camera_to_view`

bpy.ops.view3d.camera_to_view()
Set camera view to active view

### `camera_to_view_selected`

bpy.ops.view3d.camera_to_view_selected()
Move the camera so selected objects are framed

### `clear_render_border`

bpy.ops.view3d.clear_render_border()
Clear the boundaries of the border render and disable border render

### `clip_border`

bpy.ops.view3d.clip_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Set the view clipping region

### `copybuffer`

bpy.ops.view3d.copybuffer()
Copy the selected objects to the internal clipboard

### `cursor3d`

bpy.ops.view3d.cursor3d(use_depth=True, orientation='VIEW')
Set the location of the 3D cursor

### `dolly`

bpy.ops.view3d.dolly(mx=0, my=0, delta=0, use_cursor_init=True)
Dolly in/out in the view

### `drop_world`

bpy.ops.view3d.drop_world(name="", session_uid=0)
Drop a world into the scene

### `edit_mesh_extrude_individual_move`

bpy.ops.view3d.edit_mesh_extrude_individual_move()
Extrude each individual face separately along local normals

### `edit_mesh_extrude_manifold_normal`

bpy.ops.view3d.edit_mesh_extrude_manifold_normal()
Extrude manifold region along normals

### `edit_mesh_extrude_move_normal`

bpy.ops.view3d.edit_mesh_extrude_move_normal(dissolve_and_intersect=False)
Extrude region together along the average normal

### `edit_mesh_extrude_move_shrink_fatten`

bpy.ops.view3d.edit_mesh_extrude_move_shrink_fatten()
Extrude region together along local normals

### `fly`

bpy.ops.view3d.fly()
Interactively fly around the scene

### `interactive_add`

bpy.ops.view3d.interactive_add(primitive_type='CUBE', plane_origin_base='EDGE', plane_origin_depth='EDGE', plane_aspect_base='FREE', plane_aspect_depth='FREE', wait_for_input=True)
Interactively add an object

### `localview`

bpy.ops.view3d.localview(frame_selected=True)
Toggle display of selected object(s) separately and centered in view

### `localview_remove_from`

bpy.ops.view3d.localview_remove_from()
Move selected objects out of local view

### `move`

bpy.ops.view3d.move(use_cursor_init=True)
Move the view

### `navigate`

bpy.ops.view3d.navigate()
Interactively navigate around the scene (uses the mode (walk/fly) preference)

### `object_as_camera`

bpy.ops.view3d.object_as_camera()
Set the active object as the active camera for this view or scene

### `object_mode_pie_or_toggle`

bpy.ops.view3d.object_mode_pie_or_toggle()
(undocumented operator)

### `pastebuffer`

bpy.ops.view3d.pastebuffer(autoselect=True, active_collection=True)
Paste objects from the internal clipboard

### `render_border`

bpy.ops.view3d.render_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Set the boundaries of the border render and enable border render

### `rotate`

bpy.ops.view3d.rotate(use_cursor_init=True)
Rotate the view

### `ruler_add`

bpy.ops.view3d.ruler_add()
Add ruler

### `ruler_remove`

bpy.ops.view3d.ruler_remove()
(undocumented operator)

### `select`

bpy.ops.view3d.select(extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, center=False, enumerate=False, object=False, location=(0, 0))
Select and activate item(s)

### `select_box`

bpy.ops.view3d.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Select items using box selection

### `select_circle`

bpy.ops.view3d.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Select items using circle selection

### `select_lasso`

bpy.ops.view3d.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select items using lasso selection

### `select_menu`

bpy.ops.view3d.select_menu(name='<UNKNOWN ENUM>', extend=False, deselect=False, toggle=False)
Menu object selection

### `smoothview`

bpy.ops.view3d.smoothview()
(undocumented operator)

### `snap_cursor_to_active`

bpy.ops.view3d.snap_cursor_to_active()
Snap 3D cursor to the active item

### `snap_cursor_to_center`

bpy.ops.view3d.snap_cursor_to_center()
Snap 3D cursor to the world origin

### `snap_cursor_to_grid`

bpy.ops.view3d.snap_cursor_to_grid()
Snap 3D cursor to the nearest grid division

### `snap_cursor_to_selected`

bpy.ops.view3d.snap_cursor_to_selected()
Snap 3D cursor to the middle of the selected item(s)

### `snap_selected_to_active`

bpy.ops.view3d.snap_selected_to_active()
Snap selected item(s) to the active item

### `snap_selected_to_cursor`

bpy.ops.view3d.snap_selected_to_cursor(use_offset=True)
Snap selected item(s) to the 3D cursor

### `snap_selected_to_grid`

bpy.ops.view3d.snap_selected_to_grid()
Snap selected item(s) to their nearest grid division

### `toggle_matcap_flip`

bpy.ops.view3d.toggle_matcap_flip()
Flip MatCap

### `toggle_shading`

bpy.ops.view3d.toggle_shading(type='WIREFRAME')
Toggle shading type in 3D viewport

### `toggle_xray`

bpy.ops.view3d.toggle_xray()
Transparent scene display. Allow selecting through items

### `transform_gizmo_set`

bpy.ops.view3d.transform_gizmo_set(extend=False, type=set())
Set the current transform gizmo

### `view_all`

bpy.ops.view3d.view_all(use_all_regions=False, center=False)
View all objects in scene

### `view_axis`

bpy.ops.view3d.view_axis(type='LEFT', align_active=False, relative=False)
Use a preset viewpoint

### `view_camera`

bpy.ops.view3d.view_camera()
Toggle the camera view

### `view_center_camera`

bpy.ops.view3d.view_center_camera()
Center the camera view, resizing the view to fit its bounds

### `view_center_cursor`

bpy.ops.view3d.view_center_cursor()
Center the view so that the cursor is in the middle of the view

### `view_center_lock`

bpy.ops.view3d.view_center_lock()
Center the view lock offset

### `view_center_pick`

bpy.ops.view3d.view_center_pick()
Center the view to the Z-depth position under the mouse cursor

### `view_lock_clear`

bpy.ops.view3d.view_lock_clear()
Clear all view locking

### `view_lock_to_active`

bpy.ops.view3d.view_lock_to_active()
Lock the view to the active object/bone

### `view_orbit`

bpy.ops.view3d.view_orbit(angle=0, type='ORBITLEFT')
Orbit the view

### `view_pan`

bpy.ops.view3d.view_pan(type='PANLEFT')
Pan the view in a given direction

### `view_persportho`

bpy.ops.view3d.view_persportho()
Switch the current view from perspective/orthographic projection

### `view_roll`

bpy.ops.view3d.view_roll(angle=0, type='ANGLE')
Roll the view

### `view_selected`

bpy.ops.view3d.view_selected(use_all_regions=False)
Move the view to the selection center

### `walk`

bpy.ops.view3d.walk()
Interactively walk around the scene

### `zoom`

bpy.ops.view3d.zoom(mx=0, my=0, delta=0, use_cursor_init=True)
Zoom in/out in the view

### `zoom_border`

bpy.ops.view3d.zoom_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, zoom_out=False)
Zoom in the view to the nearest object contained in the border

### `zoom_camera_1_to_1`

bpy.ops.view3d.zoom_camera_1_to_1()
Match the camera to 1:1 to the render output
