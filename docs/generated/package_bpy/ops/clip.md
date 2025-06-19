# clip

Part of `bpy.ops`
Module: `bpy.ops.clip`

## Operators (91)

### `add_marker`

bpy.ops.clip.add_marker(location=(0, 0))
Place new marker at specified location

### `add_marker_at_click`

bpy.ops.clip.add_marker_at_click()
Place new marker at the desired (clicked) position

### `add_marker_move`

bpy.ops.clip.add_marker_move(CLIP_OT_add_marker={"location":(0, 0)}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Add new marker and move it on movie

### `add_marker_slide`

bpy.ops.clip.add_marker_slide(CLIP_OT_add_marker={"location":(0, 0)}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Add new marker and slide it with mouse until mouse button release

### `apply_solution_scale`

bpy.ops.clip.apply_solution_scale(distance=0)
Apply scale on solution itself to make distance between selected tracks equals to desired

### `average_tracks`

bpy.ops.clip.average_tracks(keep_original=True)
Average selected tracks into active

### `bundles_to_mesh`

bpy.ops.clip.bundles_to_mesh()
Create vertex cloud using coordinates of reconstructed tracks

### `camera_preset_add`

bpy.ops.clip.camera_preset_add(name="", remove_name=False, remove_active=False, use_focal_length=True)
Add or remove a Tracking Camera Intrinsics Preset

### `change_frame`

bpy.ops.clip.change_frame(frame=0)
Interactively change the current frame number

### `clean_tracks`

bpy.ops.clip.clean_tracks(frames=0, error=0, action='SELECT')
Clean tracks with high error values or few frames

### `clear_solution`

bpy.ops.clip.clear_solution()
Clear all calculated data

### `clear_track_path`

bpy.ops.clip.clear_track_path(action='REMAINED', clear_active=False)
Clear tracks after/before current position or clear the whole track

### `constraint_to_fcurve`

bpy.ops.clip.constraint_to_fcurve()
Create F-Curves for object which will copy object's movement caused by this constraint

### `copy_tracks`

bpy.ops.clip.copy_tracks()
Copy the selected tracks to the internal clipboard

### `create_plane_track`

bpy.ops.clip.create_plane_track()
Create new plane track out of selected point tracks

### `cursor_set`

bpy.ops.clip.cursor_set(location=(0, 0))
Set 2D cursor location

### `delete_marker`

bpy.ops.clip.delete_marker(confirm=True)
Delete marker for current frame from selected tracks

### `delete_proxy`

bpy.ops.clip.delete_proxy()
Delete movie clip proxy files from the hard drive

### `delete_track`

bpy.ops.clip.delete_track(confirm=True)
Delete selected tracks

### `detect_features`

bpy.ops.clip.detect_features(placement='FRAME', margin=16, threshold=0.5, min_distance=120)
Automatically detect features and place markers to track

### `disable_markers`

bpy.ops.clip.disable_markers(action='DISABLE')
Disable/enable selected markers

### `dopesheet_select_channel`

bpy.ops.clip.dopesheet_select_channel(location=(0, 0), extend=False)
Select movie tracking channel

### `dopesheet_view_all`

bpy.ops.clip.dopesheet_view_all()
Reset viewable area to show full keyframe range

### `filter_tracks`

bpy.ops.clip.filter_tracks(track_threshold=5)
Filter tracks which has weirdly looking spikes in motion curves

### `frame_jump`

bpy.ops.clip.frame_jump(position='PATHSTART')
Jump to special frame

### `graph_center_current_frame`

bpy.ops.clip.graph_center_current_frame()
Scroll view so current frame would be centered

### `graph_delete_curve`

bpy.ops.clip.graph_delete_curve(confirm=True)
Delete track corresponding to the selected curve

### `graph_delete_knot`

bpy.ops.clip.graph_delete_knot()
Delete curve knots

### `graph_disable_markers`

bpy.ops.clip.graph_disable_markers(action='DISABLE')
Disable/enable selected markers

### `graph_select`

bpy.ops.clip.graph_select(location=(0, 0), extend=False)
Select graph curves

### `graph_select_all_markers`

bpy.ops.clip.graph_select_all_markers(action='TOGGLE')
Change selection of all markers of active track

### `graph_select_box`

bpy.ops.clip.graph_select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, deselect=False, extend=True)
Select curve points using box selection

### `graph_view_all`

bpy.ops.clip.graph_view_all()
View all curves in editor

### `hide_tracks`

bpy.ops.clip.hide_tracks(unselected=False)
Hide selected tracks

### `hide_tracks_clear`

bpy.ops.clip.hide_tracks_clear()
Clear hide selected tracks

### `join_tracks`

bpy.ops.clip.join_tracks()
Join selected tracks

### `keyframe_delete`

bpy.ops.clip.keyframe_delete()
Delete a keyframe from selected tracks at current frame

### `keyframe_insert`

bpy.ops.clip.keyframe_insert()
Insert a keyframe to selected tracks at current frame

### `lock_selection_toggle`

bpy.ops.clip.lock_selection_toggle()
Toggle Lock Selection option of the current clip editor

### `lock_tracks`

bpy.ops.clip.lock_tracks(action='LOCK')
Lock/unlock selected tracks

### `mode_set`

bpy.ops.clip.mode_set(mode='TRACKING')
Set the clip interaction mode

### `new_image_from_plane_marker`

bpy.ops.clip.new_image_from_plane_marker()
Create new image from the content of the plane marker

### `open`

bpy.ops.clip.open(directory="", files=[], hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Load a sequence of frames or a movie file

### `paste_tracks`

bpy.ops.clip.paste_tracks()
Paste tracks from the internal clipboard

### `prefetch`

bpy.ops.clip.prefetch()
Prefetch frames from disk for faster playback/tracking

### `rebuild_proxy`

bpy.ops.clip.rebuild_proxy()
Rebuild all selected proxies and timecode indices in the background

### `refine_markers`

bpy.ops.clip.refine_markers(backwards=False)
Refine selected markers positions by running the tracker from track's reference to current frame

### `reload`

bpy.ops.clip.reload()
Reload clip

### `select`

bpy.ops.clip.select(extend=False, deselect_all=False, location=(0, 0))
Select tracking markers

### `select_all`

bpy.ops.clip.select_all(action='TOGGLE')
Change selection of all tracking markers

### `select_box`

bpy.ops.clip.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Select markers using box selection

### `select_circle`

bpy.ops.clip.select_circle(x=0, y=0, radius=25, wait_for_input=True, mode='SET')
Select markers using circle selection

### `select_grouped`

bpy.ops.clip.select_grouped(group='ESTIMATED')
Select all tracks from specified group

### `select_lasso`

bpy.ops.clip.select_lasso(path=[], use_smooth_stroke=False, smooth_stroke_factor=0.75, smooth_stroke_radius=35, mode='SET')
Select markers using lasso selection

### `set_active_clip`

bpy.ops.clip.set_active_clip()
(undocumented operator)

### `set_axis`

bpy.ops.clip.set_axis(axis='X')
Set the direction of a scene axis by rotating the camera (or its parent if present). This assumes that the selected track lies on a real axis connecting it to the origin

### `set_origin`

bpy.ops.clip.set_origin(use_median=False)
Set active marker as origin by moving camera (or its parent if present) in 3D space

### `set_plane`

bpy.ops.clip.set_plane(plane='FLOOR')
Set plane based on 3 selected bundles by moving camera (or its parent if present) in 3D space

### `set_scale`

bpy.ops.clip.set_scale(distance=0)
Set scale of scene by scaling camera (or its parent if present)

### `set_scene_frames`

bpy.ops.clip.set_scene_frames()
Set scene's start and end frame to match clip's start frame and length

### `set_solution_scale`

bpy.ops.clip.set_solution_scale(distance=0)
Set object solution scale using distance between two selected tracks

### `set_solver_keyframe`

bpy.ops.clip.set_solver_keyframe(keyframe='KEYFRAME_A')
Set keyframe used by solver

### `set_viewport_background`

bpy.ops.clip.set_viewport_background()
Set current movie clip as a camera background in 3D Viewport (works only when a 3D Viewport is visible)

### `setup_tracking_scene`

bpy.ops.clip.setup_tracking_scene()
Prepare scene for compositing 3D objects into this footage

### `slide_marker`

bpy.ops.clip.slide_marker(offset=(0, 0))
Slide marker areas

### `slide_plane_marker`

bpy.ops.clip.slide_plane_marker()
Slide plane marker areas

### `solve_camera`

bpy.ops.clip.solve_camera()
Solve camera motion from tracks

### `stabilize_2d_add`

bpy.ops.clip.stabilize_2d_add()
Add selected tracks to 2D translation stabilization

### `stabilize_2d_remove`

bpy.ops.clip.stabilize_2d_remove()
Remove selected track from translation stabilization

### `stabilize_2d_rotation_add`

bpy.ops.clip.stabilize_2d_rotation_add()
Add selected tracks to 2D rotation stabilization

### `stabilize_2d_rotation_remove`

bpy.ops.clip.stabilize_2d_rotation_remove()
Remove selected track from rotation stabilization

### `stabilize_2d_rotation_select`

bpy.ops.clip.stabilize_2d_rotation_select()
Select tracks which are used for rotation stabilization

### `stabilize_2d_select`

bpy.ops.clip.stabilize_2d_select()
Select tracks which are used for translation stabilization

### `track_color_preset_add`

bpy.ops.clip.track_color_preset_add(name="", remove_name=False, remove_active=False)
Add or remove a Clip Track Color Preset

### `track_copy_color`

bpy.ops.clip.track_copy_color()
Copy color to all selected tracks

### `track_markers`

bpy.ops.clip.track_markers(backwards=False, sequence=False)
Track selected markers

### `track_settings_as_default`

bpy.ops.clip.track_settings_as_default()
Copy tracking settings from active track to default settings

### `track_settings_to_track`

bpy.ops.clip.track_settings_to_track()
Copy tracking settings from active track to selected tracks

### `track_to_empty`

bpy.ops.clip.track_to_empty()
Create an Empty object which will be copying movement of active track

### `tracking_object_new`

bpy.ops.clip.tracking_object_new()
Add new object for tracking

### `tracking_object_remove`

bpy.ops.clip.tracking_object_remove()
Remove object for tracking

### `tracking_settings_preset_add`

bpy.ops.clip.tracking_settings_preset_add(name="", remove_name=False, remove_active=False)
Add or remove a motion tracking settings preset

### `update_image_from_plane_marker`

bpy.ops.clip.update_image_from_plane_marker()
Update current image used by plane marker from the content of the plane marker

### `view_all`

bpy.ops.clip.view_all(fit_view=False)
View whole image with markers

### `view_center_cursor`

bpy.ops.clip.view_center_cursor()
Center the view so that the cursor is in the middle of the view

### `view_pan`

bpy.ops.clip.view_pan(offset=(0, 0))
Pan the view

### `view_selected`

bpy.ops.clip.view_selected()
View all selected elements

### `view_zoom`

bpy.ops.clip.view_zoom(factor=0, use_cursor_init=True)
Zoom in/out the view

### `view_zoom_in`

bpy.ops.clip.view_zoom_in(location=(0, 0))
Zoom in the view

### `view_zoom_out`

bpy.ops.clip.view_zoom_out(location=(0, 0))
Zoom out the view

### `view_zoom_ratio`

bpy.ops.clip.view_zoom_ratio(ratio=0)
Set the zoom ratio (based on clip size)
