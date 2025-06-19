# sequencer

Part of `bpy.ops`
Module: `bpy.ops.sequencer`

## Operators (101)

### `change_effect_input`

bpy.ops.sequencer.change_effect_input()
(undocumented operator)

### `change_effect_type`

bpy.ops.sequencer.change_effect_type(type='CROSS')
(undocumented operator)

### `change_path`

bpy.ops.sequencer.change_path(filepath="", directory="", files=[], hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', use_placeholders=False)
(undocumented operator)

### `change_scene`

bpy.ops.sequencer.change_scene(scene='<UNKNOWN ENUM>')
Change Scene assigned to Strip

### `connect`

bpy.ops.sequencer.connect(toggle=True)
Link selected strips together for simplified group selection

### `copy`

bpy.ops.sequencer.copy()
Copy the selected strips to the internal clipboard

### `crossfade_sounds`

bpy.ops.sequencer.crossfade_sounds()
Do cross-fading volume animation of two selected sound strips

### `cursor_set`

bpy.ops.sequencer.cursor_set(location=(0, 0))
Set 2D cursor location

### `deinterlace_selected_movies`

bpy.ops.sequencer.deinterlace_selected_movies()
Deinterlace all selected movie sources

### `delete`

bpy.ops.sequencer.delete(delete_data=False)
Delete selected strips from the sequencer

### `disconnect`

bpy.ops.sequencer.disconnect()
Unlink selected strips so that they can be selected individually

### `duplicate`

bpy.ops.sequencer.duplicate()
Duplicate the selected strips

### `duplicate_move`

bpy.ops.sequencer.duplicate_move(SEQUENCER_OT_duplicate={}, TRANSFORM_OT_seq_slide={"value":(0, 0), "use_restore_handle_selection":False, "snap":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False})
Duplicate selected strips and move them

### `effect_strip_add`

bpy.ops.sequencer.effect_strip_add(type='CROSS', frame_start=0, frame_end=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, color=(0, 0, 0))
Add an effect to the sequencer, most are applied on top of existing strips

### `enable_proxies`

bpy.ops.sequencer.enable_proxies(proxy_25=False, proxy_50=False, proxy_75=False, proxy_100=False, overwrite=False)
Enable selected proxies on all selected Movie and Image strips

### `export_subtitles`

bpy.ops.sequencer.export_subtitles(filepath="", hide_props_region=True, check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT')
Export .srt file containing text strips

### `fades_add`

bpy.ops.sequencer.fades_add(duration_seconds=1, type='IN_OUT')
Adds or updates a fade animation for either visual or audio strips

### `fades_clear`

bpy.ops.sequencer.fades_clear()
Removes fade animation from selected sequences

### `gap_insert`

bpy.ops.sequencer.gap_insert(frames=10)
Insert gap at current frame to first strips at the right, independent of selection or locked state of strips

### `gap_remove`

bpy.ops.sequencer.gap_remove(all=False)
Remove gap at current frame to first strip at the right, independent of selection or locked state of strips

### `image_strip_add`

bpy.ops.sequencer.image_strip_add(directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', frame_start=0, frame_end=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, fit_method='FIT', set_view_transform=True, use_placeholders=False)
Add an image or image sequence to the sequencer

### `images_separate`

bpy.ops.sequencer.images_separate(length=1)
On image sequence strips, it returns a strip for each image

### `lock`

bpy.ops.sequencer.lock()
Lock strips so they can't be transformed

### `mask_strip_add`

bpy.ops.sequencer.mask_strip_add(frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, mask='<UNKNOWN ENUM>')
Add a mask strip to the sequencer

### `meta_make`

bpy.ops.sequencer.meta_make()
Group selected strips into a meta-strip

### `meta_separate`

bpy.ops.sequencer.meta_separate()
Put the contents of a meta-strip back in the sequencer

### `meta_toggle`

bpy.ops.sequencer.meta_toggle()
Toggle a meta-strip (to edit enclosed strips)

### `movie_strip_add`

bpy.ops.sequencer.movie_strip_add(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, fit_method='FIT', set_view_transform=True, adjust_playback_rate=True, sound=True, use_framerate=True)
Add a movie strip to the sequencer

### `movieclip_strip_add`

bpy.ops.sequencer.movieclip_strip_add(frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, clip='<UNKNOWN ENUM>')
Add a movieclip strip to the sequencer

### `mute`

bpy.ops.sequencer.mute(unselected=False)
Mute (un)selected strips

### `offset_clear`

bpy.ops.sequencer.offset_clear()
Clear strip offsets from the start and end frames

### `paste`

bpy.ops.sequencer.paste(keep_offset=False)
Paste strips from the internal clipboard

### `preview_duplicate_move`

bpy.ops.sequencer.preview_duplicate_move(SEQUENCER_OT_duplicate={}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate selected strips and move them

### `reassign_inputs`

bpy.ops.sequencer.reassign_inputs()
Reassign the inputs for the effect strip

### `rebuild_proxy`

bpy.ops.sequencer.rebuild_proxy()
Rebuild all selected proxies and timecode indices

### `refresh_all`

bpy.ops.sequencer.refresh_all()
Refresh the sequencer editor

### `reload`

bpy.ops.sequencer.reload(adjust_length=False)
Reload strips in the sequencer

### `rename_channel`

bpy.ops.sequencer.rename_channel()
(undocumented operator)

### `rendersize`

bpy.ops.sequencer.rendersize()
Set render size and aspect from active sequence

### `retiming_add_freeze_frame_slide`

bpy.ops.sequencer.retiming_add_freeze_frame_slide(SEQUENCER_OT_retiming_freeze_frame_add={"duration":0}, TRANSFORM_OT_seq_slide={"value":(0, 0), "use_restore_handle_selection":False, "snap":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False})
Add freeze frame and move it

### `retiming_add_transition_slide`

bpy.ops.sequencer.retiming_add_transition_slide(SEQUENCER_OT_retiming_transition_add={"duration":0}, TRANSFORM_OT_seq_slide={"value":(0, 0), "use_restore_handle_selection":False, "snap":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False})
Add smooth transition between 2 retimed segments and change its duration

### `retiming_freeze_frame_add`

bpy.ops.sequencer.retiming_freeze_frame_add(duration=0)
Add freeze frame

### `retiming_key_add`

bpy.ops.sequencer.retiming_key_add(timeline_frame=0)
Add retiming Key

### `retiming_key_delete`

bpy.ops.sequencer.retiming_key_delete()
Delete selected strips from the sequencer

### `retiming_reset`

bpy.ops.sequencer.retiming_reset()
Reset strip retiming

### `retiming_segment_speed_set`

bpy.ops.sequencer.retiming_segment_speed_set(speed=100, keep_retiming=True)
Set speed of retimed segment

### `retiming_show`

bpy.ops.sequencer.retiming_show()
Show retiming keys in selected strips

### `retiming_transition_add`

bpy.ops.sequencer.retiming_transition_add(duration=0)
Add smooth transition between 2 retimed segments

### `sample`

bpy.ops.sequencer.sample(size=1)
Use mouse to sample color in current frame

### `scene_frame_range_update`

bpy.ops.sequencer.scene_frame_range_update()
Update frame range of scene strip

### `scene_strip_add`

bpy.ops.sequencer.scene_strip_add(frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, scene='<UNKNOWN ENUM>')
Add a strip to the sequencer using a Blender scene as a source

### `scene_strip_add_new`

bpy.ops.sequencer.scene_strip_add_new(frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, type='NEW')
Create a new Strip and assign a new Scene as source

### `select`

bpy.ops.sequencer.select(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, deselect=False, toggle=False, deselect_all=False, select_passthrough=False, center=False, linked_handle=False, linked_time=False, side_of_frame=False, ignore_connections=False)
Select a strip (last selected becomes the "active strip")

### `select_all`

bpy.ops.sequencer.select_all(action='TOGGLE')
Select or deselect all strips

### `select_box`

bpy.ops.sequencer.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET', tweak=False, include_handles=False, ignore_connections=False)
Select strips using box selection

### `select_grouped`

bpy.ops.sequencer.select_grouped(type='TYPE', extend=False, use_active_channel=False)
Select all strips grouped by various properties

### `select_handle`

bpy.ops.sequencer.select_handle(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, ignore_connections=False)
Select strip handle

### `select_handles`

bpy.ops.sequencer.select_handles(side='BOTH')
Select gizmo handles on the sides of the selected strip

### `select_less`

bpy.ops.sequencer.select_less()
Shrink the current selection of adjacent selected strips

### `select_linked`

bpy.ops.sequencer.select_linked()
Select all strips adjacent to the current selection

### `select_linked_pick`

bpy.ops.sequencer.select_linked_pick(extend=False)
Select a chain of linked strips nearest to the mouse pointer

### `select_more`

bpy.ops.sequencer.select_more()
Select more strips adjacent to the current selection

### `select_side`

bpy.ops.sequencer.select_side(side='BOTH')
Select strips on the nominated side of the selected strips

### `select_side_of_frame`

bpy.ops.sequencer.select_side_of_frame(extend=False, side='LEFT')
Select strips relative to the current frame

### `set_range_to_strips`

bpy.ops.sequencer.set_range_to_strips(preview=False)
Set the frame range to the selected strips start and end

### `slip`

bpy.ops.sequencer.slip(offset=0)
Slip the contents of selected strips

### `snap`

bpy.ops.sequencer.snap(frame=0)
Frame where selected strips will be snapped

### `sound_strip_add`

bpy.ops.sequencer.sound_strip_add(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=True, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', frame_start=0, channel=1, replace_sel=True, overlap=False, overlap_shuffle_override=False, cache=False, mono=False)
Add a sound strip to the sequencer

### `split`

bpy.ops.sequencer.split(frame=0, channel=0, type='SOFT', use_cursor_position=False, side='MOUSE', ignore_selection=False)
Split the selected strips in two

### `split_multicam`

bpy.ops.sequencer.split_multicam(camera=1)
Split multicam strip and select camera

### `strip_color_tag_set`

bpy.ops.sequencer.strip_color_tag_set(color='NONE')
Set a color tag for the selected strips

### `strip_jump`

bpy.ops.sequencer.strip_jump(next=True, center=True)
Move frame to previous edit point

### `strip_modifier_add`

bpy.ops.sequencer.strip_modifier_add(type='<UNKNOWN ENUM>')
Add a modifier to the strip

### `strip_modifier_copy`

bpy.ops.sequencer.strip_modifier_copy(type='REPLACE')
Copy modifiers of the active strip to all selected strips

### `strip_modifier_equalizer_redefine`

bpy.ops.sequencer.strip_modifier_equalizer_redefine(graphs='SIMPLE', name="Name")
Redefine equalizer graphs

### `strip_modifier_move`

bpy.ops.sequencer.strip_modifier_move(name="Name", direction='UP')
Move modifier up and down in the stack

### `strip_modifier_remove`

bpy.ops.sequencer.strip_modifier_remove(name="Name")
Remove a modifier from the strip

### `strip_transform_clear`

bpy.ops.sequencer.strip_transform_clear(property='ALL')
Reset image transformation to default value

### `strip_transform_fit`

bpy.ops.sequencer.strip_transform_fit(fit_method='FIT')
(undocumented operator)

### `swap`

bpy.ops.sequencer.swap(side='RIGHT')
Swap active strip with strip to the right or left

### `swap_data`

bpy.ops.sequencer.swap_data()
Swap 2 sequencer strips

### `swap_inputs`

bpy.ops.sequencer.swap_inputs()
Swap the two inputs of the effect strip

### `text_cursor_move`

bpy.ops.sequencer.text_cursor_move(type='LINE_BEGIN', select_text=False)
Move cursor in text

### `text_cursor_set`

bpy.ops.sequencer.text_cursor_set(select_text=False)
Set cursor position in text

### `text_delete`

bpy.ops.sequencer.text_delete(type='NEXT_OR_SELECTION')
Delete text at cursor position

### `text_deselect_all`

bpy.ops.sequencer.text_deselect_all()
Deselect all characters

### `text_edit_copy`

bpy.ops.sequencer.text_edit_copy()
Copy text to clipboard

### `text_edit_cut`

bpy.ops.sequencer.text_edit_cut()
Cut text to clipboard

### `text_edit_mode_toggle`

bpy.ops.sequencer.text_edit_mode_toggle()
Toggle text editing

### `text_edit_paste`

bpy.ops.sequencer.text_edit_paste()
Paste text to clipboard

### `text_insert`

bpy.ops.sequencer.text_insert(string="")
Insert text at cursor position

### `text_line_break`

bpy.ops.sequencer.text_line_break()
Insert line break at cursor position

### `text_select_all`

bpy.ops.sequencer.text_select_all()
Select all characters

### `unlock`

bpy.ops.sequencer.unlock()
Unlock strips so they can be transformed

### `unmute`

bpy.ops.sequencer.unmute(unselected=False)
Unmute (un)selected strips

### `view_all`

bpy.ops.sequencer.view_all()
View all the strips in the sequencer

### `view_all_preview`

bpy.ops.sequencer.view_all_preview()
Zoom preview to fit in the area

### `view_frame`

bpy.ops.sequencer.view_frame()
Move the view to the current frame

### `view_ghost_border`

bpy.ops.sequencer.view_ghost_border(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True)
Set the boundaries of the border used for offset view

### `view_selected`

bpy.ops.sequencer.view_selected()
Zoom the sequencer on the selected strips

### `view_zoom_ratio`

bpy.ops.sequencer.view_zoom_ratio(ratio=1)
Change zoom ratio of sequencer preview
