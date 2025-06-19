# screen

Part of `bpy.ops`
Module: `bpy.ops.screen`

## Operators (39)

### `actionzone`

bpy.ops.screen.actionzone(modifier=0)
Handle area action zones for mouse actions/gestures

### `animation_cancel`

bpy.ops.screen.animation_cancel(restore_frame=True)
Cancel animation, returning to the original frame

### `animation_play`

bpy.ops.screen.animation_play(reverse=False, sync=False)
Play animation

### `animation_step`

bpy.ops.screen.animation_step()
Step through animation by position

### `area_close`

bpy.ops.screen.area_close()
Close selected area

### `area_dupli`

bpy.ops.screen.area_dupli()
Duplicate selected area into new window

### `area_join`

bpy.ops.screen.area_join(source_xy=(0, 0), target_xy=(0, 0))
Join selected areas into new window

### `area_move`

bpy.ops.screen.area_move(x=0, y=0, delta=0)
Move selected area edges

### `area_options`

bpy.ops.screen.area_options()
Operations for splitting and merging

### `area_split`

bpy.ops.screen.area_split(direction='HORIZONTAL', factor=0.5, cursor=(0, 0))
Split selected area into new windows

### `area_swap`

bpy.ops.screen.area_swap(cursor=(0, 0))
Swap selected areas screen positions

### `back_to_previous`

bpy.ops.screen.back_to_previous()
Revert back to the original screen layout, before fullscreen area overlay

### `delete`

bpy.ops.screen.delete()
Delete active screen

### `drivers_editor_show`

bpy.ops.screen.drivers_editor_show()
Show drivers editor in a separate window

### `frame_jump`

bpy.ops.screen.frame_jump(end=False)
Jump to first/last frame in frame range

### `frame_offset`

bpy.ops.screen.frame_offset(delta=0)
Move current frame forward/backward by a given number

### `header_toggle_menus`

bpy.ops.screen.header_toggle_menus()
Expand or collapse the header pulldown menus

### `info_log_show`

bpy.ops.screen.info_log_show()
Show info log in a separate window

### `keyframe_jump`

bpy.ops.screen.keyframe_jump(next=True)
Jump to previous/next keyframe

### `marker_jump`

bpy.ops.screen.marker_jump(next=True)
Jump to previous/next marker

### `new`

bpy.ops.screen.new()
Add a new screen

### `redo_last`

bpy.ops.screen.redo_last()
Display parameters for last action performed

### `region_blend`

bpy.ops.screen.region_blend()
Blend in and out overlapping region

### `region_context_menu`

bpy.ops.screen.region_context_menu()
Display region context menu

### `region_flip`

bpy.ops.screen.region_flip()
Toggle the region's alignment (left/right or top/bottom)

### `region_quadview`

bpy.ops.screen.region_quadview()
Split selected area into camera, front, right, and top views

### `region_scale`

bpy.ops.screen.region_scale()
Scale selected area

### `region_toggle`

bpy.ops.screen.region_toggle(region_type='WINDOW')
Hide or unhide the region

### `repeat_history`

bpy.ops.screen.repeat_history(index=0)
Display menu for previous actions performed

### `repeat_last`

bpy.ops.screen.repeat_last()
Repeat last action

### `screen_full_area`

bpy.ops.screen.screen_full_area(use_hide_panels=False)
Toggle display selected area as fullscreen/maximized

### `screen_set`

bpy.ops.screen.screen_set(delta=1)
Cycle through available screens

### `screenshot`

bpy.ops.screen.screenshot(filepath="", hide_props_region=True, check_existing=True, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Capture a picture of the whole Blender window

### `screenshot_area`

bpy.ops.screen.screenshot_area(filepath="", hide_props_region=True, check_existing=True, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Capture a picture of an editor

### `space_context_cycle`

bpy.ops.screen.space_context_cycle(direction='NEXT')
Cycle through the editor context by activating the next/previous one

### `space_type_set_or_cycle`

bpy.ops.screen.space_type_set_or_cycle(space_type='EMPTY')
Set the space type or cycle subtype

### `spacedata_cleanup`

bpy.ops.screen.spacedata_cleanup()
Remove unused settings for invisible editors

### `userpref_show`

bpy.ops.screen.userpref_show(section='INTERFACE')
Edit user preferences and system settings

### `workspace_cycle`

bpy.ops.screen.workspace_cycle(direction='NEXT')
Cycle through workspaces
