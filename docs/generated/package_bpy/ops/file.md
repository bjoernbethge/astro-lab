# file

Part of `bpy.ops`
Module: `bpy.ops.file`

## Operators (40)

### `autopack_toggle`

bpy.ops.file.autopack_toggle()
Automatically pack all external files into the .blend file

### `bookmark_add`

bpy.ops.file.bookmark_add()
Add a bookmark for the selected/active directory

### `bookmark_cleanup`

bpy.ops.file.bookmark_cleanup()
Delete all invalid bookmarks

### `bookmark_delete`

bpy.ops.file.bookmark_delete(index=-1)
Delete selected bookmark

### `bookmark_move`

bpy.ops.file.bookmark_move(direction='TOP')
Move the active bookmark up/down in the list

### `cancel`

bpy.ops.file.cancel()
Cancel file operation

### `delete`

bpy.ops.file.delete()
Move selected files to the trash or recycle bin

### `directory_new`

bpy.ops.file.directory_new(directory="", open=False, confirm=True)
Create a new directory

### `edit_directory_path`

bpy.ops.file.edit_directory_path()
Start editing directory field

### `execute`

bpy.ops.file.execute()
Execute selected file

### `external_operation`

bpy.ops.file.external_operation(filepath="", operation='OPEN')
Perform external operation on a file or folder

### `filenum`

bpy.ops.file.filenum(increment=1)
Increment number in filename

### `filepath_drop`

bpy.ops.file.filepath_drop(filepath="Path")
(undocumented operator)

### `find_missing_files`

bpy.ops.file.find_missing_files(find_all=False, directory="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=False, filter_blenlib=False, filemode=9, display_type='DEFAULT', sort_method='DEFAULT')
Try to find missing external files

### `hidedot`

bpy.ops.file.hidedot()
Toggle hide hidden dot files

### `highlight`

bpy.ops.file.highlight()
Highlight selected file(s)

### `make_paths_absolute`

bpy.ops.file.make_paths_absolute()
Make all paths to external files absolute

### `make_paths_relative`

bpy.ops.file.make_paths_relative()
Make all paths to external files relative to current .blend

### `mouse_execute`

bpy.ops.file.mouse_execute()
Perform the current execute action for the file under the cursor (e.g. open the file)

### `next`

bpy.ops.file.next()
Move to next folder

### `pack_all`

bpy.ops.file.pack_all()
Pack all used external files into this .blend

### `pack_libraries`

bpy.ops.file.pack_libraries()
Store all data-blocks linked from other .blend files in the current .blend file. Library references are preserved so the linked data-blocks can be unpacked again

### `parent`

bpy.ops.file.parent()
Move to parent directory

### `previous`

bpy.ops.file.previous()
Move to previous folder

### `refresh`

bpy.ops.file.refresh()
Refresh the file list

### `rename`

bpy.ops.file.rename()
Rename file or file directory

### `report_missing_files`

bpy.ops.file.report_missing_files()
Report all missing external files

### `reset_recent`

bpy.ops.file.reset_recent()
Reset recent files

### `select`

bpy.ops.file.select(wait_to_deselect_others=False, mouse_x=0, mouse_y=0, extend=False, fill=False, open=True, deselect_all=False, only_activate_if_selected=False, pass_through=False)
Handle mouse clicks to select and activate items

### `select_all`

bpy.ops.file.select_all(action='TOGGLE')
Select or deselect all files

### `select_bookmark`

bpy.ops.file.select_bookmark(dir="")
Select a bookmarked directory

### `select_box`

bpy.ops.file.select_box(xmin=0, xmax=0, ymin=0, ymax=0, wait_for_input=True, mode='SET')
Activate/select the file(s) contained in the border

### `select_walk`

bpy.ops.file.select_walk(direction='UP', extend=False, fill=False)
Select/Deselect files by walking through them

### `smoothscroll`

bpy.ops.file.smoothscroll()
Smooth scroll to make editable file visible

### `sort_column_ui_context`

bpy.ops.file.sort_column_ui_context()
Change sorting to use column under cursor

### `start_filter`

bpy.ops.file.start_filter()
Start entering filter text

### `unpack_all`

bpy.ops.file.unpack_all(method='USE_LOCAL')
Unpack all files packed into this .blend to external ones

### `unpack_item`

bpy.ops.file.unpack_item(method='USE_LOCAL', id_name="", id_type=19785)
Unpack this file to an external file

### `unpack_libraries`

bpy.ops.file.unpack_libraries()
Restore all packed linked data-blocks to their original locations

### `view_selected`

bpy.ops.file.view_selected()
Scroll the selected files into view
