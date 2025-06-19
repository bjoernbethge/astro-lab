# ed

Part of `bpy.ops`
Module: `bpy.ops.ed`

## Operators (13)

### `flush_edits`

bpy.ops.ed.flush_edits()
Flush edit data from active editing modes

### `lib_id_fake_user_toggle`

bpy.ops.ed.lib_id_fake_user_toggle()
Save this data-block even if it has no users

### `lib_id_generate_preview`

bpy.ops.ed.lib_id_generate_preview()
Create an automatic preview for the selected data-block

### `lib_id_generate_preview_from_object`

bpy.ops.ed.lib_id_generate_preview_from_object()
Create a preview for this asset by rendering the active object

### `lib_id_load_custom_preview`

bpy.ops.ed.lib_id_load_custom_preview(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Choose an image to help identify the data-block visually

### `lib_id_override_editable_toggle`

bpy.ops.ed.lib_id_override_editable_toggle()
Set if this library override data-block can be edited

### `lib_id_remove_preview`

bpy.ops.ed.lib_id_remove_preview()
Remove the preview of this data-block

### `lib_id_unlink`

bpy.ops.ed.lib_id_unlink()
Remove a usage of a data-block, clearing the assignment

### `redo`

bpy.ops.ed.redo()
Redo previous action

### `undo`

bpy.ops.ed.undo()
Undo previous action

### `undo_history`

bpy.ops.ed.undo_history(item=0)
Redo specific action in history

### `undo_push`

bpy.ops.ed.undo_push(message="Add an undo step *function may be moved*")
Add an undo state (internal use only)

### `undo_redo`

bpy.ops.ed.undo_redo()
Undo and redo previous action
