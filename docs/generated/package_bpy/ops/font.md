# font

Part of `bpy.ops`
Module: `bpy.ops.font`

## Operators (23)

### `case_set`

bpy.ops.font.case_set(case='LOWER')
Set font case

### `case_toggle`

bpy.ops.font.case_toggle()
Toggle font case

### `change_character`

bpy.ops.font.change_character(delta=1)
Change font character code

### `change_spacing`

bpy.ops.font.change_spacing(delta=1)
Change font spacing

### `delete`

bpy.ops.font.delete(type='PREVIOUS_CHARACTER')
Delete text by cursor position

### `line_break`

bpy.ops.font.line_break()
Insert line break at cursor position

### `move`

bpy.ops.font.move(type='LINE_BEGIN')
Move cursor to position type

### `move_select`

bpy.ops.font.move_select(type='LINE_BEGIN')
Move the cursor while selecting

### `open`

bpy.ops.font.open(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=True, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='THUMBNAIL', sort_method='FILE_SORT_ALPHA')
Load a new font from a file

### `select_all`

bpy.ops.font.select_all()
Select all text

### `select_word`

bpy.ops.font.select_word()
Select word under cursor

### `selection_set`

bpy.ops.font.selection_set()
Set cursor selection

### `style_set`

bpy.ops.font.style_set(style='BOLD', clear=False)
Set font style

### `style_toggle`

bpy.ops.font.style_toggle(style='BOLD')
Toggle font style

### `text_copy`

bpy.ops.font.text_copy()
Copy selected text to clipboard

### `text_cut`

bpy.ops.font.text_cut()
Cut selected text to clipboard

### `text_insert`

bpy.ops.font.text_insert(text="", accent=False)
Insert text at cursor position

### `text_insert_unicode`

bpy.ops.font.text_insert_unicode()
Insert Unicode Character

### `text_paste`

bpy.ops.font.text_paste(selection=False)
Paste text from clipboard

### `text_paste_from_file`

bpy.ops.font.text_paste_from_file(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=True, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, display_type='DEFAULT', sort_method='DEFAULT')
Paste contents from file

### `textbox_add`

bpy.ops.font.textbox_add()
Add a new text box

### `textbox_remove`

bpy.ops.font.textbox_remove(index=0)
Remove the text box

### `unlink`

bpy.ops.font.unlink()
Unlink active font data-block
