# text

Part of `bpy.ops`
Module: `bpy.ops.text`

## Operators (43)

### `autocomplete`

bpy.ops.text.autocomplete()
Show a list of used text in the open document

### `comment_toggle`

bpy.ops.text.comment_toggle(type='TOGGLE')
(undocumented operator)

### `convert_whitespace`

bpy.ops.text.convert_whitespace(type='SPACES')
Convert whitespaces by type

### `copy`

bpy.ops.text.copy()
Copy selected text to clipboard

### `cursor_set`

bpy.ops.text.cursor_set(x=0, y=0)
Set cursor position

### `cut`

bpy.ops.text.cut()
Cut selected text to clipboard

### `delete`

bpy.ops.text.delete(type='NEXT_CHARACTER')
Delete text by cursor position

### `duplicate_line`

bpy.ops.text.duplicate_line()
Duplicate the current line

### `find`

bpy.ops.text.find()
Find specified text

### `find_set_selected`

bpy.ops.text.find_set_selected()
Find specified text and set as selected

### `indent`

bpy.ops.text.indent()
Indent selected text

### `indent_or_autocomplete`

bpy.ops.text.indent_or_autocomplete()
Indent selected text or autocomplete

### `insert`

bpy.ops.text.insert(text="")
Insert text at cursor position

### `jump`

bpy.ops.text.jump(line=1)
Jump cursor to line

### `jump_to_file_at_point`

bpy.ops.text.jump_to_file_at_point(filepath="", line=0, column=0)
Jump to a file for the text editor

### `line_break`

bpy.ops.text.line_break()
Insert line break at cursor position

### `line_number`

bpy.ops.text.line_number()
The current line number

### `make_internal`

bpy.ops.text.make_internal()
Make active text file internal

### `move`

bpy.ops.text.move(type='LINE_BEGIN')
Move cursor to position type

### `move_lines`

bpy.ops.text.move_lines(direction='DOWN')
Move the currently selected line(s) up/down

### `move_select`

bpy.ops.text.move_select(type='LINE_BEGIN')
Move the cursor while selecting

### `new`

bpy.ops.text.new()
Create a new text data-block

### `open`

bpy.ops.text.open(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=True, filter_font=False, filter_sound=False, filter_text=True, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, display_type='DEFAULT', sort_method='DEFAULT', internal=False)
Open a new text data-block

### `overwrite_toggle`

bpy.ops.text.overwrite_toggle()
Toggle overwrite while typing

### `paste`

bpy.ops.text.paste(selection=False)
Paste text from clipboard

### `refresh_pyconstraints`

bpy.ops.text.refresh_pyconstraints()
Refresh all pyconstraints

### `reload`

bpy.ops.text.reload()
Reload active text data-block from its file

### `replace`

bpy.ops.text.replace(all=False)
Replace text with the specified text

### `replace_set_selected`

bpy.ops.text.replace_set_selected()
Replace text with specified text and set as selected

### `resolve_conflict`

bpy.ops.text.resolve_conflict(resolution='IGNORE')
When external text is out of sync, resolve the conflict

### `run_script`

bpy.ops.text.run_script()
Run active script

### `save`

bpy.ops.text.save()
Save active text data-block

### `save_as`

bpy.ops.text.save_as(filepath="", hide_props_region=True, check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=True, filter_font=False, filter_sound=False, filter_text=True, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, display_type='DEFAULT', sort_method='DEFAULT')
Save active text file with options

### `scroll`

bpy.ops.text.scroll(lines=1)
(undocumented operator)

### `scroll_bar`

bpy.ops.text.scroll_bar(lines=1)
(undocumented operator)

### `select_all`

bpy.ops.text.select_all()
Select all text

### `select_line`

bpy.ops.text.select_line()
Select text by line

### `select_word`

bpy.ops.text.select_word()
Select word under cursor

### `selection_set`

bpy.ops.text.selection_set()
Set text selection

### `start_find`

bpy.ops.text.start_find()
Start searching text

### `to_3d_object`

bpy.ops.text.to_3d_object(split_lines=False)
Create 3D text object from active text data-block

### `unindent`

bpy.ops.text.unindent()
Unindent selected text

### `unlink`

bpy.ops.text.unlink()
Unlink active text data-block
