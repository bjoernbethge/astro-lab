# console

Part of `bpy.ops`
Module: `bpy.ops.console`

## Operators (21)

### `autocomplete`

bpy.ops.console.autocomplete()
Evaluate the namespace up until the cursor and give a list of options or complete the name if there is only one

### `banner`

bpy.ops.console.banner()
Print a message when the terminal initializes

### `clear`

bpy.ops.console.clear(scrollback=True, history=False)
Clear text by type

### `clear_line`

bpy.ops.console.clear_line()
Clear the line and store in history

### `copy`

bpy.ops.console.copy(delete=False)
Copy selected text to clipboard

### `copy_as_script`

bpy.ops.console.copy_as_script()
Copy the console contents for use in a script

### `delete`

bpy.ops.console.delete(type='NEXT_CHARACTER')
Delete text by cursor position

### `execute`

bpy.ops.console.execute(interactive=False)
Execute the current console line as a Python expression

### `history_append`

bpy.ops.console.history_append(text="", current_character=0, remove_duplicates=False)
Append history at cursor position

### `history_cycle`

bpy.ops.console.history_cycle(reverse=False)
Cycle through history

### `indent`

bpy.ops.console.indent()
Add 4 spaces at line beginning

### `indent_or_autocomplete`

bpy.ops.console.indent_or_autocomplete()
Indent selected text or autocomplete

### `insert`

bpy.ops.console.insert(text="")
Insert text at cursor position

### `language`

bpy.ops.console.language(language="")
Set the current language for this console

### `move`

bpy.ops.console.move(type='LINE_BEGIN', select=False)
Move cursor position

### `paste`

bpy.ops.console.paste(selection=False)
Paste text from clipboard

### `scrollback_append`

bpy.ops.console.scrollback_append(text="", type='OUTPUT')
Append scrollback text by type

### `select_all`

bpy.ops.console.select_all()
Select all the text

### `select_set`

bpy.ops.console.select_set()
Set the console selection

### `select_word`

bpy.ops.console.select_word()
Select word at cursor position

### `unindent`

bpy.ops.console.unindent()
Delete 4 spaces from line beginning
