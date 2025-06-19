# workspace

Part of `bpy.ops`
Module: `bpy.ops.workspace`

## Operators (7)

### `add`

bpy.ops.workspace.add()
Add a new workspace by duplicating the current one or appending one from the user configuration

### `append_activate`

bpy.ops.workspace.append_activate(idname="", filepath="")
Append a workspace and make it the active one in the current window

### `delete`

bpy.ops.workspace.delete()
Delete the active workspace

### `duplicate`

bpy.ops.workspace.duplicate()
Add a new workspace

### `reorder_to_back`

bpy.ops.workspace.reorder_to_back()
Reorder workspace to be last in the list

### `reorder_to_front`

bpy.ops.workspace.reorder_to_front()
Reorder workspace to be first in the list

### `scene_pin_toggle`

bpy.ops.workspace.scene_pin_toggle()
Remember the last used scene for the current workspace and switch to it whenever this workspace is activated again
