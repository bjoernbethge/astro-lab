# gpencil

Part of `bpy.ops`
Module: `bpy.ops.gpencil`

## Operators (8)

### `annotate`

bpy.ops.gpencil.annotate(mode='DRAW', arrowstyle_start='NONE', arrowstyle_end='NONE', use_stabilizer=False, stabilizer_factor=0.75, stabilizer_radius=35, stroke=[], wait_for_input=True)
Make annotations on the active data

### `annotation_active_frame_delete`

bpy.ops.gpencil.annotation_active_frame_delete()
Delete the active frame for the active Annotation Layer

### `annotation_add`

bpy.ops.gpencil.annotation_add()
Add new Annotation data-block

### `data_unlink`

bpy.ops.gpencil.data_unlink()
Unlink active Annotation data-block

### `layer_annotation_add`

bpy.ops.gpencil.layer_annotation_add()
Add new Annotation layer or note for the active data-block

### `layer_annotation_move`

bpy.ops.gpencil.layer_annotation_move(type='UP')
Move the active Annotation layer up/down in the list

### `layer_annotation_remove`

bpy.ops.gpencil.layer_annotation_remove()
Remove active Annotation layer

### `tint_flip`

bpy.ops.gpencil.tint_flip()
Switch tint colors
