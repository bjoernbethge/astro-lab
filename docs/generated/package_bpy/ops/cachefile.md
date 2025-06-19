# cachefile

Part of `bpy.ops`
Module: `bpy.ops.cachefile`

## Operators (5)

### `layer_add`

bpy.ops.cachefile.layer_add(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=True, filter_usd=True, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT')
Add an override layer to the archive

### `layer_move`

bpy.ops.cachefile.layer_move(direction='UP')
Move layer in the list, layers further down the list will overwrite data from the layers higher up

### `layer_remove`

bpy.ops.cachefile.layer_remove()
Remove an override layer from the archive

### `open`

bpy.ops.cachefile.open(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=True, filter_usd=True, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT')
Load a cache file

### `reload`

bpy.ops.cachefile.reload()
Update objects paths list with new data from the archive
