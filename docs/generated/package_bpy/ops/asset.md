# asset

Part of `bpy.ops`
Module: `bpy.ops.asset`

## Operators (16)

### `assign_action`

bpy.ops.asset.assign_action()
Set this pose Action as active Action on the active Object

### `bundle_install`

bpy.ops.asset.bundle_install(asset_library_reference='<UNKNOWN ENUM>', filepath="", hide_props_region=True, check_existing=True, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT')
Copy the current .blend file into an Asset Library. Only works on standalone .blend files (i.e. when no other files are referenced)

### `catalog_delete`

bpy.ops.asset.catalog_delete(catalog_id="")
Remove an asset catalog from the asset library (contained assets will not be affected and show up as unassigned)

### `catalog_new`

bpy.ops.asset.catalog_new(parent_path="")
Create a new catalog to put assets in

### `catalog_redo`

bpy.ops.asset.catalog_redo()
Redo the last undone edit to the asset catalogs

### `catalog_undo`

bpy.ops.asset.catalog_undo()
Undo the last edit to the asset catalogs

### `catalog_undo_push`

bpy.ops.asset.catalog_undo_push()
Store the current state of the asset catalogs in the undo buffer

### `catalogs_save`

bpy.ops.asset.catalogs_save()
Make any edits to any catalogs permanent by writing the current set up to the asset library

### `clear`

bpy.ops.asset.clear(set_fake_user=False)
Delete all asset metadata and turn the selected asset data-blocks back into normal data-blocks

### `clear_single`

bpy.ops.asset.clear_single(set_fake_user=False)
Delete all asset metadata and turn the asset data-block back into a normal data-block

### `library_refresh`

bpy.ops.asset.library_refresh()
Reread assets and asset catalogs from the asset library on disk

### `mark`

bpy.ops.asset.mark()
Enable easier reuse of selected data-blocks through the Asset Browser, with the help of customizable metadata (like previews, descriptions and tags)

### `mark_single`

bpy.ops.asset.mark_single()
Enable easier reuse of a data-block through the Asset Browser, with the help of customizable metadata (like previews, descriptions and tags)

### `open_containing_blend_file`

bpy.ops.asset.open_containing_blend_file()
Open the blend file that contains the active asset

### `tag_add`

bpy.ops.asset.tag_add()
Add a new keyword tag to the active asset

### `tag_remove`

bpy.ops.asset.tag_remove()
Remove an existing keyword tag from the active asset
