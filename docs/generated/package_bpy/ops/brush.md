# brush

Part of `bpy.ops`
Module: `bpy.ops.brush`

## Operators (13)

### `asset_activate`

bpy.ops.brush.asset_activate(asset_library_type='LOCAL', asset_library_identifier="", relative_asset_identifier="")
Activate a brush asset as current sculpt and paint tool

### `asset_delete`

bpy.ops.brush.asset_delete()
Delete the active brush asset

### `asset_edit_metadata`

bpy.ops.brush.asset_edit_metadata(catalog_path="", author="", description="")
Edit asset information like the catalog, preview image, tags, or author

### `asset_load_preview`

bpy.ops.brush.asset_load_preview(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT')
Choose a preview image for the brush

### `asset_revert`

bpy.ops.brush.asset_revert()
Revert the active brush settings to the default values from the asset library

### `asset_save`

bpy.ops.brush.asset_save()
Update the active brush asset in the asset library with current settings

### `asset_save_as`

bpy.ops.brush.asset_save_as(name="", asset_library_reference='<UNKNOWN ENUM>', catalog_path="")
Save a copy of the active brush asset into the default asset library, and make it the active brush

### `curve_preset`

bpy.ops.brush.curve_preset(shape='SMOOTH')
Set brush shape

### `scale_size`

bpy.ops.brush.scale_size(scalar=1)
Change brush size by a scalar

### `sculpt_curves_falloff_preset`

bpy.ops.brush.sculpt_curves_falloff_preset(shape='SMOOTH')
Set Curve Falloff Preset

### `stencil_control`

bpy.ops.brush.stencil_control(mode='TRANSLATION', texmode='PRIMARY')
Control the stencil brush

### `stencil_fit_image_aspect`

bpy.ops.brush.stencil_fit_image_aspect(use_repeat=True, use_scale=True, mask=False)
When using an image texture, adjust the stencil size to fit the image aspect ratio

### `stencil_reset_transform`

bpy.ops.brush.stencil_reset_transform(mask=False)
Reset the stencil transformation to the default
