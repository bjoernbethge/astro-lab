# preferences

Part of `bpy.ops`
Module: `bpy.ops.preferences`

## Operators (35)

### `addon_disable`

bpy.ops.preferences.addon_disable(module="")
Turn off this add-on

### `addon_enable`

bpy.ops.preferences.addon_enable(module="")
Turn on this add-on

### `addon_expand`

bpy.ops.preferences.addon_expand(module="")
Display information and preferences for this add-on

### `addon_install`

bpy.ops.preferences.addon_install(overwrite=True, enable_on_install=False, target='DEFAULT', filepath="", filter_folder=True, filter_python=True, filter_glob="*.py;*.zip")
Install an add-on

### `addon_refresh`

bpy.ops.preferences.addon_refresh()
Scan add-on directories for new modules

### `addon_remove`

bpy.ops.preferences.addon_remove(module="")
Delete the add-on from the file system

### `addon_show`

bpy.ops.preferences.addon_show(module="")
Show add-on preferences

### `app_template_install`

bpy.ops.preferences.app_template_install(overwrite=True, filepath="", filter_folder=True, filter_glob="*.zip")
Install an application template

### `asset_library_add`

bpy.ops.preferences.asset_library_add(directory="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, display_type='DEFAULT', sort_method='DEFAULT')
Add a directory to be used by the Asset Browser as source of assets

### `asset_library_remove`

bpy.ops.preferences.asset_library_remove(index=0)
Remove a path to a .blend file, so the Asset Browser will not attempt to show it anymore

### `associate_blend`

bpy.ops.preferences.associate_blend()
Use this installation for .blend files and to display thumbnails

### `autoexec_path_add`

bpy.ops.preferences.autoexec_path_add()
Add path to exclude from auto-execution

### `autoexec_path_remove`

bpy.ops.preferences.autoexec_path_remove(index=0)
Remove path to exclude from auto-execution

### `copy_prev`

bpy.ops.preferences.copy_prev()
Copy settings from previous version

### `extension_repo_add`

bpy.ops.preferences.extension_repo_add(name="", remote_url="", use_access_token=False, access_token="", use_sync_on_startup=False, use_custom_directory=False, custom_directory="", type='REMOTE')
Add a new repository used to store extensions

### `extension_repo_remove`

bpy.ops.preferences.extension_repo_remove(index=0, remove_files=False)
Remove an extension repository

### `extension_url_drop`

bpy.ops.preferences.extension_url_drop(url="")
Handle dropping an extension URL

### `keyconfig_activate`

bpy.ops.preferences.keyconfig_activate(filepath="")
(undocumented operator)

### `keyconfig_export`

bpy.ops.preferences.keyconfig_export(all=False, filepath="", filter_folder=True, filter_text=True, filter_python=True)
Export key configuration to a Python script

### `keyconfig_import`

bpy.ops.preferences.keyconfig_import(filepath="keymap.py", filter_folder=True, filter_text=True, filter_python=True, keep_original=True)
Import key configuration from a Python script

### `keyconfig_remove`

bpy.ops.preferences.keyconfig_remove()
Remove key config

### `keyconfig_test`

bpy.ops.preferences.keyconfig_test()
Test key configuration for conflicts

### `keyitem_add`

bpy.ops.preferences.keyitem_add()
Add key map item

### `keyitem_remove`

bpy.ops.preferences.keyitem_remove(item_id=0)
Remove key map item

### `keyitem_restore`

bpy.ops.preferences.keyitem_restore(item_id=0)
Restore key map item

### `keymap_restore`

bpy.ops.preferences.keymap_restore(all=False)
Restore key map(s)

### `reset_default_theme`

bpy.ops.preferences.reset_default_theme()
Reset to the default theme colors

### `script_directory_add`

bpy.ops.preferences.script_directory_add(directory="", filter_folder=True)
(undocumented operator)

### `script_directory_remove`

bpy.ops.preferences.script_directory_remove(index=0)
(undocumented operator)

### `studiolight_copy_settings`

bpy.ops.preferences.studiolight_copy_settings(index=0)
Copy Studio Light settings to the Studio Light editor

### `studiolight_install`

bpy.ops.preferences.studiolight_install(files=[], directory="", filter_folder=True, filter_glob="*.png;*.jpg;*.hdr;*.exr", type='MATCAP')
Install a user defined light

### `studiolight_new`

bpy.ops.preferences.studiolight_new(filename="StudioLight")
Save custom studio light from the studio light editor settings

### `studiolight_uninstall`

bpy.ops.preferences.studiolight_uninstall(index=0)
Delete Studio Light

### `theme_install`

bpy.ops.preferences.theme_install(overwrite=True, filepath="", filter_folder=True, filter_glob="*.xml")
Load and apply a Blender XML theme file

### `unassociate_blend`

bpy.ops.preferences.unassociate_blend()
Remove this installation's associations with .blend files
