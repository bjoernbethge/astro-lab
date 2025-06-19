# extensions

Part of `bpy.ops`
Module: `bpy.ops.extensions`

## Operators (34)

### `dummy_progress`

bpy.ops.extensions.dummy_progress()
(undocumented operator)

### `package_disable`

bpy.ops.extensions.package_disable()
Turn off this extension

### `package_enable_not_installed`

bpy.ops.extensions.package_enable_not_installed()
Turn on this extension

### `package_install`

bpy.ops.extensions.package_install(repo_directory="", repo_index=-1, pkg_id="", enable_on_install=True, url="", do_legacy_replace=False)
Download and install the extension

### `package_install_files`

bpy.ops.extensions.package_install_files(filter_glob="*.zip;*.py", directory="", files=[], filepath="", repo='<UNKNOWN ENUM>', enable_on_install=True, target='DEFAULT', overwrite=True, url="")
Install extensions from files into a locally managed repository

### `package_install_marked`

bpy.ops.extensions.package_install_marked(enable_on_install=True)
(undocumented operator)

### `package_mark_clear`

bpy.ops.extensions.package_mark_clear(pkg_id="", repo_index=-1)
(undocumented operator)

### `package_mark_clear_all`

bpy.ops.extensions.package_mark_clear_all()
(undocumented operator)

### `package_mark_set`

bpy.ops.extensions.package_mark_set(pkg_id="", repo_index=-1)
(undocumented operator)

### `package_mark_set_all`

bpy.ops.extensions.package_mark_set_all()
(undocumented operator)

### `package_obsolete_marked`

bpy.ops.extensions.package_obsolete_marked()
Zeroes package versions, useful for development - to test upgrading

### `package_show_clear`

bpy.ops.extensions.package_show_clear(pkg_id="", repo_index=-1)
(undocumented operator)

### `package_show_set`

bpy.ops.extensions.package_show_set(pkg_id="", repo_index=-1)
(undocumented operator)

### `package_show_settings`

bpy.ops.extensions.package_show_settings(pkg_id="", repo_index=-1)
(undocumented operator)

### `package_theme_disable`

bpy.ops.extensions.package_theme_disable(pkg_id="", repo_index=-1)
Turn off this theme

### `package_theme_enable`

bpy.ops.extensions.package_theme_enable(pkg_id="", repo_index=-1)
Turn off this theme

### `package_uninstall`

bpy.ops.extensions.package_uninstall(repo_directory="", repo_index=-1, pkg_id="")
Disable and uninstall the extension

### `package_uninstall_marked`

bpy.ops.extensions.package_uninstall_marked()
(undocumented operator)

### `package_uninstall_system`

bpy.ops.extensions.package_uninstall_system()
(undocumented operator)

### `package_upgrade_all`

bpy.ops.extensions.package_upgrade_all(use_active_only=False)
Upgrade all the extensions to their latest version for all the remote repositories

### `repo_enable_from_drop`

bpy.ops.extensions.repo_enable_from_drop(repo_index=-1)
(undocumented operator)

### `repo_lock_all`

bpy.ops.extensions.repo_lock_all()
Lock repositories - to test locking

### `repo_refresh_all`

bpy.ops.extensions.repo_refresh_all()
Scan extension & legacy add-ons for changes to modules & meta-data (similar to restarting). Any issues are reported as warnings

### `repo_sync`

bpy.ops.extensions.repo_sync(repo_directory="", repo_index=-1)
(undocumented operator)

### `repo_sync_all`

bpy.ops.extensions.repo_sync_all(use_active_only=False)
Refresh the list of extensions for all the remote repositories

### `repo_unlock`

bpy.ops.extensions.repo_unlock()
Remove the repository file-system lock

### `repo_unlock_all`

bpy.ops.extensions.repo_unlock_all()
Unlock repositories - to test unlocking

### `status_clear`

bpy.ops.extensions.status_clear()
(undocumented operator)

### `status_clear_errors`

bpy.ops.extensions.status_clear_errors()
(undocumented operator)

### `userpref_allow_online`

bpy.ops.extensions.userpref_allow_online()
Allow internet access. Blender may access configured online extension repositories. Installed third party add-ons may access the internet for their own functionality

### `userpref_allow_online_popup`

bpy.ops.extensions.userpref_allow_online_popup()
Allow internet access. Blender may access configured online extension repositories. Installed third party add-ons may access the internet for their own functionality

### `userpref_show_for_update`

bpy.ops.extensions.userpref_show_for_update()
Open extensions preferences

### `userpref_show_online`

bpy.ops.extensions.userpref_show_online()
Show system preferences "Network" panel to allow online access

### `userpref_tags_set`

bpy.ops.extensions.userpref_tags_set(value=False, data_path="")
Set the value of all tags
