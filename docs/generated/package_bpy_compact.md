# Bpy Package Documentation

Auto-generated documentation for installed package `bpy`

## Package Information

- **Version**: 4.4.0
- **Location**: D:\astro-lab\.venv\Lib\site-packages
- **Summary**: Blender as a Python module

## Submodules

### ops
Module: `bpy.ops`

#### Nested Submodules

##### action
Module: `bpy.ops.action`

**Operators:**
`bake_keys, clean, clickselect, copy, delete`

##### anim
Module: `bpy.ops.anim`

**Operators:**
`change_frame, channel_select_keys, channel_view_pick, channels_bake, channels_clean_empty`

##### armature
Module: `bpy.ops.armature`

**Operators:**
`align, assign_to_collection, autoside_names, bone_primitive_add, calculate_roll`

##### asset
Module: `bpy.ops.asset`

**Operators:**
`assign_action, bundle_install, catalog_delete, catalog_new, catalog_redo`

##### boid
Module: `bpy.ops.boid`

**Operators:**
`rule_add, rule_del, rule_move_down, rule_move_up, state_add`

##### brush
Module: `bpy.ops.brush`

**Operators:**
`asset_activate, asset_delete, asset_edit_metadata, asset_load_preview, asset_revert`

##### buttons
Module: `bpy.ops.buttons`

**Operators:**
`clear_filter, context_menu, directory_browse, file_browse, start_filter`

##### cachefile
Module: `bpy.ops.cachefile`

**Operators:**
`layer_add, layer_move, layer_remove, open, reload`


### path
Module: `bpy.path`

This module has a similar scope to os.path, containing utility
functions for dealing with paths in Blender.

#### Functions

- **`abspath(path, *, start=None, library=None)`**
  Returns the absolute path relative to the current blend file

- **`basename(path)`**
  Equivalent to ``os.path.basename``, but skips a "//" prefix.

- **`clean_name(name, *, replace='_')`**
  Returns a name with characters replaced that

- **`display_name(name, *, has_ext=True, title_case=True)`**
  Creates a display string from name to be used menus and the user interface.

- **`display_name_from_filepath(name)`**
  Returns the path stripped of directory and extension,

- **`display_name_to_filepath(name)`**
  Performs the reverse of display_name using literal versions of characters

- **`ensure_ext(filepath, ext, *, case_sensitive=False)`**
  Return the path with the extension added if it is not already set.

- **`is_subdir(path, directory)`**
  Returns true if *path* in a subdirectory of *directory*.

- **`module_names(path, *, recursive=False, package='')`**
  Return a list of modules which can be imported from *path*.

- **`native_pathsep(path)`**
  Replace the path separator with the systems native ``os.sep``.

#### Attributes

- **`extensions_audio`** (frozenset): `frozenset({'.ogg', '.opus', '.aif', '.mka', '.wav', '.mp3', '.aiff', '.flac', '.wma', '.mp2', '.aac'...`
- **`extensions_image`** (frozenset): `frozenset({'.bmp', '.jpg', '.png', '.psb', '.dds', '.tx', '.cin', '.tiff', '.tga', '.tif', '.j2c', '...`
- **`extensions_movie`** (frozenset): `frozenset({'.ogg', '.m4v', '.m2t', '.mxf', '.avi', '.webm', '.m2ts', '.vob', '.r3d', '.ts', '.xvid',...`

#### Other Data Types

- **`extensions_audio`** (<class 'frozenset'>)
  frozenset() -> empty frozenset object
- **`extensions_image`** (<class 'frozenset'>)
  frozenset() -> empty frozenset object
- **`extensions_movie`** (<class 'frozenset'>)
  frozenset() -> empty frozenset object

### props
Module: `bpy.props`

This module defines properties to extend Blender's internal data. The result of these functions is used to assign properties to classes registered with Blender and can't be used directly.

.. note:: All parameters to these functions must be passed as keywords.

#### Important Data Types

- **`IntProperty`** (<class 'builtin_function_or_method'>)
  .. function:: IntProperty(*, name="", description="", translation_context="*", default=0, min=-2**31, max=2**31-1, soft_min=-2**31, soft_max=2**31-1, step=1, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', update=None, get=None, set=None)
  *(callable)*

- **`BoolProperty`** (<class 'builtin_function_or_method'>)
  .. function:: BoolProperty(*, name="", description="", translation_context="*", default=False, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', update=None, get=None, set=None)
  *(callable)*

- **`EnumProperty`** (<class 'builtin_function_or_method'>)
  .. function:: EnumProperty(items, *, name="", description="", translation_context="*", default=None, options={'ANIMATABLE'}, override=set(), tags=set(), update=None, get=None, set=None)
  *(callable)*

- **`FloatProperty`** (<class 'builtin_function_or_method'>)
  .. function:: FloatProperty(*, name="", description="", translation_context="*", default=0.0, min=-3.402823e+38, max=3.402823e+38, soft_min=-3.402823e+38, soft_max=3.402823e+38, step=3, precision=2, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', unit='NONE', update=None, get=None, set=None)
  *(callable)*

- **`RemoveProperty`** (<class 'builtin_function_or_method'>)
  .. function:: RemoveProperty(cls, attr)
  *(callable)*

- **`StringProperty`** (<class 'builtin_function_or_method'>)
  .. function:: StringProperty(*, name="", description="", translation_context="*", default="", maxlen=0, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', update=None, get=None, set=None, search=None, search_options={'SUGGESTION'})
  *(callable)*

- **`PointerProperty`** (<class 'builtin_function_or_method'>)
  .. function:: PointerProperty(type=None, *, name="", description="", translation_context="*", options={'ANIMATABLE'}, override=set(), tags=set(), poll=None, update=None)
  *(callable)*

- **`IntVectorProperty`** (<class 'builtin_function_or_method'>)
  .. function:: IntVectorProperty(*, name="", description="", translation_context="*", default=(0, 0, 0), min=-2**31, max=2**31-1, soft_min=-2**31, soft_max=2**31-1, step=1, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', size=3, update=None, get=None, set=None)
  *(callable)*

- **`BoolVectorProperty`** (<class 'builtin_function_or_method'>)
  .. function:: BoolVectorProperty(*, name="", description="", translation_context="*", default=(False, False, False), options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', size=3, update=None, get=None, set=None)
  *(callable)*

- **`CollectionProperty`** (<class 'builtin_function_or_method'>)
  .. function:: CollectionProperty(type=None, *, name="", description="", translation_context="*", options={'ANIMATABLE'}, override=set(), tags=set())
  *(callable)*

- **`FloatVectorProperty`** (<class 'builtin_function_or_method'>)
  .. function:: FloatVectorProperty(*, name="", description="", translation_context="*", default=(0.0, 0.0, 0.0), min=sys.float_info.min, max=sys.float_info.max, soft_min=sys.float_info.min, soft_max=sys.float_info.max, step=3, precision=2, options={'ANIMATABLE'}, override=set(), tags=set(), subtype='NONE', unit='NONE', size=3, update=None, get=None, set=None)
  *(callable)*

### types
Module: `bpy.types`

Access to internal Blender types

#### Classes

- **`ANIM_MT_keyframe_insert_pie`**
  Methods: draw

- **`ANIM_OT_clear_useless_actions`**
  Mark actions with no F-Curves for deletion after save and reload of file preserving "action libraries"
  Methods: execute

- **`ANIM_OT_keying_set_export`**
  Export Keying Set to a Python script
  Methods: execute, invoke

- **`ANIM_OT_slot_new_for_id`**
  Create a new Action Slot for an ID.
  Methods: execute

- **`ANIM_OT_slot_unassign_from_constraint`**
  Un-assign the assigned Action Slot from an Action constraint.

#### Important Data Types

- **`ID`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`AOV`** (<class 'type'>)
  *(has methods, callable)*

- **`Key`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`AOVs`** (<class 'type'>)
  *(has methods, callable)*

- **`Area`** (<class 'type'>)
  *(has methods, callable)*

- **`Bone`** (<class 'bpy_struct_meta_idprop'>)
  functions for bones, common between Armature/Pose/Edit bones.
  *(has methods, callable)*

- **`Mask`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`Menu`** (<class 'bpy_types._RNAMeta'>)
  *(has methods, callable)*

- **`Mesh`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`Node`** (<class 'bpy_types._RNAMetaPropGroup'>)
  *(has methods, callable)*

- **`Pose`** (<class 'type'>)
  *(has methods, callable)*

- **`Text`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`Addon`** (<class 'type'>)
  *(has methods, callable)*

- **`Brush`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

- **`Curve`** (<class 'bpy_struct_meta_idprop'>)
  *(has methods, callable)*

### utils
Module: `bpy.utils`

This module contains utility functions specific to blender but
not associated with blenders internal data.

#### Functions

- **`app_template_paths(*, path=None)`**
  Returns valid application template paths.

- **`execfile(filepath, *, mod=None)`**
  Execute a file path as a Python script.

- **`expose_bundled_modules()`**
  For Blender as a Python module, add bundled VFX library python bindings

- **`extension_path_user(package, *, path='', create=False)`**
  Return a user writable directory associated with an extension.

- **`is_path_builtin(path)`**
  Returns True if the path is one of the built-in paths used by Blender.

- **`is_path_extension(path)`**
  Returns True if the path is from an extensions repository.

- **`keyconfig_init()`**

- **`keyconfig_set(filepath, *, report=None)`**

- **`load_scripts(*, reload_scripts=False, refresh_scripts=False, extensions=True)`**
  Load scripts and run each modules register function.

- **`load_scripts_extensions(*, reload_scripts=False)`**
  Load extensions scripts (add-ons and app-templates)

#### Important Data Types

- **`flip_name`** (<class 'builtin_function_or_method'>)
  .. function:: flip_name(name, strip_digits=False)
  *(callable)*

- **`blend_paths`** (<class 'builtin_function_or_method'>)
  .. function:: blend_paths(absolute=False, packed=False, local=False)
  *(callable)*

- **`resource_path`** (<class 'builtin_function_or_method'>)
  .. function:: resource_path(type, major=bpy.app.version[0], minor=bpy.app.version[1])
  *(callable)*

- **`register_class`** (<class 'builtin_function_or_method'>)
  .. function:: register_class(cls)
  *(callable)*

- **`system_resource`** (<class 'builtin_function_or_method'>)
  .. function:: system_resource(type, path="")
  *(callable)*

- **`unregister_class`** (<class 'builtin_function_or_method'>)
  .. function:: unregister_class(cls)
  *(callable)*

- **`escape_identifier`** (<class 'builtin_function_or_method'>)
  .. function:: escape_identifier(string)
  *(callable)*

- **`unescape_identifier`** (<class 'builtin_function_or_method'>)
  .. function:: unescape_identifier(string)
  *(callable)*

- **`register_cli_command`** (<class 'builtin_function_or_method'>)
  .. method:: register_cli_command(id, execute)
  *(callable)*

- **`unregister_cli_command`** (<class 'builtin_function_or_method'>)
  .. method:: unregister_cli_command(handle)
  *(callable)*

## Attributes

### app
Type: `app`
Value: `bpy.app(version=(4, 4, 0), version_file=(4, 4, 30), version_string='4.4.0', version_cycle='release',...`

This module contains application values that remain unchanged during runtime.

### context
Type: `Context`
Value: `<bpy_struct, Context at 0x000002B7E816E3B8>`

### data
Type: `BlendData`
Value: `<bpy_struct, BlendData at 0x000002B7E97088A8>`
