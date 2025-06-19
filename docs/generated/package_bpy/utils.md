# utils Submodule

Part of the `bpy` package
Module: `bpy.utils`

## Description

This module contains utility functions specific to blender but
not associated with blenders internal data.

## Functions (34)

### `app_template_paths(*, path=None)`

Returns valid application template paths.

:arg path: Optional subdir.
:type path: str
:return: App template paths.
:rtype: Iterator[str]

### `execfile(filepath, *, mod=None)`

Execute a file path as a Python script.

:arg filepath: Path of the script to execute.
:type filepath: str
:arg mod: Optional cached module, the result of a previous execution.
:type mod: ModuleType | None
:return: The module which can be passed back in as ``mod``.
:rtype: ModuleType

### `expose_bundled_modules()`

For Blender as a Python module, add bundled VFX library python bindings
to ``sys.path``. These may be used instead of dedicated packages, to ensure
the libraries are compatible with Blender.

### `extension_path_user(package, *, path='', create=False)`

Return a user writable directory associated with an extension.

.. note::

   This allows each extension to have it's own user directory to store files.

   The location of the extension it self is not a suitable place to store files
   because it is cleared each upgrade and the users may not have write permissions
   to the repository (typically "System" repositories).

:arg package: The ``__package__`` of the extension.
:type package: str
:arg path: Optional subdirectory.
:type path: str
:arg create: Treat the path as a directory and create it if its not existing.
:type create: bool
:return: a path.
:rtype: str

### `is_path_builtin(path)`

Returns True if the path is one of the built-in paths used by Blender.

:arg path: Path you want to check if it is in the built-in settings directory
:type path: str
:rtype: bool

### `is_path_extension(path)`

Returns True if the path is from an extensions repository.

:arg path: Path to check if it is within an extension repository.
:type path: str
:rtype: bool

### `keyconfig_init()`

### `keyconfig_set(filepath, *, report=None)`

### `load_scripts(*, reload_scripts=False, refresh_scripts=False, extensions=True)`

Load scripts and run each modules register function.

:arg reload_scripts: Causes all scripts to have their unregister method
   called before loading.
:type reload_scripts: bool
:arg refresh_scripts: only load scripts which are not already loaded
   as modules.
:type refresh_scripts: bool
:arg extensions: Loads additional scripts (add-ons & app-templates).
:type extensions: bool

### `load_scripts_extensions(*, reload_scripts=False)`

Load extensions scripts (add-ons and app-templates)

:arg reload_scripts: Causes all scripts to have their unregister method
   called before loading.
:type reload_scripts: bool

### `make_rna_paths(struct_name, prop_name, enum_name)`

Create RNA "paths" from given names.

:arg struct_name: Name of a RNA struct (like e.g. "Scene").
:type struct_name: str
:arg prop_name: Name of a RNA struct's property.
:type prop_name: str
:arg enum_name: Name of a RNA enum identifier.
:type enum_name: str
:return: A triple of three "RNA paths"
   (most_complete_path, "struct.prop", "struct.prop:'enum'").
   If no enum_name is given, the third element will always be void.
:rtype: tuple[str, str, str]

### `manual_language_code(default='en')`

:return:
   The language code used for user manual URL component based on the current language user-preference,
   falling back to the ``default`` when unavailable.
:rtype: str

### `manual_map()`

### `modules_from_path(path, loaded_modules)`

Load all modules in a path and return them as a list.

:arg path: this path is scanned for scripts and packages.
:type path: str
:arg loaded_modules: already loaded module names, files matching these
   names will be ignored.
:type loaded_modules: set[ModuleType]
:return: all loaded modules.
:rtype: list[ModuleType]

### `preset_find(name, preset_path, *, display_name=False, ext='.py')`

### `preset_paths(subdir)`

Returns a list of paths for a specific preset.

:arg subdir: preset subdirectory (must not be an absolute path).
:type subdir: str
:return: Script paths.
:rtype: list[str]

### `refresh_script_paths()`

Run this after creating new script paths to update sys.path

### `register_classes_factory(classes)`

Utility function to create register and unregister functions
which simply registers and unregisters a sequence of classes.

### `register_manual_map(manual_hook)`

### `register_preset_path(path)`

Register a preset search path.

:arg path: preset directory (must be an absolute path).

   This path must contain a "presets" subdirectory which will typically contain presets for add-ons.

   You may call ``bpy.utils.register_preset_path(os.path.dirname(__file__))`` from an add-ons ``__init__.py`` file.
   When the ``__init__.py`` is in the same location as a ``presets`` directory.
   For example an operators preset would be located under: ``presets/operator/{operator.id}/``
   where ``operator.id`` is the ``bl_idname`` of the operator.
:type path: str
:return: success
:rtype: bool

### `register_submodule_factory(module_name, submodule_names)`

Utility function to create register and unregister functions
which simply load submodules,
calling their register & unregister functions.

.. note::

   Modules are registered in the order given,
   unregistered in reverse order.

:arg module_name: The module name, typically ``__name__``.
:type module_name: str
:arg submodule_names: List of submodule names to load and unload.
:type submodule_names: list[str]
:return: register and unregister functions.
:rtype: tuple[Callable[[], None], Callable[[], None]]

### `register_tool(tool_cls, *, after=None, separator=False, group=False)`

Register a tool in the toolbar.

:arg tool_cls: A tool subclass.
:type tool_cls: type[:class:`bpy.types.WorkSpaceTool`]
:arg after: Optional identifiers this tool will be added after.
:type after: Sequence[str] | set[str] | None
:arg separator: When true, add a separator before this tool.
:type separator: bool
:arg group: When true, add a new nested group of tools.
:type group: bool

### `script_path_user()`

returns the env var and falls back to home dir or None

### `script_paths(*, subdir=None, user_pref=True, check_all=False, use_user=True, use_system_environment=True)`

Returns a list of valid script paths.

:arg subdir: Optional subdir.
:type subdir: str
:arg user_pref: Include the user preference script paths.
:type user_pref: bool
:arg check_all: Include local, user and system paths rather just the paths Blender uses.
:type check_all: bool
:arg use_user: Include user paths
:type use_user: bool
:arg use_system_environment: Include BLENDER_SYSTEM_SCRIPTS variable path
:type use_system_environment: bool
:return: script paths.
:rtype: list[str]

### `script_paths_pref()`

Returns a list of user preference script directories.

### `script_paths_system_environment()`

Returns a list of system script directories from environment variables.

### `smpte_from_frame(frame, *, fps=None, fps_base=None)`

Returns an SMPTE formatted string from the *frame*:
``HH:MM:SS:FF``.

If *fps* and *fps_base* are not given the current scene is used.

:arg frame: frame number.
:type frame: int | float
:return: the frame string.
:rtype: str

### `smpte_from_seconds(time, *, fps=None, fps_base=None)`

Returns an SMPTE formatted string from the *time*:
``HH:MM:SS:FF``.

If *fps* and *fps_base* are not given the current scene is used.

:arg time: time in seconds.
:type time: int | float | datetime.timedelta
:return: the frame string.
:rtype: str

### `time_from_frame(frame, *, fps=None, fps_base=None)`

Returns the time from a frame number .

If *fps* and *fps_base* are not given the current scene is used.

:arg frame: number.
:type frame: int | float
:return: the time in seconds.
:rtype: datetime.timedelta

### `time_to_frame(time, *, fps=None, fps_base=None)`

Returns a float frame number from a time given in seconds or
as a datetime.timedelta object.

If *fps* and *fps_base* are not given the current scene is used.

:arg time: time in seconds.
:type time: float | int | datetime.timedelta
:return: The frame.
:rtype: float | int | datetime.timedelta

### `unregister_manual_map(manual_hook)`

### `unregister_preset_path(path)`

Unregister a preset search path.

:arg path: preset directory (must be an absolute path).

   This must match the registered path exactly.
:type path: str
:return: success
:rtype: bool

### `unregister_tool(tool_cls)`

### `user_resource(resource_type, *, path='', create=False)`

Return a user resource path (normally from the users home directory).

:arg resource_type: Resource type in ['DATAFILES', 'CONFIG', 'SCRIPTS', 'EXTENSIONS'].
:type resource_type: str
:arg path: Optional subdirectory.
:type path: str
:arg create: Treat the path as a directory and create it if its not existing.
:type create: bool
:return: a path.
:rtype: str

## Important Data Types (10)

### `flip_name`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: flip_name(name, strip_digits=False)

Flip a name between left/right sides, useful for 
mirroring bone names.

:arg name: Bone name to flip.
:type name: str
:arg strip_digits: Whether to remove ``.###`` suffix.
:type strip_digits: bool
:return: The flipped name.
:rtype: str

*(callable)*

### `blend_paths`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: blend_paths(absolute=False, packed=False, local=False)

Returns a list of paths to external files referenced by the loaded .blend file.

:arg absolute: When true the paths returned are made absolute.
:type absolute: bool
:arg packed: When true skip file paths for packed data.
:type packed: bool
:arg local: When true skip linked library paths.
:type local: bool
:return: path list.
:rtype: list[str]

*(callable)*

### `resource_path`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: resource_path(type, major=bpy.app.version[0], minor=bpy.app.version[1])

Return the base path for storing system files.

:arg type: string in ['USER', 'LOCAL', 'SYSTEM'].
:type type: str
:arg major: major version, defaults to current.
:type major: int
:arg minor: minor version, defaults to current.
:type minor: str
:return: the resource path (not necessarily existing).
:rtype: str

*(callable)*

### `register_class`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: register_class(cls)

Register a subclass of a Blender type class.

:arg cls: Registerable Blender class type.
:type cls: type[:class:`bpy.types.Panel` | :class:`bpy.types.UIList` | :class:`bpy.types.Menu` | :class:`bpy.types.Header` | :class:`bpy.types.Operator` | :class:`bpy.types.KeyingSetInfo` | :class:`bpy.types.RenderEngine` | :class:`bpy.types.AssetShelf` | :class:`bpy.types.FileHandler` | :class:`bpy.types.PropertyGroup` | :class:`bpy.types.AddonPreferences`]

:raises ValueError:
   if the class is not a subclass of a registerable blender class.

.. note::

   If the class has a *register* class method it will be called
   before registration.

*(callable)*

### `system_resource`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: system_resource(type, path="")

Return a system resource path.

:arg type: string in ['DATAFILES', 'SCRIPTS', 'EXTENSIONS', 'PYTHON'].
:type type: str
:arg path: Optional subdirectory.
:type path: str | bytes

*(callable)*

### `unregister_class`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: unregister_class(cls)

Unload the Python class from blender.

:arg cls: Blender type class, 
   see :mod:`bpy.utils.register_class` for classes which can 
   be registered.
:type cls: type[:class:`bpy.types.Panel` | :class:`bpy.types.UIList` | :class:`bpy.types.Menu` | :class:`bpy.types.Header` | :class:`bpy.types.Operator` | :class:`bpy.types.KeyingSetInfo` | :class:`bpy.types.RenderEngine` | :class:`bpy.types.AssetShelf` | :class:`bpy.types.FileHandler` | :class:`bpy.types.PropertyGroup` | :class:`bpy.types.AddonPreferences`]

.. note::

   If the class has an *unregister* class method it will be called
   before unregistering.

*(callable)*

### `escape_identifier`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: escape_identifier(string)

Simple string escaping function used for animation paths.

:arg string: text
:type string: str
:return: The escaped string.
:rtype: str

*(callable)*

### `unescape_identifier`
**Type**: `<class 'builtin_function_or_method'>`

.. function:: unescape_identifier(string)

Simple string un-escape function used for animation paths.
This performs the reverse of :func:`escape_identifier`.

:arg string: text
:type string: str
:return: The un-escaped string.
:rtype: str

*(callable)*

### `register_cli_command`
**Type**: `<class 'builtin_function_or_method'>`

.. method:: register_cli_command(id, execute)

Register a command, accessible via the (``-c`` / ``--command``) command-line argument.

:arg id: The command identifier (must pass an ``str.isidentifier`` check).

   If the ``id`` is already registered, a warning is printed and the command is inaccessible to prevent accidents invoking the wrong command.
:type id: str
:arg execute: Callback, taking a single list of strings and returns an int.
   The arguments are built from all command-line arguments following the command id.
   The return value should be 0 for success, 1 on failure (specific error codes from the ``os`` module can also be used).
:type execute: callable
:return: The command handle which can be passed to :func:`unregister_cli_command`.
:rtype: capsule

*(callable)*

### `unregister_cli_command`
**Type**: `<class 'builtin_function_or_method'>`

.. method:: unregister_cli_command(handle)

Unregister a CLI command.

:arg handle: The return value of :func:`register_cli_command`.
:type handle: capsule

*(callable)*
