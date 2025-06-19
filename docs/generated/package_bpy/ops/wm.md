# wm

Part of `bpy.ops`
Module: `bpy.ops.wm`

## Operators (115)

### `alembic_export`

bpy.ops.wm.alembic_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=True, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.abc", start=-2147483648, end=-2147483648, xsamples=1, gsamples=1, sh_open=0, sh_close=1, selected=False, visible_objects_only=False, flatten=False, collection="", uvs=True, packuv=True, normals=True, vcolors=False, orcos=True, face_sets=False, subdiv_schema=False, apply_subdiv=False, curves_as_mesh=False, use_instancing=True, global_scale=1, triangulate=False, quad_method='SHORTEST_DIAGONAL', ngon_method='BEAUTY', export_hair=True, export_particles=True, export_custom_properties=True, as_background_job=False, evaluation_mode='RENDER', init_scene_frame_range=True)
Export current scene in an Alembic archive

### `alembic_import`

bpy.ops.wm.alembic_import(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=True, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.abc", scale=1, set_frame_range=True, validate_meshes=False, always_add_cache_reader=False, is_sequence=False, as_background_job=False)
Load an Alembic archive

### `append`

bpy.ops.wm.append(filepath="", directory="", filename="", files=[], check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=True, filemode=1, display_type='DEFAULT', sort_method='DEFAULT', link=False, do_reuse_local_id=False, clear_asset_data=False, autoselect=True, active_collection=True, instance_collections=False, instance_object_data=True, set_fake=False, use_recursive=True)
Append from a Library .blend file

### `batch_rename`

bpy.ops.wm.batch_rename(data_type='OBJECT', data_source='SELECT', actions=[])
Rename multiple items at once

### `blend_strings_utf8_validate`

bpy.ops.wm.blend_strings_utf8_validate()
Check and fix all strings in current .blend file to be valid UTF-8 Unicode (needed for some old, 2.4x area files)

### `call_asset_shelf_popover`

bpy.ops.wm.call_asset_shelf_popover(name="")
Open a predefined asset shelf in a popup

### `call_menu`

bpy.ops.wm.call_menu(name="")
Open a predefined menu

### `call_menu_pie`

bpy.ops.wm.call_menu_pie(name="")
Open a predefined pie menu

### `call_panel`

bpy.ops.wm.call_panel(name="", keep_open=True)
Open a predefined panel

### `clear_recent_files`

bpy.ops.wm.clear_recent_files(remove='ALL')
Clear the recent files list

### `collada_export`

bpy.ops.wm.collada_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=True, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.dae", prop_bc_export_ui_section='main', apply_modifiers=False, export_mesh_type=0, export_mesh_type_selection='view', export_global_forward_selection='Y', export_global_up_selection='Z', apply_global_orientation=False, selected=False, include_children=False, include_armatures=False, include_shapekeys=False, deform_bones_only=False, include_animations=True, include_all_actions=True, export_animation_type_selection='sample', sampling_rate=1, keep_smooth_curves=False, keep_keyframes=False, keep_flat_curves=False, active_uv_only=False, use_texture_copies=True, triangulate=True, use_object_instantiation=True, use_blender_profile=True, sort_by_name=False, export_object_transformation_type=0, export_object_transformation_type_selection='matrix', export_animation_transformation_type=0, export_animation_transformation_type_selection='matrix', open_sim=False, limit_precision=False, keep_bind_info=False)
Save a Collada file

### `collada_import`

bpy.ops.wm.collada_import(filepath="", check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=True, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.dae", import_units=False, custom_normals=True, fix_orientation=False, find_chains=False, auto_connect=False, min_chain_length=0, keep_bind_info=False)
Load a Collada file

### `collection_export_all`

bpy.ops.wm.collection_export_all()
Invoke all configured exporters for all collections

### `console_toggle`

bpy.ops.wm.console_toggle()
Toggle System Console

### `context_collection_boolean_set`

bpy.ops.wm.context_collection_boolean_set(data_path_iter="", data_path_item="", type='TOGGLE')
Set boolean values for a collection of items

### `context_cycle_array`

bpy.ops.wm.context_cycle_array(data_path="", reverse=False)
Set a context array value (useful for cycling the active mesh edit mode)

### `context_cycle_enum`

bpy.ops.wm.context_cycle_enum(data_path="", reverse=False, wrap=False)
Toggle a context value

### `context_cycle_int`

bpy.ops.wm.context_cycle_int(data_path="", reverse=False, wrap=False)
Set a context value (useful for cycling active material, shape keys, groups, etc.)

### `context_menu_enum`

bpy.ops.wm.context_menu_enum(data_path="")
(undocumented operator)

### `context_modal_mouse`

bpy.ops.wm.context_modal_mouse(data_path_iter="", data_path_item="", header_text="", input_scale=0.01, invert=False, initial_x=0)
Adjust arbitrary values with mouse input

### `context_pie_enum`

bpy.ops.wm.context_pie_enum(data_path="")
(undocumented operator)

### `context_scale_float`

bpy.ops.wm.context_scale_float(data_path="", value=1)
Scale a float context value

### `context_scale_int`

bpy.ops.wm.context_scale_int(data_path="", value=1, always_step=True)
Scale an int context value

### `context_set_boolean`

bpy.ops.wm.context_set_boolean(data_path="", value=True)
Set a context value

### `context_set_enum`

bpy.ops.wm.context_set_enum(data_path="", value="")
Set a context value

### `context_set_float`

bpy.ops.wm.context_set_float(data_path="", value=0, relative=False)
Set a context value

### `context_set_id`

bpy.ops.wm.context_set_id(data_path="", value="")
Set a context value to an ID data-block

### `context_set_int`

bpy.ops.wm.context_set_int(data_path="", value=0, relative=False)
Set a context value

### `context_set_string`

bpy.ops.wm.context_set_string(data_path="", value="")
Set a context value

### `context_set_value`

bpy.ops.wm.context_set_value(data_path="", value="")
Set a context value

### `context_toggle`

bpy.ops.wm.context_toggle(data_path="", module="")
Toggle a context value

### `context_toggle_enum`

bpy.ops.wm.context_toggle_enum(data_path="", value_1="", value_2="")
Toggle a context value

### `debug_menu`

bpy.ops.wm.debug_menu(debug_value=0)
Open a popup to set the debug level

### `doc_view`

bpy.ops.wm.doc_view(doc_id="")
Open online reference docs in a web browser

### `doc_view_manual`

bpy.ops.wm.doc_view_manual(doc_id="")
Load online manual

### `doc_view_manual_ui_context`

bpy.ops.wm.doc_view_manual_ui_context()
View a context based online manual in a web browser

### `drop_blend_file`

bpy.ops.wm.drop_blend_file(filepath="")
(undocumented operator)

### `drop_import_file`

bpy.ops.wm.drop_import_file(directory="", files=[])
Operator that allows file handlers to receive file drops

### `grease_pencil_export_pdf`

bpy.ops.wm.grease_pencil_export_pdf(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=True, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', use_fill=True, selected_object_type='ACTIVE', stroke_sample=0, use_uniform_width=False, frame_mode='ACTIVE')
Export Grease Pencil to PDF

### `grease_pencil_export_svg`

bpy.ops.wm.grease_pencil_export_svg(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=True, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', use_fill=True, selected_object_type='ACTIVE', stroke_sample=0, use_uniform_width=False, use_clip_camera=False)
Export Grease Pencil to SVG

### `grease_pencil_import_svg`

bpy.ops.wm.grease_pencil_import_svg(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=True, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', resolution=10, scale=10, use_scene_unit=False)
Import SVG into Grease Pencil

### `interface_theme_preset_add`

bpy.ops.wm.interface_theme_preset_add(name="", remove_name=False, remove_active=False)
Add a custom theme to the preset list

### `interface_theme_preset_remove`

bpy.ops.wm.interface_theme_preset_remove(name="", remove_name=False, remove_active=True)
Remove a custom theme from the preset list

### `interface_theme_preset_save`

bpy.ops.wm.interface_theme_preset_save(name="", remove_name=False, remove_active=True)
Save a custom theme in the preset list

### `keyconfig_preset_add`

bpy.ops.wm.keyconfig_preset_add(name="", remove_name=False, remove_active=False)
Add a custom keymap configuration to the preset list

### `keyconfig_preset_remove`

bpy.ops.wm.keyconfig_preset_remove(name="", remove_name=False, remove_active=True)
Remove a custom keymap configuration from the preset list

### `lib_reload`

bpy.ops.wm.lib_reload(library="", filepath="", directory="", filename="", hide_props_region=True, check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT')
Reload the given library

### `lib_relocate`

bpy.ops.wm.lib_relocate(library="", filepath="", directory="", filename="", files=[], hide_props_region=True, check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT')
Relocate the given library to one or several others

### `link`

bpy.ops.wm.link(filepath="", directory="", filename="", files=[], check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=True, filemode=1, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', link=True, do_reuse_local_id=False, clear_asset_data=False, autoselect=True, active_collection=True, instance_collections=True, instance_object_data=True)
Link from a Library .blend file

### `memory_statistics`

bpy.ops.wm.memory_statistics()
Print memory statistics to the console

### `obj_export`

bpy.ops.wm.obj_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', export_animation=False, start_frame=-2147483648, end_frame=2147483647, forward_axis='NEGATIVE_Z', up_axis='Y', global_scale=1, apply_modifiers=True, export_eval_mode='DAG_EVAL_VIEWPORT', export_selected_objects=False, export_uv=True, export_normals=True, export_colors=False, export_materials=True, export_pbr_extensions=False, path_mode='AUTO', export_triangulated_mesh=False, export_curves_as_nurbs=False, export_object_groups=False, export_material_groups=False, export_vertex_groups=False, export_smooth_groups=False, smooth_group_bitflags=False, filter_glob="*.obj;*.mtl", collection="")
Save the scene to a Wavefront OBJ file

### `obj_import`

bpy.ops.wm.obj_import(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', global_scale=1, clamp_size=0, forward_axis='NEGATIVE_Z', up_axis='Y', use_split_objects=True, use_split_groups=False, import_vertex_groups=False, validate_meshes=True, close_spline_loops=True, collection_separator="", filter_glob="*.obj;*.mtl")
Load a Wavefront OBJ scene

### `open_mainfile`

bpy.ops.wm.open_mainfile(filepath="", hide_props_region=True, check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', load_ui=True, use_scripts=True, display_file_selector=True, state=0)
Open a Blender file

### `operator_cheat_sheet`

bpy.ops.wm.operator_cheat_sheet()
List all the operators in a text-block, useful for scripting

### `operator_defaults`

bpy.ops.wm.operator_defaults()
Set the active operator to its default values

### `operator_pie_enum`

bpy.ops.wm.operator_pie_enum(data_path="", prop_string="")
(undocumented operator)

### `operator_preset_add`

bpy.ops.wm.operator_preset_add(name="", remove_name=False, remove_active=False, operator="")
Add or remove an Operator Preset

### `operator_presets_cleanup`

bpy.ops.wm.operator_presets_cleanup(operator="", properties=[])
Remove outdated operator properties from presets that may cause problems

### `owner_disable`

bpy.ops.wm.owner_disable(owner_id="")
Disable add-on for workspace

### `owner_enable`

bpy.ops.wm.owner_enable(owner_id="")
Enable add-on for workspace

### `path_open`

bpy.ops.wm.path_open(filepath="")
Open a path in a file browser

### `ply_export`

bpy.ops.wm.ply_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', forward_axis='Y', up_axis='Z', global_scale=1, apply_modifiers=True, export_selected_objects=False, collection="", export_uv=True, export_normals=False, export_colors='SRGB', export_attributes=True, export_triangulated_mesh=False, ascii_format=False, filter_glob="*.ply")
Save the scene to a PLY file

### `ply_import`

bpy.ops.wm.ply_import(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', global_scale=1, use_scene_unit=False, forward_axis='Y', up_axis='Z', merge_verts=False, import_colors='SRGB', import_attributes=True, filter_glob="*.ply")
Import an PLY file as an object

### `previews_batch_clear`

bpy.ops.wm.previews_batch_clear(files=[], directory="", filter_blender=True, filter_folder=True, use_scenes=True, use_collections=True, use_objects=True, use_intern_data=True, use_trusted=False, use_backups=True)
Clear selected .blend file's previews

### `previews_batch_generate`

bpy.ops.wm.previews_batch_generate(files=[], directory="", filter_blender=True, filter_folder=True, use_scenes=True, use_collections=True, use_objects=True, use_intern_data=True, use_trusted=False, use_backups=True)
Generate selected .blend file's previews

### `previews_clear`

bpy.ops.wm.previews_clear(id_type=set())
Clear data-block previews (only for some types like objects, materials, textures, etc.)

### `previews_ensure`

bpy.ops.wm.previews_ensure()
Ensure data-block previews are available and up-to-date (to be saved in .blend file, only for some types like materials, textures, etc.)

### `properties_add`

bpy.ops.wm.properties_add(data_path="")
Add your own property to the data-block

### `properties_context_change`

bpy.ops.wm.properties_context_change(context="")
Jump to a different tab inside the properties editor

### `properties_edit`

bpy.ops.wm.properties_edit(data_path="", property_name="", property_type='FLOAT', is_overridable_library=False, description="", use_soft_limits=False, array_length=3, default_int=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), min_int=-10000, max_int=10000, soft_min_int=-10000, soft_max_int=10000, step_int=1, default_bool=(False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False), default_float=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), min_float=-10000, max_float=-10000, soft_min_float=-10000, soft_max_float=-10000, precision=3, step_float=0.1, subtype='NONE', default_string="", id_type='OBJECT', eval_string="")
Change a custom property's type, or adjust how it is displayed in the interface

### `properties_edit_value`

bpy.ops.wm.properties_edit_value(data_path="", property_name="", eval_string="")
Edit the value of a custom property

### `properties_remove`

bpy.ops.wm.properties_remove(data_path="", property_name="")
Internal use (edit a property data_path)

### `quit_blender`

bpy.ops.wm.quit_blender()
Quit Blender

### `radial_control`

bpy.ops.wm.radial_control(data_path_primary="", data_path_secondary="", use_secondary="", rotation_path="", color_path="", fill_color_path="", fill_color_override_path="", fill_color_override_test_path="", zoom_path="", image_id="", secondary_tex=False, release_confirm=False)
Set some size property (e.g. brush size) with mouse wheel

### `read_factory_settings`

bpy.ops.wm.read_factory_settings(use_factory_startup_app_template_only=False, app_template="Template", use_empty=False)
Load factory default startup file and preferences. To make changes permanent, use "Save Startup File" and "Save Preferences"

### `read_factory_userpref`

bpy.ops.wm.read_factory_userpref(use_factory_startup_app_template_only=False)
Load factory default preferences. To make changes to preferences permanent, use "Save Preferences"

### `read_history`

bpy.ops.wm.read_history()
Reloads history and bookmarks

### `read_homefile`

bpy.ops.wm.read_homefile(filepath="", load_ui=True, use_splash=False, use_factory_startup=False, use_factory_startup_app_template_only=False, app_template="Template", use_empty=False)
Open the default file

### `read_userpref`

bpy.ops.wm.read_userpref()
Load last saved preferences

### `recover_auto_save`

bpy.ops.wm.recover_auto_save(filepath="", hide_props_region=True, check_existing=False, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=False, filter_blenlib=False, filemode=8, display_type='LIST_VERTICAL', sort_method='FILE_SORT_TIME', use_scripts=True)
Open an automatically saved file to recover it

### `recover_last_session`

bpy.ops.wm.recover_last_session(use_scripts=True)
Open the last closed file ("quit.blend")

### `redraw_timer`

bpy.ops.wm.redraw_timer(type='DRAW', iterations=10, time_limit=0)
Simple redraw timer to test the speed of updating the interface

### `revert_mainfile`

bpy.ops.wm.revert_mainfile(use_scripts=True)
Reload the saved file

### `save_as_mainfile`

bpy.ops.wm.save_as_mainfile(filepath="", hide_props_region=True, check_existing=True, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', compress=False, relative_remap=True, copy=False)
Save the current file in the desired location

### `save_homefile`

bpy.ops.wm.save_homefile()
Make the current file the default startup file

### `save_mainfile`

bpy.ops.wm.save_mainfile(filepath="", hide_props_region=True, check_existing=True, filter_blender=True, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', compress=False, relative_remap=False, exit=False, incremental=False)
Save the current Blender file

### `save_userpref`

bpy.ops.wm.save_userpref()
Make the current preferences default

### `search_menu`

bpy.ops.wm.search_menu()
Pop-up a search over all menus in the current context

### `search_operator`

bpy.ops.wm.search_operator()
Pop-up a search over all available operators in current context

### `search_single_menu`

bpy.ops.wm.search_single_menu(menu_idname="", initial_query="")
Pop-up a search for a menu in current context

### `set_stereo_3d`

bpy.ops.wm.set_stereo_3d(display_mode='ANAGLYPH', anaglyph_type='RED_CYAN', interlace_type='ROW_INTERLEAVED', use_interlace_swap=False, use_sidebyside_crosseyed=False)
Toggle 3D stereo support for current window (or change the display mode)

### `splash`

bpy.ops.wm.splash()
Open the splash screen with release info

### `splash_about`

bpy.ops.wm.splash_about()
Open a window with information about Blender

### `stl_export`

bpy.ops.wm.stl_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', ascii_format=False, use_batch=False, export_selected_objects=False, collection="", global_scale=1, use_scene_unit=False, forward_axis='Y', up_axis='Z', apply_modifiers=True, filter_glob="*.stl")
Save the scene to an STL file

### `stl_import`

bpy.ops.wm.stl_import(filepath="", directory="", files=[], check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', global_scale=1, use_scene_unit=False, use_facet_normal=False, forward_axis='Y', up_axis='Z', use_mesh_validate=True, filter_glob="*.stl")
Import an STL file as an object

### `sysinfo`

bpy.ops.wm.sysinfo(filepath="")
Generate system information, saved into a text file

### `tool_set_by_brush_type`

bpy.ops.wm.tool_set_by_brush_type(brush_type="", space_type='EMPTY')
Look up the most appropriate tool for the given brush type and activate that

### `tool_set_by_id`

bpy.ops.wm.tool_set_by_id(name="", cycle=False, as_fallback=False, space_type='EMPTY')
Set the tool by name (for key-maps)

### `tool_set_by_index`

bpy.ops.wm.tool_set_by_index(index=0, cycle=False, expand=True, as_fallback=False, space_type='EMPTY')
Set the tool by index (for key-maps)

### `toolbar`

bpy.ops.wm.toolbar()
(undocumented operator)

### `toolbar_fallback_pie`

bpy.ops.wm.toolbar_fallback_pie()
(undocumented operator)

### `toolbar_prompt`

bpy.ops.wm.toolbar_prompt()
Leader key like functionality for accessing tools

### `url_open`

bpy.ops.wm.url_open(url="")
Open a website in the web browser

### `url_open_preset`

bpy.ops.wm.url_open_preset(type='BUG')
Open a preset website in the web browser

### `usd_export`

bpy.ops.wm.usd_export(filepath="", check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=True, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.usd", selected_objects_only=False, visible_objects_only=True, collection="", export_animation=False, export_hair=False, export_uvmaps=True, rename_uvmaps=True, export_mesh_colors=True, export_normals=True, export_materials=True, export_subdivision='BEST_MATCH', export_armatures=True, only_deform_bones=False, export_shapekeys=True, use_instancing=False, evaluation_mode='RENDER', generate_preview_surface=True, generate_materialx_network=False, convert_orientation=False, export_global_forward_selection='NEGATIVE_Z', export_global_up_selection='Y', export_textures=False, export_textures_mode='NEW', overwrite_textures=False, relative_paths=True, xform_op_mode='TRS', root_prim_path="/root", export_custom_properties=True, custom_properties_namespace="userProperties", author_blender_name=True, convert_world_material=True, allow_unicode=False, export_meshes=True, export_lights=True, export_cameras=True, export_curves=True, export_points=True, export_volumes=True, triangulate_meshes=False, quad_method='SHORTEST_DIAGONAL', ngon_method='BEAUTY', usdz_downscale_size='KEEP', usdz_downscale_custom_size=128, merge_parent_xform=False, convert_scene_units='METERS', meters_per_unit=1)
Export current scene in a USD archive

### `usd_import`

bpy.ops.wm.usd_import(filepath="", check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=True, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=8, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', filter_glob="*.usd", scale=1, set_frame_range=True, import_cameras=True, import_curves=True, import_lights=True, import_materials=True, import_meshes=True, import_volumes=True, import_shapes=True, import_skeletons=True, import_blendshapes=True, import_points=True, import_subdiv=False, support_scene_instancing=True, import_visible_only=True, create_collection=False, read_mesh_uvs=True, read_mesh_colors=True, read_mesh_attributes=True, prim_path_mask="", import_guide=False, import_proxy=False, import_render=True, import_all_materials=False, import_usd_preview=True, set_material_blend=True, light_intensity_scale=1, mtl_purpose='MTL_FULL', mtl_name_collision_mode='MAKE_UNIQUE', import_textures_mode='IMPORT_PACK', import_textures_dir="//textures/", tex_name_collision_mode='USE_EXISTING', attr_import_mode='ALL', validate_meshes=False, create_world_material=True, import_defined_only=True, merge_parent_xform=True, apply_unit_conversion_scale=True)
Import USD stage into current scene

### `window_close`

bpy.ops.wm.window_close()
Close the current window

### `window_fullscreen_toggle`

bpy.ops.wm.window_fullscreen_toggle()
Toggle the current window full-screen

### `window_new`

bpy.ops.wm.window_new()
Create a new window

### `window_new_main`

bpy.ops.wm.window_new_main()
Create a new main window with its own workspace and scene selection

### `xr_navigation_fly`

bpy.ops.wm.xr_navigation_fly(mode='VIEWER_FORWARD', lock_location_z=False, lock_direction=False, speed_frame_based=True, speed_min=0.018, speed_max=0.054, speed_interpolation0=(0, 0), speed_interpolation1=(1, 1))
Move/turn relative to the VR viewer or controller

### `xr_navigation_grab`

bpy.ops.wm.xr_navigation_grab(lock_location=False, lock_location_z=False, lock_rotation=False, lock_rotation_z=False, lock_scale=False)
Navigate the VR scene by grabbing with controllers

### `xr_navigation_reset`

bpy.ops.wm.xr_navigation_reset(location=True, rotation=True, scale=True)
Reset VR navigation deltas relative to session base pose

### `xr_navigation_teleport`

bpy.ops.wm.xr_navigation_teleport(teleport_axes=(True, True, True), interpolation=1, offset=0, selectable_only=True, distance=1.70141e+38, from_viewer=False, axis=(0, 0, -1), color=(0.35, 0.35, 1, 1))
Set VR viewer location to controller raycast hit location

### `xr_session_toggle`

bpy.ops.wm.xr_session_toggle()
Open a view for use with virtual reality headsets, or close it if already opened
