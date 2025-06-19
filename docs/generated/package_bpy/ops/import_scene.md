# import_scene

Part of `bpy.ops`
Module: `bpy.ops.import_scene`

## Operators (2)

### `fbx`

bpy.ops.import_scene.fbx(filepath="", directory="", filter_glob="*.fbx", files=[], ui_tab='MAIN', use_manual_orientation=False, global_scale=1, bake_space_transform=False, use_custom_normals=True, colors_type='SRGB', use_image_search=True, use_alpha_decals=False, decal_offset=0, use_anim=True, anim_offset=1, use_subsurf=False, use_custom_props=True, use_custom_props_enum_as_string=True, ignore_leaf_bones=False, force_connect_children=False, automatic_bone_orientation=False, primary_bone_axis='Y', secondary_bone_axis='X', use_prepost_rot=True, axis_forward='-Z', axis_up='Y')
Load a FBX file

### `gltf`

bpy.ops.import_scene.gltf(filepath="", export_import_convert_lighting_mode='SPEC', filter_glob="*.glb;*.gltf", files=[], loglevel=0, import_pack_images=True, merge_vertices=False, import_shading='NORMALS', bone_heuristic='BLENDER', disable_bone_shape=False, bone_shape_scale_factor=1, guess_original_bind_pose=True, import_webp_texture=False, import_select_created_objects=True, import_scene_extras=True)
Load a glTF 2.0 file
