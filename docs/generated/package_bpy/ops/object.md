# object

Part of `bpy.ops`
Module: `bpy.ops.object`

## Operators (237)

### `add`

bpy.ops.object.add(radius=1, type='EMPTY', enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an object to the scene

### `add_modifier_menu`

bpy.ops.object.add_modifier_menu()
(undocumented operator)

### `add_named`

bpy.ops.object.add_named(linked=False, name="", session_uid=0, matrix=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), drop_x=0, drop_y=0)
Add named object

### `align`

bpy.ops.object.align(bb_quality=True, align_mode='OPT_2', relative_to='OPT_4', align_axis=set())
Align objects

### `anim_transforms_to_deltas`

bpy.ops.object.anim_transforms_to_deltas()
Convert object animation for normal transforms to delta transforms

### `armature_add`

bpy.ops.object.armature_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an armature object to the scene

### `assign_property_defaults`

bpy.ops.object.assign_property_defaults(process_data=True, process_bones=True)
Assign the current values of custom properties as their defaults, for use as part of the rest pose state in NLA track mixing

### `bake`

bpy.ops.object.bake(type='COMBINED', pass_filter=set(), filepath="", width=512, height=512, margin=16, margin_type='EXTEND', use_selected_to_active=False, max_ray_distance=0, cage_extrusion=0, cage_object="", normal_space='TANGENT', normal_r='POS_X', normal_g='POS_Y', normal_b='POS_Z', target='IMAGE_TEXTURES', save_mode='INTERNAL', use_clear=False, use_cage=False, use_split_materials=False, use_automatic_name=False, uv_layer="")
Bake image textures of selected objects

### `bake_image`

bpy.ops.object.bake_image()
Bake image textures of selected objects

### `camera_add`

bpy.ops.object.camera_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a camera object to the scene

### `clear_override_library`

bpy.ops.object.clear_override_library()
Delete the selected local overrides and relink their usages to the linked data-blocks if possible, else reset them and mark them as non editable

### `collection_add`

bpy.ops.object.collection_add()
Add an object to a new collection

### `collection_external_asset_drop`

bpy.ops.object.collection_external_asset_drop(session_uid=0, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0), use_instance=True, drop_x=0, drop_y=0, collection='<UNKNOWN ENUM>')
Add the dragged collection to the scene

### `collection_instance_add`

bpy.ops.object.collection_instance_add(name="Collection", collection='<UNKNOWN ENUM>', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0), session_uid=0, drop_x=0, drop_y=0)
Add a collection instance

### `collection_link`

bpy.ops.object.collection_link(collection='<UNKNOWN ENUM>')
Add an object to an existing collection

### `collection_objects_select`

bpy.ops.object.collection_objects_select()
Select all objects in collection

### `collection_remove`

bpy.ops.object.collection_remove()
Remove the active object from this collection

### `collection_unlink`

bpy.ops.object.collection_unlink()
Unlink the collection from all objects

### `constraint_add`

bpy.ops.object.constraint_add(type='<UNKNOWN ENUM>')
Add a constraint to the active object

### `constraint_add_with_targets`

bpy.ops.object.constraint_add_with_targets(type='<UNKNOWN ENUM>')
Add a constraint to the active object, with target (where applicable) set to the selected objects/bones

### `constraints_clear`

bpy.ops.object.constraints_clear()
Clear all constraints from the selected objects

### `constraints_copy`

bpy.ops.object.constraints_copy()
Copy constraints to other selected objects

### `convert`

bpy.ops.object.convert(target='MESH', keep_original=False, merge_customdata=True, thickness=5, faces=True, offset=0.01)
Convert selected objects to another type

### `correctivesmooth_bind`

bpy.ops.object.correctivesmooth_bind(modifier="")
Bind base pose in Corrective Smooth modifier

### `curves_empty_hair_add`

bpy.ops.object.curves_empty_hair_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an empty curve object to the scene with the selected mesh as surface

### `curves_random_add`

bpy.ops.object.curves_random_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a curves object with random curves to the scene

### `data_instance_add`

bpy.ops.object.data_instance_add(name="", session_uid=0, type='ACTION', align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0), drop_x=0, drop_y=0)
Add an object data instance

### `data_transfer`

bpy.ops.object.data_transfer(use_reverse_transfer=False, use_freeze=False, data_type='<UNKNOWN ENUM>', use_create=True, vert_mapping='NEAREST', edge_mapping='NEAREST', loop_mapping='NEAREST_POLYNOR', poly_mapping='NEAREST', use_auto_transform=False, use_object_transform=True, use_max_distance=False, max_distance=1, ray_radius=0, islands_precision=0.1, layers_select_src='ACTIVE', layers_select_dst='ACTIVE', mix_mode='REPLACE', mix_factor=1)
Transfer data layer(s) (weights, edge sharp, etc.) from active to selected meshes

### `datalayout_transfer`

bpy.ops.object.datalayout_transfer(modifier="", data_type='<UNKNOWN ENUM>', use_delete=False, layers_select_src='ACTIVE', layers_select_dst='ACTIVE')
Transfer layout of data layer(s) from active to selected meshes

### `delete`

bpy.ops.object.delete(use_global=False, confirm=True)
Delete selected objects

### `drop_geometry_nodes`

bpy.ops.object.drop_geometry_nodes(session_uid=0, show_datablock_in_modifier=True)
(undocumented operator)

### `drop_named_material`

bpy.ops.object.drop_named_material(name="", session_uid=0)
(undocumented operator)

### `duplicate`

bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
Duplicate selected objects

### `duplicate_move`

bpy.ops.object.duplicate_move(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate the selected objects and move them

### `duplicate_move_linked`

bpy.ops.object.duplicate_move_linked(OBJECT_OT_duplicate={"linked":False, "mode":'TRANSLATION'}, TRANSFORM_OT_translate={"value":(0, 0, 0), "orient_type":'GLOBAL', "orient_matrix":((0, 0, 0), (0, 0, 0), (0, 0, 0)), "orient_matrix_type":'GLOBAL', "constraint_axis":(False, False, False), "mirror":False, "use_proportional_edit":False, "proportional_edit_falloff":'SMOOTH', "proportional_size":1, "use_proportional_connected":False, "use_proportional_projected":False, "snap":False, "snap_elements":{'INCREMENT'}, "use_snap_project":False, "snap_target":'CLOSEST', "use_snap_self":True, "use_snap_edit":True, "use_snap_nonedit":True, "use_snap_selectable":False, "snap_point":(0, 0, 0), "snap_align":False, "snap_normal":(0, 0, 0), "gpencil_strokes":False, "cursor_transform":False, "texture_space":False, "remove_on_cancel":False, "use_duplicated_keyframes":False, "view2d_edge_pan":False, "release_confirm":False, "use_accurate":False, "use_automerge_and_split":False})
Duplicate the selected objects, but not their object data, and move them

### `duplicates_make_real`

bpy.ops.object.duplicates_make_real(use_base_parent=False, use_hierarchy=False)
Make instanced objects attached to this object real

### `editmode_toggle`

bpy.ops.object.editmode_toggle()
Toggle object's edit mode

### `effector_add`

bpy.ops.object.effector_add(type='FORCE', radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an empty object with a physics effector to the scene

### `empty_add`

bpy.ops.object.empty_add(type='PLAIN_AXES', radius=1, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an empty object to the scene

### `empty_image_add`

bpy.ops.object.empty_image_add(filepath="", hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=True, filter_movie=True, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, show_multiview=False, use_multiview=False, display_type='DEFAULT', sort_method='DEFAULT', name="", session_uid=0, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0), background=False)
Add an empty image type to scene with data

### `explode_refresh`

bpy.ops.object.explode_refresh(modifier="")
Refresh data in the Explode modifier

### `forcefield_toggle`

bpy.ops.object.forcefield_toggle()
Toggle object's force field

### `geometry_node_bake_delete_single`

bpy.ops.object.geometry_node_bake_delete_single(session_uid=0, modifier_name="", bake_id=0)
Delete baked data of a single bake node or simulation

### `geometry_node_bake_pack_single`

bpy.ops.object.geometry_node_bake_pack_single(session_uid=0, modifier_name="", bake_id=0)
Pack baked data from disk into the .blend file

### `geometry_node_bake_single`

bpy.ops.object.geometry_node_bake_single(session_uid=0, modifier_name="", bake_id=0)
Bake a single bake node or simulation

### `geometry_node_bake_unpack_single`

bpy.ops.object.geometry_node_bake_unpack_single(session_uid=0, modifier_name="", bake_id=0, method='USE_LOCAL')
Unpack baked data from the .blend file to disk

### `geometry_node_tree_copy_assign`

bpy.ops.object.geometry_node_tree_copy_assign()
Copy the active geometry node group and assign it to the active modifier

### `geometry_nodes_input_attribute_toggle`

bpy.ops.object.geometry_nodes_input_attribute_toggle(input_name="", modifier_name="")
Switch between an attribute and a single value to define the data for every element

### `geometry_nodes_move_to_nodes`

bpy.ops.object.geometry_nodes_move_to_nodes(use_selected_objects=False)
Move inputs and outputs from in the modifier to a new node group

### `grease_pencil_add`

bpy.ops.object.grease_pencil_add(type='EMPTY', use_in_front=True, stroke_depth_offset=0.05, use_lights=False, stroke_depth_order='3D', radius=1, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a Grease Pencil object to the scene

### `grease_pencil_dash_modifier_segment_add`

bpy.ops.object.grease_pencil_dash_modifier_segment_add(modifier="")
Add a segment to the dash modifier

### `grease_pencil_dash_modifier_segment_move`

bpy.ops.object.grease_pencil_dash_modifier_segment_move(modifier="", type='UP')
Move the active dash segment up or down

### `grease_pencil_dash_modifier_segment_remove`

bpy.ops.object.grease_pencil_dash_modifier_segment_remove(modifier="", index=0)
Remove the active segment from the dash modifier

### `grease_pencil_time_modifier_segment_add`

bpy.ops.object.grease_pencil_time_modifier_segment_add(modifier="")
Add a segment to the time modifier

### `grease_pencil_time_modifier_segment_move`

bpy.ops.object.grease_pencil_time_modifier_segment_move(modifier="", type='UP')
Move the active time segment up or down

### `grease_pencil_time_modifier_segment_remove`

bpy.ops.object.grease_pencil_time_modifier_segment_remove(modifier="", index=0)
Remove the active segment from the time modifier

### `hide_collection`

bpy.ops.object.hide_collection(collection_index=-1, toggle=False, extend=False)
Show only objects in collection (Shift to extend)

### `hide_render_clear_all`

bpy.ops.object.hide_render_clear_all()
Reveal all render objects by setting the hide render flag

### `hide_view_clear`

bpy.ops.object.hide_view_clear(select=True)
Reveal temporarily hidden objects

### `hide_view_set`

bpy.ops.object.hide_view_set(unselected=False)
Temporarily hide objects from the viewport

### `hook_add_newob`

bpy.ops.object.hook_add_newob()
Hook selected vertices to a newly created object

### `hook_add_selob`

bpy.ops.object.hook_add_selob(use_bone=False)
Hook selected vertices to the first selected object

### `hook_assign`

bpy.ops.object.hook_assign(modifier='<UNKNOWN ENUM>')
Assign the selected vertices to a hook

### `hook_recenter`

bpy.ops.object.hook_recenter(modifier='<UNKNOWN ENUM>')
Set hook center to cursor position

### `hook_remove`

bpy.ops.object.hook_remove(modifier='<UNKNOWN ENUM>')
Remove a hook from the active object

### `hook_reset`

bpy.ops.object.hook_reset(modifier='<UNKNOWN ENUM>')
Recalculate and clear offset transformation

### `hook_select`

bpy.ops.object.hook_select(modifier='<UNKNOWN ENUM>')
Select affected vertices on mesh

### `instance_offset_from_cursor`

bpy.ops.object.instance_offset_from_cursor()
Set offset used for collection instances based on cursor position

### `instance_offset_from_object`

bpy.ops.object.instance_offset_from_object()
Set offset used for collection instances based on the active object position

### `instance_offset_to_cursor`

bpy.ops.object.instance_offset_to_cursor()
Set cursor position to the offset used for collection instances

### `isolate_type_render`

bpy.ops.object.isolate_type_render()
Hide unselected render objects of same type as active by setting the hide render flag

### `join`

bpy.ops.object.join()
Join selected objects into active object

### `join_shapes`

bpy.ops.object.join_shapes()
Copy the current resulting shape of another selected object to this one

### `join_uvs`

bpy.ops.object.join_uvs()
Transfer UV Maps from active to selected objects (needs matching geometry)

### `laplaciandeform_bind`

bpy.ops.object.laplaciandeform_bind(modifier="")
Bind mesh to system in laplacian deform modifier

### `light_add`

bpy.ops.object.light_add(type='POINT', radius=1, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a light object to the scene

### `light_linking_blocker_collection_new`

bpy.ops.object.light_linking_blocker_collection_new()
Create new light linking collection used by the active emitter

### `light_linking_blockers_link`

bpy.ops.object.light_linking_blockers_link(link_state='INCLUDE')
Light link selected blockers to the active emitter object

### `light_linking_blockers_select`

bpy.ops.object.light_linking_blockers_select()
Select all objects which block light from this emitter

### `light_linking_receiver_collection_new`

bpy.ops.object.light_linking_receiver_collection_new()
Create new light linking collection used by the active emitter

### `light_linking_receivers_link`

bpy.ops.object.light_linking_receivers_link(link_state='INCLUDE')
Light link selected receivers to the active emitter object

### `light_linking_receivers_select`

bpy.ops.object.light_linking_receivers_select()
Select all objects which receive light from this emitter

### `light_linking_unlink_from_collection`

bpy.ops.object.light_linking_unlink_from_collection()
Remove this object or collection from the light linking collection

### `lightprobe_add`

bpy.ops.object.lightprobe_add(type='SPHERE', radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a light probe object

### `lightprobe_cache_bake`

bpy.ops.object.lightprobe_cache_bake(subset='ALL')
Bake irradiance volume light cache

### `lightprobe_cache_free`

bpy.ops.object.lightprobe_cache_free(subset='SELECTED')
Delete cached indirect lighting

### `lineart_bake_strokes`

bpy.ops.object.lineart_bake_strokes(bake_all=False)
Bake Line Art for current Grease Pencil object

### `lineart_clear`

bpy.ops.object.lineart_clear(clear_all=False)
Clear all strokes in current Grease Pencil object

### `link_to_collection`

bpy.ops.object.link_to_collection(collection_index=-1, is_new=False, new_collection_name="")
Link objects to a collection

### `location_clear`

bpy.ops.object.location_clear(clear_delta=False)
Clear the object's location

### `make_dupli_face`

bpy.ops.object.make_dupli_face()
Convert objects into instanced faces

### `make_links_data`

bpy.ops.object.make_links_data(type='OBDATA')
Transfer data from active object to selected objects

### `make_links_scene`

bpy.ops.object.make_links_scene(scene='<UNKNOWN ENUM>')
Link selection to another scene

### `make_local`

bpy.ops.object.make_local(type='SELECT_OBJECT')
Make library linked data-blocks local to this file

### `make_override_library`

bpy.ops.object.make_override_library(collection=0)
Create a local override of the selected linked objects, and their hierarchy of dependencies

### `make_single_user`

bpy.ops.object.make_single_user(type='SELECTED_OBJECTS', object=False, obdata=False, material=False, animation=False, obdata_animation=False)
Make linked data local to each object

### `material_slot_add`

bpy.ops.object.material_slot_add()
Add a new material slot

### `material_slot_assign`

bpy.ops.object.material_slot_assign()
Assign active material slot to selection

### `material_slot_copy`

bpy.ops.object.material_slot_copy()
Copy material to selected objects

### `material_slot_deselect`

bpy.ops.object.material_slot_deselect()
Deselect by active material slot

### `material_slot_move`

bpy.ops.object.material_slot_move(direction='UP')
Move the active material up/down in the list

### `material_slot_remove`

bpy.ops.object.material_slot_remove()
Remove the selected material slot

### `material_slot_remove_unused`

bpy.ops.object.material_slot_remove_unused()
Remove unused material slots

### `material_slot_select`

bpy.ops.object.material_slot_select()
Select by active material slot

### `meshdeform_bind`

bpy.ops.object.meshdeform_bind(modifier="")
Bind mesh to cage in mesh deform modifier

### `metaball_add`

bpy.ops.object.metaball_add(type='BALL', radius=2, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add an metaball object to the scene

### `mode_set`

bpy.ops.object.mode_set(mode='OBJECT', toggle=False)
Sets the object interaction mode

### `mode_set_with_submode`

bpy.ops.object.mode_set_with_submode(mode='OBJECT', toggle=False, mesh_select_mode=set())
Sets the object interaction mode

### `modifier_add`

bpy.ops.object.modifier_add(type='SUBSURF', use_selected_objects=False)
Add a procedural operation/effect to the active object

### `modifier_add_node_group`

bpy.ops.object.modifier_add_node_group(asset_library_type='LOCAL', asset_library_identifier="", relative_asset_identifier="", session_uid=0, use_selected_objects=False)
Add a procedural operation/effect to the active object

### `modifier_apply`

bpy.ops.object.modifier_apply(modifier="", report=False, merge_customdata=True, single_user=False, all_keyframes=False, use_selected_objects=False)
Apply modifier and remove from the stack

### `modifier_apply_as_shapekey`

bpy.ops.object.modifier_apply_as_shapekey(keep_modifier=False, modifier="", report=False, use_selected_objects=False)
Apply modifier as a new shape key and remove from the stack

### `modifier_convert`

bpy.ops.object.modifier_convert(modifier="")
Convert particles to a mesh object

### `modifier_copy`

bpy.ops.object.modifier_copy(modifier="", use_selected_objects=False)
Duplicate modifier at the same position in the stack

### `modifier_copy_to_selected`

bpy.ops.object.modifier_copy_to_selected(modifier="")
Copy the modifier from the active object to all selected objects

### `modifier_move_down`

bpy.ops.object.modifier_move_down(modifier="")
Move modifier down in the stack

### `modifier_move_to_index`

bpy.ops.object.modifier_move_to_index(modifier="", index=0, use_selected_objects=False)
Change the modifier's index in the stack so it evaluates after the set number of others

### `modifier_move_up`

bpy.ops.object.modifier_move_up(modifier="")
Move modifier up in the stack

### `modifier_remove`

bpy.ops.object.modifier_remove(modifier="", report=False, use_selected_objects=False)
Remove a modifier from the active object

### `modifier_set_active`

bpy.ops.object.modifier_set_active(modifier="")
Activate the modifier to use as the context

### `modifiers_clear`

bpy.ops.object.modifiers_clear()
Clear all modifiers from the selected objects

### `modifiers_copy_to_selected`

bpy.ops.object.modifiers_copy_to_selected()
Copy modifiers to other selected objects

### `move_to_collection`

bpy.ops.object.move_to_collection(collection_index=-1, is_new=False, new_collection_name="")
Move objects to a collection

### `multires_base_apply`

bpy.ops.object.multires_base_apply(modifier="")
Modify the base mesh to conform to the displaced mesh

### `multires_external_pack`

bpy.ops.object.multires_external_pack()
Pack displacements from an external file

### `multires_external_save`

bpy.ops.object.multires_external_save(filepath="", hide_props_region=True, check_existing=True, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=True, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=False, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', modifier="")
Save displacements to an external file

### `multires_higher_levels_delete`

bpy.ops.object.multires_higher_levels_delete(modifier="")
Deletes the higher resolution mesh, potential loss of detail

### `multires_rebuild_subdiv`

bpy.ops.object.multires_rebuild_subdiv(modifier="")
Rebuilds all possible subdivisions levels to generate a lower resolution base mesh

### `multires_reshape`

bpy.ops.object.multires_reshape(modifier="")
Copy vertex coordinates from other object

### `multires_subdivide`

bpy.ops.object.multires_subdivide(modifier="", mode='CATMULL_CLARK')
Add a new level of subdivision

### `multires_unsubdivide`

bpy.ops.object.multires_unsubdivide(modifier="")
Rebuild a lower subdivision level of the current base mesh

### `ocean_bake`

bpy.ops.object.ocean_bake(modifier="", free=False)
Bake an image sequence of ocean data

### `origin_clear`

bpy.ops.object.origin_clear()
Clear the object's origin

### `origin_set`

bpy.ops.object.origin_set(type='GEOMETRY_ORIGIN', center='MEDIAN')
Set the object's origin, by either moving the data, or set to center of data, or use 3D cursor

### `parent_clear`

bpy.ops.object.parent_clear(type='CLEAR')
Clear the object's parenting

### `parent_inverse_apply`

bpy.ops.object.parent_inverse_apply()
Apply the object's parent inverse to its data

### `parent_no_inverse_set`

bpy.ops.object.parent_no_inverse_set(keep_transform=False)
Set the object's parenting without setting the inverse parent correction

### `parent_set`

bpy.ops.object.parent_set(type='OBJECT', xmirror=False, keep_transform=False)
Set the object's parenting

### `particle_system_add`

bpy.ops.object.particle_system_add()
Add a particle system

### `particle_system_remove`

bpy.ops.object.particle_system_remove()
Remove the selected particle system

### `paths_calculate`

bpy.ops.object.paths_calculate(display_type='RANGE', range='SCENE')
Generate motion paths for the selected objects

### `paths_clear`

bpy.ops.object.paths_clear(only_selected=False)
(undocumented operator)

### `paths_update`

bpy.ops.object.paths_update()
Recalculate motion paths for selected objects

### `paths_update_visible`

bpy.ops.object.paths_update_visible()
Recalculate all visible motion paths for objects and poses

### `pointcloud_add`

bpy.ops.object.pointcloud_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a point cloud object to the scene

### `posemode_toggle`

bpy.ops.object.posemode_toggle()
Enable or disable posing/selecting bones

### `quadriflow_remesh`

bpy.ops.object.quadriflow_remesh(use_mesh_symmetry=True, use_preserve_sharp=False, use_preserve_boundary=False, preserve_attributes=False, smooth_normals=False, mode='FACES', target_ratio=1, target_edge_length=0.1, target_faces=4000, mesh_area=-1, seed=0)
Create a new quad based mesh using the surface data of the current mesh. All data layers will be lost

### `quick_explode`

bpy.ops.object.quick_explode(style='EXPLODE', amount=100, frame_duration=50, frame_start=1, frame_end=10, velocity=1, fade=True)
Make selected objects explode

### `quick_fur`

bpy.ops.object.quick_fur(density='MEDIUM', length=0.1, radius=0.001, view_percentage=1, apply_hair_guides=True, use_noise=True, use_frizz=True)
Add a fur setup to the selected objects

### `quick_liquid`

bpy.ops.object.quick_liquid(show_flows=False)
Make selected objects liquid

### `quick_smoke`

bpy.ops.object.quick_smoke(style='SMOKE', show_flows=False)
Use selected objects as smoke emitters

### `randomize_transform`

bpy.ops.object.randomize_transform(random_seed=0, use_delta=False, use_loc=True, loc=(0, 0, 0), use_rot=True, rot=(0, 0, 0), use_scale=True, scale_even=False, scale=(1, 1, 1))
Randomize objects location, rotation, and scale

### `reset_override_library`

bpy.ops.object.reset_override_library()
Reset the selected local overrides to their linked references values

### `rotation_clear`

bpy.ops.object.rotation_clear(clear_delta=False)
Clear the object's rotation

### `scale_clear`

bpy.ops.object.scale_clear(clear_delta=False)
Clear the object's scale

### `select_all`

bpy.ops.object.select_all(action='TOGGLE')
Change selection of all visible objects in scene

### `select_by_type`

bpy.ops.object.select_by_type(extend=False, type='MESH')
Select all visible objects that are of a type

### `select_camera`

bpy.ops.object.select_camera(extend=False)
Select the active camera

### `select_grouped`

bpy.ops.object.select_grouped(extend=False, type='CHILDREN_RECURSIVE')
Select all visible objects grouped by various properties

### `select_hierarchy`

bpy.ops.object.select_hierarchy(direction='PARENT', extend=False)
Select object relative to the active object's position in the hierarchy

### `select_less`

bpy.ops.object.select_less()
Deselect objects at the boundaries of parent/child relationships

### `select_linked`

bpy.ops.object.select_linked(extend=False, type='OBDATA')
Select all visible objects that are linked

### `select_mirror`

bpy.ops.object.select_mirror(extend=False)
Select the mirror objects of the selected object e.g. "L.sword" and "R.sword"

### `select_more`

bpy.ops.object.select_more()
Select connected parent/child objects

### `select_pattern`

bpy.ops.object.select_pattern(pattern="*", case_sensitive=False, extend=True)
Select objects matching a naming pattern

### `select_random`

bpy.ops.object.select_random(ratio=0.5, seed=0, action='SELECT')
Select or deselect random visible objects

### `select_same_collection`

bpy.ops.object.select_same_collection(collection="")
Select object in the same collection

### `shade_auto_smooth`

bpy.ops.object.shade_auto_smooth(use_auto_smooth=True, angle=0.523599)
Add modifier to automatically set the sharpness of mesh edges based on the angle between the neighboring faces

### `shade_flat`

bpy.ops.object.shade_flat(keep_sharp_edges=True)
Render and display faces uniform, using face normals

### `shade_smooth`

bpy.ops.object.shade_smooth(keep_sharp_edges=True)
Render and display faces smooth, using interpolated vertex normals

### `shade_smooth_by_angle`

bpy.ops.object.shade_smooth_by_angle(angle=0.523599, keep_sharp_edges=True)
Set the sharpness of mesh edges based on the angle between the neighboring faces

### `shaderfx_add`

bpy.ops.object.shaderfx_add(type='FX_BLUR')
Add a visual effect to the active object

### `shaderfx_copy`

bpy.ops.object.shaderfx_copy(shaderfx="")
Duplicate effect at the same position in the stack

### `shaderfx_move_down`

bpy.ops.object.shaderfx_move_down(shaderfx="")
Move effect down in the stack

### `shaderfx_move_to_index`

bpy.ops.object.shaderfx_move_to_index(shaderfx="", index=0)
Change the effect's position in the list so it evaluates after the set number of others

### `shaderfx_move_up`

bpy.ops.object.shaderfx_move_up(shaderfx="")
Move effect up in the stack

### `shaderfx_remove`

bpy.ops.object.shaderfx_remove(shaderfx="", report=False)
Remove a effect from the active Grease Pencil object

### `shape_key_add`

bpy.ops.object.shape_key_add(from_mix=True)
Add shape key to the object

### `shape_key_clear`

bpy.ops.object.shape_key_clear()
Reset the weights of all shape keys to 0 or to the closest value respecting the limits

### `shape_key_lock`

bpy.ops.object.shape_key_lock(action='LOCK')
Change the lock state of all shape keys of active object

### `shape_key_mirror`

bpy.ops.object.shape_key_mirror(use_topology=False)
Mirror the current shape key along the local X axis

### `shape_key_move`

bpy.ops.object.shape_key_move(type='TOP')
Move the active shape key up/down in the list

### `shape_key_remove`

bpy.ops.object.shape_key_remove(all=False, apply_mix=False)
Remove shape key from the object

### `shape_key_retime`

bpy.ops.object.shape_key_retime()
Resets the timing for absolute shape keys

### `shape_key_transfer`

bpy.ops.object.shape_key_transfer(mode='OFFSET', use_clamp=False)
Copy the active shape key of another selected object to this one

### `simulation_nodes_cache_bake`

bpy.ops.object.simulation_nodes_cache_bake(selected=False)
Bake simulations in geometry nodes modifiers

### `simulation_nodes_cache_calculate_to_frame`

bpy.ops.object.simulation_nodes_cache_calculate_to_frame(selected=False)
Calculate simulations in geometry nodes modifiers from the start to current frame

### `simulation_nodes_cache_delete`

bpy.ops.object.simulation_nodes_cache_delete(selected=False)
Delete cached/baked simulations in geometry nodes modifiers

### `skin_armature_create`

bpy.ops.object.skin_armature_create(modifier="")
Create an armature that parallels the skin layout

### `skin_loose_mark_clear`

bpy.ops.object.skin_loose_mark_clear(action='MARK')
Mark/clear selected vertices as loose

### `skin_radii_equalize`

bpy.ops.object.skin_radii_equalize()
Make skin radii of selected vertices equal on each axis

### `skin_root_mark`

bpy.ops.object.skin_root_mark()
Mark selected vertices as roots

### `speaker_add`

bpy.ops.object.speaker_add(enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a speaker object to the scene

### `subdivision_set`

bpy.ops.object.subdivision_set(level=1, relative=False)
Sets a Subdivision Surface level (1 to 5)

### `surfacedeform_bind`

bpy.ops.object.surfacedeform_bind(modifier="")
Bind mesh to target in surface deform modifier

### `text_add`

bpy.ops.object.text_add(radius=1, enter_editmode=False, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a text object to the scene

### `track_clear`

bpy.ops.object.track_clear(type='CLEAR')
Clear tracking constraint or flag from object

### `track_set`

bpy.ops.object.track_set(type='DAMPTRACK')
Make the object track another object, using various methods/constraints

### `transfer_mode`

bpy.ops.object.transfer_mode(use_flash_on_transfer=True)
Switches the active object and assigns the same mode to a new one under the mouse cursor, leaving the active mode in the current one

### `transform_apply`

bpy.ops.object.transform_apply(location=True, rotation=True, scale=True, properties=True, isolate_users=False)
Apply the object's transformation to its data

### `transform_axis_target`

bpy.ops.object.transform_axis_target()
Interactively point cameras and lights to a location (Ctrl translates)

### `transform_to_mouse`

bpy.ops.object.transform_to_mouse(name="", session_uid=0, matrix=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)), drop_x=0, drop_y=0)
Snap selected item(s) to the mouse location

### `transforms_to_deltas`

bpy.ops.object.transforms_to_deltas(mode='ALL', reset_values=True)
Convert normal object transforms to delta transforms, any existing delta transforms will be included as well

### `unlink_data`

bpy.ops.object.unlink_data()
(undocumented operator)

### `vertex_group_add`

bpy.ops.object.vertex_group_add()
Add a new vertex group to the active object

### `vertex_group_assign`

bpy.ops.object.vertex_group_assign()
Assign the selected vertices to the active vertex group

### `vertex_group_assign_new`

bpy.ops.object.vertex_group_assign_new()
Assign the selected vertices to a new vertex group

### `vertex_group_clean`

bpy.ops.object.vertex_group_clean(group_select_mode='ACTIVE', limit=0, keep_single=False)
Remove vertex group assignments which are not required

### `vertex_group_copy`

bpy.ops.object.vertex_group_copy()
Make a copy of the active vertex group

### `vertex_group_copy_to_selected`

bpy.ops.object.vertex_group_copy_to_selected()
Replace vertex groups of selected objects by vertex groups of active object

### `vertex_group_deselect`

bpy.ops.object.vertex_group_deselect()
Deselect all selected vertices assigned to the active vertex group

### `vertex_group_invert`

bpy.ops.object.vertex_group_invert(group_select_mode='ACTIVE', auto_assign=True, auto_remove=True)
Invert active vertex group's weights

### `vertex_group_levels`

bpy.ops.object.vertex_group_levels(group_select_mode='ACTIVE', offset=0, gain=1)
Add some offset and multiply with some gain the weights of the active vertex group

### `vertex_group_limit_total`

bpy.ops.object.vertex_group_limit_total(group_select_mode='ALL', limit=4)
Limit deform weights associated with a vertex to a specified number by removing lowest weights

### `vertex_group_lock`

bpy.ops.object.vertex_group_lock(action='TOGGLE', mask='ALL')
Change the lock state of all or some vertex groups of active object

### `vertex_group_mirror`

bpy.ops.object.vertex_group_mirror(mirror_weights=True, flip_group_names=True, all_groups=False, use_topology=False)
Mirror vertex group, flip weights and/or names, editing only selected vertices, flipping when both sides are selected otherwise copy from unselected

### `vertex_group_move`

bpy.ops.object.vertex_group_move(direction='UP')
Move the active vertex group up/down in the list

### `vertex_group_normalize`

bpy.ops.object.vertex_group_normalize()
Normalize weights of the active vertex group, so that the highest ones are now 1.0

### `vertex_group_normalize_all`

bpy.ops.object.vertex_group_normalize_all(group_select_mode='ALL', lock_active=True)
Normalize all weights of all vertex groups, so that for each vertex, the sum of all weights is 1.0

### `vertex_group_quantize`

bpy.ops.object.vertex_group_quantize(group_select_mode='ACTIVE', steps=4)
Set weights to a fixed number of steps

### `vertex_group_remove`

bpy.ops.object.vertex_group_remove(all=False, all_unlocked=False)
Delete the active or all vertex groups from the active object

### `vertex_group_remove_from`

bpy.ops.object.vertex_group_remove_from(use_all_groups=False, use_all_verts=False)
Remove the selected vertices from active or all vertex group(s)

### `vertex_group_select`

bpy.ops.object.vertex_group_select()
Select all the vertices assigned to the active vertex group

### `vertex_group_set_active`

bpy.ops.object.vertex_group_set_active(group='<UNKNOWN ENUM>')
Set the active vertex group

### `vertex_group_smooth`

bpy.ops.object.vertex_group_smooth(group_select_mode='ACTIVE', factor=0.5, repeat=1, expand=0)
Smooth weights for selected vertices

### `vertex_group_sort`

bpy.ops.object.vertex_group_sort(sort_type='NAME')
Sort vertex groups

### `vertex_parent_set`

bpy.ops.object.vertex_parent_set()
Parent selected objects to the selected vertices

### `vertex_weight_copy`

bpy.ops.object.vertex_weight_copy()
Copy weights from active to selected

### `vertex_weight_delete`

bpy.ops.object.vertex_weight_delete(weight_group=-1)
Delete this weight from the vertex (disabled if vertex group is locked)

### `vertex_weight_normalize_active_vertex`

bpy.ops.object.vertex_weight_normalize_active_vertex()
Normalize active vertex's weights

### `vertex_weight_paste`

bpy.ops.object.vertex_weight_paste(weight_group=-1)
Copy this group's weight to other selected vertices (disabled if vertex group is locked)

### `vertex_weight_set_active`

bpy.ops.object.vertex_weight_set_active(weight_group=-1)
Set as active vertex group

### `visual_transform_apply`

bpy.ops.object.visual_transform_apply()
Apply the object's visual transformation to its data

### `volume_add`

bpy.ops.object.volume_add(align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Add a volume object to the scene

### `volume_import`

bpy.ops.object.volume_import(filepath="", directory="", files=[], hide_props_region=True, check_existing=False, filter_blender=False, filter_backup=False, filter_image=False, filter_movie=False, filter_python=False, filter_font=False, filter_sound=False, filter_text=False, filter_archive=False, filter_btx=False, filter_collada=False, filter_alembic=False, filter_usd=False, filter_obj=False, filter_volume=True, filter_folder=True, filter_blenlib=False, filemode=9, relative_path=True, display_type='DEFAULT', sort_method='DEFAULT', use_sequence_detection=True, align='WORLD', location=(0, 0, 0), rotation=(0, 0, 0), scale=(0, 0, 0))
Import OpenVDB volume file

### `voxel_remesh`

bpy.ops.object.voxel_remesh()
Calculates a new manifold mesh based on the volume of the current mesh. All data layers will be lost

### `voxel_size_edit`

bpy.ops.object.voxel_size_edit()
Modify the mesh voxel size interactively used in the voxel remesher
