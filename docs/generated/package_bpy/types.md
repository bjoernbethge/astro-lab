# types Submodule

Part of the `bpy` package
Module: `bpy.types`

## Description

Access to internal Blender types

## Important Data Types (15)

### `ID`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `AOV`
**Type**: `<class 'type'>`

*(has methods, callable)*

### `Key`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `AOVs`
**Type**: `<class 'type'>`

*(has methods, callable)*

### `Area`
**Type**: `<class 'type'>`

*(has methods, callable)*

### `Bone`
**Type**: `<class 'bpy_struct_meta_idprop'>`

functions for bones, common between Armature/Pose/Edit bones.
internal subclassing use only.

*(has methods, callable)*

### `Mask`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `Menu`
**Type**: `<class 'bpy_types._RNAMeta'>`

*(has methods, callable)*

### `Mesh`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `Node`
**Type**: `<class 'bpy_types._RNAMetaPropGroup'>`

*(has methods, callable)*

### `Pose`
**Type**: `<class 'type'>`

*(has methods, callable)*

### `Text`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `Addon`
**Type**: `<class 'type'>`

*(has methods, callable)*

### `Brush`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

### `Curve`
**Type**: `<class 'bpy_struct_meta_idprop'>`

*(has methods, callable)*

## Classes (3653)

### `ANIM_MT_keyframe_insert_pie`

#### Methods

- **`draw(self, _context)`**

### `ANIM_OT_clear_useless_actions`

Mark actions with no F-Curves for deletion after save and reload of file preserving "action libraries"

#### Methods

- **`execute(self, _context)`**

### `ANIM_OT_keying_set_export`

Export Keying Set to a Python script

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `ANIM_OT_slot_new_for_id`

Create a new Action Slot for an ID.

Note that _which_ ID should get this slot must be set in the 'animated_id' context pointer, using:

>>> layout.context_pointer_set("animated_id", animated_id)

#### Methods

- **`execute(self, context)`**

### `ANIM_OT_slot_unassign_from_constraint`

Un-assign the assigned Action Slot from an Action constraint.

Note that _which_ constraint should get this slot unassigned must be set in
the "constraint" context pointer, using:

>>> layout.context_pointer_set("constraint", constraint)

### `ANIM_OT_slot_unassign_from_id`

Un-assign the assigned Action Slot from an ID.

Note that _which_ ID should get this slot unassigned must be set in the
"animated_id" context pointer, using:

>>> layout.context_pointer_set("animated_id", animated_id)

#### Methods

- **`execute(self, context)`**

### `ANIM_OT_slot_unassign_from_nla_strip`

Un-assign the assigned Action Slot from an NLA strip.

Note that _which_ NLA strip should get this slot unassigned must be set in
the "nla_strip" context pointer, using:

>>> layout.context_pointer_set("nla_strip", nla_strip)

### `ANIM_OT_update_animated_transform_constraints`

Update f-curves/drivers affecting Transform constraints (use it with files from 2.70 and earlier)

#### Methods

- **`execute(self, context)`**

### `AOV`

### `AOVs`

### `ARMATURE_MT_collection_context_menu`

#### Methods

- **`draw(self, context)`**

### `ARMATURE_MT_collection_tree_context_menu`

#### Methods

- **`draw(self, context)`**

### `ARMATURE_OT_collection_remove_unused`

Remove all bone collections that have neither bones nor children. This is done recursively, so bone collections that only have unused children are also removed

#### Methods

- **`execute(self, context)`**

- **`execute_edit_mode(self, context)`**

- **`visit(self, bcoll, bcolls_with_bones, bcolls_to_remove)`**

- **`remove_bcolls(self, armature, bcolls_to_remove)`**

### `ARMATURE_OT_collection_show_all`

Show all bone collections

#### Methods

- **`execute(self, context)`**

### `ARMATURE_OT_collection_unsolo_all`

Clear the 'solo' setting on all bone collections

#### Methods

- **`execute(self, context)`**

### `ARMATURE_OT_copy_bone_color_to_selected`

Copy the bone color of the active bone to all selected bones

#### Methods

- **`execute(self, context)`**

### `ASSETBROWSER_MT_asset`

#### Methods

- **`draw(self, context: bpy_types.Context) -> None`**

### `ASSETBROWSER_MT_catalog`

#### Methods

- **`draw(self, _context)`**

### `ASSETBROWSER_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_MT_metadata_preview_menu`

#### Methods

- **`draw(self, _context)`**

### `ASSETBROWSER_MT_select`

#### Methods

- **`draw(self, _context)`**

### `ASSETBROWSER_MT_view`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_PT_display`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_PT_filter`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_PT_metadata`

#### Methods

- **`metadata_prop(layout, asset_metadata, propname)`**
  Only display properties that are either set or can be modified (i.e. the

- **`draw(self, context)`**

### `ASSETBROWSER_PT_metadata_preview`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_PT_metadata_tags`

#### Methods

- **`draw(self, context)`**

### `ASSETBROWSER_UL_metadata_tags`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `ASSETSHELF_PT_display`

#### Methods

- **`draw(self, context)`**

### `ASSET_OT_assign_action`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

### `ASSET_OT_open_containing_blend_file`

Open the blend file that contains the active asset

#### Methods

- **`execute(self, context)`**

- **`modal(self, context, event)`**

- **`cancel(self, context)`**

- **`open_in_new_blender(self, filepath)`**

### `ASSET_OT_tag_add`

Add a new keyword tag to the active asset

#### Methods

- **`execute(self, context)`**

### `ASSET_OT_tag_remove`

Remove an existing keyword tag from the active asset

#### Methods

- **`execute(self, context)`**

### `Action`

### `ActionChannelbag`

### `ActionChannelbagFCurves`

### `ActionChannelbagGroups`

### `ActionChannelbags`

### `ActionConstraint`

### `ActionFCurves`

### `ActionGroup`

### `ActionGroups`

### `ActionKeyframeStrip`

### `ActionLayer`

### `ActionLayers`

### `ActionPoseMarkers`

### `ActionSlot`

### `ActionSlots`

### `ActionStrip`

### `ActionStrips`

### `AddStrip`

### `Addon`

### `AddonPreferences`

### `Addons`

#### Methods

- **`new(*args, **kwargs)`**
  Addons.new()

- **`remove(*args, **kwargs)`**
  Addons.remove(addon)

### `AdjustmentStrip`

### `AlphaOverStrip`

### `AlphaUnderStrip`

### `AnimData`

### `AnimDataDrivers`

### `AnimViz`

### `AnimVizMotionPaths`

### `AnyType`

### `Area`

### `AreaLight`

### `AreaSpaces`

### `Armature`

### `ArmatureBones`

### `ArmatureConstraint`

### `ArmatureConstraintTargets`

### `ArmatureEditBones`

### `ArmatureModifier`

### `ArrayModifier`

### `AssetCatalogPath`

### `AssetHandle`

### `AssetLibraryCollection`

#### Methods

- **`new(*args, **kwargs)`**
  AssetLibraryCollection.new(name="", directory="")

- **`remove(*args, **kwargs)`**
  AssetLibraryCollection.remove(library)

### `AssetLibraryReference`

### `AssetMetaData`

### `AssetRepresentation`

### `AssetShelf`

### `AssetTag`

### `AssetTags`

### `AssetWeakReference`

### `Attribute`

### `AttributeGroupCurves`

### `AttributeGroupGreasePencil`

### `AttributeGroupGreasePencilDrawing`

### `AttributeGroupMesh`

### `AttributeGroupPointCloud`

### `BONE_PT_bActionConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bActionConstraint_action`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bActionConstraint_target`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bArmatureConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bArmatureConstraint_bones`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bCameraSolverConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bChildOfConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bClampToConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bDampTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bDistLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bFollowPathConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bFollowTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bKinematicConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bLocLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bLocateLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bLockTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bMinMaxConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bObjectSolverConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bPivotConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bPythonConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bRotLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bRotateLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSameVolumeConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bShrinkwrapConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSizeLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSizeLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSplineIKConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSplineIKConstraint_chain_scaling`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bSplineIKConstraint_fitting`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bStretchToConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTrackToConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformCacheConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformCacheConstraint_layers`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformCacheConstraint_procedural`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformCacheConstraint_time`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformCacheConstraint_velocity`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformConstraint`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformConstraint_from`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_bTransformConstraint_to`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_collections`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_constraints`

#### Methods

- **`draw(self, _context)`**

### `BONE_PT_context_bone`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_curved`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

#### Methods

- **`draw(self, context)`**

### `BONE_PT_deform`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `BONE_PT_display`

#### Methods

- **`draw(self, context)`**

- **`draw_bone(self, context, layout)`**

- **`draw_edit_bone(self, context, layout)`**

- **`draw_bone_color_ui(self, layout, bone_color)`**

### `BONE_PT_display_custom_shape`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_inverse_kinematics`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_relations`

#### Methods

- **`draw(self, context)`**

### `BONE_PT_transform`

#### Methods

- **`draw(self, context)`**

### `BVH_PT_export_animation`

#### Methods

- **`draw(self, context)`**

### `BVH_PT_export_transform`

#### Methods

- **`draw(self, context)`**

### `BVH_PT_import_animation`

#### Methods

- **`draw(self, context)`**

### `BVH_PT_import_main`

#### Methods

- **`draw(self, context)`**

### `BVH_PT_import_transform`

#### Methods

- **`draw(self, context)`**

### `BakeSettings`

### `BevelModifier`

### `BezierSplinePoint`

### `BlendData`

#### Methods

- **`user_map(...)`**
  .. method:: user_map(subset, key_types, value_types)

- **`file_path_map(...)`**
  .. method:: file_path_map(subset=None, key_types=None, include_libraries=False)

- **`batch_remove(...)`**
  .. method:: batch_remove(ids)

- **`orphans_purge(...)`**
  .. method:: orphans_purge()

- **`temp_data(...)`**
  .. method:: temp_data(filepath=None)

### `BlendDataActions`

### `BlendDataArmatures`

### `BlendDataBrushes`

### `BlendDataCacheFiles`

### `BlendDataCameras`

### `BlendDataCollections`

### `BlendDataCurves`

### `BlendDataFonts`

### `BlendDataGreasePencils`

### `BlendDataGreasePencilsV3`

### `BlendDataHairCurves`

### `BlendDataImages`

### `BlendDataLattices`

### `BlendDataLibraries`

#### Methods

- **`load(...)`**
  .. method:: load(filepath, link=False, relative=False, assets_only=False, create_liboverrides=False, reuse_liboverrides=False, create_liboverrides_runtime=False)

- **`write(...)`**
  .. method:: write(filepath, datablocks, path_remap=False, fake_user=False, compress=False)

### `BlendDataLights`

### `BlendDataLineStyles`

### `BlendDataMasks`

### `BlendDataMaterials`

### `BlendDataMeshes`

### `BlendDataMetaBalls`

### `BlendDataMovieClips`

### `BlendDataNodeTrees`

### `BlendDataObjects`

### `BlendDataPaintCurves`

### `BlendDataPalettes`

### `BlendDataParticles`

### `BlendDataPointClouds`

### `BlendDataProbes`

### `BlendDataScenes`

### `BlendDataScreens`

### `BlendDataSounds`

### `BlendDataSpeakers`

### `BlendDataTexts`

### `BlendDataTextures`

### `BlendDataVolumes`

### `BlendDataWindowManagers`

### `BlendDataWorkSpaces`

### `BlendDataWorlds`

### `BlendImportContext`

### `BlendImportContextItem`

### `BlendImportContextItems`

### `BlendImportContextLibraries`

### `BlendImportContextLibrary`

### `BlendTexture`

### `BlenderRNA`

### `BoidRule`

### `BoidRuleAverageSpeed`

### `BoidRuleAvoid`

### `BoidRuleAvoidCollision`

### `BoidRuleFight`

### `BoidRuleFollowLeader`

### `BoidRuleGoal`

### `BoidSettings`

### `BoidState`

### `Bone`

functions for bones, common between Armature/Pose/Edit bones.
internal subclassing use only.

#### Methods

- **`MatrixFromAxisRoll(*args, **kwargs)`**
  Bone.MatrixFromAxisRoll(axis, roll)

- **`AxisRollFromMatrix(*args, **kwargs)`**
  Bone.AxisRollFromMatrix(matrix, axis=(0, 0, 0))

### `BoneCollection`

### `BoneCollectionMemberships`

### `BoneCollections`

### `BoneColor`

### `BoolAttribute`

### `BoolAttributeValue`

### `BoolProperty`

### `BooleanModifier`

### `BrightContrastModifier`

### `Brush`

### `BrushCapabilities`

### `BrushCapabilitiesImagePaint`

### `BrushCapabilitiesSculpt`

### `BrushCapabilitiesVertexPaint`

### `BrushCapabilitiesWeightPaint`

### `BrushCurvesSculptSettings`

### `BrushGpencilSettings`

### `BrushTextureSlot`

### `BuildModifier`

### `ByteColorAttribute`

### `ByteColorAttributeValue`

### `ByteIntAttribute`

### `ByteIntAttributeValue`

### `CAMERA_OT_preset_add`

Add or remove a Camera Preset

### `CAMERA_OT_safe_areas_preset_add`

Add or remove a Safe Areas Preset

### `CAMERA_PT_presets`

### `CAMERA_PT_safe_areas_presets`

### `CLIP_HT_header`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_clip`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_marker_pie`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_masking_editor_menus`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_pivot_pie`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_plane_track_image_context_menu`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_proxy`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_reconstruction`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_reconstruction_pie`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_select`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_select_graph`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_select_grouped`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_solving_pie`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_stabilize_2d_context_menu`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_stabilize_2d_rotation_context_menu`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_track_animation`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_cleanup`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_clear`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_motion`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_refine`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_transform`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_track_visibility`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_tracking_context_menu`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_tracking_editor_menus`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_tracking_pie`

#### Methods

- **`draw(self, _context)`**

### `CLIP_MT_view`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `CLIP_MT_view_zoom`

#### Methods

- **`draw(self, context)`**

### `CLIP_OT_bundles_to_mesh`

Create vertex cloud using coordinates of reconstructed tracks

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_camera_preset_add`

Add or remove a Tracking Camera Intrinsics Preset

### `CLIP_OT_constraint_to_fcurve`

Create F-Curves for object which will copy object's movement caused by this constraint

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_delete_proxy`

Delete movie clip proxy files from the hard drive

#### Methods

- **`invoke(self, context, event)`**

- **`execute(self, context)`**

### `CLIP_OT_filter_tracks`

Filter tracks which has weirdly looking spikes in motion curves

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_set_active_clip`

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_set_viewport_background`

Set current movie clip as a camera background in 3D Viewport (works only when a 3D Viewport is visible)

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_setup_tracking_scene`

Prepare scene for compositing 3D objects into this footage

#### Methods

- **`createCollection(context, collection_name)`**

- **`execute(self, context)`**

### `CLIP_OT_track_color_preset_add`

Add or remove a Clip Track Color Preset

### `CLIP_OT_track_settings_as_default`

Copy tracking settings from active track to default settings

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_track_settings_to_track`

Copy tracking settings from active track to selected tracks

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_track_to_empty`

Create an Empty object which will be copying movement of active track

#### Methods

- **`execute(self, context)`**

### `CLIP_OT_tracking_settings_preset_add`

Add or remove a motion tracking settings preset

### `CLIP_PT_2d_cursor`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_active_mask_point`

### `CLIP_PT_active_mask_spline`

### `CLIP_PT_animation`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_annotation`

### `CLIP_PT_camera_presets`

Predefined tracking camera intrinsics

### `CLIP_PT_clip_display`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_display`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_footage`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_gizmo_display`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_marker`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_marker_display`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_mask`

### `CLIP_PT_mask_animation`

### `CLIP_PT_mask_display`

### `CLIP_PT_mask_layers`

### `CLIP_PT_objects`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_plane_track`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_proportional_edit`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_proxy`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CLIP_PT_stabilization`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CLIP_PT_tools_cleanup`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tools_clip`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_tools_geometry`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_tools_grease_pencil_draw`

### `CLIP_PT_tools_marker`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_tools_mask_tools`

### `CLIP_PT_tools_mask_transforms`

### `CLIP_PT_tools_object`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tools_orientation`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tools_plane_tracking`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_tools_scenesetup`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_tools_solve`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tools_tracking`

#### Methods

- **`draw(self, _context)`**

### `CLIP_PT_track`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_track_color_presets`

Predefined track color

### `CLIP_PT_track_settings`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_track_settings_extras`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tracking_camera`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `CLIP_PT_tracking_lens`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tracking_settings`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `CLIP_PT_tracking_settings_extras`

#### Methods

- **`draw(self, context)`**

### `CLIP_PT_tracking_settings_presets`

Predefined tracking settings

### `CLIP_UL_tracking_objects`

#### Methods

- **`draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index)`**

### `CLOTH_OT_preset_add`

Add or remove a Cloth Preset

### `CLOTH_PT_presets`

### `COLLECTION_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `COLLECTION_MT_context_menu_instance_offset`

#### Methods

- **`draw(self, _context)`**

### `COLLECTION_PT_collection_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `COLLECTION_PT_collection_flags`

#### Methods

- **`draw(self, context)`**

### `COLLECTION_PT_exporters`

#### Methods

- **`draw(self, context)`**

### `COLLECTION_PT_instancing`

#### Methods

- **`draw(self, context)`**

### `COLLECTION_PT_lineart_collection`

#### Methods

- **`draw(self, context)`**

### `CONSOLE_HT_header`

#### Methods

- **`draw(self, context)`**

### `CONSOLE_MT_console`

#### Methods

- **`draw(self, _context)`**

### `CONSOLE_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `CONSOLE_MT_editor_menus`

#### Methods

- **`draw(self, _context)`**

### `CONSOLE_MT_language`

#### Methods

- **`draw(self, _context)`**

### `CONSOLE_MT_view`

#### Methods

- **`draw(self, _context)`**

### `CONSOLE_OT_autocomplete`

Evaluate the namespace up until the cursor and give a list of options or complete the name if there is only one

#### Methods

- **`execute(self, context)`**

### `CONSOLE_OT_banner`

Print a message when the terminal initializes

#### Methods

- **`execute(self, context)`**

### `CONSOLE_OT_copy_as_script`

Copy the console contents for use in a script

#### Methods

- **`execute(self, context)`**

### `CONSOLE_OT_execute`

Execute the current console line as a Python expression

#### Methods

- **`execute(self, context)`**

### `CONSOLE_OT_language`

Set the current language for this console

#### Methods

- **`execute(self, context)`**

### `CONSTRAINT_OT_add_target`

Add a target to the constraint

#### Methods

- **`execute(self, context)`**

### `CONSTRAINT_OT_disable_keep_transform`

Set the influence of this constraint to zero while trying to maintain the object's transformation. Other active constraints can still influence the final transformation

#### Methods

- **`execute(self, context)`**
  Disable constraint while maintaining the visual transform

### `CONSTRAINT_OT_normalize_target_weights`

Normalize weights of all target bones

#### Methods

- **`execute(self, context)`**

### `CONSTRAINT_OT_remove_target`

Remove the target from the constraint

#### Methods

- **`execute(self, context)`**

### `CURVES_MT_add_attribute`

#### Methods

- **`add_standard_attribute(layout, curves, name, data_type, domain)`**

- **`draw(self, context)`**

### `CURVES_UL_attributes`

#### Methods

- **`filter_items(self, _context, data, property)`**

- **`draw_item(self, _context, layout, _data, attribute, _icon, _active_data, _active_propname, _index)`**

### `CYCLES_CAMERA_PT_dof`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_CAMERA_PT_dof_aperture`

#### Methods

- **`draw(self, context)`**

### `CYCLES_LIGHT_PT_beam_shape`

#### Methods

- **`draw(self, context)`**

### `CYCLES_LIGHT_PT_light`

#### Methods

- **`draw(self, context)`**

### `CYCLES_LIGHT_PT_nodes`

#### Methods

- **`draw(self, context)`**

### `CYCLES_LIGHT_PT_preview`

#### Methods

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_displacement`

#### Methods

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_preview`

#### Methods

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_settings`

#### Methods

- **`draw_shared(self, mat)`**

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_settings_surface`

#### Methods

- **`draw_shared(self, mat)`**

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_settings_volume`

#### Methods

- **`draw_shared(self, context, mat)`**

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_surface`

#### Methods

- **`draw(self, context)`**

### `CYCLES_MATERIAL_PT_volume`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_lightgroup`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_motion_blur`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_shading_caustics`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_shading_gi_approximation`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_shading_shadow_terminator`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_visibility`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_visibility_culling`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OBJECT_PT_visibility_ray_visibility`

#### Methods

- **`draw(self, context)`**

### `CYCLES_OT_denoise_animation`

Denoise rendered animation sequence using current scene and view layer settings. Requires denoising data passes and output to OpenEXR multilayer files

#### Methods

- **`execute(self, context)`**

### `CYCLES_OT_merge_images`

Combine OpenEXR multi-layer images rendered with different sample ranges into one image with reduced noise

#### Methods

- **`execute(self, context)`**

### `CYCLES_OT_use_shading_nodes`

Enable nodes on a material, world or light

#### Methods

- **`execute(self, context)`**

### `CYCLES_PT_context_material`

#### Methods

- **`draw(self, context)`**

### `CYCLES_PT_integrator_presets`

### `CYCLES_PT_performance_presets`

### `CYCLES_PT_post_processing`

#### Methods

- **`draw(self, context)`**

### `CYCLES_PT_sampling_presets`

### `CYCLES_PT_viewport_sampling_presets`

### `CYCLES_RENDER_PT_bake`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_bake_influence`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_bake_output`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_bake_output_margin`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_bake_selected_to_active`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_curves`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_curves_viewport_display`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_debug`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_film`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_film_pixel_filter`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_film_transparency`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_filter`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_light_paths`

#### Methods

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_light_paths_caustics`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_light_paths_clamping`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_light_paths_fast_gi`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_light_paths_max_bounces`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_motion_blur`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_motion_blur_curve`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_override`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_passes`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_passes_aov`

### `CYCLES_RENDER_PT_passes_crypto`

### `CYCLES_RENDER_PT_passes_data`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_passes_light`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_passes_lightgroups`

### `CYCLES_RENDER_PT_performance`

#### Methods

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_performance_acceleration_structure`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_performance_compositor`

### `CYCLES_RENDER_PT_performance_compositor_denoise_settings`

### `CYCLES_RENDER_PT_performance_final_render`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_performance_memory`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_performance_threads`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_performance_viewport`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_advanced`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_advanced_sample_subset`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_lights`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_path_guiding`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_path_guiding_debug`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_render`

#### Methods

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_render_denoise`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_viewport`

#### Methods

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_sampling_viewport_denoise`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_simplify`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_simplify_culling`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_simplify_render`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_simplify_viewport`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_subdivision`

#### Methods

- **`draw(self, context)`**

### `CYCLES_RENDER_PT_volumes`

#### Methods

- **`draw(self, context)`**

### `CYCLES_VIEW3D_PT_shading_debug`

#### Methods

- **`draw(self, context)`**

### `CYCLES_VIEW3D_PT_shading_lighting`

#### Methods

- **`draw(self, context)`**

### `CYCLES_VIEW3D_PT_shading_render_pass`

#### Methods

- **`draw(self, context)`**

### `CYCLES_VIEW3D_PT_simplify_greasepencil`

### `CYCLES_WORLD_PT_mist`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_preview`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_ray_visibility`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_settings`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_settings_light_group`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_settings_surface`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_settings_volume`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_surface`

#### Methods

- **`draw(self, context)`**

### `CYCLES_WORLD_PT_volume`

#### Methods

- **`draw(self, context)`**

### `CacheFile`

### `CacheFileLayer`

### `CacheFileLayers`

### `CacheObjectPath`

### `CacheObjectPaths`

### `Camera`

### `CameraBackgroundImage`

### `CameraBackgroundImages`

### `CameraDOFSettings`

### `CameraSolverConstraint`

### `CameraStereoData`

### `CastModifier`

### `ChannelDriverVariables`

### `ChildOfConstraint`

### `ChildParticle`

### `ClampToConstraint`

### `ClothCollisionSettings`

### `ClothModifier`

### `ClothSettings`

### `ClothSolverResult`

### `CloudsTexture`

### `Collection`

### `CollectionChild`

### `CollectionChildren`

### `CollectionExport`

### `CollectionLightLinking`

### `CollectionObject`

### `CollectionObjects`

### `CollectionProperty`

### `CollisionModifier`

### `CollisionSettings`

### `ColorBalanceModifier`

### `ColorManagedDisplaySettings`

### `ColorManagedInputColorspaceSettings`

### `ColorManagedSequencerColorspaceSettings`

### `ColorManagedViewSettings`

### `ColorMapping`

### `ColorMixStrip`

### `ColorRamp`

### `ColorRampElement`

### `ColorRampElements`

### `ColorStrip`

### `CompositorNode`

#### Methods

- **`update(self)`**

### `CompositorNodeAlphaOver`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeAlphaOver.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeAlphaOver.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeAlphaOver.output_template(index)

### `CompositorNodeAntiAliasing`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeAntiAliasing.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeAntiAliasing.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeAntiAliasing.output_template(index)

### `CompositorNodeBilateralblur`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBilateralblur.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBilateralblur.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBilateralblur.output_template(index)

### `CompositorNodeBlur`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBlur.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBlur.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBlur.output_template(index)

### `CompositorNodeBokehBlur`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBokehBlur.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBokehBlur.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBokehBlur.output_template(index)

### `CompositorNodeBokehImage`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBokehImage.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBokehImage.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBokehImage.output_template(index)

### `CompositorNodeBoxMask`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBoxMask.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBoxMask.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBoxMask.output_template(index)

### `CompositorNodeBrightContrast`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeBrightContrast.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeBrightContrast.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeBrightContrast.output_template(index)

### `CompositorNodeChannelMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeChannelMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeChannelMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeChannelMatte.output_template(index)

### `CompositorNodeChromaMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeChromaMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeChromaMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeChromaMatte.output_template(index)

### `CompositorNodeColorBalance`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeColorBalance.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeColorBalance.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeColorBalance.output_template(index)

### `CompositorNodeColorCorrection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeColorCorrection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeColorCorrection.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeColorCorrection.output_template(index)

### `CompositorNodeColorMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeColorMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeColorMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeColorMatte.output_template(index)

### `CompositorNodeColorSpill`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeColorSpill.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeColorSpill.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeColorSpill.output_template(index)

### `CompositorNodeCombHSVA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombHSVA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombHSVA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombHSVA.output_template(index)

### `CompositorNodeCombRGBA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombRGBA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombRGBA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombRGBA.output_template(index)

### `CompositorNodeCombYCCA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombYCCA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombYCCA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombYCCA.output_template(index)

### `CompositorNodeCombYUVA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombYUVA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombYUVA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombYUVA.output_template(index)

### `CompositorNodeCombineColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombineColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombineColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombineColor.output_template(index)

### `CompositorNodeCombineXYZ`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCombineXYZ.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCombineXYZ.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCombineXYZ.output_template(index)

### `CompositorNodeComposite`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeComposite.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeComposite.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeComposite.output_template(index)

### `CompositorNodeConvertColorSpace`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeConvertColorSpace.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeConvertColorSpace.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeConvertColorSpace.output_template(index)

### `CompositorNodeCornerPin`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCornerPin.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCornerPin.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCornerPin.output_template(index)

### `CompositorNodeCrop`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCrop.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCrop.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCrop.output_template(index)

### `CompositorNodeCryptomatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCryptomatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCryptomatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCryptomatte.output_template(index)

### `CompositorNodeCryptomatteV2`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCryptomatteV2.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCryptomatteV2.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCryptomatteV2.output_template(index)

### `CompositorNodeCurveRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCurveRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCurveRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCurveRGB.output_template(index)

### `CompositorNodeCurveVec`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeCurveVec.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeCurveVec.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeCurveVec.output_template(index)

### `CompositorNodeCustomGroup`

### `CompositorNodeDBlur`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDBlur.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDBlur.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDBlur.output_template(index)

### `CompositorNodeDefocus`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDefocus.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDefocus.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDefocus.output_template(index)

### `CompositorNodeDenoise`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDenoise.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDenoise.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDenoise.output_template(index)

### `CompositorNodeDespeckle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDespeckle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDespeckle.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDespeckle.output_template(index)

### `CompositorNodeDiffMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDiffMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDiffMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDiffMatte.output_template(index)

### `CompositorNodeDilateErode`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDilateErode.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDilateErode.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDilateErode.output_template(index)

### `CompositorNodeDisplace`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDisplace.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDisplace.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDisplace.output_template(index)

### `CompositorNodeDistanceMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDistanceMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDistanceMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDistanceMatte.output_template(index)

### `CompositorNodeDoubleEdgeMask`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeDoubleEdgeMask.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeDoubleEdgeMask.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeDoubleEdgeMask.output_template(index)

### `CompositorNodeEllipseMask`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeEllipseMask.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeEllipseMask.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeEllipseMask.output_template(index)

### `CompositorNodeExposure`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeExposure.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeExposure.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeExposure.output_template(index)

### `CompositorNodeFilter`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeFilter.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeFilter.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeFilter.output_template(index)

### `CompositorNodeFlip`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeFlip.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeFlip.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeFlip.output_template(index)

### `CompositorNodeGamma`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeGamma.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeGamma.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeGamma.output_template(index)

### `CompositorNodeGlare`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeGlare.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeGlare.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeGlare.output_template(index)

### `CompositorNodeGroup`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeGroup.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeGroup.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeGroup.output_template(index)

### `CompositorNodeHueCorrect`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeHueCorrect.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeHueCorrect.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeHueCorrect.output_template(index)

### `CompositorNodeHueSat`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeHueSat.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeHueSat.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeHueSat.output_template(index)

### `CompositorNodeIDMask`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeIDMask.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeIDMask.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeIDMask.output_template(index)

### `CompositorNodeImage`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeImage.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeImage.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeImage.output_template(index)

### `CompositorNodeInpaint`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeInpaint.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeInpaint.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeInpaint.output_template(index)

### `CompositorNodeInvert`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeInvert.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeInvert.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeInvert.output_template(index)

### `CompositorNodeKeying`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeKeying.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeKeying.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeKeying.output_template(index)

### `CompositorNodeKeyingScreen`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeKeyingScreen.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeKeyingScreen.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeKeyingScreen.output_template(index)

### `CompositorNodeKuwahara`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeKuwahara.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeKuwahara.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeKuwahara.output_template(index)

### `CompositorNodeLensdist`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeLensdist.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeLensdist.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeLensdist.output_template(index)

### `CompositorNodeLevels`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeLevels.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeLevels.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeLevels.output_template(index)

### `CompositorNodeLumaMatte`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeLumaMatte.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeLumaMatte.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeLumaMatte.output_template(index)

### `CompositorNodeMapRange`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMapRange.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMapRange.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMapRange.output_template(index)

### `CompositorNodeMapUV`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMapUV.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMapUV.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMapUV.output_template(index)

### `CompositorNodeMapValue`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMapValue.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMapValue.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMapValue.output_template(index)

### `CompositorNodeMask`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMask.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMask.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMask.output_template(index)

### `CompositorNodeMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMath.output_template(index)

### `CompositorNodeMixRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMixRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMixRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMixRGB.output_template(index)

### `CompositorNodeMovieClip`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMovieClip.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMovieClip.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMovieClip.output_template(index)

### `CompositorNodeMovieDistortion`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeMovieDistortion.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeMovieDistortion.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeMovieDistortion.output_template(index)

### `CompositorNodeNormal`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeNormal.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeNormal.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeNormal.output_template(index)

### `CompositorNodeNormalize`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeNormalize.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeNormalize.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeNormalize.output_template(index)

### `CompositorNodeOutputFile`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeOutputFile.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeOutputFile.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeOutputFile.output_template(index)

### `CompositorNodeOutputFileFileSlots`

### `CompositorNodeOutputFileLayerSlots`

### `CompositorNodePixelate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodePixelate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodePixelate.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodePixelate.output_template(index)

### `CompositorNodePlaneTrackDeform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodePlaneTrackDeform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodePlaneTrackDeform.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodePlaneTrackDeform.output_template(index)

### `CompositorNodePosterize`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodePosterize.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodePosterize.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodePosterize.output_template(index)

### `CompositorNodePremulKey`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodePremulKey.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodePremulKey.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodePremulKey.output_template(index)

### `CompositorNodeRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeRGB.output_template(index)

### `CompositorNodeRGBToBW`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeRGBToBW.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeRGBToBW.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeRGBToBW.output_template(index)

### `CompositorNodeRLayers`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeRLayers.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeRLayers.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeRLayers.output_template(index)

### `CompositorNodeRotate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeRotate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeRotate.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeRotate.output_template(index)

### `CompositorNodeScale`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeScale.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeScale.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeScale.output_template(index)

### `CompositorNodeSceneTime`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSceneTime.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSceneTime.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSceneTime.output_template(index)

### `CompositorNodeSepHSVA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSepHSVA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSepHSVA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSepHSVA.output_template(index)

### `CompositorNodeSepRGBA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSepRGBA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSepRGBA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSepRGBA.output_template(index)

### `CompositorNodeSepYCCA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSepYCCA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSepYCCA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSepYCCA.output_template(index)

### `CompositorNodeSepYUVA`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSepYUVA.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSepYUVA.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSepYUVA.output_template(index)

### `CompositorNodeSeparateColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSeparateColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSeparateColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSeparateColor.output_template(index)

### `CompositorNodeSeparateXYZ`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSeparateXYZ.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSeparateXYZ.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSeparateXYZ.output_template(index)

### `CompositorNodeSetAlpha`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSetAlpha.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSetAlpha.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSetAlpha.output_template(index)

### `CompositorNodeSplit`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSplit.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSplit.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSplit.output_template(index)

### `CompositorNodeStabilize`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeStabilize.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeStabilize.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeStabilize.output_template(index)

### `CompositorNodeSunBeams`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSunBeams.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSunBeams.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSunBeams.output_template(index)

### `CompositorNodeSwitch`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSwitch.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSwitch.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSwitch.output_template(index)

### `CompositorNodeSwitchView`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeSwitchView.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeSwitchView.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeSwitchView.output_template(index)

### `CompositorNodeTexture`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTexture.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTexture.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTexture.output_template(index)

### `CompositorNodeTime`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTime.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTime.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTime.output_template(index)

### `CompositorNodeTonemap`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTonemap.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTonemap.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTonemap.output_template(index)

### `CompositorNodeTrackPos`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTrackPos.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTrackPos.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTrackPos.output_template(index)

### `CompositorNodeTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTransform.output_template(index)

### `CompositorNodeTranslate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeTranslate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeTranslate.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeTranslate.output_template(index)

### `CompositorNodeTree`

### `CompositorNodeValToRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeValToRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeValToRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeValToRGB.output_template(index)

### `CompositorNodeValue`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeValue.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeValue.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeValue.output_template(index)

### `CompositorNodeVecBlur`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeVecBlur.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeVecBlur.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeVecBlur.output_template(index)

### `CompositorNodeViewer`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeViewer.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeViewer.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeViewer.output_template(index)

### `CompositorNodeZcombine`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  CompositorNodeZcombine.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  CompositorNodeZcombine.input_template(index)

- **`output_template(*args, **kwargs)`**
  CompositorNodeZcombine.output_template(index)

### `ConsoleLine`

### `Constraint`

### `ConstraintTarget`

### `ConstraintTargetBone`

### `Context`

#### Methods

- **`path_resolve(self, path, coerce=True)`**
  Returns the property from the path, raise an exception when not found.

- **`copy(self)`**

- **`temp_override(...)`**
  .. method:: temp_override(*, window=None, area=None, region=None, **keywords)

### `CopyLocationConstraint`

### `CopyRotationConstraint`

### `CopyScaleConstraint`

### `CopyTransformsConstraint`

### `CorrectiveSmoothModifier`

### `CrossStrip`

### `CryptomatteEntry`

### `Curve`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `CurveMap`

### `CurveMapPoint`

### `CurveMapPoints`

### `CurveMapping`

### `CurveModifier`

### `CurvePaintSettings`

### `CurvePoint`

### `CurveProfile`

### `CurveProfilePoint`

### `CurveProfilePoints`

### `CurveSlice`

### `CurveSplines`

### `Curves`

### `CurvesModifier`

### `CurvesSculpt`

### `DATA_PT_CURVES_attributes`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_EEVEE_light`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_EEVEE_light_distance`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_EEVEE_light_influence`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_EEVEE_light_shadow`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_active_spline`

Same as above but for curves only

#### Methods

- **`draw(self, context)`**

### `DATA_PT_armature_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_bone_collections`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_camera`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `DATA_PT_camera_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_camera_background_image`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_camera_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_camera_display_composition_guides`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_camera_dof`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_camera_dof_aperture`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_camera_safe_areas`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `DATA_PT_camera_safe_areas_center_cut`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_camera_stereoscopy`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_cone`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_arm`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_camera`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_curve`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_curves`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_grease_pencil`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_lattice`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_light`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_lightprobe`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_mesh`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_metaball`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_pointcloud`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_speaker`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_context_volume`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_curve_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `DATA_PT_curve_texture_space`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_curves_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_curves_surface`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_custom_props_arm`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_bcoll`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_camera`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_curve`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_curves`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_lattice`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_light`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_mesh`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_metaball`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_pointcloud`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_speaker`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_custom_props_volume`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_customdata`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_distance`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_empty`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_empty_image`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_font`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_font_transform`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_geometry_curve`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_geometry_curve_bevel`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_geometry_curve_start_end`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_grease_pencil_attributes`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DATA_PT_grease_pencil_layer_adjustments`

### `DATA_PT_grease_pencil_layer_display`

### `DATA_PT_grease_pencil_layer_group_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_layer_masks`

### `DATA_PT_grease_pencil_layer_relations`

### `DATA_PT_grease_pencil_layer_transform`

### `DATA_PT_grease_pencil_layers`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_onion_skinning`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_onion_skinning_custom_colors`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_onion_skinning_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_grease_pencil_settings`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_iksolver_itasc`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lattice`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lattice_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lens`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_light`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_light_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_lightprobe_bake`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_bake_capture`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_bake_clamping`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_bake_offset`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_bake_resolution`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_capture`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_display_eevee_next`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_eevee_next`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_lightprobe_parallax`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_lightprobe_visibility`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_mball_texture_space`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_mesh_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `DATA_PT_mesh_attributes`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_metaball`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_metaball_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_metaball_element`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_modifiers`

#### Methods

- **`draw(self, _context)`**

### `DATA_PT_motion_paths`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_motion_paths_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_paragraph`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_paragraph_alignment`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_paragraph_spacing`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_pathanim`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_PT_pointcloud_attributes`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_pose`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_preview`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_remesh`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_shader_fx`

#### Methods

- **`draw(self, _context)`**

### `DATA_PT_shape_curve`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_shape_keys`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_speaker`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_speaker_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_spot`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_text_boxes`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_texture_space`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_uv_texture`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_vertex_colors`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_vertex_groups`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_volume_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `DATA_PT_volume_file`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_volume_grids`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_volume_render`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_volume_viewport_display`

#### Methods

- **`draw(self, context)`**

### `DATA_PT_volume_viewport_display_slicing`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `DATA_UL_bone_collections`

#### Methods

- **`draw_item(self, _context, layout, armature, bcoll, _icon, _active_data, _active_propname, _index)`**

### `DOPESHEET_HT_header`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_action`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_channel`

#### Methods

- **`draw(self, _context)`**

### `DOPESHEET_MT_channel_context_menu`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_delete`

#### Methods

- **`draw(self, _context)`**

### `DOPESHEET_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_gpencil_channel`

#### Methods

- **`draw(self, _context)`**

### `DOPESHEET_MT_key`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_key_transform`

#### Methods

- **`draw(self, _context)`**

### `DOPESHEET_MT_marker`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_select`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `DOPESHEET_MT_view`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_action`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_action_slot`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_asset_panel`

#### Methods

- **`draw(self, context: bpy_types.Context) -> None`**

### `DOPESHEET_PT_custom_props_action`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `DOPESHEET_PT_filters`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_grease_pencil_layer_adjustments`

### `DOPESHEET_PT_grease_pencil_layer_display`

### `DOPESHEET_PT_grease_pencil_layer_masks`

### `DOPESHEET_PT_grease_pencil_layer_relations`

### `DOPESHEET_PT_grease_pencil_layer_transform`

### `DOPESHEET_PT_grease_pencil_mode`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_proportional_edit`

#### Methods

- **`draw(self, context)`**

### `DOPESHEET_PT_snapping`

#### Methods

- **`draw(self, context)`**

### `DampedTrackConstraint`

### `DataTransferModifier`

### `DecimateModifier`

### `Depsgraph`

### `DepsgraphObjectInstance`

### `DepsgraphUpdate`

### `DisplaceModifier`

### `DisplaySafeAreas`

### `DistortedNoiseTexture`

### `DopeSheet`

### `Driver`

### `DriverTarget`

### `DriverVariable`

### `DynamicPaintBrushSettings`

### `DynamicPaintCanvasSettings`

### `DynamicPaintModifier`

### `DynamicPaintSurface`

### `DynamicPaintSurfaces`

### `EEVEE_MATERIAL_PT_context_material`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_displacement`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_settings`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_surface`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_thickness`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_viewport_settings`

#### Methods

- **`draw(self, context)`**

### `EEVEE_MATERIAL_PT_volume`

#### Methods

- **`draw(self, context)`**

### `EEVEE_NEXT_MATERIAL_PT_settings`

#### Methods

- **`draw(self, context)`**

### `EEVEE_NEXT_MATERIAL_PT_settings_surface`

#### Methods

- **`draw(self, context)`**

### `EEVEE_NEXT_MATERIAL_PT_settings_volume`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_lightprobe`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_mist`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_settings`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_sun`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_sun_shadow`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_surface`

#### Methods

- **`draw(self, context)`**

### `EEVEE_WORLD_PT_volume`

#### Methods

- **`draw(self, context)`**

### `EQCurveMappingData`

### `EXPORT_ANIM_OT_bvh`

Save a BVH motion capture file from an armature

#### Methods

- **`invoke(self, context, event)`**

- **`execute(self, context)`**

- **`draw(self, context)`**

### `EXPORT_SCENE_OT_fbx`

Write a FBX file

#### Methods

- **`draw(self, context)`**

- **`execute(self, context)`**

### `EXPORT_SCENE_OT_gltf`

Export scene as glTF 2.0 file

### `EXTENSIONS_OT_dummy_progress`

Utility to execute mix-in.

Sub-class must define.
- bl_idname
- bl_label
- exec_command_iter
- exec_command_finish

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_package_disable`

Turn off this extension

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_enable_not_installed`

Turn on this extension

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_install`

Download and install the extension

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

- **`invoke(self, context, event)`**

- **`draw(self, context)`**

### `EXTENSIONS_OT_package_install_files`

Install extensions from files into a locally managed repository

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

- **`exec_legacy(self, filepath)`**

- **`invoke(self, context, event)`**

- **`draw(self, context)`**

### `EXTENSIONS_OT_package_install_marked`

Utility to execute mix-in.

Sub-class must define.
- bl_idname
- bl_label
- exec_command_iter
- exec_command_finish

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_package_mark_clear`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_mark_clear_all`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_mark_set`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_mark_set_all`

#### Methods

- **`execute(self, context)`**

### `EXTENSIONS_OT_package_obsolete_marked`

Zeroes package versions, useful for development - to test upgrading

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_show_clear`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_show_set`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_show_settings`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_theme_disable`

Turn off this theme

#### Methods

- **`execute(self, context)`**

### `EXTENSIONS_OT_package_theme_enable`

Turn off this theme

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_uninstall`

Disable and uninstall the extension

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_package_uninstall_marked`

Utility to execute mix-in.

Sub-class must define.
- bl_idname
- bl_label
- exec_command_iter
- exec_command_finish

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_package_uninstall_system`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_package_upgrade_all`

Upgrade all the extensions to their latest version for all the remote repositories

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_repo_enable_from_drop`

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, _context)`**

- **`draw(self, _context)`**

### `EXTENSIONS_OT_repo_lock_all`

Lock repositories - to test locking

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_repo_refresh_all`

Scan extension & legacy add-ons for changes to modules & meta-data (similar to restarting). Any issues are reported as warnings

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_repo_sync`

Utility to execute mix-in.

Sub-class must define.
- bl_idname
- bl_label
- exec_command_iter
- exec_command_finish

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_repo_sync_all`

Refresh the list of extensions for all the remote repositories

#### Methods

- **`exec_command_iter(self, is_modal)`**

- **`exec_command_finish(self, canceled)`**

### `EXTENSIONS_OT_repo_unlock`

Remove the repository file-system lock

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, _context)`**

- **`draw(self, _context)`**

### `EXTENSIONS_OT_repo_unlock_all`

Unlock repositories - to test unlocking

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_status_clear`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_status_clear_errors`

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_userpref_allow_online`

Allow internet access. Blender may access configured online extension repositories. Installed third party add-ons may access the internet for their own functionality

#### Methods

- **`execute(self, context)`**

### `EXTENSIONS_OT_userpref_allow_online_popup`

Allow internet access. Blender may access configured online extension repositories. Installed third party add-ons may access the internet for their own functionality

#### Methods

- **`execute(self, _context)`**

- **`invoke(self, context, _event)`**

- **`draw(self, _context)`**

### `EXTENSIONS_OT_userpref_show_for_update`

Open extensions preferences

#### Methods

- **`execute(self, context)`**

### `EXTENSIONS_OT_userpref_show_online`

Show system preferences "Network" panel to allow online access

#### Methods

- **`execute(self, _context)`**

### `EXTENSIONS_OT_userpref_tags_set`

Set the value of all tags

#### Methods

- **`execute(self, context)`**

### `EdgeSplitModifier`

### `EditBone`

functions for bones, common between Armature/Pose/Edit bones.
internal subclassing use only.

#### Methods

- **`align_orientation(self, other)`**
  Align this bone to another by moving its tail and settings its roll

- **`transform(self, matrix, *, scale=True, roll=True)`**
  Transform the bones head, tail, roll and envelope

### `EffectStrip`

### `EffectorWeights`

### `EnumProperty`

### `EnumPropertyItem`

### `Event`

### `ExplodeModifier`

### `FCurve`

### `FCurveKeyframePoints`

### `FCurveModifiers`

### `FCurveSample`

### `FFmpegSettings`

### `FILEBROWSER_HT_header`

#### Methods

- **`draw_asset_browser_buttons(self, context)`**

- **`draw(self, context)`**

### `FILEBROWSER_MT_bookmarks_context_menu`

#### Methods

- **`draw(self, _context)`**

### `FILEBROWSER_MT_bookmarks_recents_specials_menu`

#### Methods

- **`draw(self, _context)`**

### `FILEBROWSER_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_MT_editor_menus`

#### Methods

- **`draw(self, _context)`**

### `FILEBROWSER_MT_select`

#### Methods

- **`draw(self, _context)`**

### `FILEBROWSER_MT_view`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_advanced_filter`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_bookmarks_favorites`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_bookmarks_recents`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_bookmarks_system`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_bookmarks_volumes`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_directory_path`

#### Methods

- **`is_header_visible(self, context)`**

- **`draw(self, context)`**

### `FILEBROWSER_PT_display`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_PT_filter`

#### Methods

- **`draw(self, context)`**

### `FILEBROWSER_UL_dir`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `FLUID_OT_preset_add`

Add or remove a Fluid Preset

### `FLUID_PT_presets`

### `FModifier`

### `FModifierCycles`

### `FModifierEnvelope`

### `FModifierEnvelopeControlPoint`

### `FModifierEnvelopeControlPoints`

### `FModifierFunctionGenerator`

### `FModifierGenerator`

### `FModifierLimits`

### `FModifierNoise`

### `FModifierStepped`

### `FieldSettings`

### `FileAssetSelectIDFilter`

### `FileAssetSelectParams`

### `FileBrowserFSMenuEntry`

### `FileHandler`

### `FileSelectEntry`

### `FileSelectIDFilter`

### `FileSelectParams`

### `Float2Attribute`

### `Float2AttributeValue`

### `Float4x4Attribute`

### `Float4x4AttributeValue`

### `FloatAttribute`

### `FloatAttributeValue`

### `FloatColorAttribute`

### `FloatColorAttributeValue`

### `FloatProperty`

### `FloatVectorAttribute`

### `FloatVectorAttributeValue`

### `FloatVectorValueReadOnly`

### `FloorConstraint`

### `FluidDomainSettings`

### `FluidEffectorSettings`

### `FluidFlowSettings`

### `FluidModifier`

### `FollowPathConstraint`

### `FollowTrackConstraint`

### `ForeachGeometryElementGenerationItem`

### `ForeachGeometryElementInputItem`

### `ForeachGeometryElementMainItem`

### `ForeachGeometryElementZoneViewerPathElem`

### `FreestyleLineSet`

### `FreestyleLineStyle`

### `FreestyleModuleSettings`

### `FreestyleModules`

### `FreestyleSettings`

### `Function`

### `FunctionNode`

### `FunctionNodeAlignEulerToVector`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeAlignEulerToVector.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeAlignEulerToVector.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeAlignEulerToVector.output_template(index)

### `FunctionNodeAlignRotationToVector`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeAlignRotationToVector.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeAlignRotationToVector.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeAlignRotationToVector.output_template(index)

### `FunctionNodeAxesToRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeAxesToRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeAxesToRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeAxesToRotation.output_template(index)

### `FunctionNodeAxisAngleToRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeAxisAngleToRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeAxisAngleToRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeAxisAngleToRotation.output_template(index)

### `FunctionNodeBooleanMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeBooleanMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeBooleanMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeBooleanMath.output_template(index)

### `FunctionNodeCombineColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeCombineColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeCombineColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeCombineColor.output_template(index)

### `FunctionNodeCombineMatrix`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeCombineMatrix.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeCombineMatrix.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeCombineMatrix.output_template(index)

### `FunctionNodeCombineTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeCombineTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeCombineTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeCombineTransform.output_template(index)

### `FunctionNodeCompare`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeCompare.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeCompare.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeCompare.output_template(index)

### `FunctionNodeEulerToRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeEulerToRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeEulerToRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeEulerToRotation.output_template(index)

### `FunctionNodeFindInString`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeFindInString.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeFindInString.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeFindInString.output_template(index)

### `FunctionNodeFloatToInt`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeFloatToInt.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeFloatToInt.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeFloatToInt.output_template(index)

### `FunctionNodeHashValue`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeHashValue.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeHashValue.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeHashValue.output_template(index)

### `FunctionNodeInputBool`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputBool.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputBool.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputBool.output_template(index)

### `FunctionNodeInputColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputColor.output_template(index)

### `FunctionNodeInputInt`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputInt.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputInt.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputInt.output_template(index)

### `FunctionNodeInputRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputRotation.output_template(index)

### `FunctionNodeInputSpecialCharacters`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputSpecialCharacters.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputSpecialCharacters.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputSpecialCharacters.output_template(index)

### `FunctionNodeInputString`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputString.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputString.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputString.output_template(index)

### `FunctionNodeInputVector`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInputVector.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInputVector.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInputVector.output_template(index)

### `FunctionNodeIntegerMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeIntegerMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeIntegerMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeIntegerMath.output_template(index)

### `FunctionNodeInvertMatrix`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInvertMatrix.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInvertMatrix.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInvertMatrix.output_template(index)

### `FunctionNodeInvertRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeInvertRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeInvertRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeInvertRotation.output_template(index)

### `FunctionNodeMatrixDeterminant`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeMatrixDeterminant.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeMatrixDeterminant.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeMatrixDeterminant.output_template(index)

### `FunctionNodeMatrixMultiply`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeMatrixMultiply.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeMatrixMultiply.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeMatrixMultiply.output_template(index)

### `FunctionNodeProjectPoint`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeProjectPoint.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeProjectPoint.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeProjectPoint.output_template(index)

### `FunctionNodeQuaternionToRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeQuaternionToRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeQuaternionToRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeQuaternionToRotation.output_template(index)

### `FunctionNodeRandomValue`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRandomValue.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRandomValue.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRandomValue.output_template(index)

### `FunctionNodeReplaceString`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeReplaceString.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeReplaceString.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeReplaceString.output_template(index)

### `FunctionNodeRotateEuler`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotateEuler.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotateEuler.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotateEuler.output_template(index)

### `FunctionNodeRotateRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotateRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotateRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotateRotation.output_template(index)

### `FunctionNodeRotateVector`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotateVector.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotateVector.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotateVector.output_template(index)

### `FunctionNodeRotationToAxisAngle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotationToAxisAngle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotationToAxisAngle.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotationToAxisAngle.output_template(index)

### `FunctionNodeRotationToEuler`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotationToEuler.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotationToEuler.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotationToEuler.output_template(index)

### `FunctionNodeRotationToQuaternion`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeRotationToQuaternion.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeRotationToQuaternion.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeRotationToQuaternion.output_template(index)

### `FunctionNodeSeparateColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeSeparateColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeSeparateColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeSeparateColor.output_template(index)

### `FunctionNodeSeparateMatrix`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeSeparateMatrix.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeSeparateMatrix.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeSeparateMatrix.output_template(index)

### `FunctionNodeSeparateTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeSeparateTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeSeparateTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeSeparateTransform.output_template(index)

### `FunctionNodeSliceString`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeSliceString.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeSliceString.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeSliceString.output_template(index)

### `FunctionNodeStringLength`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeStringLength.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeStringLength.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeStringLength.output_template(index)

### `FunctionNodeTransformDirection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeTransformDirection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeTransformDirection.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeTransformDirection.output_template(index)

### `FunctionNodeTransformPoint`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeTransformPoint.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeTransformPoint.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeTransformPoint.output_template(index)

### `FunctionNodeTransposeMatrix`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeTransposeMatrix.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeTransposeMatrix.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeTransposeMatrix.output_template(index)

### `FunctionNodeValueToString`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  FunctionNodeValueToString.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  FunctionNodeValueToString.input_template(index)

- **`output_template(*args, **kwargs)`**
  FunctionNodeValueToString.output_template(index)

### `GPENCIL_MT_material_context_menu`

#### Methods

- **`draw(self, _context)`**

### `GPENCIL_OT_tint_flip`

Switch tint colors

#### Methods

- **`execute(self, context)`**

### `GPENCIL_UL_annotation_layer`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `GPENCIL_UL_layer`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `GPENCIL_UL_masks`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `GPENCIL_UL_matslots`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `GPencilFrame`

### `GPencilFrames`

### `GPencilInterpolateSettings`

### `GPencilLayer`

### `GPencilSculptGuide`

### `GPencilSculptSettings`

### `GPencilStroke`

### `GPencilStrokePoint`

### `GRAPH_HT_header`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_channel`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_delete`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_key`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_key_blending`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_key_density`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_key_smoothing`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_key_snap`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_key_transform`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_marker`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_pivot_pie`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_select`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `GRAPH_MT_view`

#### Methods

- **`draw(self, context)`**

### `GRAPH_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `GRAPH_PT_filters`

#### Methods

- **`draw(self, context)`**

### `GRAPH_PT_proportional_edit`

#### Methods

- **`draw(self, context)`**

### `GRAPH_PT_snapping`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_Layers`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_draw_delete`

#### Methods

- **`draw(self, _context)`**

### `GREASE_PENCIL_MT_grease_pencil_add_layer_extra`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_group_context_menu`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_layer_active`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_layer_mask_add`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_move_to_layer`

#### Methods

- **`draw(self, context)`**

### `GREASE_PENCIL_MT_snap`

#### Methods

- **`draw(self, _context)`**

### `GREASE_PENCIL_MT_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `GREASE_PENCIL_MT_stroke_simplify`

#### Methods

- **`draw(self, _context)`**

### `GREASE_PENCIL_UL_attributes`

#### Methods

- **`filter_items(self, _context, data, property)`**

- **`draw_item(self, _context, layout, _data, attribute, _icon, _active_data, _active_propname, _index)`**

### `GREASE_PENCIL_UL_masks`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `GammaCrossStrip`

### `GaussianBlurStrip`

### `GeometryNode`

### `GeometryNodeAccumulateField`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeAccumulateField.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeAccumulateField.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeAccumulateField.output_template(index)

### `GeometryNodeAttributeDomainSize`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeAttributeDomainSize.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeAttributeDomainSize.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeAttributeDomainSize.output_template(index)

### `GeometryNodeAttributeStatistic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeAttributeStatistic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeAttributeStatistic.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeAttributeStatistic.output_template(index)

### `GeometryNodeBake`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeBake.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeBake.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeBake.output_template(index)

### `GeometryNodeBlurAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeBlurAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeBlurAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeBlurAttribute.output_template(index)

### `GeometryNodeBoundBox`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeBoundBox.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeBoundBox.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeBoundBox.output_template(index)

### `GeometryNodeCaptureAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCaptureAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCaptureAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCaptureAttribute.output_template(index)

### `GeometryNodeCollectionInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCollectionInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCollectionInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCollectionInfo.output_template(index)

### `GeometryNodeConvexHull`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeConvexHull.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeConvexHull.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeConvexHull.output_template(index)

### `GeometryNodeCornersOfEdge`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCornersOfEdge.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCornersOfEdge.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCornersOfEdge.output_template(index)

### `GeometryNodeCornersOfFace`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCornersOfFace.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCornersOfFace.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCornersOfFace.output_template(index)

### `GeometryNodeCornersOfVertex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCornersOfVertex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCornersOfVertex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCornersOfVertex.output_template(index)

### `GeometryNodeCurveArc`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveArc.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveArc.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveArc.output_template(index)

### `GeometryNodeCurveEndpointSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveEndpointSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveEndpointSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveEndpointSelection.output_template(index)

### `GeometryNodeCurveHandleTypeSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveHandleTypeSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveHandleTypeSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveHandleTypeSelection.output_template(index)

### `GeometryNodeCurveLength`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveLength.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveLength.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveLength.output_template(index)

### `GeometryNodeCurveOfPoint`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveOfPoint.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveOfPoint.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveOfPoint.output_template(index)

### `GeometryNodeCurvePrimitiveBezierSegment`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveBezierSegment.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveBezierSegment.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveBezierSegment.output_template(index)

### `GeometryNodeCurvePrimitiveCircle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveCircle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveCircle.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveCircle.output_template(index)

### `GeometryNodeCurvePrimitiveLine`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveLine.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveLine.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveLine.output_template(index)

### `GeometryNodeCurvePrimitiveQuadrilateral`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveQuadrilateral.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveQuadrilateral.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurvePrimitiveQuadrilateral.output_template(index)

### `GeometryNodeCurveQuadraticBezier`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveQuadraticBezier.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveQuadraticBezier.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveQuadraticBezier.output_template(index)

### `GeometryNodeCurveSetHandles`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveSetHandles.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveSetHandles.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveSetHandles.output_template(index)

### `GeometryNodeCurveSpiral`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveSpiral.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveSpiral.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveSpiral.output_template(index)

### `GeometryNodeCurveSplineType`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveSplineType.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveSplineType.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveSplineType.output_template(index)

### `GeometryNodeCurveStar`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveStar.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveStar.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveStar.output_template(index)

### `GeometryNodeCurveToMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveToMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveToMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveToMesh.output_template(index)

### `GeometryNodeCurveToPoints`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurveToPoints.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurveToPoints.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurveToPoints.output_template(index)

### `GeometryNodeCurvesToGreasePencil`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeCurvesToGreasePencil.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeCurvesToGreasePencil.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeCurvesToGreasePencil.output_template(index)

### `GeometryNodeCustomGroup`

### `GeometryNodeDeformCurvesOnSurface`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDeformCurvesOnSurface.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDeformCurvesOnSurface.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDeformCurvesOnSurface.output_template(index)

### `GeometryNodeDeleteGeometry`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDeleteGeometry.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDeleteGeometry.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDeleteGeometry.output_template(index)

### `GeometryNodeDistributePointsInGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDistributePointsInGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDistributePointsInGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDistributePointsInGrid.output_template(index)

### `GeometryNodeDistributePointsInVolume`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDistributePointsInVolume.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDistributePointsInVolume.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDistributePointsInVolume.output_template(index)

### `GeometryNodeDistributePointsOnFaces`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDistributePointsOnFaces.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDistributePointsOnFaces.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDistributePointsOnFaces.output_template(index)

### `GeometryNodeDualMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDualMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDualMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDualMesh.output_template(index)

### `GeometryNodeDuplicateElements`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeDuplicateElements.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeDuplicateElements.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeDuplicateElements.output_template(index)

### `GeometryNodeEdgePathsToCurves`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeEdgePathsToCurves.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeEdgePathsToCurves.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeEdgePathsToCurves.output_template(index)

### `GeometryNodeEdgePathsToSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeEdgePathsToSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeEdgePathsToSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeEdgePathsToSelection.output_template(index)

### `GeometryNodeEdgesOfCorner`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeEdgesOfCorner.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeEdgesOfCorner.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeEdgesOfCorner.output_template(index)

### `GeometryNodeEdgesOfVertex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeEdgesOfVertex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeEdgesOfVertex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeEdgesOfVertex.output_template(index)

### `GeometryNodeEdgesToFaceGroups`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeEdgesToFaceGroups.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeEdgesToFaceGroups.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeEdgesToFaceGroups.output_template(index)

### `GeometryNodeExtrudeMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeExtrudeMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeExtrudeMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeExtrudeMesh.output_template(index)

### `GeometryNodeFaceOfCorner`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFaceOfCorner.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFaceOfCorner.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFaceOfCorner.output_template(index)

### `GeometryNodeFieldAtIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFieldAtIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFieldAtIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFieldAtIndex.output_template(index)

### `GeometryNodeFieldOnDomain`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFieldOnDomain.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFieldOnDomain.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFieldOnDomain.output_template(index)

### `GeometryNodeFillCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFillCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFillCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFillCurve.output_template(index)

### `GeometryNodeFilletCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFilletCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFilletCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFilletCurve.output_template(index)

### `GeometryNodeFlipFaces`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeFlipFaces.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeFlipFaces.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeFlipFaces.output_template(index)

### `GeometryNodeForeachGeometryElementInput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementInput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementInput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementInput.output_template(index)

### `GeometryNodeForeachGeometryElementOutput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementOutput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementOutput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeForeachGeometryElementOutput.output_template(index)

### `GeometryNodeGeometryToInstance`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGeometryToInstance.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGeometryToInstance.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGeometryToInstance.output_template(index)

### `GeometryNodeGetNamedGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGetNamedGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGetNamedGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGetNamedGrid.output_template(index)

### `GeometryNodeGizmoDial`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGizmoDial.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGizmoDial.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGizmoDial.output_template(index)

### `GeometryNodeGizmoLinear`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGizmoLinear.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGizmoLinear.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGizmoLinear.output_template(index)

### `GeometryNodeGizmoTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGizmoTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGizmoTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGizmoTransform.output_template(index)

### `GeometryNodeGreasePencilToCurves`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGreasePencilToCurves.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGreasePencilToCurves.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGreasePencilToCurves.output_template(index)

### `GeometryNodeGridToMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGridToMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGridToMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGridToMesh.output_template(index)

### `GeometryNodeGroup`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeGroup.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeGroup.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeGroup.output_template(index)

### `GeometryNodeImageInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeImageInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeImageInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeImageInfo.output_template(index)

### `GeometryNodeImageTexture`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeImageTexture.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeImageTexture.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeImageTexture.output_template(index)

### `GeometryNodeImportOBJ`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeImportOBJ.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeImportOBJ.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeImportOBJ.output_template(index)

### `GeometryNodeImportPLY`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeImportPLY.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeImportPLY.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeImportPLY.output_template(index)

### `GeometryNodeImportSTL`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeImportSTL.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeImportSTL.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeImportSTL.output_template(index)

### `GeometryNodeIndexOfNearest`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeIndexOfNearest.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeIndexOfNearest.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeIndexOfNearest.output_template(index)

### `GeometryNodeIndexSwitch`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeIndexSwitch.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeIndexSwitch.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeIndexSwitch.output_template(index)

### `GeometryNodeInputActiveCamera`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputActiveCamera.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputActiveCamera.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputActiveCamera.output_template(index)

### `GeometryNodeInputCollection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputCollection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputCollection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputCollection.output_template(index)

### `GeometryNodeInputCurveHandlePositions`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputCurveHandlePositions.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputCurveHandlePositions.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputCurveHandlePositions.output_template(index)

### `GeometryNodeInputCurveTilt`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputCurveTilt.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputCurveTilt.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputCurveTilt.output_template(index)

### `GeometryNodeInputEdgeSmooth`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputEdgeSmooth.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputEdgeSmooth.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputEdgeSmooth.output_template(index)

### `GeometryNodeInputID`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputID.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputID.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputID.output_template(index)

### `GeometryNodeInputImage`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputImage.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputImage.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputImage.output_template(index)

### `GeometryNodeInputIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputIndex.output_template(index)

### `GeometryNodeInputInstanceRotation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputInstanceRotation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputInstanceRotation.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputInstanceRotation.output_template(index)

### `GeometryNodeInputInstanceScale`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputInstanceScale.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputInstanceScale.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputInstanceScale.output_template(index)

### `GeometryNodeInputMaterial`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMaterial.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMaterial.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMaterial.output_template(index)

### `GeometryNodeInputMaterialIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMaterialIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMaterialIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMaterialIndex.output_template(index)

### `GeometryNodeInputMeshEdgeAngle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeAngle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeAngle.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeAngle.output_template(index)

### `GeometryNodeInputMeshEdgeNeighbors`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeNeighbors.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeNeighbors.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeNeighbors.output_template(index)

### `GeometryNodeInputMeshEdgeVertices`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeVertices.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeVertices.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshEdgeVertices.output_template(index)

### `GeometryNodeInputMeshFaceArea`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshFaceArea.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceArea.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceArea.output_template(index)

### `GeometryNodeInputMeshFaceIsPlanar`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshFaceIsPlanar.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceIsPlanar.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceIsPlanar.output_template(index)

### `GeometryNodeInputMeshFaceNeighbors`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshFaceNeighbors.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceNeighbors.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshFaceNeighbors.output_template(index)

### `GeometryNodeInputMeshIsland`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshIsland.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshIsland.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshIsland.output_template(index)

### `GeometryNodeInputMeshVertexNeighbors`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputMeshVertexNeighbors.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputMeshVertexNeighbors.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputMeshVertexNeighbors.output_template(index)

### `GeometryNodeInputNamedAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputNamedAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputNamedAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputNamedAttribute.output_template(index)

### `GeometryNodeInputNamedLayerSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputNamedLayerSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputNamedLayerSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputNamedLayerSelection.output_template(index)

### `GeometryNodeInputNormal`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputNormal.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputNormal.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputNormal.output_template(index)

### `GeometryNodeInputObject`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputObject.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputObject.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputObject.output_template(index)

### `GeometryNodeInputPosition`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputPosition.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputPosition.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputPosition.output_template(index)

### `GeometryNodeInputRadius`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputRadius.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputRadius.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputRadius.output_template(index)

### `GeometryNodeInputSceneTime`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputSceneTime.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputSceneTime.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputSceneTime.output_template(index)

### `GeometryNodeInputShadeSmooth`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputShadeSmooth.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputShadeSmooth.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputShadeSmooth.output_template(index)

### `GeometryNodeInputShortestEdgePaths`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputShortestEdgePaths.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputShortestEdgePaths.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputShortestEdgePaths.output_template(index)

### `GeometryNodeInputSplineCyclic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputSplineCyclic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputSplineCyclic.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputSplineCyclic.output_template(index)

### `GeometryNodeInputSplineResolution`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputSplineResolution.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputSplineResolution.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputSplineResolution.output_template(index)

### `GeometryNodeInputTangent`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInputTangent.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInputTangent.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInputTangent.output_template(index)

### `GeometryNodeInstanceOnPoints`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInstanceOnPoints.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInstanceOnPoints.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInstanceOnPoints.output_template(index)

### `GeometryNodeInstanceTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInstanceTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInstanceTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInstanceTransform.output_template(index)

### `GeometryNodeInstancesToPoints`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInstancesToPoints.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInstancesToPoints.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInstancesToPoints.output_template(index)

### `GeometryNodeInterpolateCurves`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeInterpolateCurves.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeInterpolateCurves.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeInterpolateCurves.output_template(index)

### `GeometryNodeIsViewport`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeIsViewport.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeIsViewport.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeIsViewport.output_template(index)

### `GeometryNodeJoinGeometry`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeJoinGeometry.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeJoinGeometry.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeJoinGeometry.output_template(index)

### `GeometryNodeMaterialSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMaterialSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMaterialSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMaterialSelection.output_template(index)

### `GeometryNodeMenuSwitch`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMenuSwitch.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMenuSwitch.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMenuSwitch.output_template(index)

### `GeometryNodeMergeByDistance`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMergeByDistance.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMergeByDistance.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMergeByDistance.output_template(index)

### `GeometryNodeMergeLayers`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMergeLayers.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMergeLayers.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMergeLayers.output_template(index)

### `GeometryNodeMeshBoolean`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshBoolean.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshBoolean.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshBoolean.output_template(index)

### `GeometryNodeMeshCircle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshCircle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshCircle.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshCircle.output_template(index)

### `GeometryNodeMeshCone`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshCone.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshCone.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshCone.output_template(index)

### `GeometryNodeMeshCube`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshCube.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshCube.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshCube.output_template(index)

### `GeometryNodeMeshCylinder`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshCylinder.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshCylinder.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshCylinder.output_template(index)

### `GeometryNodeMeshFaceSetBoundaries`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshFaceSetBoundaries.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshFaceSetBoundaries.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshFaceSetBoundaries.output_template(index)

### `GeometryNodeMeshGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshGrid.output_template(index)

### `GeometryNodeMeshIcoSphere`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshIcoSphere.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshIcoSphere.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshIcoSphere.output_template(index)

### `GeometryNodeMeshLine`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshLine.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshLine.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshLine.output_template(index)

### `GeometryNodeMeshToCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshToCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshToCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshToCurve.output_template(index)

### `GeometryNodeMeshToDensityGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshToDensityGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshToDensityGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshToDensityGrid.output_template(index)

### `GeometryNodeMeshToPoints`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshToPoints.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshToPoints.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshToPoints.output_template(index)

### `GeometryNodeMeshToSDFGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshToSDFGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshToSDFGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshToSDFGrid.output_template(index)

### `GeometryNodeMeshToVolume`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshToVolume.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshToVolume.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshToVolume.output_template(index)

### `GeometryNodeMeshUVSphere`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeMeshUVSphere.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeMeshUVSphere.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeMeshUVSphere.output_template(index)

### `GeometryNodeObjectInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeObjectInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeObjectInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeObjectInfo.output_template(index)

### `GeometryNodeOffsetCornerInFace`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeOffsetCornerInFace.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeOffsetCornerInFace.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeOffsetCornerInFace.output_template(index)

### `GeometryNodeOffsetPointInCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeOffsetPointInCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeOffsetPointInCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeOffsetPointInCurve.output_template(index)

### `GeometryNodePoints`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePoints.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePoints.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePoints.output_template(index)

### `GeometryNodePointsOfCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePointsOfCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePointsOfCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePointsOfCurve.output_template(index)

### `GeometryNodePointsToCurves`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePointsToCurves.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePointsToCurves.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePointsToCurves.output_template(index)

### `GeometryNodePointsToSDFGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePointsToSDFGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePointsToSDFGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePointsToSDFGrid.output_template(index)

### `GeometryNodePointsToVertices`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePointsToVertices.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePointsToVertices.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePointsToVertices.output_template(index)

### `GeometryNodePointsToVolume`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodePointsToVolume.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodePointsToVolume.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodePointsToVolume.output_template(index)

### `GeometryNodeProximity`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeProximity.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeProximity.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeProximity.output_template(index)

### `GeometryNodeRaycast`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRaycast.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRaycast.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRaycast.output_template(index)

### `GeometryNodeRealizeInstances`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRealizeInstances.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRealizeInstances.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRealizeInstances.output_template(index)

### `GeometryNodeRemoveAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRemoveAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRemoveAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRemoveAttribute.output_template(index)

### `GeometryNodeRepeatInput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRepeatInput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRepeatInput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRepeatInput.output_template(index)

### `GeometryNodeRepeatOutput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRepeatOutput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRepeatOutput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRepeatOutput.output_template(index)

### `GeometryNodeReplaceMaterial`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeReplaceMaterial.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeReplaceMaterial.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeReplaceMaterial.output_template(index)

### `GeometryNodeResampleCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeResampleCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeResampleCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeResampleCurve.output_template(index)

### `GeometryNodeReverseCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeReverseCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeReverseCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeReverseCurve.output_template(index)

### `GeometryNodeRotateInstances`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeRotateInstances.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeRotateInstances.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeRotateInstances.output_template(index)

### `GeometryNodeSDFGridBoolean`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSDFGridBoolean.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSDFGridBoolean.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSDFGridBoolean.output_template(index)

### `GeometryNodeSampleCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleCurve.output_template(index)

### `GeometryNodeSampleGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleGrid.output_template(index)

### `GeometryNodeSampleGridIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleGridIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleGridIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleGridIndex.output_template(index)

### `GeometryNodeSampleIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleIndex.output_template(index)

### `GeometryNodeSampleNearest`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleNearest.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleNearest.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleNearest.output_template(index)

### `GeometryNodeSampleNearestSurface`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleNearestSurface.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleNearestSurface.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleNearestSurface.output_template(index)

### `GeometryNodeSampleUVSurface`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSampleUVSurface.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSampleUVSurface.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSampleUVSurface.output_template(index)

### `GeometryNodeScaleElements`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeScaleElements.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeScaleElements.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeScaleElements.output_template(index)

### `GeometryNodeScaleInstances`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeScaleInstances.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeScaleInstances.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeScaleInstances.output_template(index)

### `GeometryNodeSelfObject`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSelfObject.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSelfObject.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSelfObject.output_template(index)

### `GeometryNodeSeparateComponents`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSeparateComponents.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSeparateComponents.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSeparateComponents.output_template(index)

### `GeometryNodeSeparateGeometry`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSeparateGeometry.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSeparateGeometry.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSeparateGeometry.output_template(index)

### `GeometryNodeSetCurveHandlePositions`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetCurveHandlePositions.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetCurveHandlePositions.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetCurveHandlePositions.output_template(index)

### `GeometryNodeSetCurveNormal`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetCurveNormal.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetCurveNormal.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetCurveNormal.output_template(index)

### `GeometryNodeSetCurveRadius`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetCurveRadius.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetCurveRadius.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetCurveRadius.output_template(index)

### `GeometryNodeSetCurveTilt`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetCurveTilt.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetCurveTilt.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetCurveTilt.output_template(index)

### `GeometryNodeSetGeometryName`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetGeometryName.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetGeometryName.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetGeometryName.output_template(index)

### `GeometryNodeSetID`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetID.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetID.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetID.output_template(index)

### `GeometryNodeSetInstanceTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetInstanceTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetInstanceTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetInstanceTransform.output_template(index)

### `GeometryNodeSetMaterial`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetMaterial.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetMaterial.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetMaterial.output_template(index)

### `GeometryNodeSetMaterialIndex`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetMaterialIndex.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetMaterialIndex.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetMaterialIndex.output_template(index)

### `GeometryNodeSetPointRadius`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetPointRadius.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetPointRadius.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetPointRadius.output_template(index)

### `GeometryNodeSetPosition`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetPosition.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetPosition.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetPosition.output_template(index)

### `GeometryNodeSetShadeSmooth`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetShadeSmooth.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetShadeSmooth.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetShadeSmooth.output_template(index)

### `GeometryNodeSetSplineCyclic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetSplineCyclic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetSplineCyclic.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetSplineCyclic.output_template(index)

### `GeometryNodeSetSplineResolution`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSetSplineResolution.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSetSplineResolution.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSetSplineResolution.output_template(index)

### `GeometryNodeSimulationInput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSimulationInput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSimulationInput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSimulationInput.output_template(index)

### `GeometryNodeSimulationOutput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSimulationOutput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSimulationOutput.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSimulationOutput.output_template(index)

### `GeometryNodeSortElements`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSortElements.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSortElements.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSortElements.output_template(index)

### `GeometryNodeSplineLength`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSplineLength.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSplineLength.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSplineLength.output_template(index)

### `GeometryNodeSplineParameter`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSplineParameter.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSplineParameter.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSplineParameter.output_template(index)

### `GeometryNodeSplitEdges`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSplitEdges.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSplitEdges.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSplitEdges.output_template(index)

### `GeometryNodeSplitToInstances`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSplitToInstances.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSplitToInstances.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSplitToInstances.output_template(index)

### `GeometryNodeStoreNamedAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeStoreNamedAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeStoreNamedAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeStoreNamedAttribute.output_template(index)

### `GeometryNodeStoreNamedGrid`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeStoreNamedGrid.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeStoreNamedGrid.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeStoreNamedGrid.output_template(index)

### `GeometryNodeStringJoin`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeStringJoin.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeStringJoin.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeStringJoin.output_template(index)

### `GeometryNodeStringToCurves`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeStringToCurves.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeStringToCurves.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeStringToCurves.output_template(index)

### `GeometryNodeSubdivideCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSubdivideCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSubdivideCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSubdivideCurve.output_template(index)

### `GeometryNodeSubdivideMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSubdivideMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSubdivideMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSubdivideMesh.output_template(index)

### `GeometryNodeSubdivisionSurface`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSubdivisionSurface.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSubdivisionSurface.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSubdivisionSurface.output_template(index)

### `GeometryNodeSwitch`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeSwitch.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeSwitch.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeSwitch.output_template(index)

### `GeometryNodeTool3DCursor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeTool3DCursor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeTool3DCursor.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeTool3DCursor.output_template(index)

### `GeometryNodeToolActiveElement`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolActiveElement.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolActiveElement.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolActiveElement.output_template(index)

### `GeometryNodeToolFaceSet`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolFaceSet.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolFaceSet.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolFaceSet.output_template(index)

### `GeometryNodeToolMousePosition`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolMousePosition.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolMousePosition.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolMousePosition.output_template(index)

### `GeometryNodeToolSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolSelection.output_template(index)

### `GeometryNodeToolSetFaceSet`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolSetFaceSet.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolSetFaceSet.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolSetFaceSet.output_template(index)

### `GeometryNodeToolSetSelection`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeToolSetSelection.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeToolSetSelection.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeToolSetSelection.output_template(index)

### `GeometryNodeTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeTransform.output_template(index)

### `GeometryNodeTranslateInstances`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeTranslateInstances.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeTranslateInstances.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeTranslateInstances.output_template(index)

### `GeometryNodeTree`

### `GeometryNodeTriangulate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeTriangulate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeTriangulate.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeTriangulate.output_template(index)

### `GeometryNodeTrimCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeTrimCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeTrimCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeTrimCurve.output_template(index)

### `GeometryNodeUVPackIslands`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeUVPackIslands.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeUVPackIslands.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeUVPackIslands.output_template(index)

### `GeometryNodeUVUnwrap`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeUVUnwrap.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeUVUnwrap.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeUVUnwrap.output_template(index)

### `GeometryNodeVertexOfCorner`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeVertexOfCorner.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeVertexOfCorner.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeVertexOfCorner.output_template(index)

### `GeometryNodeViewer`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeViewer.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeViewer.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeViewer.output_template(index)

### `GeometryNodeViewportTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeViewportTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeViewportTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeViewportTransform.output_template(index)

### `GeometryNodeVolumeCube`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeVolumeCube.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeVolumeCube.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeVolumeCube.output_template(index)

### `GeometryNodeVolumeToMesh`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeVolumeToMesh.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeVolumeToMesh.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeVolumeToMesh.output_template(index)

### `GeometryNodeWarning`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  GeometryNodeWarning.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  GeometryNodeWarning.input_template(index)

- **`output_template(*args, **kwargs)`**
  GeometryNodeWarning.output_template(index)

### `Gizmo`

#### Methods

- **`target_set_handler(...)`**
  .. method:: target_set_handler(target, get, set, range=None):

- **`target_get_value(...)`**
  .. method:: target_get_value(target):

- **`target_set_value(...)`**
  .. method:: target_set_value(target):

- **`target_get_range(...)`**
  .. method:: target_get_range(target):

- **`draw_custom_shape(self, shape, *, matrix=None, select_id=None)`**
  Draw a shape created form :class:`Gizmo.draw_custom_shape`.

### `GizmoGroup`

### `GizmoGroupProperties`

### `GizmoProperties`

### `Gizmos`

### `GlowStrip`

### `GpPaint`

### `GpSculptPaint`

### `GpVertexPaint`

### `GpWeightPaint`

### `GreasePencil`

### `GreasePencilArmatureModifier`

### `GreasePencilArrayModifier`

### `GreasePencilBuildModifier`

### `GreasePencilColorModifier`

### `GreasePencilDashModifierData`

### `GreasePencilDashModifierSegment`

### `GreasePencilDrawing`

### `GreasePencilEnvelopeModifier`

### `GreasePencilFrame`

### `GreasePencilFrames`

### `GreasePencilHookModifier`

### `GreasePencilLatticeModifier`

### `GreasePencilLayer`

### `GreasePencilLayerGroup`

### `GreasePencilLayerMask`

### `GreasePencilLayerMasks`

### `GreasePencilLayers`

### `GreasePencilLengthModifier`

### `GreasePencilLineartModifier`

### `GreasePencilMirrorModifier`

### `GreasePencilMultiplyModifier`

### `GreasePencilNoiseModifier`

### `GreasePencilOffsetModifier`

### `GreasePencilOpacityModifier`

### `GreasePencilOutlineModifier`

### `GreasePencilShrinkwrapModifier`

### `GreasePencilSimplifyModifier`

### `GreasePencilSmoothModifier`

### `GreasePencilSubdivModifier`

### `GreasePencilTextureModifier`

### `GreasePencilThickModifierData`

### `GreasePencilTimeModifier`

### `GreasePencilTimeModifierSegment`

### `GreasePencilTintModifier`

### `GreasePencilWeightAngleModifier`

### `GreasePencilWeightProximityModifier`

### `GreasePencilv3`

### `GreasePencilv3LayerGroup`

### `GreasePencilv3Layers`

### `GroupNodeViewerPathElem`

### `Header`

### `Histogram`

### `HookModifier`

### `HueCorrectModifier`

### `HydraRenderEngine`

#### Methods

- **`get_render_settings(self, engine_type: str)`**
  Provide render settings for `HdRenderDelegate`.

- **`update(self, data, depsgraph)`**

- **`render(self, depsgraph)`**

- **`view_update(self, context, depsgraph)`**

- **`view_draw(self, context, depsgraph)`**

### `ID`

### `IDMaterials`

### `IDOverrideLibrary`

### `IDOverrideLibraryProperties`

### `IDOverrideLibraryProperty`

### `IDOverrideLibraryPropertyOperation`

### `IDOverrideLibraryPropertyOperations`

### `IDPropertyWrapPtr`

### `IDViewerPathElem`

### `IKParam`

### `IMAGE_AST_brush_paint`

### `IMAGE_FH_drop_handler`

### `IMAGE_HT_header`

#### Methods

- **`draw_xform_template(layout, context)`**

- **`draw(self, context)`**

### `IMAGE_HT_tool_header`

#### Methods

- **`draw(self, context)`**

- **`draw_tool_settings(self, context)`**

- **`draw_mode_settings(self, context)`**

### `IMAGE_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_image`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_image_invert`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_image_transform`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_mask_context_menu`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_pivot_pie`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_select`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_select_linked`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_uvs_align`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_context_menu`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_uvs_merge`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_mirror`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_select_mode`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_uvs_showhide`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_snap`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_split`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_transform`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_uvs_unwrap`

#### Methods

- **`draw(self, _context)`**

### `IMAGE_MT_view`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `IMAGE_MT_view_zoom`

#### Methods

- **`draw(self, context)`**

### `IMAGE_OT_convert_to_mesh_plane`

Convert selected reference images to textured mesh plane

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, context)`**

- **`draw(self, context)`**

### `IMAGE_OT_external_edit`

Edit image in an external application

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `IMAGE_OT_import_as_mesh_planes`

Create mesh plane(s) from image files with the appropriate aspect ratio

#### Methods

- **`update_size_mode(self, _context)`**

- **`draw_import_config(self, _context)`**

- **`draw_spatial_config(self, _context)`**

- **`draw(self, context)`**

- **`invoke(self, context, _event)`**

### `IMAGE_OT_open_images`

#### Methods

- **`execute(self, context)`**

### `IMAGE_OT_project_apply`

Project edited image back onto the object

#### Methods

- **`execute(self, _context)`**

### `IMAGE_OT_project_edit`

Edit a snapshot of the 3D Viewport in an external image editor

#### Methods

- **`execute(self, context)`**

### `IMAGE_PT_active_mask_point`

### `IMAGE_PT_active_mask_spline`

### `IMAGE_PT_active_tool`

### `IMAGE_PT_annotation`

### `IMAGE_PT_gizmo_display`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_image_properties`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_mask`

### `IMAGE_PT_mask_animation`

### `IMAGE_PT_mask_display`

### `IMAGE_PT_mask_layers`

### `IMAGE_PT_overlay`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_overlay_guides`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_overlay_image`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_overlay_texture_paint`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_overlay_uv_edit_geometry`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_overlay_uv_stretch`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_paint_clone`

### `IMAGE_PT_paint_color`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_paint_curve`

### `IMAGE_PT_paint_select`

### `IMAGE_PT_paint_settings`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_paint_settings_advanced`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_paint_stroke`

### `IMAGE_PT_paint_stroke_smooth_stroke`

### `IMAGE_PT_paint_swatches`

### `IMAGE_PT_proportional_edit`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_render_slots`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_sample_line`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_scope_sample`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_snapping`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_tools_active`

Generic Class, can be used for any toolbar.

- keymap_prefix:
  The text prefix for each key-map for this spaces tools.
- tools_all():
  Generator (context_mode, tools) tuple pairs for all tools defined.
- tools_from_context(context, mode=None):
  A generator for all tools available in the current context.

Tool Sequence Structure
=======================

Sequences of tools as returned by tools_all() and tools_from_context() are comprised of:

- A `ToolDef` instance (representing a tool that can be activated).
- None (a visual separator in the tool list).
- A tuple of `ToolDef` or None values
  (representing a group of tools that can be selected between using a click-drag action).
  Note that only a single level of nesting is supported (groups cannot contain sub-groups).
- A callable which takes a single context argument and returns a tuple of values described above.
  When the context is None, all potential tools must be returned.

### `IMAGE_PT_tools_brush_display`

### `IMAGE_PT_tools_brush_texture`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_tools_imagepaint_symmetry`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_tools_mask_texture`

### `IMAGE_PT_udim_tiles`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_uv_cursor`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_uv_sculpt_curve`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_uv_sculpt_options`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_view_display`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_view_histogram`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_view_vectorscope`

#### Methods

- **`draw(self, context)`**

### `IMAGE_PT_view_waveform`

#### Methods

- **`draw(self, context)`**

### `IMAGE_UL_render_slots`

#### Methods

- **`draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index)`**

### `IMAGE_UL_udim_tiles`

#### Methods

- **`draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index)`**

### `IMPORT_ANIM_OT_bvh`

Load a BVH motion capture file

#### Methods

- **`execute(self, context)`**

- **`draw(self, context)`**

### `IMPORT_CURVE_OT_svg`

Load a SVG file

#### Methods

- **`execute(self, context)`**

### `IMPORT_SCENE_OT_fbx`

Load a FBX file

#### Methods

- **`draw(self, context)`**

- **`execute(self, context)`**

- **`invoke(self, context, event)`**

### `IMPORT_SCENE_OT_gltf`

Load a glTF 2.0 file

#### Methods

- **`draw(self, context)`**

- **`invoke(self, context, event)`**

- **`execute(self, context)`**

- **`import_gltf2(self, context)`**

- **`unit_import(self, filename, import_settings)`**

### `INFO_HT_header`

#### Methods

- **`draw(self, context)`**

### `INFO_MT_area`

#### Methods

- **`draw(self, context)`**

### `INFO_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `INFO_MT_editor_menus`

#### Methods

- **`draw(self, _context)`**

### `INFO_MT_info`

#### Methods

- **`draw(self, _context)`**

### `INFO_MT_view`

#### Methods

- **`draw(self, _context)`**

### `IO_FH_fbx`

### `IO_FH_gltf2`

### `Image`

### `ImageFormatSettings`

### `ImagePackedFile`

### `ImagePaint`

### `ImagePreview`

### `ImageStrip`

### `ImageTexture`

### `ImageUser`

### `IndexSwitchItem`

### `Int2Attribute`

### `Int2AttributeValue`

### `IntAttribute`

### `IntAttributeValue`

### `IntProperty`

### `Itasc`

### `Key`

### `KeyConfig`

### `KeyConfigPreferences`

### `KeyConfigurations`

### `KeyMap`

### `KeyMapItem`

### `KeyMapItems`

### `KeyMaps`

### `Keyframe`

### `KeyingSet`

### `KeyingSetInfo`

### `KeyingSetPath`

### `KeyingSetPaths`

### `KeyingSets`

### `KeyingSetsAll`

### `KinematicConstraint`

### `LaplacianDeformModifier`

### `LaplacianSmoothModifier`

### `Lattice`

### `LatticeModifier`

### `LatticePoint`

### `LayerCollection`

### `LayerObjects`

### `LayoutPanelState`

### `Library`

### `LibraryWeakReference`

### `Light`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `LightProbe`

### `LightProbePlane`

### `LightProbeSphere`

### `LightProbeVolume`

### `Lightgroup`

### `Lightgroups`

### `LimitDistanceConstraint`

### `LimitLocationConstraint`

### `LimitRotationConstraint`

### `LimitScaleConstraint`

### `LineStyleAlphaModifier`

### `LineStyleAlphaModifier_AlongStroke`

### `LineStyleAlphaModifier_CreaseAngle`

### `LineStyleAlphaModifier_Curvature_3D`

### `LineStyleAlphaModifier_DistanceFromCamera`

### `LineStyleAlphaModifier_DistanceFromObject`

### `LineStyleAlphaModifier_Material`

### `LineStyleAlphaModifier_Noise`

### `LineStyleAlphaModifier_Tangent`

### `LineStyleAlphaModifiers`

### `LineStyleColorModifier`

### `LineStyleColorModifier_AlongStroke`

### `LineStyleColorModifier_CreaseAngle`

### `LineStyleColorModifier_Curvature_3D`

### `LineStyleColorModifier_DistanceFromCamera`

### `LineStyleColorModifier_DistanceFromObject`

### `LineStyleColorModifier_Material`

### `LineStyleColorModifier_Noise`

### `LineStyleColorModifier_Tangent`

### `LineStyleColorModifiers`

### `LineStyleGeometryModifier`

### `LineStyleGeometryModifier_2DOffset`

### `LineStyleGeometryModifier_2DTransform`

### `LineStyleGeometryModifier_BackboneStretcher`

### `LineStyleGeometryModifier_BezierCurve`

### `LineStyleGeometryModifier_Blueprint`

### `LineStyleGeometryModifier_GuidingLines`

### `LineStyleGeometryModifier_PerlinNoise1D`

### `LineStyleGeometryModifier_PerlinNoise2D`

### `LineStyleGeometryModifier_Polygonalization`

### `LineStyleGeometryModifier_Sampling`

### `LineStyleGeometryModifier_Simplification`

### `LineStyleGeometryModifier_SinusDisplacement`

### `LineStyleGeometryModifier_SpatialNoise`

### `LineStyleGeometryModifier_TipRemover`

### `LineStyleGeometryModifiers`

### `LineStyleModifier`

### `LineStyleTextureSlot`

### `LineStyleTextureSlots`

#### Methods

- **`add(*args, **kwargs)`**
  LineStyleTextureSlots.add()

- **`create(*args, **kwargs)`**
  LineStyleTextureSlots.create(index)

- **`clear(*args, **kwargs)`**
  LineStyleTextureSlots.clear(index)

### `LineStyleThicknessModifier`

### `LineStyleThicknessModifier_AlongStroke`

### `LineStyleThicknessModifier_Calligraphy`

### `LineStyleThicknessModifier_CreaseAngle`

### `LineStyleThicknessModifier_Curvature_3D`

### `LineStyleThicknessModifier_DistanceFromCamera`

### `LineStyleThicknessModifier_DistanceFromObject`

### `LineStyleThicknessModifier_Material`

### `LineStyleThicknessModifier_Noise`

### `LineStyleThicknessModifier_Tangent`

### `LineStyleThicknessModifiers`

### `Linesets`

### `LockedTrackConstraint`

### `LoopColors`

### `MASK_MT_add`

#### Methods

- **`draw(self, _context)`**

### `MASK_MT_animation`

#### Methods

- **`draw(self, _context)`**

### `MASK_MT_mask`

#### Methods

- **`draw(self, _context)`**

### `MASK_MT_select`

#### Methods

- **`draw(self, _context)`**

### `MASK_MT_transform`

#### Methods

- **`draw(self, _context)`**

### `MASK_MT_visibility`

#### Methods

- **`draw(self, _context)`**

### `MASK_UL_layers`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `MATERIAL_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `MATERIAL_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `MATERIAL_PT_freestyle_line`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_gpencil_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `MATERIAL_PT_gpencil_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `MATERIAL_PT_gpencil_fillcolor`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `MATERIAL_PT_gpencil_material_presets`

Material settings

### `MATERIAL_PT_gpencil_preview`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_gpencil_settings`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_gpencil_slots`

### `MATERIAL_PT_gpencil_strokecolor`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `MATERIAL_PT_gpencil_surface`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, _context)`**

### `MATERIAL_PT_lineart`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_preview`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_PT_viewport`

#### Methods

- **`draw(self, context)`**

### `MATERIAL_UL_matslots`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `MESH_MT_attribute_context_menu`

#### Methods

- **`draw(self, _context)`**

### `MESH_MT_color_attribute_context_menu`

#### Methods

- **`draw(self, _context)`**

### `MESH_MT_shape_key_context_menu`

#### Methods

- **`draw(self, _context)`**

### `MESH_MT_vertex_group_context_menu`

#### Methods

- **`draw(self, _context)`**

### `MESH_OT_faces_mirror_uv`

Copy mirror UV coordinates on the X axis based on a mirrored mesh

#### Methods

- **`do_mesh_mirror_UV(self, mesh, DIR)`**

- **`execute(self, context)`**

### `MESH_OT_primitive_torus_add`

Construct a torus mesh

#### Methods

- **`mode_update_callback(self, _context)`**

- **`draw(self, _context)`**

- **`invoke(self, context, _event)`**

- **`execute(self, context)`**

### `MESH_OT_select_next_item`

Select the next element (using selection order)

#### Methods

- **`execute(self, context)`**

### `MESH_OT_select_prev_item`

Select the previous element (using selection order)

#### Methods

- **`execute(self, context)`**

### `MESH_UL_attributes`

#### Methods

- **`filter_items(self, _context, data, property)`**

- **`draw_item(self, _context, layout, _data, attribute, _icon, _active_data, _active_propname, _index)`**

### `MESH_UL_color_attributes`

#### Methods

- **`draw_item(self, _context, layout, data, attribute, _icon, _active_data, _active_propname, _index)`**

### `MESH_UL_color_attributes_selector`

#### Methods

- **`draw_item(self, _context, layout, _data, attribute, _icon, _active_data, _active_propname, _index)`**

### `MESH_UL_shape_keys`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, active_data, _active_propname, index)`**

### `MESH_UL_uvmaps`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `MESH_UL_vgroups`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data_, _active_propname, _index)`**

### `Macro`

### `MagicTexture`

### `MaintainVolumeConstraint`

### `MarbleTexture`

### `Mask`

### `MaskLayer`

### `MaskLayers`

### `MaskModifier`

### `MaskParent`

### `MaskSpline`

### `MaskSplinePoint`

### `MaskSplinePointUW`

### `MaskSplinePoints`

### `MaskSplines`

### `MaskStrip`

### `Material`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `MaterialGPencilStyle`

### `MaterialLineArt`

### `MaterialSlot`

### `Menu`

#### Methods

- **`path_menu(self, searchpaths, operator, *, props_default=None, prop_filepath='filepath', filter_ext=None, filter_path=None, display_name=None, add_operator=None, add_operator_props=None)`**
  Populate a menu from a list of paths.

- **`draw_preset(self, _context)`**
  Define these on the subclass:

### `Mesh`

#### Methods

- **`from_pydata(self, vertices, edges, faces, shade_flat=True)`**
  Make a mesh from a list of vertices/edges/faces

- **`vertex_creases_ensure(self)`**

- **`vertex_creases_remove(self)`**

- **`edge_creases_ensure(self)`**

- **`edge_creases_remove(self)`**

### `MeshCacheModifier`

### `MeshDeformModifier`

### `MeshEdge`

### `MeshEdges`

### `MeshLoop`

### `MeshLoopColor`

### `MeshLoopColorLayer`

### `MeshLoopTriangle`

### `MeshLoopTriangles`

### `MeshLoops`

### `MeshNormalValue`

### `MeshPolygon`

### `MeshPolygons`

### `MeshSequenceCacheModifier`

### `MeshSkinVertex`

### `MeshSkinVertexLayer`

### `MeshStatVis`

### `MeshToVolumeModifier`

### `MeshUVLoop`

### `MeshUVLoopLayer`

### `MeshVertex`

### `MeshVertices`

### `MetaBall`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `MetaBallElements`

### `MetaElement`

### `MetaStrip`

### `MirrorModifier`

### `Modifier`

### `ModifierViewerPathElem`

### `MotionPath`

### `MotionPathVert`

### `MovieClip`

### `MovieClipProxy`

### `MovieClipScopes`

### `MovieClipStrip`

### `MovieClipUser`

### `MovieReconstructedCamera`

### `MovieStrip`

### `MovieTracking`

### `MovieTrackingCamera`

### `MovieTrackingDopesheet`

### `MovieTrackingMarker`

### `MovieTrackingMarkers`

### `MovieTrackingObject`

### `MovieTrackingObjectPlaneTracks`

### `MovieTrackingObjectTracks`

### `MovieTrackingObjects`

### `MovieTrackingPlaneMarker`

### `MovieTrackingPlaneMarkers`

### `MovieTrackingPlaneTrack`

### `MovieTrackingPlaneTracks`

### `MovieTrackingReconstructedCameras`

### `MovieTrackingReconstruction`

### `MovieTrackingSettings`

### `MovieTrackingStabilization`

### `MovieTrackingTrack`

### `MovieTrackingTracks`

### `MulticamStrip`

### `MultiplyStrip`

### `MultiresModifier`

### `MusgraveTexture`

### `NLA_HT_header`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_add`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_channel_context_menu`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_marker`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_marker_select`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_select`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_strips`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_strips_transform`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_tracks`

#### Methods

- **`draw(self, _context)`**

### `NLA_MT_view`

#### Methods

- **`draw(self, context)`**

### `NLA_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `NLA_OT_bake`

Bake all selected objects location/scale/rotation animation to an action

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `NLA_PT_action`

#### Methods

- **`draw(self, context)`**

### `NLA_PT_filters`

#### Methods

- **`draw(self, context)`**

### `NLA_PT_snapping`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_LIGHT_PT_beam_shape`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_LIGHT_PT_light`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_MATERIAL_PT_settings`

#### Methods

- **`draw_shared(self, mat)`**

- **`draw(self, context)`**

### `NODE_CYCLES_MATERIAL_PT_settings_surface`

#### Methods

- **`draw_shared(self, mat)`**

- **`draw(self, context)`**

### `NODE_CYCLES_MATERIAL_PT_settings_volume`

#### Methods

- **`draw_shared(self, context, mat)`**

- **`draw(self, context)`**

### `NODE_CYCLES_WORLD_PT_ray_visibility`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_WORLD_PT_settings`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_WORLD_PT_settings_surface`

#### Methods

- **`draw(self, context)`**

### `NODE_CYCLES_WORLD_PT_settings_volume`

#### Methods

- **`draw(self, context)`**

### `NODE_DATA_PT_EEVEE_light`

#### Methods

- **`draw(self, context)`**

### `NODE_DATA_PT_light`

#### Methods

- **`draw(self, context)`**

### `NODE_EEVEE_NEXT_MATERIAL_PT_settings`

#### Methods

- **`draw(self, context)`**

### `NODE_EEVEE_NEXT_MATERIAL_PT_settings_surface`

#### Methods

- **`draw(self, context)`**

### `NODE_EEVEE_NEXT_MATERIAL_PT_settings_volume`

#### Methods

- **`draw(self, context)`**

### `NODE_FH_image_node`

### `NODE_HT_header`

#### Methods

- **`draw(self, context)`**

### `NODE_MATERIAL_PT_viewport`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_add`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_GEO_GROUP`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_GEO_OUTPUT`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_POINT`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_GEO_TEXT`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_TEXTURE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_UTILITIES`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_GEO_UTILITIES_DEPRECATED`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_GEO_UTILITIES_FIELD`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_UTILITIES_MATH`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_UTILITIES_ROTATION`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_UV`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_VECTOR`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_GEO_VOLUME`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_PRIMITIVES_MESH`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_color`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_color_adjust`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_color_mix`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_filter`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_filter_blur`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_group`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_compositor_input`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_compositor_input_constant`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_input_scene`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_keying`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_mask`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_output`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_compositor_tracking`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_transform`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_utilities`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_compositor_vector`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_import`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_layout`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_shader_color`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_shader_converter`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_shader_group`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_shader_input`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_shader_output`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_shader_script`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_shader_shader`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_shader_texture`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_shader_vector`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_simulation`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_color`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_converter`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_distort`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_group`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_category_texture_input`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_output`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_pattern`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_texture_texture`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_category_utilities_matrix`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_compositor_node_add_all`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_context_menu_select_menu`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_context_menu_show_hide_menu`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_editor_menus`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_ATTRIBUTE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_COLOR`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_CURVE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_CURVE_OPERATIONS`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_CURVE_READ`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_CURVE_SAMPLE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_CURVE_WRITE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_GEOMETRY`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_GEOMETRY_OPERATIONS`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_GEOMETRY_READ`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_GEOMETRY_SAMPLE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_GEOMETRY_WRITE`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_INPUT`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_INPUT_CONSTANT`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_INPUT_GIZMO`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_INPUT_GROUP`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_INPUT_SCENE`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_INSTANCE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_MATERIAL`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_MESH`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_MESH_OPERATIONS`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_MESH_READ`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_MESH_SAMPLE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_MESH_WRITE`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_PRIMITIVES_CURVE`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_GEO_VOLUME_OPERATIONS`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_VOLUME_PRIMITIVES`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_VOLUME_READ`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_GEO_VOLUME_WRITE`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_add_all`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_geometry_node_curve_topology`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_mesh_topology`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_geometry_node_volume_sample`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_node`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_node_color_context_menu`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_node_tree_interface_context_menu`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_select`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_shader_node_add_all`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_texture_node_add_all`

#### Methods

- **`draw(self, _context)`**

### `NODE_MT_view`

#### Methods

- **`draw(self, context)`**

### `NODE_MT_view_pie`

#### Methods

- **`draw(self, _context)`**

### `NODE_OT_add_foreach_geometry_element_zone`

Add a For Each Geometry Element zone that allows executing nodes e.g. for each vertex separately

### `NODE_OT_add_node`

Add a node to the active tree

#### Methods

- **`execute(self, context)`**

### `NODE_OT_add_repeat_zone`

Add a repeat zone that allows executing nodes a dynamic number of times

### `NODE_OT_add_simulation_zone`

Add simulation zone input and output nodes to the active tree

### `NODE_OT_collapse_hide_unused_toggle`

Toggle collapsed nodes and hide unused sockets

#### Methods

- **`execute(self, context)`**

### `NODE_OT_connect_to_output`

#### Methods

- **`get_output_sockets(node_tree)`**

- **`init_shader_variables(self, space, shader_type)`**
  Get correct output node in shader editor

- **`ensure_viewer_socket(self, node_tree, socket_type, connect_socket=None)`**
  Check if a viewer output already exists in a node group, otherwise create it

- **`ensure_group_output(node_tree)`**
  Check if a group output node exists, otherwise create it

- **`remove_socket(tree, socket)`**

### `NODE_OT_gltf_settings_node_operator`

#### Methods

- **`execute(self, context)`**

### `NODE_OT_interface_item_duplicate`

Add a copy of the active item to the interface

#### Methods

- **`execute(self, context)`**

### `NODE_OT_interface_item_new`

Add a new item to the interface

#### Methods

- **`find_valid_socket_type(tree)`**

- **`execute(self, context)`**

### `NODE_OT_interface_item_remove`

Remove active item from the interface

#### Methods

- **`execute(self, context)`**

### `NODE_OT_new_geometry_node_group_assign`

Create a new geometry node group and assign it to the active modifier

#### Methods

- **`execute(self, context)`**

### `NODE_OT_new_geometry_node_group_tool`

Create a new geometry node group for a tool

#### Methods

- **`execute(self, context)`**

### `NODE_OT_new_geometry_nodes_modifier`

Create a new modifier with a new geometry node group

#### Methods

- **`execute(self, context)`**

### `NODE_OT_node_color_preset_add`

Add or remove a Node Color Preset

### `NODE_OT_tree_path_parent`

Go to parent node tree

#### Methods

- **`execute(self, context)`**

### `NODE_OT_viewer_shortcut_get`

Activate a specific compositor viewer node using 1,2,..,9 keys

#### Methods

- **`execute(self, context)`**

### `NODE_OT_viewer_shortcut_set`

Create a compositor viewer shortcut for the selected node by pressing ctrl+1,2,..9

#### Methods

- **`get_connected_viewer(self, node)`**

- **`execute(self, context)`**

### `NODE_PT_active_node_color`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `NODE_PT_active_node_generic`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_active_node_properties`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_active_tool`

### `NODE_PT_annotation`

### `NODE_PT_backdrop`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `NODE_PT_geometry_node_tool_mode`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_geometry_node_tool_object_types`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_geometry_node_tool_options`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_material_slots`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `NODE_PT_node_color_presets`

Predefined node color

### `NODE_PT_node_tree_interface`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_node_tree_properties`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_overlay`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_quality`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_texture_mapping`

#### Methods

- **`draw(self, context)`**

### `NODE_PT_tools_active`

Generic Class, can be used for any toolbar.

- keymap_prefix:
  The text prefix for each key-map for this spaces tools.
- tools_all():
  Generator (context_mode, tools) tuple pairs for all tools defined.
- tools_from_context(context, mode=None):
  A generator for all tools available in the current context.

Tool Sequence Structure
=======================

Sequences of tools as returned by tools_all() and tools_from_context() are comprised of:

- A `ToolDef` instance (representing a tool that can be activated).
- None (a visual separator in the tool list).
- A tuple of `ToolDef` or None values
  (representing a group of tools that can be selected between using a click-drag action).
  Note that only a single level of nesting is supported (groups cannot contain sub-groups).
- A callable which takes a single context argument and returns a tuple of values described above.
  When the context is None, all potential tools must be returned.

### `NODE_WORLD_PT_viewport_display`

#### Methods

- **`draw(self, context)`**

### `NlaStrip`

### `NlaStripFCurves`

### `NlaStrips`

### `NlaTrack`

### `NlaTracks`

### `Node`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  Node.is_registered_node_type()

### `NodeCustomGroup`

### `NodeEnumItem`

### `NodeFrame`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  NodeFrame.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  NodeFrame.input_template(index)

- **`output_template(*args, **kwargs)`**
  NodeFrame.output_template(index)

### `NodeGeometryBakeItem`

### `NodeGeometryBakeItems`

### `NodeGeometryCaptureAttributeItem`

### `NodeGeometryCaptureAttributeItems`

### `NodeGeometryForeachGeometryElementGenerationItems`

### `NodeGeometryForeachGeometryElementInputItems`

### `NodeGeometryForeachGeometryElementMainItems`

### `NodeGeometryRepeatOutputItems`

### `NodeGeometrySimulationOutputItems`

### `NodeGroup`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  NodeGroup.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  NodeGroup.input_template(index)

- **`output_template(*args, **kwargs)`**
  NodeGroup.output_template(index)

### `NodeGroupInput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  NodeGroupInput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  NodeGroupInput.input_template(index)

- **`output_template(*args, **kwargs)`**
  NodeGroupInput.output_template(index)

### `NodeGroupOutput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  NodeGroupOutput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  NodeGroupOutput.input_template(index)

- **`output_template(*args, **kwargs)`**
  NodeGroupOutput.output_template(index)

### `NodeIndexSwitchItems`

### `NodeInputs`

### `NodeInstanceHash`

### `NodeInternal`

#### Methods

- **`poll(*args, **kwargs)`**
  NodeInternal.poll(node_tree)

### `NodeInternalSocketTemplate`

### `NodeLink`

### `NodeLinks`

### `NodeMenuSwitchItems`

### `NodeOutputFileSlotFile`

### `NodeOutputFileSlotLayer`

### `NodeOutputs`

### `NodeReroute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  NodeReroute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  NodeReroute.input_template(index)

- **`output_template(*args, **kwargs)`**
  NodeReroute.output_template(index)

### `NodeSocket`

### `NodeSocketBool`

### `NodeSocketCollection`

### `NodeSocketColor`

### `NodeSocketFloat`

### `NodeSocketFloatAngle`

### `NodeSocketFloatColorTemperature`

### `NodeSocketFloatDistance`

### `NodeSocketFloatFactor`

### `NodeSocketFloatFrequency`

### `NodeSocketFloatPercentage`

### `NodeSocketFloatTime`

### `NodeSocketFloatTimeAbsolute`

### `NodeSocketFloatUnsigned`

### `NodeSocketFloatWavelength`

### `NodeSocketGeometry`

### `NodeSocketImage`

### `NodeSocketInt`

### `NodeSocketIntFactor`

### `NodeSocketIntPercentage`

### `NodeSocketIntUnsigned`

### `NodeSocketMaterial`

### `NodeSocketMatrix`

### `NodeSocketMenu`

### `NodeSocketObject`

### `NodeSocketRotation`

### `NodeSocketShader`

### `NodeSocketStandard`

### `NodeSocketString`

### `NodeSocketStringFilePath`

### `NodeSocketTexture`

### `NodeSocketVector`

### `NodeSocketVectorAcceleration`

### `NodeSocketVectorDirection`

### `NodeSocketVectorEuler`

### `NodeSocketVectorTranslation`

### `NodeSocketVectorVelocity`

### `NodeSocketVectorXYZ`

### `NodeSocketVirtual`

### `NodeTree`

### `NodeTreeInterface`

### `NodeTreeInterfaceItem`

### `NodeTreeInterfacePanel`

### `NodeTreeInterfaceSocket`

### `NodeTreeInterfaceSocketBool`

### `NodeTreeInterfaceSocketCollection`

### `NodeTreeInterfaceSocketColor`

### `NodeTreeInterfaceSocketFloat`

### `NodeTreeInterfaceSocketFloatAngle`

### `NodeTreeInterfaceSocketFloatColorTemperature`

### `NodeTreeInterfaceSocketFloatDistance`

### `NodeTreeInterfaceSocketFloatFactor`

### `NodeTreeInterfaceSocketFloatFrequency`

### `NodeTreeInterfaceSocketFloatPercentage`

### `NodeTreeInterfaceSocketFloatTime`

### `NodeTreeInterfaceSocketFloatTimeAbsolute`

### `NodeTreeInterfaceSocketFloatUnsigned`

### `NodeTreeInterfaceSocketFloatWavelength`

### `NodeTreeInterfaceSocketGeometry`

### `NodeTreeInterfaceSocketImage`

### `NodeTreeInterfaceSocketInt`

### `NodeTreeInterfaceSocketIntFactor`

### `NodeTreeInterfaceSocketIntPercentage`

### `NodeTreeInterfaceSocketIntUnsigned`

### `NodeTreeInterfaceSocketMaterial`

### `NodeTreeInterfaceSocketMatrix`

### `NodeTreeInterfaceSocketMenu`

### `NodeTreeInterfaceSocketObject`

### `NodeTreeInterfaceSocketRotation`

### `NodeTreeInterfaceSocketShader`

### `NodeTreeInterfaceSocketString`

### `NodeTreeInterfaceSocketStringFilePath`

### `NodeTreeInterfaceSocketTexture`

### `NodeTreeInterfaceSocketVector`

### `NodeTreeInterfaceSocketVectorAcceleration`

### `NodeTreeInterfaceSocketVectorDirection`

### `NodeTreeInterfaceSocketVectorEuler`

### `NodeTreeInterfaceSocketVectorTranslation`

### `NodeTreeInterfaceSocketVectorVelocity`

### `NodeTreeInterfaceSocketVectorXYZ`

### `NodeTreePath`

### `Nodes`

### `NodesModifier`

### `NodesModifierBake`

### `NodesModifierBakeDataBlocks`

### `NodesModifierBakes`

### `NodesModifierDataBlock`

### `NodesModifierPanel`

### `NodesModifierPanels`

### `NodesModifierWarning`

### `NoiseTexture`

### `NormalEditModifier`

### `OBJECT_MT_light_linking_context_menu`

#### Methods

- **`draw(self, _context)`**

### `OBJECT_MT_modifier_add`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_color`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_deform`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_edit`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_generate`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_normals`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_modifier_add_physics`

#### Methods

- **`draw(self, context)`**

### `OBJECT_MT_shadow_linking_context_menu`

#### Methods

- **`draw(self, _context)`**

### `OBJECT_OT_add_modifier_menu`

#### Methods

- **`invoke(self, _context, _event)`**

### `OBJECT_OT_align`

Align objects

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_anim_transforms_to_deltas`

Convert object animation for normal transforms to delta transforms

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_assign_property_defaults`

Assign the current values of custom properties as their defaults, for use as part of the rest pose state in NLA track mixing

#### Methods

- **`assign_defaults(obj)`**

- **`execute(self, context)`**

### `OBJECT_OT_geometry_nodes_move_to_nodes`

Move inputs and outputs from in the modifier to a new node group

#### Methods

- **`invoke(self, context, event)`**

- **`execute(self, context)`**

### `OBJECT_OT_hide_render_clear_all`

Reveal all render objects by setting the hide render flag

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_instance_offset_from_cursor`

Set offset used for collection instances based on cursor position

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_instance_offset_from_object`

Set offset used for collection instances based on the active object position

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_instance_offset_to_cursor`

Set cursor position to the offset used for collection instances

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_isolate_type_render`

Hide unselected render objects of same type as active by setting the hide render flag

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_join_uvs`

Transfer UV Maps from active to selected objects (needs matching geometry)

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_make_dupli_face`

Convert objects into instanced faces

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_quick_explode`

Make selected objects explode

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `OBJECT_OT_quick_fur`

Add a fur setup to the selected objects

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_quick_liquid`

Make selected objects liquid

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_quick_smoke`

Use selected objects as smoke emitters

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_randomize_transform`

Randomize objects location, rotation, and scale

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_select_camera`

Select the active camera

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_select_hierarchy`

Select object relative to the active object's position in the hierarchy

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_select_pattern`

Select objects matching a naming pattern

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, event)`**

- **`draw(self, _context)`**

### `OBJECT_OT_shape_key_transfer`

Copy the active shape key of another selected object to this one

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_subdivision_set`

Sets a Subdivision Surface level (1 to 5)

#### Methods

- **`execute(self, context)`**

### `OBJECT_OT_transforms_to_deltas`

Convert normal object transforms to delta transforms, any existing delta transforms will be included as well

#### Methods

- **`execute(self, context)`**

- **`transfer_location(self, obj)`**

- **`transfer_rotation(self, obj)`**

- **`transfer_scale(self, obj)`**

### `OBJECT_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `OBJECT_PT_bActionConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bActionConstraint_action`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bActionConstraint_target`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bArmatureConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bArmatureConstraint_bones`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bCameraSolverConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bChildOfConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bClampToConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bDampTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bDistLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bFollowPathConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bFollowTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bKinematicConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bLocLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bLocateLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bLockTrackConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bMinMaxConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bObjectSolverConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bPivotConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bPythonConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bRotLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bRotateLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bSameVolumeConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bShrinkwrapConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bSizeLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bSizeLimitConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bStretchToConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTrackToConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransLikeConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformCacheConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformCacheConstraint_layers`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformCacheConstraint_procedural`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformCacheConstraint_time`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformCacheConstraint_velocity`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformConstraint`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformConstraint_destination`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_bTransformConstraint_source`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_collections`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_constraints`

#### Methods

- **`draw(self, _context)`**

### `OBJECT_PT_context_object`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `OBJECT_PT_delta_transform`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_display`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_instancing`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_instancing_size`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `OBJECT_PT_light_linking`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_lineart`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_motion_paths`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_motion_paths_display`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_relations`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_shading`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_shadow_linking`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_transform`

#### Methods

- **`draw(self, context)`**

### `OBJECT_PT_visibility`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_HT_header`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_MT_asset`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_MT_collection`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_MT_collection_new`

#### Methods

- **`draw_without_context_menu(_context, layout)`**

- **`draw(self, context)`**

### `OUTLINER_MT_collection_view_layer`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_MT_collection_visibility`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_MT_context_menu`

#### Methods

- **`draw_common_operators(layout)`**

- **`draw(self, context)`**

### `OUTLINER_MT_context_menu_view`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_MT_edit_datablocks`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_MT_liboverride`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_MT_object`

#### Methods

- **`draw(self, context)`**

### `OUTLINER_MT_view_pie`

#### Methods

- **`draw(self, _context)`**

### `OUTLINER_PT_filter`

#### Methods

- **`draw(self, context)`**

### `Object`

#### Methods

- **`selection_sets(*args, **kwargs)`**
  Intermediate storage for properties before registration.

- **`active_selection_set(*args, **kwargs)`**
  Intermediate storage for properties before registration.

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `ObjectBase`

### `ObjectConstraints`

### `ObjectDisplay`

### `ObjectLightLinking`

### `ObjectLineArt`

### `ObjectModifiers`

### `ObjectShaderFx`

### `ObjectSolverConstraint`

### `OceanModifier`

### `Operator`

#### Methods

- **`as_keywords(self, *, ignore=())`**
  Return a copy of the properties as a dictionary

- **`poll_message_set(...)`**
  .. classmethod:: poll_message_set(message, *args)

### `OperatorFileListElement`

### `OperatorMacro`

### `OperatorMousePath`

### `OperatorOptions`

### `OperatorProperties`

### `OperatorStrokeElement`

### `OverDropStrip`

### `PAINT_OT_vertex_color_dirt`

Generate a dirt map gradient based on cavity

#### Methods

- **`execute(self, context)`**

### `PARTICLE_MT_context_menu`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_OT_hair_dynamics_preset_add`

Add or remove a Hair Dynamics Preset

### `PARTICLE_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `PARTICLE_PT_boidbrain`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_cache`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_children`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_children_clumping`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_children_clumping_noise`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PARTICLE_PT_children_kink`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_children_parting`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_children_roughness`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_context_particles`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `PARTICLE_PT_draw`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_emission`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_emission_source`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_field_weights`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_force_fields`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_force_fields_type1`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_force_fields_type1_falloff`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_force_fields_type2`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_force_fields_type2_falloff`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_hair_dynamics`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `PARTICLE_PT_hair_dynamics_collision`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_hair_dynamics_presets`

### `PARTICLE_PT_hair_dynamics_structure`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_hair_dynamics_volume`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_hair_shape`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_boids_battle`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_boids_misc`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_boids_movement`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_deflection`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_fluid_advanced`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_fluid_interaction`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_fluid_springs`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_fluid_springs_advanced`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_fluid_springs_viscoelastic`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PARTICLE_PT_physics_forces`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_integration`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_physics_relations`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render_collection`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render_collection_use_count`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PARTICLE_PT_render_extra`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render_object`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render_path`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_render_path_timing`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_rotation`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PARTICLE_PT_rotation_angular_velocity`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_textures`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_velocity`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_PT_vertexgroups`

#### Methods

- **`draw(self, context)`**

### `PARTICLE_UL_particle_systems`

#### Methods

- **`draw_item(self, _context, layout, data, item, icon, _active_data, _active_propname, _index, _flt_flag)`**

### `PHYSICS_PT_adaptive_domain`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_add`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_borders`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cache`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_cache`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_collision`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_damping`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_field_weights`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_internal_springs`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_object_collision`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_physical_properties`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_pressure`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_property_weights`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_self_collision`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_shape`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_cloth_stiffness`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_collections`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_collision`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_collision_particle`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_collision_softbody`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_diffusion`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_source`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_source_color_ramp`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_velocity`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_velocity_color_ramp`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_velocity_smudge`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_brush_wave`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_cache`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_canvas_initial_color`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_canvas_output`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_canvas_output_paintmaps`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_canvas_output_wetmaps`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_effects`

#### Methods

- **`draw(self, _context)`**

### `PHYSICS_PT_dp_effects_drip`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_effects_drip_weights`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_effects_shrink`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_effects_spread`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_surface_canvas`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dp_surface_canvas_paint_dissolve`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dp_surface_canvas_paint_dry`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_dynamic_paint`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_dynamic_paint_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_export`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_falloff`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_falloff_angular`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_falloff_radial`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_settings_kink`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_settings_texture_select`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_field_weights`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_fire`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_flow_initial_velocity`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_flow_source`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_flow_texture`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_fluid`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_fluid_domain_render`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_geometry_nodes`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_guide`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_liquid`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_mesh`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_noise`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_particles`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_collisions`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_collisions_collections`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_collisions_sensitivity`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_collisions_surface`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_limits`

#### Methods

- **`draw(self, _context)`**

### `PHYSICS_PT_rigid_body_constraint_limits_angular`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_limits_linear`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_motor`

#### Methods

- **`draw(self, _context)`**

### `PHYSICS_PT_rigid_body_constraint_motor_angular`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_motor_linear`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_objects`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_override_iterations`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_springs`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_springs_angular`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_constraint_springs_linear`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_dynamics`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_dynamics_deactivation`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_rigid_body_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_smoke`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_smoke_dissolve`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_softbody`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_cache`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_collision`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_edge`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_edge_aerodynamics`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_edge_stiffness`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_field_weights`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_goal`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_goal_settings`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_goal_strengths`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_object`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_simulation`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_solver`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_solver_diagnostics`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_softbody_solver_helpers`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_viewport_display`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_viewport_display_advanced`

#### Methods

- **`draw(self, context)`**

### `PHYSICS_PT_viewport_display_color`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_viewport_display_debug`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_viewport_display_slicing`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_PT_viscosity`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `PHYSICS_UL_dynapaint_surfaces`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `POINTCLOUD_MT_add_attribute`

#### Methods

- **`add_standard_attribute(layout, pointcloud, name, data_type, domain)`**

- **`draw(self, context)`**

### `POINTCLOUD_UL_attributes`

#### Methods

- **`filter_items(self, _context, data, property)`**

- **`draw_item(self, _context, layout, _data, attribute, _icon, _active_data, _active_propname, _index)`**

### `POSELIB_OT_convert_old_object_poselib`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

### `POSELIB_OT_convert_old_poselib`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

### `POSELIB_OT_copy_as_asset`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

- **`save_datablock(self, action: bpy.types.Action) -> pathlib.Path`**

### `POSELIB_OT_paste_asset`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

### `POSELIB_OT_pose_asset_select_bones`

#### Methods

- **`use_pose(self, context: bpy_types.Context, pose_asset: bpy.types.Action) -> Set[str]`**

### `POSELIB_OT_restore_previous_action`

#### Methods

- **`execute(self, context: bpy_types.Context) -> Set[str]`**

- **`modal(self, context, event)`**

### `POSE_MT_selection_set_create`

#### Methods

- **`draw(self, _context)`**

### `POSE_MT_selection_sets_context_menu`

#### Methods

- **`draw(self, context)`**

### `POSE_MT_selection_sets_select`

#### Methods

- **`draw(self, context)`**

### `POSE_OT_selection_set_add`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_add_and_assign`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_assign`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, context)`**

### `POSE_OT_selection_set_copy`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_delete_all`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_deselect`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_move`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_paste`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_remove`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_remove_bones`

Operator only available for objects of type armature in pose mode.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_select`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_OT_selection_set_unassign`

Operator only available if the armature has a selected selection set.

#### Methods

- **`execute(self, context)`**

### `POSE_PT_selection_sets`

#### Methods

- **`draw(self, context)`**

### `POSE_UL_selection_set`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `PREFERENCES_OT_addon_disable`

Turn off this add-on

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_addon_enable`

Turn on this add-on

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_addon_expand`

Display information and preferences for this add-on

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_addon_install`

Install an add-on

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_addon_refresh`

Scan add-on directories for new modules

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_addon_remove`

Delete the add-on from the file system

#### Methods

- **`path_from_addon(module)`**

- **`execute(self, context)`**

- **`draw(self, _context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_addon_show`

Show add-on preferences

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_app_template_install`

Install an application template

#### Methods

- **`execute(self, _context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_copy_prev`

Copy settings from previous version

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_keyconfig_activate`

#### Methods

- **`execute(self, _context)`**

### `PREFERENCES_OT_keyconfig_export`

Export key configuration to a Python script

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_keyconfig_import`

Import key configuration from a Python script

#### Methods

- **`execute(self, _context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_keyconfig_remove`

Remove key config

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_keyconfig_test`

Test key configuration for conflicts

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_keyitem_add`

Add key map item

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_keyitem_remove`

Remove key map item

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_keyitem_restore`

Restore key map item

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_keymap_restore`

Restore key map(s)

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_script_directory_add`

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_script_directory_remove`

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_studiolight_copy_settings`

Copy Studio Light settings to the Studio Light editor

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_studiolight_install`

Install a user defined light

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_studiolight_new`

Save custom studio light from the studio light editor settings

#### Methods

- **`execute(self, context)`**

- **`draw(self, _context)`**

- **`invoke(self, context, _event)`**

### `PREFERENCES_OT_studiolight_uninstall`

Delete Studio Light

#### Methods

- **`execute(self, context)`**

### `PREFERENCES_OT_theme_install`

Load and apply a Blender XML theme file

#### Methods

- **`execute(self, _context)`**

- **`invoke(self, context, _event)`**

### `PROPERTIES_HT_header`

#### Methods

- **`draw(self, context)`**

### `PROPERTIES_PT_navigation_bar`

#### Methods

- **`draw(self, context)`**

### `PROPERTIES_PT_options`

Show options for the properties editor

#### Methods

- **`draw(self, context)`**

### `PackedFile`

### `Paint`

### `PaintCurve`

### `PaintModeSettings`

### `Palette`

### `PaletteColor`

### `PaletteColors`

### `Panel`

### `Particle`

### `ParticleBrush`

### `ParticleDupliWeight`

### `ParticleEdit`

### `ParticleHairKey`

### `ParticleInstanceModifier`

### `ParticleKey`

### `ParticleSettings`

### `ParticleSettingsTextureSlot`

### `ParticleSettingsTextureSlots`

#### Methods

- **`add(*args, **kwargs)`**
  ParticleSettingsTextureSlots.add()

- **`create(*args, **kwargs)`**
  ParticleSettingsTextureSlots.create(index)

- **`clear(*args, **kwargs)`**
  ParticleSettingsTextureSlots.clear(index)

### `ParticleSystem`

### `ParticleSystemModifier`

### `ParticleSystems`

### `ParticleTarget`

### `PathCompare`

### `PathCompareCollection`

#### Methods

- **`new(*args, **kwargs)`**
  PathCompareCollection.new()

- **`remove(*args, **kwargs)`**
  PathCompareCollection.remove(pathcmp)

### `PivotConstraint`

### `Point`

### `PointCache`

### `PointCacheItem`

### `PointCaches`

### `PointCloud`

### `PointLight`

### `PointerProperty`

### `Pose`

#### Methods

- **`apply_pose_from_action(*args, **kwargs)`**
  Pose.apply_pose_from_action(action, evaluation_time=0)

- **`blend_pose_from_action(*args, **kwargs)`**
  Pose.blend_pose_from_action(action, blend_factor=1, evaluation_time=0)

- **`backup_create(*args, **kwargs)`**
  Pose.backup_create(action)

- **`backup_restore(*args, **kwargs)`**
  Pose.backup_restore()

- **`backup_clear(*args, **kwargs)`**
  Pose.backup_clear()

### `PoseBone`

functions for bones, common between Armature/Pose/Edit bones.
internal subclassing use only.

### `PoseBoneConstraints`

### `Preferences`

### `PreferencesApps`

### `PreferencesEdit`

### `PreferencesExperimental`

### `PreferencesExtensions`

### `PreferencesFilePaths`

### `PreferencesInput`

### `PreferencesKeymap`

### `PreferencesSystem`

### `PreferencesView`

### `PrimitiveBoolean`

### `PrimitiveFloat`

### `PrimitiveInt`

### `PrimitiveString`

### `Property`

### `PropertyGroup`

### `PropertyGroupItem`

### `PythonConstraint`

### `QuaternionAttribute`

### `QuaternionAttributeValue`

### `RENDER_MT_framerate_presets`

#### Methods

- **`draw(self, _context)`**
  Define these on the subclass:

### `RENDER_MT_lineset_context_menu`

#### Methods

- **`draw(self, _context)`**

### `RENDER_OT_color_management_white_balance_preset_add`

Add or remove a white balance preset

### `RENDER_OT_cycles_integrator_preset_add`

Add an Integrator Preset

### `RENDER_OT_cycles_performance_preset_add`

Add an Performance Preset

### `RENDER_OT_cycles_sampling_preset_add`

Add a Sampling Preset

### `RENDER_OT_cycles_viewport_sampling_preset_add`

Add a Viewport Sampling Preset

### `RENDER_OT_eevee_raytracing_preset_add`

Add or remove an EEVEE ray-tracing preset

### `RENDER_OT_play_rendered_anim`

Play back rendered frames/movies using an external player

#### Methods

- **`execute(self, context)`**

### `RENDER_OT_preset_add`

Add or remove a Render Preset

### `RENDER_PT_color_management`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_color_management_curves`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_color_management_display_settings`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_color_management_white_balance`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_color_management_white_balance_presets`

### `RENDER_PT_context`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_hair`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_clamping`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_clamping_surface`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_clamping_volume`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_denoise`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_depth_of_field`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_film`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_gi_approximation`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_motion_blur`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_motion_blur_curve`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_raytracing`

#### Methods

- **`draw_header(self, context)`**

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_raytracing_presets`

### `RENDER_PT_eevee_next_sampling`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_sampling_advanced`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_sampling_render`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_sampling_shadows`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_sampling_viewport`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_screen_trace`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_volumes`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_next_volumes_range`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_eevee_performance`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_performance_compositor`

### `RENDER_PT_eevee_performance_compositor_denoise_settings`

### `RENDER_PT_eevee_performance_memory`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_eevee_performance_viewport`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_encoding`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `RENDER_PT_encoding_audio`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_encoding_video`

#### Methods

- **`draw(self, context)`**

- **`draw_vcodec(self, context)`**
  Video codec options.

### `RENDER_PT_ffmpeg_presets`

### `RENDER_PT_format`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw_framerate(layout, rd)`**

- **`draw(self, context)`**

### `RENDER_PT_format_presets`

### `RENDER_PT_frame_range`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_freestyle`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_gpencil`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_hydra_debug`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_opengl_color`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_opengl_film`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_opengl_lighting`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_opengl_options`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_opengl_sampling`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_output`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_output_color_management`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_output_views`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_post_processing`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_simplify`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_simplify_greasepencil`

### `RENDER_PT_simplify_render`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_simplify_viewport`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_stamp`

#### Methods

- **`draw(self, context)`**

### `RENDER_PT_stamp_burn`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_stamp_note`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_stereoscopy`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `RENDER_PT_time_stretching`

#### Methods

- **`draw(self, context)`**

### `RENDER_UL_renderviews`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, index)`**

### `RIGIDBODY_OT_bake_to_keyframes`

Bake rigid body transformations of selected objects to keyframes

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `RIGIDBODY_OT_connect`

Create rigid body constraints between selected rigid bodies

#### Methods

- **`execute(self, context)`**

### `RIGIDBODY_OT_object_settings_copy`

Copy Rigid Body settings from active object to selected

#### Methods

- **`execute(self, context)`**

### `RaytraceEEVEE`

### `ReadOnlyInteger`

### `Region`

### `RegionView3D`

### `RemeshModifier`

### `RenderEngine`

### `RenderLayer`

### `RenderPass`

### `RenderPasses`

### `RenderResult`

### `RenderSettings`

### `RenderSlot`

### `RenderSlots`

### `RenderView`

### `RenderViews`

### `RepeatItem`

### `RepeatZoneViewerPathElem`

### `RetimingKey`

### `RetimingKeys`

### `RigidBodyConstraint`

### `RigidBodyObject`

### `RigidBodyWorld`

### `SCENE_OT_freestyle_add_edge_marks_to_keying_set`

Add the data paths to the Freestyle Edge Mark property of selected edges to the active keying set

#### Methods

- **`execute(self, context)`**

### `SCENE_OT_freestyle_add_face_marks_to_keying_set`

Add the data paths to the Freestyle Face Mark property of selected polygons to the active keying set

#### Methods

- **`execute(self, context)`**

### `SCENE_OT_freestyle_fill_range_by_selection`

Fill the Range Min/Max entries by the min/max distance between selected mesh objects and the source object (either a user-specified object or the active camera)

#### Methods

- **`execute(self, context)`**

### `SCENE_OT_freestyle_module_open`

Open a style module file

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, _context)`**

### `SCENE_OT_gltf2_action_filter_refresh`

Refresh list of actions

#### Methods

- **`execute(self, context)`**

### `SCENE_OT_gpencil_brush_preset_add`

Add or remove grease pencil brush preset

### `SCENE_OT_gpencil_material_preset_add`

Add or remove Grease Pencil material preset

### `SCENE_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_audio`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `SCENE_PT_eevee_next_light_probes`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_keyframing_settings`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_keying_set_paths`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_keying_sets`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_physics`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SCENE_PT_rigid_body_cache`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_rigid_body_field_weights`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_rigid_body_world`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SCENE_PT_rigid_body_world_settings`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_scene`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_simulation`

#### Methods

- **`draw(self, context)`**

### `SCENE_PT_unit`

#### Methods

- **`draw(self, context)`**

### `SCENE_UL_gltf2_filter_action`

#### Methods

- **`draw_item(self, context, layout, data, item, icon, active_data, active_propname, index)`**

### `SCENE_UL_keying_set_paths`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `SCRIPT_OT_execute_preset`

Load a preset

#### Methods

- **`execute(self, context)`**

### `SEQUENCER_FH_image_strip`

### `SEQUENCER_FH_movie_strip`

### `SEQUENCER_FH_sound_strip`

### `SEQUENCER_HT_header`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_HT_tool_header`

#### Methods

- **`draw(self, context)`**

- **`draw_tool_settings(self, context)`**

### `SEQUENCER_MT_add`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_add_effect`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_add_empty`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_add_scene`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_add_transitions`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_change`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_color_tag_picker`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_context_menu`

#### Methods

- **`draw_generic(self, context)`**

- **`draw_retime(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_image`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_image_apply`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_image_clear`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_image_transform`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_marker`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_navigation`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_pivot_pie`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_preview_context_menu`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_preview_view_pie`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_preview_zoom`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_proxy`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_range`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_retiming`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_select`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_select_channel`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_select_handle`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_strip`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_strip_effect`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_strip_input`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_strip_lock_mute`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_strip_movie`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_MT_strip_retiming`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_strip_text`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_strip_transform`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_view`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_MT_view_pie`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_OT_crossfade_sounds`

Do cross-fading volume animation of two selected sound strips

#### Methods

- **`execute(self, context)`**

### `SEQUENCER_OT_deinterlace_selected_movies`

Deinterlace all selected movie sources

#### Methods

- **`execute(self, context)`**

### `SEQUENCER_OT_fades_add`

Adds or updates a fade animation for either visual or audio strips

#### Methods

- **`execute(self, context)`**

- **`calculate_fade_duration(self, context, sequence)`**

- **`is_long_enough(self, sequence, duration=0.0)`**

- **`calculate_fades(self, sequence, fade_fcurve, animated_property, duration)`**
  Returns a list of Fade objects

- **`fade_find_or_create_fcurve(self, context, sequence, animated_property)`**
  Iterates over all the fcurves until it finds an fcurve with a data path

### `SEQUENCER_OT_fades_clear`

Removes fade animation from selected sequences

#### Methods

- **`execute(self, context)`**

### `SEQUENCER_OT_split_multicam`

Split multicam strip and select camera

#### Methods

- **`execute(self, context)`**

### `SEQUENCER_PT_active_tool`

### `SEQUENCER_PT_adjust_color`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_adjust_comp`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_adjust_crop`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_adjust_sound`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_adjust_transform`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_adjust_video`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_annotation`

#### Methods

- **`has_preview(context)`**

### `SEQUENCER_PT_annotation_onion`

#### Methods

- **`has_preview(context)`**

### `SEQUENCER_PT_cache_settings`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_cache_view_settings`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_color_tag_picker`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `SEQUENCER_PT_effect`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_effect_text_box`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_effect_text_layout`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_effect_text_outline`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_effect_text_shadow`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_effect_text_style`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_frame_overlay`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_gizmo_display`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_mask`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_modifiers`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_movie_clip`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_overlay`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_PT_preview`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_preview_overlay`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_preview_snapping`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_proxy_settings`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_scene`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_scene_sound`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_sequencer_overlay`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_sequencer_overlay_strips`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_sequencer_overlay_waveforms`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_sequencer_snapping`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_snapping`

#### Methods

- **`draw(self, _context)`**

### `SEQUENCER_PT_source`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_strip`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_strip_cache`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_strip_proxy`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_time`

#### Methods

- **`draw_header_preset(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_tools_active`

Generic Class, can be used for any toolbar.

- keymap_prefix:
  The text prefix for each key-map for this spaces tools.
- tools_all():
  Generator (context_mode, tools) tuple pairs for all tools defined.
- tools_from_context(context, mode=None):
  A generator for all tools available in the current context.

Tool Sequence Structure
=======================

Sequences of tools as returned by tools_all() and tools_from_context() are comprised of:

- A `ToolDef` instance (representing a tool that can be activated).
- None (a visual separator in the tool list).
- A tuple of `ToolDef` or None values
  (representing a group of tools that can be selected between using a click-drag action).
  Note that only a single level of nesting is supported (groups cannot contain sub-groups).
- A callable which takes a single context argument and returns a tuple of values described above.
  When the context is None, all potential tools must be returned.

### `SEQUENCER_PT_view`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_view_cursor`

#### Methods

- **`draw(self, context)`**

### `SEQUENCER_PT_view_safe_areas`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SEQUENCER_PT_view_safe_areas_center_cut`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `SPHFluidSettings`

### `SPREADSHEET_HT_header`

#### Methods

- **`draw(self, context)`**

- **`draw_without_viewer_path(self, layout)`**

- **`draw_full_viewer_path(self, context, layout, viewer_path)`**

- **`draw_collapsed_viewer_path(self, context, layout, viewer_path)`**

- **`draw_spreadsheet_context(self, layout, ctx)`**

### `SPREADSHEET_OT_toggle_pin`

Turn on or off pinning

#### Methods

- **`execute(self, context)`**

- **`pin(self, context)`**

- **`unpin(self, context)`**

### `STATUSBAR_HT_header`

#### Methods

- **`draw(self, _context)`**

### `Scene`

#### Methods

- **`update_render_engine(*args, **kwargs)`**
  Scene.update_render_engine()

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

- **`cycles_curves(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `SceneDisplay`

### `SceneEEVEE`

### `SceneGpencil`

### `SceneHydra`

### `SceneObjects`

### `SceneRenderView`

### `SceneStrip`

### `Scopes`

### `Screen`

### `ScrewModifier`

### `ScriptDirectory`

### `ScriptDirectoryCollection`

#### Methods

- **`new(*args, **kwargs)`**
  ScriptDirectoryCollection.new()

- **`remove(*args, **kwargs)`**
  ScriptDirectoryCollection.remove(script_directory)

### `Sculpt`

### `SelectedUvElement`

### `SequenceEditor`

### `SequenceTimelineChannel`

### `SequencerCacheOverlay`

### `SequencerPreviewOverlay`

### `SequencerTimelineOverlay`

### `SequencerTonemapModifierData`

### `SequencerToolSettings`

### `ShaderFx`

### `ShaderFxBlur`

### `ShaderFxColorize`

### `ShaderFxFlip`

### `ShaderFxGlow`

### `ShaderFxPixel`

### `ShaderFxRim`

### `ShaderFxShadow`

### `ShaderFxSwirl`

### `ShaderFxWave`

### `ShaderNode`

### `ShaderNodeAddShader`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeAddShader.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeAddShader.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeAddShader.output_template(index)

### `ShaderNodeAmbientOcclusion`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeAmbientOcclusion.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeAmbientOcclusion.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeAmbientOcclusion.output_template(index)

### `ShaderNodeAttribute`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeAttribute.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeAttribute.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeAttribute.output_template(index)

### `ShaderNodeBackground`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBackground.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBackground.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBackground.output_template(index)

### `ShaderNodeBevel`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBevel.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBevel.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBevel.output_template(index)

### `ShaderNodeBlackbody`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBlackbody.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBlackbody.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBlackbody.output_template(index)

### `ShaderNodeBrightContrast`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBrightContrast.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBrightContrast.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBrightContrast.output_template(index)

### `ShaderNodeBsdfAnisotropic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfAnisotropic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfAnisotropic.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfAnisotropic.output_template(index)

### `ShaderNodeBsdfDiffuse`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfDiffuse.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfDiffuse.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfDiffuse.output_template(index)

### `ShaderNodeBsdfGlass`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfGlass.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfGlass.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfGlass.output_template(index)

### `ShaderNodeBsdfHair`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfHair.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfHair.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfHair.output_template(index)

### `ShaderNodeBsdfHairPrincipled`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfHairPrincipled.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfHairPrincipled.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfHairPrincipled.output_template(index)

### `ShaderNodeBsdfMetallic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfMetallic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfMetallic.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfMetallic.output_template(index)

### `ShaderNodeBsdfPrincipled`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfPrincipled.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfPrincipled.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfPrincipled.output_template(index)

### `ShaderNodeBsdfRayPortal`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfRayPortal.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfRayPortal.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfRayPortal.output_template(index)

### `ShaderNodeBsdfRefraction`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfRefraction.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfRefraction.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfRefraction.output_template(index)

### `ShaderNodeBsdfSheen`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfSheen.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfSheen.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfSheen.output_template(index)

### `ShaderNodeBsdfToon`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfToon.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfToon.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfToon.output_template(index)

### `ShaderNodeBsdfTranslucent`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfTranslucent.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfTranslucent.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfTranslucent.output_template(index)

### `ShaderNodeBsdfTransparent`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBsdfTransparent.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBsdfTransparent.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBsdfTransparent.output_template(index)

### `ShaderNodeBump`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeBump.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeBump.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeBump.output_template(index)

### `ShaderNodeCameraData`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeCameraData.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeCameraData.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeCameraData.output_template(index)

### `ShaderNodeClamp`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeClamp.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeClamp.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeClamp.output_template(index)

### `ShaderNodeCombineColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeCombineColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeCombineColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeCombineColor.output_template(index)

### `ShaderNodeCombineHSV`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeCombineHSV.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeCombineHSV.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeCombineHSV.output_template(index)

### `ShaderNodeCombineRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeCombineRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeCombineRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeCombineRGB.output_template(index)

### `ShaderNodeCombineXYZ`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeCombineXYZ.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeCombineXYZ.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeCombineXYZ.output_template(index)

### `ShaderNodeCustomGroup`

### `ShaderNodeDisplacement`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeDisplacement.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeDisplacement.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeDisplacement.output_template(index)

### `ShaderNodeEeveeSpecular`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeEeveeSpecular.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeEeveeSpecular.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeEeveeSpecular.output_template(index)

### `ShaderNodeEmission`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeEmission.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeEmission.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeEmission.output_template(index)

### `ShaderNodeFloatCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeFloatCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeFloatCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeFloatCurve.output_template(index)

### `ShaderNodeFresnel`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeFresnel.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeFresnel.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeFresnel.output_template(index)

### `ShaderNodeGamma`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeGamma.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeGamma.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeGamma.output_template(index)

### `ShaderNodeGroup`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeGroup.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeGroup.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeGroup.output_template(index)

### `ShaderNodeHairInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeHairInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeHairInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeHairInfo.output_template(index)

### `ShaderNodeHoldout`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeHoldout.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeHoldout.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeHoldout.output_template(index)

### `ShaderNodeHueSaturation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeHueSaturation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeHueSaturation.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeHueSaturation.output_template(index)

### `ShaderNodeInvert`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeInvert.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeInvert.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeInvert.output_template(index)

### `ShaderNodeLayerWeight`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeLayerWeight.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeLayerWeight.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeLayerWeight.output_template(index)

### `ShaderNodeLightFalloff`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeLightFalloff.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeLightFalloff.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeLightFalloff.output_template(index)

### `ShaderNodeLightPath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeLightPath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeLightPath.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeLightPath.output_template(index)

### `ShaderNodeMapRange`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMapRange.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMapRange.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMapRange.output_template(index)

### `ShaderNodeMapping`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMapping.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMapping.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMapping.output_template(index)

### `ShaderNodeMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMath.output_template(index)

### `ShaderNodeMix`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMix.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMix.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMix.output_template(index)

### `ShaderNodeMixRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMixRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMixRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMixRGB.output_template(index)

### `ShaderNodeMixShader`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeMixShader.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeMixShader.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeMixShader.output_template(index)

### `ShaderNodeNewGeometry`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeNewGeometry.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeNewGeometry.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeNewGeometry.output_template(index)

### `ShaderNodeNormal`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeNormal.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeNormal.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeNormal.output_template(index)

### `ShaderNodeNormalMap`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeNormalMap.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeNormalMap.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeNormalMap.output_template(index)

### `ShaderNodeObjectInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeObjectInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeObjectInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeObjectInfo.output_template(index)

### `ShaderNodeOutputAOV`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeOutputAOV.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeOutputAOV.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeOutputAOV.output_template(index)

### `ShaderNodeOutputLight`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeOutputLight.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeOutputLight.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeOutputLight.output_template(index)

### `ShaderNodeOutputLineStyle`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeOutputLineStyle.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeOutputLineStyle.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeOutputLineStyle.output_template(index)

### `ShaderNodeOutputMaterial`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeOutputMaterial.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeOutputMaterial.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeOutputMaterial.output_template(index)

### `ShaderNodeOutputWorld`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeOutputWorld.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeOutputWorld.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeOutputWorld.output_template(index)

### `ShaderNodeParticleInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeParticleInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeParticleInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeParticleInfo.output_template(index)

### `ShaderNodePointInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodePointInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodePointInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodePointInfo.output_template(index)

### `ShaderNodeRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeRGB.output_template(index)

### `ShaderNodeRGBCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeRGBCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeRGBCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeRGBCurve.output_template(index)

### `ShaderNodeRGBToBW`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeRGBToBW.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeRGBToBW.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeRGBToBW.output_template(index)

### `ShaderNodeScript`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeScript.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeScript.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeScript.output_template(index)

### `ShaderNodeSeparateColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSeparateColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSeparateColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSeparateColor.output_template(index)

### `ShaderNodeSeparateHSV`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSeparateHSV.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSeparateHSV.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSeparateHSV.output_template(index)

### `ShaderNodeSeparateRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSeparateRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSeparateRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSeparateRGB.output_template(index)

### `ShaderNodeSeparateXYZ`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSeparateXYZ.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSeparateXYZ.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSeparateXYZ.output_template(index)

### `ShaderNodeShaderToRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeShaderToRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeShaderToRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeShaderToRGB.output_template(index)

### `ShaderNodeSqueeze`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSqueeze.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSqueeze.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSqueeze.output_template(index)

### `ShaderNodeSubsurfaceScattering`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeSubsurfaceScattering.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeSubsurfaceScattering.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeSubsurfaceScattering.output_template(index)

### `ShaderNodeTangent`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTangent.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTangent.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTangent.output_template(index)

### `ShaderNodeTexBrick`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexBrick.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexBrick.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexBrick.output_template(index)

### `ShaderNodeTexChecker`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexChecker.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexChecker.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexChecker.output_template(index)

### `ShaderNodeTexCoord`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexCoord.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexCoord.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexCoord.output_template(index)

### `ShaderNodeTexEnvironment`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexEnvironment.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexEnvironment.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexEnvironment.output_template(index)

### `ShaderNodeTexGabor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexGabor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexGabor.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexGabor.output_template(index)

### `ShaderNodeTexGradient`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexGradient.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexGradient.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexGradient.output_template(index)

### `ShaderNodeTexIES`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexIES.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexIES.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexIES.output_template(index)

### `ShaderNodeTexImage`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexImage.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexImage.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexImage.output_template(index)

### `ShaderNodeTexMagic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexMagic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexMagic.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexMagic.output_template(index)

### `ShaderNodeTexNoise`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexNoise.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexNoise.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexNoise.output_template(index)

### `ShaderNodeTexPointDensity`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexPointDensity.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexPointDensity.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexPointDensity.output_template(index)

### `ShaderNodeTexSky`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexSky.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexSky.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexSky.output_template(index)

### `ShaderNodeTexVoronoi`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexVoronoi.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexVoronoi.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexVoronoi.output_template(index)

### `ShaderNodeTexWave`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexWave.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexWave.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexWave.output_template(index)

### `ShaderNodeTexWhiteNoise`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeTexWhiteNoise.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeTexWhiteNoise.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeTexWhiteNoise.output_template(index)

### `ShaderNodeTree`

### `ShaderNodeUVAlongStroke`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeUVAlongStroke.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeUVAlongStroke.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeUVAlongStroke.output_template(index)

### `ShaderNodeUVMap`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeUVMap.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeUVMap.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeUVMap.output_template(index)

### `ShaderNodeValToRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeValToRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeValToRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeValToRGB.output_template(index)

### `ShaderNodeValue`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeValue.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeValue.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeValue.output_template(index)

### `ShaderNodeVectorCurve`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVectorCurve.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVectorCurve.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVectorCurve.output_template(index)

### `ShaderNodeVectorDisplacement`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVectorDisplacement.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVectorDisplacement.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVectorDisplacement.output_template(index)

### `ShaderNodeVectorMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVectorMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVectorMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVectorMath.output_template(index)

### `ShaderNodeVectorRotate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVectorRotate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVectorRotate.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVectorRotate.output_template(index)

### `ShaderNodeVectorTransform`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVectorTransform.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVectorTransform.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVectorTransform.output_template(index)

### `ShaderNodeVertexColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVertexColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVertexColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVertexColor.output_template(index)

### `ShaderNodeVolumeAbsorption`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVolumeAbsorption.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVolumeAbsorption.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVolumeAbsorption.output_template(index)

### `ShaderNodeVolumeInfo`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVolumeInfo.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVolumeInfo.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVolumeInfo.output_template(index)

### `ShaderNodeVolumePrincipled`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVolumePrincipled.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVolumePrincipled.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVolumePrincipled.output_template(index)

### `ShaderNodeVolumeScatter`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeVolumeScatter.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeVolumeScatter.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeVolumeScatter.output_template(index)

### `ShaderNodeWavelength`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeWavelength.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeWavelength.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeWavelength.output_template(index)

### `ShaderNodeWireframe`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  ShaderNodeWireframe.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  ShaderNodeWireframe.input_template(index)

- **`output_template(*args, **kwargs)`**
  ShaderNodeWireframe.output_template(index)

### `ShapeKey`

### `ShapeKeyBezierPoint`

### `ShapeKeyCurvePoint`

### `ShapeKeyPoint`

### `Short2Attribute`

### `Short2AttributeValue`

### `ShrinkwrapConstraint`

### `ShrinkwrapModifier`

### `SimpleDeformModifier`

### `SimulationStateItem`

### `SimulationZoneViewerPathElem`

### `SkinModifier`

### `SmoothModifier`

### `SoftBodyModifier`

### `SoftBodySettings`

### `SolidifyModifier`

### `Sound`

### `SoundEqualizerModifier`

### `SoundStrip`

### `Space`

### `SpaceClipEditor`

### `SpaceConsole`

### `SpaceDopeSheetEditor`

### `SpaceFileBrowser`

### `SpaceGraphEditor`

### `SpaceImageEditor`

### `SpaceImageOverlay`

### `SpaceInfo`

### `SpaceNLA`

### `SpaceNodeEditor`

### `SpaceNodeEditorPath`

### `SpaceNodeOverlay`

### `SpaceOutliner`

### `SpacePreferences`

### `SpaceProperties`

### `SpaceSequenceEditor`

### `SpaceSpreadsheet`

### `SpaceTextEditor`

### `SpaceUVEditor`

### `SpaceView3D`

### `Speaker`

### `SpeedControlStrip`

### `Spline`

### `SplineBezierPoints`

### `SplineIKConstraint`

### `SplinePoint`

### `SplinePoints`

### `SpotLight`

### `SpreadsheetColumn`

### `SpreadsheetColumnID`

### `SpreadsheetRowFilter`

### `Stereo3dDisplay`

### `Stereo3dFormat`

### `StretchToConstraint`

### `StringAttribute`

### `StringAttributeValue`

### `StringProperty`

### `Strip`

### `StripColorBalance`

### `StripColorBalanceData`

### `StripCrop`

### `StripElement`

### `StripElements`

### `StripModifier`

### `StripModifiers`

### `StripProxy`

### `StripTransform`

### `StripsMeta`

### `StripsTopLevel`

### `Struct`

### `StucciTexture`

### `StudioLight`

### `StudioLights`

### `SubsurfModifier`

### `SubtractStrip`

### `SunLight`

### `SurfaceCurve`

### `SurfaceDeformModifier`

### `SurfaceModifier`

### `TEXTURE_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `TEXTURE_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_blend`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_clouds`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_colors`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_colors_ramp`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `TEXTURE_PT_context`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `TEXTURE_PT_distortednoise`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_image`

#### Methods

- **`draw(self, _context)`**

### `TEXTURE_PT_image_alpha`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `TEXTURE_PT_image_mapping`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_image_mapping_crop`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_image_sampling`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_image_settings`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_influence`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_magic`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_mapping`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_marble`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_musgrave`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_node`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_preview`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_stucci`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_voronoi`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_voronoi_feature_weights`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_PT_wood`

#### Methods

- **`draw(self, context)`**

### `TEXTURE_UL_texpaintslots`

#### Methods

- **`draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index)`**

### `TEXTURE_UL_texslots`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `TEXT_EDITOR_OT_preset_add`

Add or remove a Text Editor Preset

### `TEXT_HT_footer`

#### Methods

- **`draw(self, context)`**

### `TEXT_HT_header`

#### Methods

- **`draw(self, context)`**

### `TEXT_MT_context_menu`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_edit`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_edit_to3d`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `TEXT_MT_format`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_select`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_templates`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_templates_osl`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_templates_py`

#### Methods

- **`draw(self, _context)`**

### `TEXT_MT_text`

#### Methods

- **`draw(self, context)`**

### `TEXT_MT_view`

#### Methods

- **`draw(self, context)`**

### `TEXT_MT_view_navigation`

#### Methods

- **`draw(self, _context)`**

### `TEXT_PT_find`

#### Methods

- **`draw(self, context)`**

### `TEXT_PT_properties`

#### Methods

- **`draw(self, context)`**

### `TIME_MT_cache`

#### Methods

- **`draw(self, context)`**

### `TIME_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `TIME_MT_marker`

#### Methods

- **`draw(self, context)`**

### `TIME_MT_view`

#### Methods

- **`draw(self, context)`**

### `TIME_PT_auto_keyframing`

#### Methods

- **`draw(self, context)`**

### `TIME_PT_keyframing_settings`

#### Methods

- **`draw(self, context)`**

### `TIME_PT_playback`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_HT_upper_bar`

#### Methods

- **`draw(self, context)`**

- **`draw_left(self, context)`**

- **`draw_right(self, context)`**

### `TOPBAR_MT_blender`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_blender_system`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_edit`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_edit_armature_add`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_edit_curve_add`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_file`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_file_cleanup`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_file_context_menu`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_file_defaults`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_file_export`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_file_external_data`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_file_import`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_file_new`

#### Methods

- **`app_template_paths()`**

- **`draw_ex(layout, _context, *, use_splash=False, use_more=False)`**

- **`draw(self, context)`**

### `TOPBAR_MT_file_previews`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_file_recover`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_MT_help`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_render`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_templates_more`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_window`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_MT_workspace_menu`

#### Methods

- **`draw(self, _context)`**

### `TOPBAR_PT_annotation_layers`

### `TOPBAR_PT_gpencil_primitive`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_PT_grease_pencil_layers`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_PT_grease_pencil_materials`

### `TOPBAR_PT_grease_pencil_vertex_color`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_PT_name`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_PT_name_marker`

#### Methods

- **`is_using_pose_markers(context)`**

- **`get_selected_marker(context)`**

- **`row_with_icon(layout, icon)`**

- **`draw(self, context)`**

### `TOPBAR_PT_tool_fallback`

#### Methods

- **`draw(self, context)`**

### `TOPBAR_PT_tool_settings_extra`

Popover panel for adding extra options that don't fit in the tool settings header

#### Methods

- **`draw(self, context)`**

### `TexMapping`

### `TexPaintSlot`

### `Text`

#### Methods

- **`as_module(self)`**

- **`region_as_string(...)`**
  .. method:: region_as_string(range=None)

- **`region_from_string(...)`**
  .. method:: region_from_string(body, range=None)

### `TextBox`

### `TextCharacterFormat`

### `TextCurve`

### `TextLine`

### `TextStrip`

### `Texture`

### `TextureNode`

### `TextureNodeAt`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeAt.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeAt.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeAt.output_template(index)

### `TextureNodeBricks`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeBricks.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeBricks.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeBricks.output_template(index)

### `TextureNodeChecker`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeChecker.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeChecker.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeChecker.output_template(index)

### `TextureNodeCombineColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeCombineColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeCombineColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeCombineColor.output_template(index)

### `TextureNodeCompose`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeCompose.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeCompose.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeCompose.output_template(index)

### `TextureNodeCoordinates`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeCoordinates.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeCoordinates.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeCoordinates.output_template(index)

### `TextureNodeCurveRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeCurveRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeCurveRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeCurveRGB.output_template(index)

### `TextureNodeCurveTime`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeCurveTime.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeCurveTime.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeCurveTime.output_template(index)

### `TextureNodeDecompose`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeDecompose.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeDecompose.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeDecompose.output_template(index)

### `TextureNodeDistance`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeDistance.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeDistance.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeDistance.output_template(index)

### `TextureNodeGroup`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeGroup.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeGroup.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeGroup.output_template(index)

### `TextureNodeHueSaturation`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeHueSaturation.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeHueSaturation.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeHueSaturation.output_template(index)

### `TextureNodeImage`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeImage.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeImage.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeImage.output_template(index)

### `TextureNodeInvert`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeInvert.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeInvert.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeInvert.output_template(index)

### `TextureNodeMath`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeMath.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeMath.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeMath.output_template(index)

### `TextureNodeMixRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeMixRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeMixRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeMixRGB.output_template(index)

### `TextureNodeOutput`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeOutput.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeOutput.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeOutput.output_template(index)

### `TextureNodeRGBToBW`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeRGBToBW.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeRGBToBW.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeRGBToBW.output_template(index)

### `TextureNodeRotate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeRotate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeRotate.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeRotate.output_template(index)

### `TextureNodeScale`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeScale.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeScale.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeScale.output_template(index)

### `TextureNodeSeparateColor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeSeparateColor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeSeparateColor.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeSeparateColor.output_template(index)

### `TextureNodeTexBlend`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexBlend.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexBlend.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexBlend.output_template(index)

### `TextureNodeTexClouds`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexClouds.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexClouds.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexClouds.output_template(index)

### `TextureNodeTexDistNoise`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexDistNoise.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexDistNoise.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexDistNoise.output_template(index)

### `TextureNodeTexMagic`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexMagic.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexMagic.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexMagic.output_template(index)

### `TextureNodeTexMarble`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexMarble.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexMarble.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexMarble.output_template(index)

### `TextureNodeTexMusgrave`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexMusgrave.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexMusgrave.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexMusgrave.output_template(index)

### `TextureNodeTexNoise`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexNoise.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexNoise.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexNoise.output_template(index)

### `TextureNodeTexStucci`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexStucci.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexStucci.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexStucci.output_template(index)

### `TextureNodeTexVoronoi`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexVoronoi.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexVoronoi.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexVoronoi.output_template(index)

### `TextureNodeTexWood`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexWood.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexWood.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexWood.output_template(index)

### `TextureNodeTexture`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTexture.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTexture.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTexture.output_template(index)

### `TextureNodeTranslate`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeTranslate.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeTranslate.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeTranslate.output_template(index)

### `TextureNodeTree`

### `TextureNodeValToNor`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeValToNor.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeValToNor.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeValToNor.output_template(index)

### `TextureNodeValToRGB`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeValToRGB.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeValToRGB.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeValToRGB.output_template(index)

### `TextureNodeViewer`

#### Methods

- **`is_registered_node_type(*args, **kwargs)`**
  TextureNodeViewer.is_registered_node_type()

- **`input_template(*args, **kwargs)`**
  TextureNodeViewer.input_template(index)

- **`output_template(*args, **kwargs)`**
  TextureNodeViewer.output_template(index)

### `TextureSlot`

### `Theme`

### `ThemeAssetShelf`

### `ThemeBoneColorSet`

### `ThemeClipEditor`

### `ThemeCollectionColor`

### `ThemeConsole`

### `ThemeDopeSheet`

### `ThemeFileBrowser`

### `ThemeFontStyle`

### `ThemeGradientColors`

### `ThemeGraphEditor`

### `ThemeImageEditor`

### `ThemeInfo`

### `ThemeNLAEditor`

### `ThemeNodeEditor`

### `ThemeOutliner`

### `ThemePanelColors`

### `ThemePreferences`

### `ThemeProperties`

### `ThemeSequenceEditor`

### `ThemeSpaceGeneric`

### `ThemeSpaceGradient`

### `ThemeSpaceListGeneric`

### `ThemeSpreadsheet`

### `ThemeStatusBar`

### `ThemeStripColor`

### `ThemeStyle`

### `ThemeTextEditor`

### `ThemeTopBar`

### `ThemeUserInterface`

### `ThemeView3D`

### `ThemeWidgetColors`

### `ThemeWidgetStateColors`

### `TimelineMarker`

### `TimelineMarkers`

### `Timer`

### `ToolSettings`

### `TrackToConstraint`

### `TransformCacheConstraint`

### `TransformConstraint`

### `TransformOrientation`

### `TransformOrientationSlot`

### `TransformStrip`

### `TriangulateModifier`

### `UDIMTile`

### `UDIMTiles`

### `UILIST_OT_entry_add`

Add an entry to the list after the current active item

#### Methods

- **`execute(self, context)`**

### `UILIST_OT_entry_move`

Move an entry in the list up or down

#### Methods

- **`execute(self, context)`**

### `UILIST_OT_entry_remove`

Remove the selected entry from the list

#### Methods

- **`execute(self, context)`**

### `UILayout`

#### Methods

- **`icon(*args, **kwargs)`**
  UILayout.icon(data)

- **`enum_item_name(*args, **kwargs)`**
  UILayout.enum_item_name(data, property, identifier)

- **`enum_item_description(*args, **kwargs)`**
  UILayout.enum_item_description(data, property, identifier)

- **`enum_item_icon(*args, **kwargs)`**
  UILayout.enum_item_icon(data, property, identifier)

- **`introspect(...)`**
  .. method:: introspect()

### `UIList`

### `UIPieMenu`

### `UIPopover`

### `UIPopupMenu`

### `UI_MT_button_context_menu`

UI button context menu definition. Scripts can append/prepend this to
add own operators to the context menu. They must check context though, so
their items only draw in a valid context and for the correct buttons.

#### Methods

- **`draw(self, _context)`**

### `UI_MT_list_item_context_menu`

UI List item context menu definition. Scripts can append/prepend this to
add own operators to the context menu. They must check context though, so
their items only draw in a valid context and for the correct UI list.

#### Methods

- **`draw(self, _context)`**

### `UI_UL_list`

#### Methods

- **`filter_items_by_name(pattern, bitflag, items, propname='name', flags=None, reverse=False)`**
  Set FILTER_ITEM for items which name matches filter_name one (case-insensitive).

- **`sort_items_helper(sort_data, key, reverse=False)`**
  Common sorting utility. Returns a neworder list mapping org_idx -> new_idx.

### `USDHook`

### `USERPREF_HT_header`

#### Methods

- **`draw_buttons(layout, context)`**

- **`draw(self, context)`**

### `USERPREF_MT_addons_settings`

#### Methods

- **`draw(self, _context)`**

### `USERPREF_MT_editor_menus`

#### Methods

- **`draw(self, _context)`**

### `USERPREF_MT_extensions_active_repo`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_extensions_active_repo_extra`

#### Methods

- **`draw(self, _context)`**

### `USERPREF_MT_extensions_active_repo_remove`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_extensions_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_extensions_settings`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_interface_theme_presets`

#### Methods

- **`draw(self, context)`**

- **`reset_cb(_context, _filepath)`**

- **`post_cb(context, filepath)`**

### `USERPREF_MT_keyconfigs`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_save_load`

#### Methods

- **`draw(self, context)`**

### `USERPREF_MT_view`

#### Methods

- **`draw(self, _context)`**

### `USERPREF_PT_addons`

#### Methods

- **`is_user_addon(mod, user_addon_paths)`**

- **`draw_addon_preferences(layout, context, addon_preferences)`**

- **`draw_error(layout, message)`**

- **`draw(self, context)`**

### `USERPREF_PT_addons_filter`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_addons_tags`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_animation_fcurves`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_animation_keyframes`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_animation_timeline`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_annotations`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_cursor`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_gpencil`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_misc`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_node_editor`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_objects`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_edit_objects_duplicate_data`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_objects_new`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_sequence_editor`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_text_editor`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_edit_weight_paint`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_experimental_debugging`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_experimental_new_features`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_experimental_prototypes`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_extensions`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_extensions_repos`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_extensions_tags`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_applications`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_asset_libraries`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_data`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_development`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_render`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_file_paths_script_directories`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_input_keyboard`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_input_mouse`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_input_ndof`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_input_tablet`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_input_touchpad`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_display`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_editors`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_menus`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_interface_menus_mouse_over`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_menus_pie`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_statusbar`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_temporary_windows`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_text`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_interface_translation`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_keymap`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_navigation_bar`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_navigation_fly_walk`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_navigation_fly_walk_gravity`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_navigation_fly_walk_navigation`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_navigation_orbit`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_navigation_zoom`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_ndof_settings`

#### Methods

- **`draw_settings(layout, props, show_3dview_settings=True)`**

- **`draw(self, context)`**

### `USERPREF_PT_save_preferences`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_saveload_autorun`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `USERPREF_PT_saveload_blend`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_saveload_file_browser`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_studiolight_light_editor`

#### Methods

- **`opengl_light_buttons(layout, light)`**

- **`draw(self, context)`**

### `USERPREF_PT_studiolight_lights`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`get_error_message(self)`**

### `USERPREF_PT_studiolight_matcaps`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`get_error_message(self)`**

### `USERPREF_PT_studiolight_world`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`get_error_message(self)`**

### `USERPREF_PT_system_cycles_devices`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_display_graphics`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_memory`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_network`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_os_settings`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_sound`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_system_video_sequencer`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_text_editor`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_text_editor_presets`

### `USERPREF_PT_theme`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_bone_color_sets`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, _context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_clip_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_clip_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_clip_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_clip_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_collection_colors`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, _context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_console`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_console_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_console_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_dopesheet_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_dopesheet_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_dopesheet_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_dopesheet_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_file_browser`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_file_browser_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_file_browser_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_graph_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_graph_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_graph_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_graph_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_image_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_image_editor_asset_shelf`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_image_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_image_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_info`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_info_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_info_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_gizmos`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_interface_icons`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_interface_shade_wcol_box`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_list_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_menu`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_menu_back`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_menu_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_num`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_numslider`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_option`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_pie_menu`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_progress`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_pulldown`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_radio`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_regular`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_scroll`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_tab`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_text`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_toggle`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_tool`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_toolbar_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_shade_wcol_tooltip`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_state`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_interface_styles`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_interface_transparent_checker`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_interface_wcol_box`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_list_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_menu`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_menu_back`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_menu_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_num`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_numslider`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_option`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_pie_menu`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_progress`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_pulldown`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_radio`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_regular`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_scroll`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_tab`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_text`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_toggle`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_tool`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_toolbar_item`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_interface_wcol_tooltip`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_nla_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_nla_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_nla_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_nla_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_node_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_node_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_node_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_node_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_outliner`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_outliner_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_outliner_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_preferences`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_preferences_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_preferences_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_properties`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_properties_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_properties_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_sequence_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_sequence_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_sequence_editor_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_sequence_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_spreadsheet`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_spreadsheet_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_spreadsheet_space_list`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_spreadsheet_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_statusbar`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_statusbar_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_statusbar_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_strip_colors`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, _context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_text_editor`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_text_editor_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_text_editor_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_text_style`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, _context)`**

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_theme_topbar`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_topbar_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_topbar_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_user_interface`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_view_3d`

#### Methods

- **`draw_header(self, _context)`**

- **`draw(self, context)`**

### `USERPREF_PT_theme_view_3d_asset_shelf`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_view_3d_space`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_view_3d_space_gradients`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_theme_view_3d_space_panelcolors`

#### Methods

- **`draw(self, context)`**

### `USERPREF_PT_viewport_display`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_viewport_quality`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_viewport_selection`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_viewport_subdivision`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_PT_viewport_textures`

Base class for panels to center align contents with some horizontal margin.
Deriving classes need to implement a ``draw_centered(context, layout)`` function.

#### Methods

- **`draw_centered(self, context, layout)`**

### `USERPREF_UL_asset_libraries`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

### `USERPREF_UL_extension_repos`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, _index)`**

- **`filter_items(self, _context, data, propname)`**

### `UVLoopLayers`

### `UVProjectModifier`

### `UVProjector`

### `UVWarpModifier`

### `UV_OT_align_rotation`

Align the UV island's rotation

#### Methods

- **`execute(self, context)`**

- **`draw(self, _context)`**

### `UV_OT_export_layout`

Export UV layout to file

#### Methods

- **`invoke(self, context, event)`**

- **`get_default_file_name(self, context)`**

- **`check(self, context)`**

- **`execute(self, context)`**

- **`iter_meshes_to_export(self, context)`**

### `UV_OT_follow_active_quads`

Follow UVs from active quads along continuous face loops

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `UV_OT_lightmap_pack`

Pack each face's UVs into the UV bounds

#### Methods

- **`draw(self, context)`**

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `UV_OT_randomize_uv_transform`

Randomize the UV island's location, rotation, and scale

#### Methods

- **`execute(self, context)`**

### `UnifiedPaintSettings`

### `UnitSettings`

### `UnknownType`

### `UserAssetLibrary`

### `UserExtensionRepo`

### `UserExtensionRepoCollection`

#### Methods

- **`new(*args, **kwargs)`**
  UserExtensionRepoCollection.new(name="", module="", custom_directory="", remote_url="", source='USER')

- **`remove(*args, **kwargs)`**
  UserExtensionRepoCollection.remove(repo)

### `UserSolidLight`

### `UvSculpt`

### `VIEW3D_AST_brush_gpencil_paint`

### `VIEW3D_AST_brush_gpencil_sculpt`

### `VIEW3D_AST_brush_gpencil_vertex`

### `VIEW3D_AST_brush_gpencil_weight`

### `VIEW3D_AST_brush_sculpt`

### `VIEW3D_AST_brush_sculpt_curves`

### `VIEW3D_AST_brush_texture_paint`

### `VIEW3D_AST_brush_vertex_paint`

### `VIEW3D_AST_brush_weight_paint`

### `VIEW3D_AST_pose_library`

### `VIEW3D_FH_camera_background_image`

### `VIEW3D_FH_empty_image`

### `VIEW3D_FH_vdb_volume`

### `VIEW3D_HT_header`

#### Methods

- **`draw_xform_template(layout, context)`**

- **`draw(self, context)`**

### `VIEW3D_HT_tool_header`

#### Methods

- **`draw(self, context)`**

- **`draw_tool_settings(self, context)`**

- **`draw_mode_settings(self, context)`**

### `VIEW3D_MT_add`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_armature_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_armature_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_bone_collections`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_bone_options_disable`

### `VIEW3D_MT_bone_options_enable`

### `VIEW3D_MT_bone_options_toggle`

### `VIEW3D_MT_brush_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_brush_gpencil_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_camera_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_curve_add`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_armature`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_armature_delete`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_armature_names`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_armature_parent`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_armature_roll`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve_clean`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve_ctrlpoints`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_curve_delete`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve_segments`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curve_showhide`

### `VIEW3D_MT_edit_curves`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curves_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curves_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curves_control_points`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curves_segments`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_curves_select_more_less`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_font`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_font_chars`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_font_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_font_delete`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_font_kerning`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_greasepencil`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_greasepencil_animation`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_greasepencil_cleanup`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_greasepencil_delete`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_greasepencil_point`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_greasepencil_showhide`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_greasepencil_stroke`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_lattice`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_lattice_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_clean`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_mesh_delete`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_edges`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_extrude`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_mesh_faces`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_mesh_faces_data`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_merge`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_normals`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_normals_average`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_normals_select_strength`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_normals_set_strength`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_select_by_trait`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_mesh_select_linked`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_select_loops`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_select_mode`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_select_more_less`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_select_similar`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_shading`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_showhide`

### `VIEW3D_MT_edit_mesh_split`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_vertices`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_mesh_weights`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_meta`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_meta_showhide`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_metaball_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_edit_pointcloud`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_edit_surface`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_editor_menus`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_empty_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_face_sets`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_face_sets_init`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_grease_pencil_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_grease_pencil_assign_material`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_grease_pencil_sculpt_automasking_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_greasepencil_edit_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_greasepencil_material_active`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_greasepencil_vertex_group`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_hook`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_image_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_light_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_lightprobe_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_make_links`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_make_single_user`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_mask`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_mesh_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_metaball_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_mirror`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_object_animation`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_apply`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_asset`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_cleanup`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_clear`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_collection`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_constraints`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_object_convert`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_object_liboverride`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_mode_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_modifiers`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_parent`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_quick_effects`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_relations`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_rigid_body`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_shading`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_showhide`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_object_track`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_orientations_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_paint_grease_pencil`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_paint_vertex`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_paint_vertex_grease_pencil`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_paint_weight`

#### Methods

- **`draw_generic(layout, is_editmode=False)`**

- **`draw(self, _context)`**

### `VIEW3D_MT_paint_weight_lock`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_particle`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_particle_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_particle_showhide`

### `VIEW3D_MT_pivot_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_pose`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_apply`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_constraints`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_context_menu`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_ik`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_modify`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_motion`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_names`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_propagate`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_showhide`

### `VIEW3D_MT_pose_slide`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_pose_transform`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_proportional_editing_falloff_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_random_mask`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_sculpt_automasking_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_sculpt_curves`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_face_sets_edit_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_mask_edit_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_set_pivot`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_showhide`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_transform`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_sculpt_trim`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_armature`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_curve`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_curves`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_grease_pencil`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_select_edit_lattice`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_mesh`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_metaball`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_point_cloud`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_surface`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_edit_text`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_object`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_object_more_less`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_paint_mask`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_paint_mask_vertex`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_particle`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_pose`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_pose_more_less`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_select_sculpt_curves`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_shading_ex_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_shading_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_snap`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_snap_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_surface_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_tools_projectpaint_clone`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_tools_projectpaint_stencil`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_tools_projectpaint_uvlayer`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_transform`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_transform_armature`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_transform_gizmo_pie`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_transform_object`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_uv_map`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_vertex_group`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_view`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_MT_view_align`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_align_selected`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_cameras`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_local`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_navigation`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_regions`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_view_viewpoint`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_volume_add`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_weight_grease_pencil`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_MT_wpaint_vgroup_lock_pie`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_OT_edit_mesh_extrude_individual_move`

Extrude each individual face separately along local normals

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `VIEW3D_OT_edit_mesh_extrude_manifold_normal`

Extrude manifold region along normals

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `VIEW3D_OT_edit_mesh_extrude_move_normal`

Extrude region together along the average normal

#### Methods

- **`extrude_region(operator, context, use_vert_normals, dissolve_and_intersect)`**

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `VIEW3D_OT_edit_mesh_extrude_move_shrink_fatten`

Extrude region together along local normals

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

### `VIEW3D_OT_transform_gizmo_set`

Set the current transform gizmo

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, event)`**

### `VIEW3D_PT_active_tool`

### `VIEW3D_PT_active_tool_duplicate`

### `VIEW3D_PT_annotation_onion`

### `VIEW3D_PT_brush_asset_shelf_filter`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_collections`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_context_properties`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_curves_sculpt_add_shape`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_curves_sculpt_grow_shrink_scaling`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_curves_sculpt_parameter_falloff`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_curves_sculpt_symmetry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_curves_sculpt_symmetry_for_topbar`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_gizmo_display`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_gpencil_brush_presets`

Brush settings

### `VIEW3D_PT_grease_pencil`

### `VIEW3D_PT_grease_pencil_guide`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_grease_pencil_lock`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_grease_pencil_multi_frame`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_grease_pencil_origin`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_grease_pencil_sculpt_automasking`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_greasepencil_draw_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_greasepencil_sculpt_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_greasepencil_vertex_paint_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_greasepencil_weight_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_mask`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_object_type_visibility`

#### Methods

- **`draw_ex(self, _context, view, show_select)`**

- **`draw(self, context)`**

### `VIEW3D_PT_overlay`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_overlay_bones`

#### Methods

- **`is_using_wireframe(context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_curve`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_curves`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_mesh`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_mesh_freestyle`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_mesh_measurement`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_mesh_normals`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_edit_mesh_shading`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_geometry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_grease_pencil_canvas_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_grease_pencil_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_guides`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_motion_tracking`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_object`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_sculpt`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_sculpt_curves`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_texture_paint`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_vertex_paint`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_viewer_node`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_overlay_weight_paint`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_paint_texture_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_paint_vertex_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_paint_weight_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_proportional_edit`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_quad_view`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_automasking`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_context_menu`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_dyntopo`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_options_gravity`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_snapping`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_symmetry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_symmetry_for_topbar`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_sculpt_voxel_remesh`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_shading_color`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_compositor`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_lighting`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_options_shadow`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_options_ssao`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_shading_render_pass`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_slots_color_attributes`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_slots_paint_canvas`

#### Methods

- **`get_mode_settings(self, context)`**

- **`draw_image_interpolation(self, **_kwargs)`**

- **`draw_header(self, context)`**

### `VIEW3D_PT_slots_projectpaint`

#### Methods

- **`get_mode_settings(self, context)`**

- **`draw_image_interpolation(self, layout, mode_settings)`**

- **`draw_header(self, context)`**

### `VIEW3D_PT_slots_vertex_groups`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_snapping`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_stencil_projectpaint`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_active`

Generic Class, can be used for any toolbar.

- keymap_prefix:
  The text prefix for each key-map for this spaces tools.
- tools_all():
  Generator (context_mode, tools) tuple pairs for all tools defined.
- tools_from_context(context, mode=None):
  A generator for all tools available in the current context.

Tool Sequence Structure
=======================

Sequences of tools as returned by tools_all() and tools_from_context() are comprised of:

- A `ToolDef` instance (representing a tool that can be activated).
- None (a visual separator in the tool list).
- A tuple of `ToolDef` or None values
  (representing a group of tools that can be selected between using a click-drag action).
  Note that only a single level of nesting is supported (groups cannot contain sub-groups).
- A callable which takes a single context argument and returns a tuple of values described above.
  When the context is None, all potential tools must be returned.

### `VIEW3D_PT_tools_armatureedit_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_clone`

### `VIEW3D_PT_tools_brush_color`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_display`

### `VIEW3D_PT_tools_brush_falloff`

### `VIEW3D_PT_tools_brush_falloff_frontface`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_falloff_normal`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_select`

### `VIEW3D_PT_tools_brush_settings`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_settings_advanced`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_brush_stroke`

### `VIEW3D_PT_tools_brush_stroke_smooth_stroke`

### `VIEW3D_PT_tools_brush_swatches`

### `VIEW3D_PT_tools_brush_texture`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_advanced`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_eraser`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_gap_closure`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_mix_palette`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_mixcolor`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_paint_falloff`

### `VIEW3D_PT_tools_grease_pencil_brush_post_processing`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_random`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_sculpt_falloff`

### `VIEW3D_PT_tools_grease_pencil_brush_select`

### `VIEW3D_PT_tools_grease_pencil_brush_settings`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_stabilizer`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_stroke`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_vertex_color`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_vertex_falloff`

### `VIEW3D_PT_tools_grease_pencil_brush_vertex_palette`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_brush_weight_falloff`

### `VIEW3D_PT_tools_grease_pencil_paint_appearance`

### `VIEW3D_PT_tools_grease_pencil_sculpt_appearance`

### `VIEW3D_PT_tools_grease_pencil_sculpt_brush_advanced`

### `VIEW3D_PT_tools_grease_pencil_sculpt_brush_popover`

### `VIEW3D_PT_tools_grease_pencil_sculpt_select`

### `VIEW3D_PT_tools_grease_pencil_sculpt_settings`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_advanced`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_eraser`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_gap_closure`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_mix_palette`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_mixcolor`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_post_processing`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_random`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_select`

### `VIEW3D_PT_tools_grease_pencil_v3_brush_settings`

#### Methods

- **`draw_header_preset(self, _context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_stabilizer`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_v3_brush_stroke`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_tools_grease_pencil_vertex_appearance`

### `VIEW3D_PT_tools_grease_pencil_vertex_paint_select`

### `VIEW3D_PT_tools_grease_pencil_vertex_paint_settings`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_weight_appearance`

### `VIEW3D_PT_tools_grease_pencil_weight_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_grease_pencil_weight_paint_select`

### `VIEW3D_PT_tools_grease_pencil_weight_paint_settings`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_imagepaint_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_imagepaint_options_cavity`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEW3D_PT_tools_imagepaint_options_external`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_imagepaint_symmetry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_mask_texture`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_meshedit_options`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_tools_meshedit_options_transform`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_meshedit_options_uvs`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_object_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_object_options_transform`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_particlemode`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_particlemode_options`

Default tools for particle mode

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_particlemode_options_display`

Default tools for particle mode

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_particlemode_options_shapecut`

Default tools for particle mode

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_posemode_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_vertexpaint_options`

#### Methods

- **`draw(self, _context)`**

### `VIEW3D_PT_tools_vertexpaint_symmetry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_vertexpaint_symmetry_for_topbar`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_weight_gradient`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_weightpaint_options`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_weightpaint_symmetry`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_tools_weightpaint_symmetry_for_topbar`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_transform_orientations`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_view3d_cursor`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_view3d_lock`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_view3d_properties`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_view3d_stereo`

#### Methods

- **`draw(self, context)`**

### `VIEW3D_PT_viewport_debug`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_MT_lightgroup_sync`

#### Methods

- **`draw(self, _context)`**

### `VIEWLAYER_PT_eevee_next_layer_passes_data`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_eevee_next_layer_passes_light`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_filter`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

### `VIEWLAYER_PT_freestyle_edge_detection`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_lineset`

#### Methods

- **`draw_edge_type_buttons(self, box, lineset, edge_type)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_lineset_collection`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_lineset_edgetype`

#### Methods

- **`draw_header(self, context)`**

- **`draw_edge_type_buttons(self, box, lineset, edge_type)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_lineset_facemarks`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_lineset_visibilty`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_alpha`

#### Methods

- **`draw_alpha_modifier(self, context, modifier)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_color`

#### Methods

- **`draw_color_modifier(self, context, modifier)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_geometry`

#### Methods

- **`draw_geometry_modifier(self, _context, modifier)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_chaining`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_dashedline`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_selection`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_sorting`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_splitting`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_strokes_splitting_pattern`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_texture`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_linestyle_thickness`

#### Methods

- **`draw_thickness_modifier(self, context, modifier)`**

- **`draw(self, context)`**

### `VIEWLAYER_PT_freestyle_style_modules`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_layer`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_layer_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `VIEWLAYER_PT_layer_passes`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_PT_layer_passes_aov`

### `VIEWLAYER_PT_layer_passes_cryptomatte`

### `VIEWLAYER_PT_layer_passes_lightgroups`

### `VIEWLAYER_PT_workbench_layer_passes_data`

#### Methods

- **`draw(self, context)`**

### `VIEWLAYER_UL_aov`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname)`**

### `VIEWLAYER_UL_linesets`

#### Methods

- **`draw_item(self, _context, layout, _data, item, icon, _active_data, _active_propname, index)`**

### `VOLUME_UL_grids`

#### Methods

- **`draw_item(self, _context, layout, _data, grid, _icon, _active_data, _active_propname, _index)`**

### `VectorFont`

### `VertexGroup`

### `VertexGroupElement`

### `VertexGroups`

### `VertexPaint`

### `VertexWeightEditModifier`

### `VertexWeightMixModifier`

### `VertexWeightProximityModifier`

### `View2D`

### `View3DCursor`

### `View3DOverlay`

### `View3DShading`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `ViewLayer`

#### Methods

- **`update_render_passes(*args, **kwargs)`**
  ViewLayer.update_render_passes()

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `ViewLayerEEVEE`

### `ViewLayers`

### `ViewerNodeViewerPathElem`

### `ViewerPath`

### `ViewerPathElem`

### `Volume`

### `VolumeDisplaceModifier`

### `VolumeDisplay`

### `VolumeGrid`

### `VolumeGrids`

### `VolumeRender`

### `VolumeToMeshModifier`

### `VoronoiTexture`

### `WM_MT_operator_presets`

#### Methods

- **`draw(self, context)`**

### `WM_MT_region_toggle_pie`

#### Methods

- **`draw(self, context)`**

### `WM_MT_splash`

#### Methods

- **`draw(self, context)`**

### `WM_MT_splash_about`

#### Methods

- **`draw(self, context)`**

### `WM_MT_splash_quick_setup`

#### Methods

- **`draw(self, context)`**

### `WM_MT_toolsystem_submenu`

#### Methods

- **`draw(self, context)`**

### `WM_OT_batch_rename`

Rename multiple items at once

#### Methods

- **`draw(self, context)`**

- **`check(self, context)`**

- **`execute(self, context)`**

- **`invoke(self, context, event)`**

### `WM_OT_blend_strings_utf8_validate`

Check and fix all strings in current .blend file to be valid UTF-8 Unicode (needed for some old, 2.4x area files)

#### Methods

- **`validate_strings(self, item, done_items)`**

- **`execute(self, _context)`**

### `WM_OT_context_collection_boolean_set`

Set boolean values for a collection of items

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_cycle_array`

Set a context array value (useful for cycling the active mesh edit mode)

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_cycle_enum`

Toggle a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_cycle_int`

Set a context value (useful for cycling active material, shape keys, groups, etc.)

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_menu_enum`

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_modal_mouse`

Adjust arbitrary values with mouse input

#### Methods

- **`modal(self, context, event)`**

- **`invoke(self, context, event)`**

### `WM_OT_context_pie_enum`

#### Methods

- **`invoke(self, context, event)`**

### `WM_OT_context_scale_float`

Scale a float context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_scale_int`

Scale an int context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_boolean`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_enum`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_float`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_id`

Set a context value to an ID data-block

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_int`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_string`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_set_value`

Set a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_toggle`

Toggle a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_context_toggle_enum`

Toggle a context value

#### Methods

- **`execute(self, context)`**

### `WM_OT_doc_view`

Open online reference docs in a web browser

#### Methods

- **`execute(self, _context)`**

### `WM_OT_doc_view_manual`

Load online manual

#### Methods

- **`execute(self, _context)`**

### `WM_OT_drop_blend_file`

#### Methods

- **`invoke(self, context, _event)`**

- **`draw_menu(self, menu, _context)`**

### `WM_OT_interface_theme_preset_add`

Add a custom theme to the preset list

#### Methods

- **`post_cb(self, context, filepath)`**

### `WM_OT_interface_theme_preset_remove`

Remove a custom theme from the preset list

#### Methods

- **`invoke(self, context, event)`**

- **`post_cb(self, context, _filepath)`**

### `WM_OT_interface_theme_preset_save`

Save a custom theme in the preset list

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, event)`**

### `WM_OT_keyconfig_preset_add`

Add a custom keymap configuration to the preset list

#### Methods

- **`add(self, _context, filepath)`**

### `WM_OT_keyconfig_preset_remove`

Remove a custom keymap configuration from the preset list

#### Methods

- **`pre_cb(self, context)`**

- **`post_cb(self, context, _filepath)`**

- **`invoke(self, context, event)`**

### `WM_OT_operator_cheat_sheet`

List all the operators in a text-block, useful for scripting

#### Methods

- **`execute(self, _context)`**

### `WM_OT_operator_pie_enum`

#### Methods

- **`invoke(self, context, event)`**

### `WM_OT_operator_preset_add`

Add or remove an Operator Preset

#### Methods

- **`operator_path(operator)`**

### `WM_OT_operator_presets_cleanup`

Remove outdated operator properties from presets that may cause problems

#### Methods

- **`execute(self, context)`**

### `WM_OT_owner_disable`

Disable add-on for workspace

#### Methods

- **`execute(self, context)`**

### `WM_OT_owner_enable`

Enable add-on for workspace

#### Methods

- **`execute(self, context)`**

### `WM_OT_path_open`

Open a path in a file browser

#### Methods

- **`execute(self, _context)`**

### `WM_OT_previews_batch_clear`

Clear selected .blend file's previews

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, context)`**

### `WM_OT_previews_batch_generate`

Generate selected .blend file's previews

#### Methods

- **`invoke(self, context, _event)`**

- **`execute(self, context)`**

### `WM_OT_properties_add`

Add your own property to the data-block

#### Methods

- **`execute(self, context)`**

### `WM_OT_properties_context_change`

Jump to a different tab inside the properties editor

#### Methods

- **`execute(self, context)`**

### `WM_OT_properties_edit`

Change a custom property's type, or adjust how it is displayed in the interface

#### Methods

- **`subtype_items_cb(self, context)`**

- **`property_type_update_cb(self, context)`**

- **`convert_custom_property_to_string(item, name)`**

- **`get_property_type(item, property_name)`**

- **`get_property_id_type(item, property_name)`**

### `WM_OT_properties_edit_value`

Edit the value of a custom property

#### Methods

- **`execute(self, context)`**

- **`invoke(self, context, _event)`**

- **`draw(self, context)`**

### `WM_OT_properties_remove`

Internal use (edit a property data_path)

#### Methods

- **`execute(self, context)`**

### `WM_OT_sysinfo`

Generate system information, saved into a text file

#### Methods

- **`execute(self, _context)`**

- **`invoke(self, context, _event)`**

### `WM_OT_tool_set_by_brush_type`

Look up the most appropriate tool for the given brush type and activate that

#### Methods

- **`execute(self, context)`**

### `WM_OT_tool_set_by_id`

Set the tool by name (for key-maps)

#### Methods

- **`space_type_from_operator(op, context)`**

- **`execute(self, context)`**

### `WM_OT_tool_set_by_index`

Set the tool by index (for key-maps)

#### Methods

- **`execute(self, context)`**

### `WM_OT_toolbar`

#### Methods

- **`keymap_from_toolbar(context, space_type, *, use_fallback_keys=True, use_reset=True)`**

- **`execute(self, context)`**

### `WM_OT_toolbar_fallback_pie`

#### Methods

- **`invoke(self, context, event)`**

### `WM_OT_toolbar_prompt`

Leader key like functionality for accessing tools

#### Methods

- **`modal(self, context, event)`**

- **`invoke(self, context, event)`**

### `WM_OT_url_open`

Open a website in the web browser

#### Methods

- **`execute(self, _context)`**

### `WM_OT_url_open_preset`

Open a preset website in the web browser

#### Methods

- **`execute(self, context)`**

### `WM_PT_operator_presets`

#### Methods

- **`draw(self, context)`**

### `WORKSPACE_PT_addons`

#### Methods

- **`draw_header(self, context)`**

- **`draw(self, context)`**

### `WORKSPACE_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `WORKSPACE_PT_main`

#### Methods

- **`draw(self, context)`**

### `WORKSPACE_UL_addons_items`

#### Methods

- **`filter_items(self, _context, data, property)`**

- **`draw_item(self, context, layout, _data, addon, _icon, _active_data, _active_propname, _index)`**

### `WORLD_OT_convert_volume_to_mesh`

Convert the volume of a world to a mesh. The world's volume used to be rendered by EEVEE Legacy. Conversion is needed for it to render properly

#### Methods

- **`execute(self, context)`**

### `WORLD_PT_animation`

Mix-in class for Animation panels.

This class can be used to show a generic 'Animation' panel for IDs shown in
the properties editor. Specific ID types need specific subclasses.

For an example, see DATA_PT_camera_animation in properties_data_camera.py

#### Methods

- **`draw(self, context)`**

### `WORLD_PT_context_world`

#### Methods

- **`draw(self, context)`**

### `WORLD_PT_custom_props`

The subclass should have its own poll function
and the variable '_context_path' MUST be set.

### `WORLD_PT_viewport_display`

#### Methods

- **`draw(self, context)`**

### `WalkNavigation`

### `WarpModifier`

### `WaveModifier`

### `WeightedNormalModifier`

### `WeldModifier`

### `WhiteBalanceModifier`

### `Window`

### `WindowManager`

#### Methods

- **`popup_menu(self, draw_func, *, title='', icon='NONE')`**

- **`popover(self, draw_func, *, ui_units_x=0, keymap=None, from_active_button=False)`**

- **`popup_menu_pie(self, event, draw_func, *, title='', icon='NONE')`**

- **`fileselect_add(*args, **kwargs)`**
  WindowManager.fileselect_add(operator)

- **`modal_handler_add(*args, **kwargs)`**
  WindowManager.modal_handler_add(operator)

### `WipeStrip`

### `WireframeModifier`

### `WoodTexture`

### `WorkSpace`

#### Methods

- **`status_text_set(self, text)`**
  Set the status text or None to clear,

- **`status_text_set_internal(*args, **kwargs)`**
  WorkSpace.status_text_set_internal(text)

- **`active_addon(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `WorkSpaceTool`

### `World`

#### Methods

- **`cycles(*args, **kwargs)`**
  Intermediate storage for properties before registration.

- **`cycles_visibility(*args, **kwargs)`**
  Intermediate storage for properties before registration.

### `WorldLighting`

### `WorldMistSettings`

### `XrActionMap`

### `XrActionMapBinding`

### `XrActionMapBindings`

### `XrActionMapItem`

### `XrActionMapItems`

### `XrActionMaps`

#### Methods

- **`new(*args, **kwargs)`**
  XrActionMaps.new(xr_session_state, name, replace_existing)

- **`new_from_actionmap(*args, **kwargs)`**
  XrActionMaps.new_from_actionmap(xr_session_state, actionmap)

- **`remove(*args, **kwargs)`**
  XrActionMaps.remove(xr_session_state, actionmap)

- **`find(*args, **kwargs)`**
  XrActionMaps.find(xr_session_state, name)

### `XrComponentPath`

### `XrComponentPaths`

### `XrEventData`

### `XrSessionSettings`

### `XrSessionState`

#### Methods

- **`is_running(*args, **kwargs)`**
  XrSessionState.is_running(context)

- **`reset_to_base_pose(*args, **kwargs)`**
  XrSessionState.reset_to_base_pose(context)

- **`action_set_create(*args, **kwargs)`**
  XrSessionState.action_set_create(context, actionmap)

- **`action_create(*args, **kwargs)`**
  XrSessionState.action_create(context, actionmap, actionmap_item)

- **`action_binding_create(*args, **kwargs)`**
  XrSessionState.action_binding_create(context, actionmap, actionmap_item, actionmap_binding)

### `XrUserPath`

### `XrUserPaths`

### `bpy_func`

### `bpy_prop`

#### Methods

- **`path_from_id(...)`**
  .. method:: path_from_id()

- **`as_bytes(...)`**
  .. method:: as_bytes()

- **`update(...)`**
  .. method:: update()

### `bpy_prop_array`

#### Methods

- **`foreach_get(...)`**
  .. method:: foreach_get(seq)

- **`foreach_set(...)`**
  .. method:: foreach_set(seq)

### `bpy_prop_collection`

#### Methods

- **`foreach_get(...)`**
  .. method:: foreach_get(attr, seq)

- **`foreach_set(...)`**
  .. method:: foreach_set(attr, seq)

- **`keys(...)`**
  .. method:: keys()

- **`items(...)`**
  .. method:: items()

- **`values(...)`**
  .. method:: values()

### `bpy_prop_collection_idprop`

#### Methods

- **`add(...)`**
  .. method:: add()

- **`remove(...)`**
  .. method:: remove(index)

- **`clear(...)`**
  .. method:: clear()

- **`move(...)`**
  .. method:: move(src_index, dst_index)

### `bpy_struct`

#### Methods

- **`keys(...)`**
  .. method:: keys()

- **`values(...)`**
  .. method:: values()

- **`items(...)`**
  .. method:: items()

- **`get(...)`**
  .. method:: get(key, default=None)

- **`pop(...)`**
  .. method:: pop(key, default=None)

### `bpy_struct_meta_idprop`

type(object) -> the object's type
type(name, bases, dict, **kwds) -> a new type

### `wmOwnerID`

### `wmOwnerIDs`

### `wmTools`
