# poselib

Part of `bpy.ops`
Module: `bpy.ops.poselib`

## Operators (11)

### `apply_pose_asset`

bpy.ops.poselib.apply_pose_asset(blend_factor=1, flipped=False)
Apply the given Pose Action to the rig

### `asset_delete`

bpy.ops.poselib.asset_delete()
Delete the selected Pose Asset

### `asset_modify`

bpy.ops.poselib.asset_modify(mode='ADJUST')
Update the selected pose asset in the asset library from the currently selected bones. The mode defines how the asset is updated

### `blend_pose_asset`

bpy.ops.poselib.blend_pose_asset(blend_factor=0, flipped=False, release_confirm=False)
Blend the given Pose Action to the rig

### `convert_old_object_poselib`

bpy.ops.poselib.convert_old_object_poselib()
Create a pose asset for each pose marker in this legacy pose library data-block

### `convert_old_poselib`

bpy.ops.poselib.convert_old_poselib()
Create a pose asset for each pose marker in the current action

### `copy_as_asset`

bpy.ops.poselib.copy_as_asset()
Create a new pose asset on the clipboard, to be pasted into an Asset Browser

### `create_pose_asset`

bpy.ops.poselib.create_pose_asset(pose_name="", asset_library_reference='<UNKNOWN ENUM>', catalog_path="", activate_new_action=False)
Create a new asset from the selected bones in the scene

### `paste_asset`

bpy.ops.poselib.paste_asset()
Paste the Asset that was previously copied using Copy As Asset

### `pose_asset_select_bones`

bpy.ops.poselib.pose_asset_select_bones(select=True, flipped=False)
Select those bones that are used in this pose

### `restore_previous_action`

bpy.ops.poselib.restore_previous_action()
Switch back to the previous Action, after creating a pose asset
