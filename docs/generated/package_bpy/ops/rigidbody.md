# rigidbody

Part of `bpy.ops`
Module: `bpy.ops.rigidbody`

## Operators (13)

### `bake_to_keyframes`

bpy.ops.rigidbody.bake_to_keyframes(frame_start=1, frame_end=250, step=1)
Bake rigid body transformations of selected objects to keyframes

### `connect`

bpy.ops.rigidbody.connect(con_type='FIXED', pivot_type='CENTER', connection_pattern='SELECTED_TO_ACTIVE')
Create rigid body constraints between selected rigid bodies

### `constraint_add`

bpy.ops.rigidbody.constraint_add(type='FIXED')
Add Rigid Body Constraint to active object

### `constraint_remove`

bpy.ops.rigidbody.constraint_remove()
Remove Rigid Body Constraint from Object

### `mass_calculate`

bpy.ops.rigidbody.mass_calculate(material='Air', density=1)
Automatically calculate mass values for Rigid Body Objects based on volume

### `object_add`

bpy.ops.rigidbody.object_add(type='ACTIVE')
Add active object as Rigid Body

### `object_remove`

bpy.ops.rigidbody.object_remove()
Remove Rigid Body settings from Object

### `object_settings_copy`

bpy.ops.rigidbody.object_settings_copy()
Copy Rigid Body settings from active object to selected

### `objects_add`

bpy.ops.rigidbody.objects_add(type='ACTIVE')
Add selected objects as Rigid Bodies

### `objects_remove`

bpy.ops.rigidbody.objects_remove()
Remove selected objects from Rigid Body simulation

### `shape_change`

bpy.ops.rigidbody.shape_change(type='MESH')
Change collision shapes for selected Rigid Body Objects

### `world_add`

bpy.ops.rigidbody.world_add()
Add Rigid Body simulation world to the current scene

### `world_remove`

bpy.ops.rigidbody.world_remove()
Remove Rigid Body simulation world from the current scene
