# particle

Part of `bpy.ops`
Module: `bpy.ops.particle`

## Operators (36)

### `brush_edit`

bpy.ops.particle.brush_edit(stroke=[], pen_flip=False)
Apply a stroke of brush to the particles

### `connect_hair`

bpy.ops.particle.connect_hair(all=False)
Connect hair to the emitter mesh

### `copy_particle_systems`

bpy.ops.particle.copy_particle_systems(space='OBJECT', remove_target_particles=True, use_active=False)
Copy particle systems from the active object to selected objects

### `delete`

bpy.ops.particle.delete(type='PARTICLE')
Delete selected particles or keys

### `disconnect_hair`

bpy.ops.particle.disconnect_hair(all=False)
Disconnect hair from the emitter mesh

### `duplicate_particle_system`

bpy.ops.particle.duplicate_particle_system(use_duplicate_settings=False)
Duplicate particle system within the active object

### `dupliob_copy`

bpy.ops.particle.dupliob_copy()
Duplicate the current instance object

### `dupliob_move_down`

bpy.ops.particle.dupliob_move_down()
Move instance object down in the list

### `dupliob_move_up`

bpy.ops.particle.dupliob_move_up()
Move instance object up in the list

### `dupliob_refresh`

bpy.ops.particle.dupliob_refresh()
Refresh list of instance objects and their weights

### `dupliob_remove`

bpy.ops.particle.dupliob_remove()
Remove the selected instance object

### `edited_clear`

bpy.ops.particle.edited_clear()
Undo all edition performed on the particle system

### `hair_dynamics_preset_add`

bpy.ops.particle.hair_dynamics_preset_add(name="", remove_name=False, remove_active=False)
Add or remove a Hair Dynamics Preset

### `hide`

bpy.ops.particle.hide(unselected=False)
Hide selected particles

### `mirror`

bpy.ops.particle.mirror()
Duplicate and mirror the selected particles along the local X axis

### `new`

bpy.ops.particle.new()
Add new particle settings

### `new_target`

bpy.ops.particle.new_target()
Add a new particle target

### `particle_edit_toggle`

bpy.ops.particle.particle_edit_toggle()
Toggle particle edit mode

### `rekey`

bpy.ops.particle.rekey(keys_number=2)
Change the number of keys of selected particles (root and tip keys included)

### `remove_doubles`

bpy.ops.particle.remove_doubles(threshold=0.0002)
Remove selected particles close enough of others

### `reveal`

bpy.ops.particle.reveal(select=True)
Show hidden particles

### `select_all`

bpy.ops.particle.select_all(action='TOGGLE')
(De)select all particles' keys

### `select_less`

bpy.ops.particle.select_less()
Deselect boundary selected keys of each particle

### `select_linked`

bpy.ops.particle.select_linked()
Select all keys linked to already selected ones

### `select_linked_pick`

bpy.ops.particle.select_linked_pick(deselect=False, location=(0, 0))
Select nearest particle from mouse pointer

### `select_more`

bpy.ops.particle.select_more()
Select keys linked to boundary selected keys of each particle

### `select_random`

bpy.ops.particle.select_random(ratio=0.5, seed=0, action='SELECT', type='HAIR')
Select a randomly distributed set of hair or points

### `select_roots`

bpy.ops.particle.select_roots(action='SELECT')
Select roots of all visible particles

### `select_tips`

bpy.ops.particle.select_tips(action='SELECT')
Select tips of all visible particles

### `shape_cut`

bpy.ops.particle.shape_cut()
Cut hair to conform to the set shape object

### `subdivide`

bpy.ops.particle.subdivide()
Subdivide selected particles segments (adds keys)

### `target_move_down`

bpy.ops.particle.target_move_down()
Move particle target down in the list

### `target_move_up`

bpy.ops.particle.target_move_up()
Move particle target up in the list

### `target_remove`

bpy.ops.particle.target_remove()
Remove the selected particle target

### `unify_length`

bpy.ops.particle.unify_length()
Make selected hair the same length

### `weight_set`

bpy.ops.particle.weight_set(factor=1)
Set the weight of selected keys
