# constraint

Part of `bpy.ops`
Module: `bpy.ops.constraint`

## Operators (18)

### `add_target`

bpy.ops.constraint.add_target()
Add a target to the constraint

### `apply`

bpy.ops.constraint.apply(constraint="", owner='OBJECT', report=False)
Apply constraint and remove from the stack

### `childof_clear_inverse`

bpy.ops.constraint.childof_clear_inverse(constraint="", owner='OBJECT')
Clear inverse correction for Child Of constraint

### `childof_set_inverse`

bpy.ops.constraint.childof_set_inverse(constraint="", owner='OBJECT')
Set inverse correction for Child Of constraint

### `copy`

bpy.ops.constraint.copy(constraint="", owner='OBJECT', report=False)
Duplicate constraint at the same position in the stack

### `copy_to_selected`

bpy.ops.constraint.copy_to_selected(constraint="", owner='OBJECT')
Copy constraint to other selected objects/bones

### `delete`

bpy.ops.constraint.delete(constraint="", owner='OBJECT', report=False)
Remove constraint from constraint stack

### `disable_keep_transform`

bpy.ops.constraint.disable_keep_transform()
Set the influence of this constraint to zero while trying to maintain the object's transformation. Other active constraints can still influence the final transformation

### `followpath_path_animate`

bpy.ops.constraint.followpath_path_animate(constraint="", owner='OBJECT', frame_start=1, length=100)
Add default animation for path used by constraint if it isn't animated already

### `limitdistance_reset`

bpy.ops.constraint.limitdistance_reset(constraint="", owner='OBJECT')
Reset limiting distance for Limit Distance Constraint

### `move_down`

bpy.ops.constraint.move_down(constraint="", owner='OBJECT')
Move constraint down in constraint stack

### `move_to_index`

bpy.ops.constraint.move_to_index(constraint="", owner='OBJECT', index=0)
Change the constraint's position in the list so it evaluates after the set number of others

### `move_up`

bpy.ops.constraint.move_up(constraint="", owner='OBJECT')
Move constraint up in constraint stack

### `normalize_target_weights`

bpy.ops.constraint.normalize_target_weights()
Normalize weights of all target bones

### `objectsolver_clear_inverse`

bpy.ops.constraint.objectsolver_clear_inverse(constraint="", owner='OBJECT')
Clear inverse correction for Object Solver constraint

### `objectsolver_set_inverse`

bpy.ops.constraint.objectsolver_set_inverse(constraint="", owner='OBJECT')
Set inverse correction for Object Solver constraint

### `remove_target`

bpy.ops.constraint.remove_target(index=0)
Remove the target from the constraint

### `stretchto_reset`

bpy.ops.constraint.stretchto_reset(constraint="", owner='OBJECT')
Reset original length of bone for Stretch To Constraint
