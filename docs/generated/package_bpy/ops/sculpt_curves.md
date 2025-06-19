# sculpt_curves

Part of `bpy.ops`
Module: `bpy.ops.sculpt_curves`

## Operators (4)

### `brush_stroke`

bpy.ops.sculpt_curves.brush_stroke(stroke=[], mode='NORMAL', pen_flip=False)
Sculpt curves using a brush

### `min_distance_edit`

bpy.ops.sculpt_curves.min_distance_edit()
Change the minimum distance used by the density brush

### `select_grow`

bpy.ops.sculpt_curves.select_grow(distance=0.1)
Select curves which are close to curves that are selected already

### `select_random`

bpy.ops.sculpt_curves.select_random(seed=0, partial=False, probability=0.5, min=0, constant_per_curve=True)
Randomizes existing selection or create new random selection
