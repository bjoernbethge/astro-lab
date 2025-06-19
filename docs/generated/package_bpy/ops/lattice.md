# lattice

Part of `bpy.ops`
Module: `bpy.ops.lattice`

## Operators (8)

### `flip`

bpy.ops.lattice.flip(axis='U')
Mirror all control points without inverting the lattice deform

### `make_regular`

bpy.ops.lattice.make_regular()
Set UVW control points a uniform distance apart

### `select_all`

bpy.ops.lattice.select_all(action='TOGGLE')
Change selection of all UVW control points

### `select_less`

bpy.ops.lattice.select_less()
Deselect vertices at the boundary of each selection region

### `select_mirror`

bpy.ops.lattice.select_mirror(axis={'X'}, extend=False)
Select mirrored lattice points

### `select_more`

bpy.ops.lattice.select_more()
Select vertex directly linked to already selected ones

### `select_random`

bpy.ops.lattice.select_random(ratio=0.5, seed=0, action='SELECT')
Randomly select UVW control points

### `select_ungrouped`

bpy.ops.lattice.select_ungrouped(extend=False)
Select vertices without a group
