# Physics Module

Auto-generated documentation for `astro_lab.utils.blender.advanced.physics`

## Functions

### create_binary_stars()

Create binary star system.

### create_solar_system()

Create a simple solar system.

## Classes

### GravitationalSimulation

N-body gravitational simulations

#### Methods

**`create_n_body_system(bodies: List[Dict[str, Any]]) -> List[bpy_types.Object]`**

Create N-body gravitational simulation.

Args:
bodies: List of body parameters with mass, position, velocity

Returns:
List of created body objects

**`create_binary_system(primary_mass: float, secondary_mass: float, separation: float) -> Tuple[bpy_types.Object, bpy_types.Object]`**

Create gravitationally bound binary system.

Args:
primary_mass: Mass of primary star
secondary_mass: Mass of secondary star
separation: Orbital separation

Returns:
Tuple of (primary, secondary) objects

### OrbitalMechanics

Create realistic orbital mechanics visualizations

#### Methods

**`create_orbital_system(center_obj: bpy_types.Object, orbits: List[Dict[str, float]]) -> List[bpy_types.Object]`**

Create a system of orbiting objects with trails.

Args:
center_obj: Central object (star/planet)
orbits: List of orbit parameters

Returns:
List of created orbital objects

### PhysicsShaders

Shaders for astrophysical objects

#### Methods

**`create_body_material(mass: float, body_type: str) -> bpy.types.Material`**

Create material for astronomical body.

**`create_stellar_material(mass: float) -> bpy.types.Material`**

Create stellar material based on mass.
