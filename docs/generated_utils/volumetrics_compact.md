# Volumetrics Module

Auto-generated documentation for `utils.blender.advanced.volumetrics`

## Functions

### create_nebula_complex()

Create a complex nebula with multiple emission regions.

### create_stellar_system_with_winds()

Create stellar system with stellar winds.

## Classes

### VolumetricAstronomy

Create volumetric astronomical phenomena

#### Methods

**`create_emission_nebula(center: Vector = Vector((0.0, 0.0, 0.0)), size: float = 10.0, nebula_type: str = 'h_alpha', density: float = 0.1) -> bpy_types.Object`**

Create an emission nebula with realistic structure.

Args:
center: Nebula center position
size: Nebula size
nebula_type: Type ('h_alpha', 'oxygen', 'planetary', 'supernova')
density: Base density value

Returns:
Created nebula object

**`create_stellar_wind(star_obj: bpy_types.Object, wind_speed: float = 500.0, mass_loss_rate: float = 1e-06, wind_radius: float = 5.0) -> bpy_types.Object`**

Create stellar wind visualization around a star.

Args:
star_obj: Star object to create wind around
wind_speed: Wind velocity in km/s
mass_loss_rate: Mass loss rate in solar masses per year
wind_radius: Maximum wind radius

Returns:
Created stellar wind object

**`create_planetary_atmosphere(planet_obj: bpy_types.Object, atmosphere_type: str = 'earth_like', thickness: float = 0.5) -> bpy_types.Object`**

Create layered planetary atmosphere with scattering.

Args:
planet_obj: Planet object
atmosphere_type: Type ('earth_like', 'mars_like', 'venus_like', 'gas_giant')
thickness: Atmosphere thickness relative to planet radius

Returns:
Created atmosphere object

**`create_galactic_dust_lane(start_pos: Vector, end_pos: Vector, width: float = 2.0, dust_density: float = 0.05) -> bpy_types.Object`**

Create galactic dust lane with absorption and scattering.

Args:
start_pos: Start position of dust lane
end_pos: End position of dust lane
width: Width of the dust lane
dust_density: Dust density

Returns:
Created dust lane object

### VolumetricShaders

Volumetric shaders for astronomical phenomena

#### Methods

**`create_emission_nebula_material(nebula_type: str) -> bpy.types.Material`**

Create emission nebula material based on spectral lines.

Args:
nebula_type: Type of emission nebula

Returns:
Created material

**`create_stellar_wind_material(wind_speed: float) -> bpy.types.Material`**

Create stellar wind material with velocity-based opacity.

Args:
wind_speed: Wind velocity in km/s

Returns:
Created material

**`create_atmospheric_material(atmosphere_type: str) -> bpy.types.Material`**

Create planetary atmosphere material with Rayleigh scattering.

Args:
atmosphere_type: Type of atmosphere

Returns:
Created material

**`create_dust_lane_material(dust_density: float) -> bpy.types.Material`**

Create galactic dust lane material with absorption.

Args:
dust_density: Dust density

Returns:
Created material
