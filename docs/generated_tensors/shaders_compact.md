# Shaders Module

Auto-generated documentation for `astro_lab.utils.blender.advanced.shaders`

## Functions

### create_planetary_system()

Create system with different planetary types.

### create_stellar_showcase()

Create showcase of different stellar types.

## Classes

### AstronomicalShaders

Create scientifically accurate astronomical shaders

#### Methods

**`create_stellar_blackbody_shader(temperature: float, luminosity: float = 1.0, stellar_class: Optional[str] = None) -> bpy.types.Material`**

Create physically accurate stellar shader based on blackbody radiation.

Args:
temperature: Stellar temperature in Kelvin
luminosity: Stellar luminosity relative to Sun
stellar_class: Optional spectral class (O, B, A, F, G, K, M)

Returns:
Created stellar material

**`create_planetary_surface_shader(planet_type: str, composition: Dict[str, float] = None) -> bpy.types.Material`**

Create planetary surface shader based on composition.

Args:
planet_type: Type of planet ('terrestrial', 'gas_giant', 'ice_giant', 'moon')
composition: Compositional percentages {'rock': 0.7, 'ice': 0.3, etc.}

Returns:
Created planetary material

**`create_nebula_emission_shader(emission_lines: List[str], density_variation: float = 0.5) -> bpy.types.Material`**

Create nebula shader based on emission line spectra.

Args:
emission_lines: List of emission lines ['H_alpha', 'O_III', 'H_beta', etc.]
density_variation: Amount of density variation (0-1)

Returns:
Created nebula material

**`create_atmospheric_scattering_shader(atmosphere_type: str, scale_height: float = 8.5) -> bpy.types.Material`**

Create atmospheric scattering shader (Rayleigh + Mie).

Args:
atmosphere_type: Type of atmosphere ('earth', 'mars', 'venus', 'titan')
scale_height: Atmospheric scale height in km

Returns:
Created atmospheric material
