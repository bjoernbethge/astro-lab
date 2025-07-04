"""
Galaxy Calculation Utilities
============================

Comprehensive galaxy physics calculations for astronomical visualization.
Includes morphology, dynamics, photometry, and evolution calculations.
"""

import logging
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
SOLAR_MASS = 1.989e30  # kg
SOLAR_LUMINOSITY = 3.828e26  # watts
PC_TO_M = 3.086e16  # meters per parsec
C_LIGHT = 2.998e8  # m/s
H0 = 70.0  # km/s/Mpc (Hubble constant)


def calculate_galaxy_luminosity_distance(
    redshift: float, H0: float = 70.0, omega_m: float = 0.3, omega_lambda: float = 0.7
) -> float:
    """
    Calculate luminosity distance to galaxy using cosmological parameters.

    Args:
        redshift: Galaxy redshift
        H0: Hubble constant in km/s/Mpc
        omega_m: Matter density parameter
        omega_lambda: Dark energy density parameter

    Returns:
        Luminosity distance in Mpc
    """
    # Simple approximation for low redshifts
    if redshift < 0.1:
        return (C_LIGHT * redshift) / (H0 * 1000)  # Convert to Mpc

    # More accurate calculation for higher redshifts
    z = redshift

    # Numerical integration (simplified)
    z_array = np.linspace(0, z, 100)
    dz = z_array[1] - z_array[0]
    E_z_array = np.sqrt(omega_m * (1 + z_array) ** 3 + omega_lambda)

    integral = np.trapz(1.0 / E_z_array, dx=dz)

    D_L = (C_LIGHT * (1 + z) * integral) / (H0 * 1000)  # Mpc

    return D_L


def calculate_galaxy_angular_size(
    physical_size_kpc: float, distance_mpc: float
) -> float:
    """
    Calculate angular size of galaxy in arcseconds.

    Args:
        physical_size_kpc: Physical size in kiloparsecs
        distance_mpc: Distance in megaparsecs

    Returns:
        Angular size in arcseconds
    """
    # Convert to same units
    distance_kpc = distance_mpc * 1000

    # Angular size in radians
    angular_size_rad = physical_size_kpc / distance_kpc

    # Convert to arcseconds
    angular_size_arcsec = angular_size_rad * 206265

    return angular_size_arcsec


def calculate_galaxy_distance_modulus(distance_mpc: float) -> float:
    """
    Calculate distance modulus for galaxy.

    Args:
        distance_mpc: Distance in megaparsecs

    Returns:
        Distance modulus in magnitudes
    """
    distance_pc = distance_mpc * 1e6
    return 5 * np.log10(distance_pc) - 5


def calculate_galaxy_mass_from_luminosity(
    luminosity_solar: float, mass_to_light_ratio: float = 2.0
) -> float:
    """
    Calculate galaxy mass from luminosity using mass-to-light ratio.

    Args:
        luminosity_solar: Luminosity in solar units
        mass_to_light_ratio: Mass-to-light ratio in solar units

    Returns:
        Galaxy mass in solar masses
    """
    return luminosity_solar * mass_to_light_ratio


def calculate_galaxy_rotation_curve(
    radius_kpc: np.ndarray, total_mass_solar: float, disk_scale_length: float = 3.0
) -> np.ndarray:
    """
    Calculate galaxy rotation curve using simple disk + halo model.

    Args:
        radius_kpc: Radii in kiloparsecs
        total_mass_solar: Total galaxy mass in solar masses
        disk_scale_length: Disk scale length in kpc

    Returns:
        Rotation velocities in km/s
    """
    # Convert to SI units
    radius_m = radius_kpc * 1000 * PC_TO_M
    total_mass_kg = total_mass_solar * SOLAR_MASS

    # Simple exponential disk + dark matter halo
    disk_fraction = 0.3
    halo_fraction = 0.7

    # Disk component (exponential profile)
    disk_mass = total_mass_kg * disk_fraction
    scale_length_m = disk_scale_length * 1000 * PC_TO_M

    x = radius_m / scale_length_m
    disk_mass_enclosed = disk_mass * (1 - np.exp(-x) * (1 + x))

    # Halo component (NFW profile approximation)
    halo_mass = total_mass_kg * halo_fraction
    scale_radius_m = 20 * 1000 * PC_TO_M  # 20 kpc typical scale radius

    halo_mass_enclosed = halo_mass * (radius_m / (radius_m + scale_radius_m))

    # Total enclosed mass
    total_enclosed = disk_mass_enclosed + halo_mass_enclosed

    # Rotation velocity (circular velocity)
    G = 6.674e-11  # m³/kg/s²
    v_rot = np.sqrt(G * total_enclosed / radius_m) / 1000  # km/s

    return v_rot


def calculate_galaxy_density_profile(
    radius_kpc: np.ndarray,
    central_density: float = 1e9,
    scale_length: float = 3.0,
    profile_type: str = "exponential",
) -> np.ndarray:
    """
    Calculate galaxy surface density profile.

    Args:
        radius_kpc: Radii in kiloparsecs
        central_density: Central surface density in Msun/pc²
        scale_length: Scale length in kpc
        profile_type: "exponential", "sersic", or "deVaucouleurs"

    Returns:
        Surface density in Msun/pc²
    """
    if profile_type == "exponential":
        # Exponential disk profile
        density = central_density * np.exp(-radius_kpc / scale_length)

    elif profile_type == "sersic":
        # Sersic profile (n=1 is exponential, n=4 is de Vaucouleurs)
        n = 2.0  # Sersic index
        b_n = 1.9992 * n - 0.3271  # Approximation for b_n

        density = central_density * np.exp(
            -b_n * ((radius_kpc / scale_length) ** (1 / n) - 1)
        )

    elif profile_type == "deVaucouleurs":
        # de Vaucouleurs profile (Sersic n=4)
        b_4 = 7.6692
        density = central_density * np.exp(
            -b_4 * ((radius_kpc / scale_length) ** 0.25 - 1)
        )

    else:
        raise ValueError(f"Unknown profile type: {profile_type}")

    return density


def calculate_star_formation_rate(
    galaxy_mass_solar: float, galaxy_type: str = "Sb", redshift: float = 0.0
) -> float:
    """
    Calculate star formation rate using empirical relations.

    Args:
        galaxy_mass_solar: Galaxy stellar mass in solar masses
        galaxy_type: Galaxy morphological type
        redshift: Galaxy redshift

    Returns:
        Star formation rate in Msun/year
    """
    # Main sequence relation (log-linear)
    log_mass = np.log10(galaxy_mass_solar)

    # Base star formation rate
    if log_mass < 9.5:
        log_sfr_base = 0.8 * log_mass - 7.2
    else:
        log_sfr_base = 0.3 * log_mass - 2.5

    # Morphology correction
    morph_correction = {
        "E0": -1.5,
        "E3": -1.2,
        "E7": -1.0,
        "S0": -0.8,
        "Sa": -0.3,
        "Sb": 0.0,
        "Sc": 0.3,
        "SBa": -0.2,
        "SBb": 0.1,
        "SBc": 0.4,
        "Irr": 0.6,
    }

    correction = morph_correction.get(galaxy_type, 0.0)

    # Redshift evolution (more star formation at higher z)
    z_evolution = 0.5 * redshift

    log_sfr = log_sfr_base + correction + z_evolution

    return 10**log_sfr


def calculate_galaxy_color_index(
    stellar_mass: float, star_formation_rate: float, metallicity: float = 0.02
) -> float:
    """
    Calculate galaxy color index (g-r) from physical properties.

    Args:
        stellar_mass: Stellar mass in solar masses
        star_formation_rate: Star formation rate in Msun/year
        metallicity: Metallicity (fraction)

    Returns:
        g-r color index in magnitudes
    """
    # Specific star formation rate
    log_ssfr = np.log10(star_formation_rate / stellar_mass)

    # Mass dependence (more massive galaxies are redder)
    log_mass = np.log10(stellar_mass)
    mass_term = 0.15 * (log_mass - 10.0)

    # Star formation dependence (more star formation = bluer)
    sfr_term = -0.25 * (log_ssfr + 10.0)

    # Metallicity dependence (higher metallicity = redder)
    metal_term = 5.0 * (metallicity - 0.02)

    # Base color for typical galaxy
    base_color = 0.65

    color_gr = base_color + mass_term + sfr_term + metal_term

    return np.clip(color_gr, 0.2, 1.5)


def calculate_galaxy_metallicity(
    stellar_mass: float, star_formation_rate: float
) -> float:
    """
    Calculate galaxy metallicity using mass-metallicity relation.

    Args:
        stellar_mass: Stellar mass in solar masses
        star_formation_rate: Star formation rate in Msun/year

    Returns:
        Metallicity as fraction of solar (Z/Z_sun)
    """
    log_mass = np.log10(stellar_mass)
    log_sfr = np.log10(star_formation_rate)

    # Fundamental metallicity relation
    # Based on Mannucci et al. (2010)
    mu = log_mass - 0.32 * log_sfr

    if mu < 9.5:
        metallicity_solar = 0.1 * 10 ** (0.6 * mu - 5.4)
    else:
        metallicity_solar = 0.6 * 10 ** (0.3 * mu - 2.85)

    return metallicity_solar


def calculate_galaxy_age_from_color(color_gr: float, metallicity: float = 1.0) -> float:
    """
    Estimate galaxy age from color index.

    Args:
        color_gr: g-r color index
        metallicity: Metallicity relative to solar

    Returns:
        Age in Gyr
    """
    # Empirical relation from stellar population models
    # Redder galaxies are typically older

    base_age = 2.0 + 8.0 * (color_gr - 0.3)

    # Metallicity correction (higher metallicity = younger for given color)
    metal_correction = -1.0 * np.log10(metallicity)

    age_gyr = base_age + metal_correction

    return np.clip(age_gyr, 0.1, 13.8)  # Age of universe


def calculate_galaxy_size_from_mass(
    stellar_mass: float, galaxy_type: str = "Sb"
) -> float:
    """
    Calculate galaxy effective radius from stellar mass.

    Args:
        stellar_mass: Stellar mass in solar masses
        galaxy_type: Galaxy morphological type

    Returns:
        Effective radius in kpc
    """
    log_mass = np.log10(stellar_mass)

    # Mass-size relation (different for early/late types)
    if galaxy_type in ["E0", "E3", "E7", "S0"]:
        # Early-type galaxies
        log_re = 0.56 * log_mass - 4.54
    else:
        # Late-type galaxies
        log_re = 0.22 * log_mass - 1.43

    effective_radius = 10**log_re

    return effective_radius


def calculate_galaxy_surface_brightness(
    luminosity: float, effective_radius_kpc: float, band: str = "r"
) -> float:
    """
    Calculate galaxy surface brightness.

    Args:
        luminosity: Luminosity in solar units
        effective_radius_kpc: Effective radius in kpc
        band: Photometric band

    Returns:
        Surface brightness in mag/arcsec²
    """
    # Convert effective radius to arcsec at 10 pc (for absolute magnitude)
    # This is a simplification - normally would need distance
    re_arcsec = effective_radius_kpc * 1000 / 10 * 206265  # rough approximation

    # Effective area
    area_arcsec2 = np.pi * re_arcsec**2

    # Surface brightness
    absolute_mag = -2.5 * np.log10(luminosity) + 4.83  # Solar absolute magnitude
    surface_brightness = absolute_mag + 2.5 * np.log10(area_arcsec2)

    return surface_brightness


def calculate_bulge_to_disk_ratio(galaxy_type: str, stellar_mass: float) -> float:
    """
    Calculate bulge-to-disk mass ratio.

    Args:
        galaxy_type: Galaxy morphological type
        stellar_mass: Total stellar mass

    Returns:
        Bulge-to-disk ratio
    """
    # Base ratios by type
    base_ratios = {
        "E0": 100.0,
        "E3": 100.0,
        "E7": 100.0,  # Pure bulge
        "S0": 1.5,
        "Sa": 0.8,
        "Sb": 0.3,
        "Sc": 0.1,
        "SBa": 0.6,
        "SBb": 0.25,
        "SBc": 0.08,
        "Irr": 0.02,
    }

    base_ratio = base_ratios.get(galaxy_type, 0.3)

    # Mass dependence (more massive galaxies have larger bulges)
    log_mass = np.log10(stellar_mass)
    mass_correction = 10 ** (0.2 * (log_mass - 10.5))

    return base_ratio * mass_correction


def get_galaxy_properties(
    galaxy_type: str, stellar_mass: float, redshift: float = 0.0
) -> Dict[str, float]:
    """
    Calculate comprehensive galaxy properties.

    Args:
        galaxy_type: Galaxy morphological type
        stellar_mass: Stellar mass in solar masses
        redshift: Galaxy redshift

    Returns:
        Dict containing all galaxy properties
    """
    # Calculate derived properties
    sfr = calculate_star_formation_rate(stellar_mass, galaxy_type, redshift)
    metallicity = calculate_galaxy_metallicity(stellar_mass, sfr)
    color_gr = calculate_galaxy_color_index(stellar_mass, sfr, metallicity)
    age = calculate_galaxy_age_from_color(color_gr, metallicity)
    size = calculate_galaxy_size_from_mass(stellar_mass, galaxy_type)
    distance = calculate_galaxy_luminosity_distance(redshift) if redshift > 0 else 10.0

    # Angular size
    angular_size = calculate_galaxy_angular_size(size, distance)

    # Surface brightness
    luminosity = stellar_mass / 2.0  # Rough M/L ratio
    surface_brightness = calculate_galaxy_surface_brightness(luminosity, size)

    # Bulge-to-disk ratio
    bulge_disk_ratio = calculate_bulge_to_disk_ratio(galaxy_type, stellar_mass)

    return {
        "stellar_mass": stellar_mass,
        "star_formation_rate": sfr,
        "metallicity": metallicity,
        "color_gr": color_gr,
        "age_gyr": age,
        "effective_radius_kpc": size,
        "distance_mpc": distance,
        "angular_size_arcsec": angular_size,
        "surface_brightness": surface_brightness,
        "bulge_disk_ratio": bulge_disk_ratio,
        "luminosity": luminosity,
    }


def get_galaxy_morphology_params(galaxy_type: str) -> Dict[str, float]:
    """
    Get morphological parameters for galaxy type.

    Args:
        galaxy_type: Galaxy morphological type

    Returns:
        Dict containing morphological parameters
    """
    from .astronomical_data import GALAXY_TYPES

    if galaxy_type not in GALAXY_TYPES:
        logger.warning(f"Unknown galaxy type: {galaxy_type}, using Sb")
        galaxy_type = "Sb"

    params = GALAXY_TYPES[galaxy_type].copy()

    # Add derived parameters
    params["is_early_type"] = galaxy_type in ["E0", "E3", "E7", "S0"]
    params["has_spiral_arms"] = galaxy_type in ["Sa", "Sb", "Sc", "SBa", "SBb", "SBc"]
    params["has_bar"] = galaxy_type.startswith("SB")
    params["is_irregular"] = galaxy_type == "Irr"

    return params
