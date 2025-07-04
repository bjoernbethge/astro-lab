"""
Nebula Calculation Utilities
============================

Comprehensive nebula physics calculations for astronomical visualization.
Includes emission line calculations, ionization physics, and morphology.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Physical constants
PC_TO_M = 3.086e16  # meters per parsec
SOLAR_MASS = 1.989e30  # kg
BOLTZMANN_K = 1.381e-23  # J/K
PLANCK_H = 6.626e-34  # J⋅s
C_LIGHT = 2.998e8  # m/s
ELECTRON_CHARGE = 1.602e-19  # C

# Emission line wavelengths (Angstroms)
EMISSION_LINES = {
    "H_alpha": 6562.8,
    "H_beta": 4861.3,
    "H_gamma": 4340.5,
    "H_delta": 4101.7,
    "O_III_4959": 4958.9,
    "O_III_5007": 5006.8,
    "N_II_6548": 6548.0,
    "N_II_6584": 6583.5,
    "S_II_6716": 6716.4,
    "S_II_6731": 6730.8,
    "O_II_3727": 3726.0,
    "O_II_3729": 3728.8,
    "He_II_4686": 4685.7,
    "Ar_III_7136": 7135.8,
}


def calculate_nebula_temperature(
    ionization_parameter: float, central_star_temp: float = 50000
) -> float:
    """
    Calculate nebula electron temperature from ionization conditions.

    Args:
        ionization_parameter: log(U), dimensionless
        central_star_temp: Central star temperature in K

    Returns:
        Electron temperature in K
    """
    # Empirical relation based on photoionization models
    log_U = ionization_parameter

    # Base temperature from ionization balance
    if log_U > -2:
        # High ionization
        T_e = 12000 + 3000 * (log_U + 2)
    elif log_U > -3:
        # Medium ionization
        T_e = 8000 + 4000 * (log_U + 3)
    else:
        # Low ionization
        T_e = 6000 + 2000 * (log_U + 4)

    # Central star temperature influence
    stellar_factor = (central_star_temp / 50000) ** 0.3
    T_e *= stellar_factor

    return np.clip(T_e, 5000, 20000)


def calculate_nebula_density(
    nebula_type: str, size_pc: float, total_mass_solar: float = 1.0
) -> float:
    """
    Calculate nebula particle density.

    Args:
        nebula_type: Type of nebula
        size_pc: Nebula size in parsecs
        total_mass_solar: Total nebula mass in solar masses

    Returns:
        Number density in particles/cm³
    """
    # Convert to SI units
    size_m = size_pc * PC_TO_M
    mass_kg = total_mass_solar * SOLAR_MASS

    # Volume (assuming spherical)
    volume_m3 = (4 / 3) * np.pi * (size_m / 2) ** 3

    # Mass density
    mass_density = mass_kg / volume_m3  # kg/m³

    # Convert to number density (assuming mostly hydrogen)
    proton_mass = 1.673e-27  # kg
    number_density_m3 = mass_density / proton_mass

    # Convert to cm⁻³
    number_density_cm3 = number_density_m3 / 1e6

    # Typical densities by nebula type
    typical_densities = {
        "H_II": 1e2,
        "planetary": 1e4,
        "supernova_remnant": 1e1,
        "dark": 1e3,
        "reflection": 1e2,
    }

    # Use typical value if calculated value is unrealistic
    typical = typical_densities.get(nebula_type, 1e2)

    if number_density_cm3 < typical * 0.1 or number_density_cm3 > typical * 100:
        return typical

    return number_density_cm3


def calculate_nebula_mass(
    size_pc: float, density_cm3: float, mean_molecular_weight: float = 1.4
) -> float:
    """
    Calculate nebula mass from size and density.

    Args:
        size_pc: Nebula diameter in parsecs
        density_cm3: Number density in particles/cm³
        mean_molecular_weight: Mean molecular weight

    Returns:
        Mass in solar masses
    """
    # Convert to SI units
    size_m = size_pc * PC_TO_M
    density_m3 = density_cm3 * 1e6

    # Volume (spherical)
    volume_m3 = (4 / 3) * np.pi * (size_m / 2) ** 3

    # Mass
    proton_mass = 1.673e-27  # kg
    mass_kg = density_m3 * volume_m3 * proton_mass * mean_molecular_weight

    # Convert to solar masses
    mass_solar = mass_kg / SOLAR_MASS

    return mass_solar


def calculate_nebula_luminosity(
    density_cm3: float,
    temperature_K: float,
    size_pc: float,
    ionization_fraction: float = 1.0,
) -> float:
    """
    Calculate nebula H-alpha luminosity.

    Args:
        density_cm3: Electron density in cm⁻³
        temperature_K: Electron temperature in K
        size_pc: Nebula size in parsecs
        ionization_fraction: Fraction of hydrogen ionized

    Returns:
        H-alpha luminosity in erg/s
    """
    # Recombination coefficient for H-alpha (Case B)
    # T4 = temperature_K / 1e4  # Removed unused variable
    alpha_B = 2.59e-13 * (T4 ** (-0.833 - 0.034 * np.log(T4)))  # cm³/s

    # H-alpha emissivity
    h_alpha_fraction = 0.45  # Fraction of recombinations producing H-alpha

    # Volume
    size_m = size_pc * PC_TO_M
    volume_m3 = (4 / 3) * np.pi * (size_m / 2) ** 3
    volume_cm3 = volume_m3 * 1e6

    # Number of recombinations per second
    n_e = density_cm3 * ionization_fraction
    n_p = density_cm3 * ionization_fraction

    recombination_rate = alpha_B * n_e * n_p * volume_cm3  # s⁻¹

    # H-alpha photon production rate
    h_alpha_rate = recombination_rate * h_alpha_fraction

    # Energy per H-alpha photon
    wavelength_m = 6.563e-7  # H-alpha wavelength in meters
    photon_energy = PLANCK_H * C_LIGHT / wavelength_m  # J

    # Luminosity
    luminosity_watts = h_alpha_rate * photon_energy
    luminosity_erg_s = luminosity_watts * 1e7  # Convert to erg/s

    return luminosity_erg_s


def calculate_nebula_expansion_velocity(
    nebula_type: str, age_years: float, energy_erg: float = 1e51
) -> float:
    """
    Calculate nebula expansion velocity.

    Args:
        nebula_type: Type of nebula
        age_years: Age in years
        energy_erg: Input energy in erg

    Returns:
        Expansion velocity in km/s
    """
    age_s = age_years * 3.156e7  # Convert to seconds

    if nebula_type == "supernova_remnant":
        # Sedov-Taylor blast wave solution
        # v = (2/5) * R/t for adiabatic expansion
        # R = (E*t²/(ρ))^(1/5)

        # Typical ISM density
        rho_ism = 1.673e-24  # kg/m³ (1 particle/cm³)

        # Radius from energy and time
        radius_m = (energy_erg * 1e-7 * age_s**2 / rho_ism) ** (1 / 5)

        # Expansion velocity
        velocity_ms = (2 / 5) * radius_m / age_s
        velocity_kms = velocity_ms / 1000

    elif nebula_type == "planetary":
        # Typical planetary nebula expansion
        # Roughly constant velocity after initial acceleration
        velocity_kms = 20 + 10 * np.random.random()  # 20-30 km/s typical

    elif nebula_type == "H_II":
        # H II regions expand due to pressure gradients
        # Much slower than SNRs
        velocity_kms = 5 + 5 * np.random.random()  # 5-10 km/s typical

    else:
        # Default expansion velocity
        velocity_kms = 10.0

    return velocity_kms


def calculate_nebula_angular_size(physical_size_pc: float, distance_pc: float) -> float:
    """
    Calculate angular size of nebula in arcseconds.

    Args:
        physical_size_pc: Physical size in parsecs
        distance_pc: Distance in parsecs

    Returns:
        Angular size in arcseconds
    """
    # Angular size in radians
    angular_size_rad = physical_size_pc / distance_pc

    # Convert to arcseconds
    angular_size_arcsec = angular_size_rad * 206265

    return angular_size_arcsec


def calculate_nebula_distance_modulus(distance_pc: float) -> float:
    """
    Calculate distance modulus for nebula.

    Args:
        distance_pc: Distance in parsecs

    Returns:
        Distance modulus in magnitudes
    """
    return 5 * np.log10(distance_pc) - 5


def calculate_nebula_color_index(
    temperature_K: float, nebula_type: str = "H_II"
) -> Tuple[float, float, float]:
    """
    Calculate nebula color based on temperature and type.

    Args:
        temperature_K: Electron temperature in K
        nebula_type: Type of nebula

    Returns:
        RGB color tuple (0-1 range)
    """
    if nebula_type == "H_II":
        # Emission nebulae - dominated by H-alpha (red) and O III (blue-green)
        # Temperature affects ionization balance
        if temperature_K > 12000:
            # High ionization - more O III
            r, g, b = 0.8, 0.9, 1.0
        elif temperature_K > 8000:
            # Medium ionization - balanced
            r, g, b = 1.0, 0.6, 0.8
        else:
            # Low ionization - more H-alpha
            r, g, b = 1.0, 0.3, 0.3

    elif nebula_type == "planetary":
        # Planetary nebulae - often blue-green from O III
        r, g, b = 0.4, 0.8, 1.0

    elif nebula_type == "supernova_remnant":
        # SNRs - often red from shocked H-alpha and S II
        r, g, b = 1.0, 0.4, 0.6

    elif nebula_type == "reflection":
        # Reflection nebulae - blue scattered light
        r, g, b = 0.6, 0.7, 1.0

    elif nebula_type == "dark":
        # Dark nebulae - no emission, just absorption
        r, g, b = 0.1, 0.1, 0.1

    else:
        # Default
        r, g, b = 0.8, 0.6, 0.9

    return (r, g, b)


def calculate_nebula_metallicity(
    nebula_type: str, galactic_radius_kpc: float = 8.0
) -> float:
    """
    Calculate nebula metallicity based on galactic position.

    Args:
        nebula_type: Type of nebula
        galactic_radius_kpc: Galactocentric radius in kpc

    Returns:
        Metallicity relative to solar
    """
    # Galactic metallicity gradient
    # Solar circle at 8 kpc
    # solar_metallicity = 1.0  # Removed unused variable

    # Typical gradient: -0.05 dex/kpc
    gradient = -0.05  # dex/kpc

    metallicity_dex = gradient * (galactic_radius_kpc - 8.0)
    metallicity_relative = 10**metallicity_dex

    # Type-specific adjustments
    if nebula_type == "planetary":
        # Planetary nebulae can show enrichment from stellar evolution
        metallicity_relative *= 1.0 + 0.5 * np.random.random()
    elif nebula_type == "supernova_remnant":
        # SNRs can show mixing with enriched material
        metallicity_relative *= 1.0 + np.random.random()

    return metallicity_relative


def calculate_nebula_size(
    nebula_type: str, age_years: float = 1e4, expansion_velocity_kms: float = 20.0
) -> float:
    """
    Calculate nebula size from age and expansion velocity.

    Args:
        nebula_type: Type of nebula
        age_years: Age in years
        expansion_velocity_kms: Expansion velocity in km/s

    Returns:
        Size in parsecs
    """
    # Convert to consistent units
    age_s = age_years * 3.156e7  # seconds
    velocity_ms = expansion_velocity_kms * 1000  # m/s

    # Distance traveled
    distance_m = velocity_ms * age_s

    # Convert to parsecs
    size_pc = distance_m / PC_TO_M

    # Apply type-specific factors
    if nebula_type == "planetary":
        # Planetary nebulae have complex evolution
        # Size may not grow linearly with age
        if age_years > 1e4:
            size_pc *= 0.8  # Deceleration phase
    elif nebula_type == "supernova_remnant":
        # SNRs go through different phases
        if age_years > 1e5:
            size_pc *= 0.5  # Momentum conserving phase

    return size_pc


def calculate_nebula_ionization_parameter(
    stellar_luminosity: float, distance_to_star_pc: float, density_cm3: float
) -> float:
    """
    Calculate ionization parameter for nebula.

    Args:
        stellar_luminosity: Ionizing luminosity in erg/s
        distance_to_star_pc: Distance to ionizing star in pc
        density_cm3: Gas density in cm⁻³

    Returns:
        log(U) ionization parameter
    """
    # Convert distance to cm
    distance_cm = distance_to_star_pc * PC_TO_M * 100

    # Ionizing photon flux at nebula location
    flux_photons = stellar_luminosity / (4 * np.pi * distance_cm**2)

    # Assume average ionizing photon energy ~20 eV
    photon_energy_erg = 20 * 1.602e-12  # erg
    flux_photons_s = flux_photons / photon_energy_erg

    # Ionization parameter U = n_photons / (n_gas * c)
    U = flux_photons_s / (density_cm3 * C_LIGHT * 100)  # dimensionless

    return np.log10(U)


def calculate_emission_line_ratios(
    temperature_K: float, density_cm3: float, ionization_parameter: float
) -> Dict[str, float]:
    """
    Calculate emission line ratios for diagnostic purposes.

    Args:
        temperature_K: Electron temperature in K
        density_cm3: Electron density in cm⁻³
        ionization_parameter: log(U)

    Returns:
        Dict of emission line ratios
    """
    # Temperature-sensitive ratios
    # T4 = temperature_K / 1e4  # Removed unused variable

    # [O III] 5007/4959 ratio (fixed by atomic physics)
    oiii_ratio = 2.98

    # [N II]/H-alpha ratio (metallicity and ionization sensitive)
    log_U = ionization_parameter
    nii_ha_base = -1.5 + 0.8 * log_U  # Empirical relation
    nii_ha = 10**nii_ha_base

    # [S II] 6716/6731 ratio (density sensitive)
    # Empirical fit to collision strengths
    if density_cm3 < 100:
        sii_ratio = 1.4
    elif density_cm3 < 1000:
        sii_ratio = 1.2 - 0.3 * np.log10(density_cm3 / 100)
    else:
        sii_ratio = 0.8

    # [O III]/H-beta ratio (ionization sensitive)
    oiii_hb_base = 0.5 + 1.2 * log_U
    oiii_hb = 10**oiii_hb_base

    # [O II]/[O III] ratio (ionization sensitive)
    oii_oiii_base = -2.0 - 1.5 * log_U
    oii_oiii = 10**oii_oiii_base

    return {
        "O_III_5007_4959": oiii_ratio,
        "N_II_H_alpha": nii_ha,
        "S_II_6716_6731": sii_ratio,
        "O_III_H_beta": oiii_hb,
        "O_II_O_III": oii_oiii,
    }


def get_nebula_properties(
    nebula_type: str,
    size_pc: float = 1.0,
    age_years: float = 1e4,
    distance_pc: float = 1000,
) -> Dict[str, Any]:
    """
    Calculate comprehensive nebula properties.

    Args:
        nebula_type: Type of nebula
        size_pc: Size in parsecs
        age_years: Age in years
        distance_pc: Distance in parsecs

    Returns:
        Dict containing all nebula properties
    """
    # Basic physical properties
    density = calculate_nebula_density(nebula_type, size_pc)
    mass = calculate_nebula_mass(size_pc, density)

    # Ionization and temperature
    ionization_param = -2.5  # Typical value
    temperature = calculate_nebula_temperature(ionization_param)

    # Dynamics
    expansion_vel = calculate_nebula_expansion_velocity(nebula_type, age_years)

    # Observational properties
    angular_size = calculate_nebula_angular_size(size_pc, distance_pc)
    distance_modulus = calculate_nebula_distance_modulus(distance_pc)
    luminosity = calculate_nebula_luminosity(density, temperature, size_pc)

    # Colors and appearance
    color_rgb = calculate_nebula_color_index(temperature, nebula_type)
    metallicity = calculate_nebula_metallicity(nebula_type)

    # Emission line diagnostics
    line_ratios = calculate_emission_line_ratios(temperature, density, ionization_param)

    return {
        "nebula_type": nebula_type,
        "size_pc": size_pc,
        "age_years": age_years,
        "distance_pc": distance_pc,
        "density_cm3": density,
        "mass_solar": mass,
        "temperature_K": temperature,
        "expansion_velocity_kms": expansion_vel,
        "angular_size_arcsec": angular_size,
        "distance_modulus": distance_modulus,
        "h_alpha_luminosity_erg_s": luminosity,
        "color_rgb": color_rgb,
        "metallicity": metallicity,
        "ionization_parameter": ionization_param,
        "emission_line_ratios": line_ratios,
    }
