"""
Astronomical Constants for astro_lab.tensors
===========================================

Central repository for all astronomical constants used in tensor operations.
Eliminates magic numbers and provides documented, consistent values.

All values are in SI units unless explicitly noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class PhysicalConstants:
    """Fundamental physical constants."""

    # Speed of light
    SPEED_OF_LIGHT_M_S: float = 299792458.0  # m/s (exact)
    SPEED_OF_LIGHT_KM_S: float = 299792.458  # km/s

    # Gravitational constant
    G_M3_KG_S2: float = 6.67430e-11  # m³/kg/s² (2018 CODATA)

    # Stefan-Boltzmann constant
    STEFAN_BOLTZMANN: float = 5.670374419e-8  # W/m²/K⁴

@dataclass(frozen=True)
class AstronomicalConstants:
    """Astronomical constants and conversion factors."""

    # Distance units
    AU_M: float = 149597870700.0  # m (exact, IAU 2012)
    AU_KM: float = 149597870.7  # km
    PARSEC_M: float = 648000 / 3.14159265359 * AU_M  # m
    PARSEC_LY: float = 3.26156  # light-years per parsec
    LY_M: float = 299792458.0 * 365.25 * 24 * 3600  # m (light-year in meters)

    # Angular conversions
    ARCSEC_TO_RAD: float = 4.84813681109536e-6  # radians per arcsecond
    ARCMIN_TO_RAD: float = 2.90888208665722e-4  # radians per arcminute
    DEG_TO_RAD: float = 0.017453292519943295  # radians per degree

    # Time units
    JULIAN_YEAR_S: float = 365.25 * 24 * 3600  # seconds
    SIDEREAL_YEAR_S: float = 365.25636 * 24 * 3600  # seconds

    # Solar system masses and radii
    SOLAR_MASS_KG: float = 1.9884e30  # kg (IAU 2015)
    EARTH_MASS_KG: float = 5.9722e24  # kg
    LUNAR_MASS_KG: float = 7.342e22  # kg

    SOLAR_RADIUS_M: float = 6.957e8  # m (IAU 2015)
    SOLAR_RADIUS_KM: float = 695700.0  # km
    EARTH_RADIUS_M: float = 6.3781e6  # m (equatorial)
    EARTH_RADIUS_KM: float = 6378.1  # km (equatorial)
    LUNAR_RADIUS_KM: float = 1737.4  # km

@dataclass(frozen=True)
class GravitationalParameters:
    """Standard gravitational parameters (GM) in km³/s²."""

    # Major bodies
    SUN: float = 132712440018.0  # km³/s² (IAU 2015)
    EARTH: float = 398600.4418  # km³/s² (WGS 84)
    MOON: float = 4902.7779  # km³/s²

    # Planets
    MERCURY: float = 22032.09  # km³/s²
    VENUS: float = 324858.592  # km³/s²
    MARS: float = 42828.37  # km³/s²
    JUPITER: float = 126686534.9  # km³/s²
    SATURN: float = 37931187.0  # km³/s²
    URANUS: float = 5793939.0  # km³/s²
    NEPTUNE: float = 6836529.0  # km³/s²

    # Default for generic "star" systems
    SOLAR_MASS: float = SUN  # Alias for stellar systems

@dataclass(frozen=True)
class SpectroscopyConstants:
    """Constants for spectroscopic calculations."""

    # Rest wavelengths of common lines (Angstrom)
    H_ALPHA: float = 6562.801  # Hydrogen Balmer alpha
    H_BETA: float = 4861.363  # Hydrogen Balmer beta
    H_GAMMA: float = 4340.462  # Hydrogen Balmer gamma

    LY_ALPHA: float = 1215.670  # Lyman alpha

    # Metal lines
    CA_II_K: float = 3933.664  # Calcium II K line
    CA_II_H: float = 3968.470  # Calcium II H line
    MG_B: float = 5183.604  # Magnesium b line

    # Units
    ANGSTROM_TO_M: float = 1e-10  # meters per Angstrom
    ANGSTROM_TO_NM: float = 0.1  # nanometers per Angstrom

@dataclass(frozen=True)
class PhotometryConstants:
    """Constants for photometric calculations."""

    # Zero point fluxes (Jy) for AB magnitude system
    AB_ZERO_POINT_JY: float = 3631.0  # Jansky

    # Solar magnitudes in various systems (Vega)
    SOLAR_MAG_V: float = -26.74  # V-band apparent magnitude
    SOLAR_MAG_B: float = -26.09  # B-band apparent magnitude
    SOLAR_MAG_R: float = -27.10  # R-band apparent magnitude

    # Extinction coefficients (typical values, mag/airmass)
    EXTINCTION_U: float = 0.6  # U-band
    EXTINCTION_B: float = 0.4  # B-band
    EXTINCTION_V: float = 0.2  # V-band
    EXTINCTION_R: float = 0.1  # R-band
    EXTINCTION_I: float = 0.08  # I-band

@dataclass(frozen=True)
class CosmologyConstants:
    """Standard cosmological parameters (Planck 2018)."""

    # Hubble constant
    H0_KM_S_MPC: float = 67.4  # km/s/Mpc

    # Density parameters (at z=0)
    OMEGA_M: float = 0.315  # Matter density
    OMEGA_LAMBDA: float = 0.685  # Dark energy density
    OMEGA_B: float = 0.049  # Baryon density
    OMEGA_K: float = 0.0  # Curvature (flat universe)

    # Other parameters
    SIGMA_8: float = 0.811  # Amplitude of fluctuations
    N_S: float = 0.965  # Spectral index

# Convenience collections
CONSTANTS = PhysicalConstants()
ASTRO = AstronomicalConstants()
GRAVITY = GravitationalParameters()
SPECTROSCOPY = SpectroscopyConstants()
PHOTOMETRY = PhotometryConstants()
COSMOLOGY = CosmologyConstants()

# Alternative mappings for convenience
def get_mu(attractor: str) -> float:
    """
    Get gravitational parameter for given attractor.

    Args:
        attractor: Name of central body

    Returns:
        Standard gravitational parameter in km³/s²
    """
    mu_values = {
        "earth": GRAVITY.EARTH,
        "sun": GRAVITY.SUN,
        "moon": GRAVITY.MOON,
        "mercury": GRAVITY.MERCURY,
        "venus": GRAVITY.VENUS,
        "mars": GRAVITY.MARS,
        "jupiter": GRAVITY.JUPITER,
        "saturn": GRAVITY.SATURN,
        "uranus": GRAVITY.URANUS,
        "neptune": GRAVITY.NEPTUNE,
        "star": GRAVITY.SOLAR_MASS,  # Default for stellar systems
    }
    return mu_values.get(attractor.lower(), GRAVITY.EARTH)

def get_planet_radius(planet: str) -> float:
    """
    Get planetary radius in km.

    Args:
        planet: Planet name

    Returns:
        Radius in km
    """
    radii = {
        "earth": ASTRO.EARTH_RADIUS_KM,
        "sun": ASTRO.SOLAR_RADIUS_KM,
        "moon": ASTRO.LUNAR_RADIUS_KM,
        # Add more as needed
    }
    return radii.get(planet.lower(), ASTRO.EARTH_RADIUS_KM)
