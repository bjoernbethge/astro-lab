"""
Survey Data to SpatialTensorDict Converters
===========================================

Integration with existing config system - no duplicate configurations!
"""

import logging

import numpy as np
import polars as pl
import torch
from astropy import units as u
from astropy.coordinates import ICRS, Distance, SkyCoord
from astropy.cosmology import Planck18

from ..config.surveys import get_survey_config
from ..tensors import SpatialTensorDict

logger = logging.getLogger(__name__)


def create_spatial_tensor_from_survey(
    df: pl.DataFrame, 
    survey: str,
    use_memory_mapping: bool = False,
    chunk_size: int = 1_000_000,
) -> SpatialTensorDict:
    """
    Create SpatialTensorDict using existing survey config system.
    
    Args:
        df: Survey data
        survey: Survey name
        use_memory_mapping: Use memory-mapped tensors for large data
        chunk_size: Chunk size for memory-mapped processing

    This is the main converter that uses the existing config infrastructure
    instead of duplicating configuration logic.
    """
    config = get_survey_config(survey)
    logger.debug(f"Converting {len(df)} {config['name']} objects")

    # Extract coordinates using config
    coord_cols = config["coord_cols"]

    if len(coord_cols) == 2:  # RA/Dec
        ra_col, dec_col = coord_cols
        ra = df[ra_col].to_numpy() * u.Unit("deg")
        dec = df[dec_col].to_numpy() * u.Unit("deg")

        # Handle distance based on survey type
        distance = _get_distance_for_survey(df, survey, config)

        # Create coordinates
        coords = SkyCoord(ra=ra, dec=dec, distance=distance, frame=ICRS)

        # Transform to appropriate frame
        target_frame = _get_target_frame(survey, config)
        if target_frame != "icrs":
            coords = coords.transform_to(target_frame)

    elif len(coord_cols) == 3:  # x,y,z (simulations)
        x_col, y_col, z_col = coord_cols
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        z = df[z_col].to_numpy()

        # Handle simulation units
        if survey == "tng50":
            h = 0.6774  # TNG Hubble parameter
            x = x / h * 1000  # ckpc/h to pc
            y = y / h * 1000
            z = z / h * 1000

        # Create tensor directly
        coords_tensor = torch.tensor(np.column_stack([x, y, z]), dtype=torch.float32)

        spatial_tensor = SpatialTensorDict(
            coords_tensor,
            coordinate_system=config.get("coordinate_system", "comoving"),
            unit=u.Unit("pc"),
            use_memory_mapping=use_memory_mapping,
            chunk_size=chunk_size,
        )

        if survey == "tng50":
            spatial_tensor.meta["simulation"] = "TNG50"
            spatial_tensor.meta["hubble_param"] = h

        return spatial_tensor
    else:
        raise ValueError(f"Invalid coordinate columns for {survey}: {coord_cols}")

    # Create SpatialTensorDict with memory mapping option
    spatial_tensor = SpatialTensorDict(
        coords,
        coordinate_system=config.get("coordinate_system", "icrs"),
        unit=u.Unit("pc"),
        use_memory_mapping=use_memory_mapping,
        chunk_size=chunk_size,
    )

    # Add survey-specific metadata
    spatial_tensor.meta["survey"] = survey
    spatial_tensor.meta["data_release"] = config.get("data_release", "unknown")
    spatial_tensor.meta["filter_system"] = config.get("filter_system", "unknown")

    return spatial_tensor


def _get_distance_for_survey(df: pl.DataFrame, survey: str, config: dict) -> u.Quantity:
    """Get distances using survey-specific logic."""

    # Check for redshift first (galaxy surveys)
    z_cols = ["z", "redshift", "z_spec", "z_phot"]
    z_col = None
    for col in z_cols:
        if col in df.columns:
            z_col = col
            break

    if z_col is not None:
        # Galaxy survey with redshifts
        redshift = df[z_col].to_numpy()
        valid_mask = (redshift > 0) & (redshift < 10.0)

        if np.any(valid_mask):
            # Use cosmological distances
            comoving_dist = Planck18.comoving_distance(redshift[valid_mask]).to_value(
                "pc"
            )

            if not np.all(valid_mask):
                # Fill invalid redshifts with default
                full_distance = np.full(len(redshift), _get_default_distance(survey))
                full_distance[valid_mask] = comoving_dist
                distance = full_distance * u.Unit("pc")
            else:
                distance = comoving_dist * u.Unit("pc")

            return distance

    # Check for parallax (stellar surveys)
    plx_cols = ["parallax", "plx", "par"]
    plx_col = None
    for col in plx_cols:
        if col in df.columns:
            plx_col = col
            break

    if plx_col is not None:
        parallax = df[plx_col].to_numpy()
        # Handle negative/zero parallaxes
        parallax_safe = np.where(parallax > 0, parallax, 0.1)
        distance = Distance(parallax=parallax_safe * u.mas)
        return distance

    # Check for direct distance
    dist_cols = ["distance", "dist", "sy_dist"]
    dist_col = None
    for col in dist_cols:
        if col in df.columns:
            dist_col = col
            break

    if dist_col is not None:
        distance = df[dist_col].to_numpy() * u.Unit("pc")
        return distance

    # Photometric distance estimation
    distance = _estimate_photometric_distance(df, survey, config)
    return distance * u.Unit("pc")


def _estimate_photometric_distance(
    df: pl.DataFrame, survey: str, config: dict
) -> np.ndarray:
    """Estimate distances from photometry using config."""
    config.get("mag_cols", [])

    # Survey-specific magnitude to absolute magnitude mappings
    mag_abs_mapping = {
        "gaia": [("phot_g_mean_mag", 4.5)],
        "twomass": [("ks_m", 7.0), ("j_m", 8.0)],
        "wise": [("w1mpro", 8.0)],
        "panstarrs": [("r_mean_psf_mag", 4.5), ("g_mean_psf_mag", 5.0)],
        "des": [("r", 4.5), ("g", 5.0)],
        "euclid": [("VIS", 4.0)],
    }

    mappings = mag_abs_mapping.get(survey, [])

    for mag_col, abs_mag in mappings:
        if mag_col in df.columns:
            app_mag = df[mag_col].to_numpy()
            distance_modulus = app_mag - abs_mag
            distances = 10 ** (distance_modulus / 5 + 1)  # pc
            distances = np.clip(distances, 10, 100_000)  # Reasonable bounds
            return distances

    # Default distance if no photometry available
    return np.full(len(df), _get_default_distance(survey))


def _get_default_distance(survey: str) -> float:
    """Get default distances based on survey type."""
    defaults = {
        # Stellar surveys (parsec)
        "gaia": 100.0,
        "exoplanet": 100.0,
        "twomass": 1000.0,
        "wise": 2000.0,
        "panstarrs": 1500.0,
        # Galaxy surveys (Mpc in parsec units)
        "nsa": 100_000_000.0,  # 100 Mpc
        "sdss": 100_000_000.0,
        "des": 150_000_000.0,
        "euclid": 200_000_000.0,
        # Simulation (kpc in parsec units)
        "tng50": 50_000.0,  # 50 kpc
    }
    return defaults.get(survey, 1000.0)  # 1 kpc default


def _get_target_frame(survey: str, config: dict) -> str:
    """Get target coordinate frame for survey."""
    # Stellar surveys usually want galactocentric
    stellar_surveys = ["gaia", "exoplanet", "twomass", "wise", "panstarrs"]
    if survey in stellar_surveys:
        return "galactocentric"

    # Galaxy surveys stay in ICRS
    galaxy_surveys = ["nsa", "sdss", "des", "euclid"]
    if survey in galaxy_surveys:
        return "icrs"

    # Simulations use their own coordinate system
    if survey == "tng50":
        return "comoving"

    # Default to config or ICRS
    return config.get("coordinate_system", "icrs")


# Backward compatibility functions - delegate to main converter
def gaia_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "gaia")


def nsa_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "nsa")


def sdss_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "sdss")


def exoplanet_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "exoplanet")


def tng50_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "tng50")


def twomass_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "twomass")


def wise_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "wise")


def panstarrs_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "panstarrs")


def des_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "des")


def euclid_to_spatial_tensor(df: pl.DataFrame) -> SpatialTensorDict:
    return create_spatial_tensor_from_survey(df, "euclid")


# Registry using existing config system
CONVERTERS = {
    survey: lambda df, s=survey: create_spatial_tensor_from_survey(df, s)
    for survey in [
        "gaia",
        "nsa",
        "sdss",
        "exoplanet",
        "tng50",
        "twomass",
        "wise",
        "panstarrs",
        "des",
        "euclid",
    ]
}

# Function registry for backward compatibility
CONVERTER_FUNCTIONS = {
    "gaia": gaia_to_spatial_tensor,
    "nsa": nsa_to_spatial_tensor,
    "sdss": sdss_to_spatial_tensor,
    "exoplanet": exoplanet_to_spatial_tensor,
    "tng50": tng50_to_spatial_tensor,
    "twomass": twomass_to_spatial_tensor,
    "wise": wise_to_spatial_tensor,
    "panstarrs": panstarrs_to_spatial_tensor,
    "des": des_to_spatial_tensor,
    "euclid": euclid_to_spatial_tensor,
}


def get_converter(survey: str):
    """Get coordinate converter for survey using existing config system."""
    from ..config.surveys import get_available_surveys

    available_surveys = get_available_surveys()
    if survey not in available_surveys:
        available = ", ".join(sorted(available_surveys))
        raise ValueError(f"Unknown survey: {survey}. Available: {available}")

    return CONVERTER_FUNCTIONS.get(
        survey, lambda df: create_spatial_tensor_from_survey(df, survey)
    )


def validate_spatial_coordinates(
    spatial_tensor: SpatialTensorDict, survey: str
) -> bool:
    """Validate spatial coordinates using survey config."""
    coords = spatial_tensor["coordinates"]

    if not torch.isfinite(coords).all():
        logger.error("Non-finite coordinates found")
        return False

    max_distance = torch.norm(coords, dim=1).max().item()

    # Use survey config for validation
    config = get_survey_config(survey)
    coord_system = config.get("coordinate_system", "icrs")

    # Validation based on coordinate system
    if coord_system == "galactocentric":
        if max_distance > 100_000:
            logger.warning(
                f"Large distances for galactocentric survey: {max_distance:.0f} pc"
            )
    elif coord_system == "icrs":
        if max_distance > 1e12:
            logger.warning(f"Very large cosmological distances: {max_distance:.0e} pc")

    logger.debug(f"Validation passed: {len(coords)} objects, max {max_distance:.0f} pc")
    return True
