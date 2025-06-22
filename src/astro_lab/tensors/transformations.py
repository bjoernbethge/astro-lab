"""
Transformation Registry for Survey Tensors
==========================================

Registry system for survey transformations following the refactoring guide.
Eliminates hardcoded transformations and provides extensible pattern.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from .constants import PHOTOMETRY, SPECTROSCOPY

logger = logging.getLogger(__name__)

class TransformationRegistry:
    """
    Registry for survey transformations.

    Provides decorator-based registration and lookup of transformation functions
    between different astronomical surveys.
    """

    _transformations: Dict[Tuple[str, str], Callable] = {}
    _metadata: Dict[Tuple[str, str], Dict[str, Any]] = {}

    @classmethod
    def register(
        cls,
        from_survey: str,
        to_survey: str,
        description: Optional[str] = None,
        requires_columns: Optional[list] = None,
        accuracy_note: Optional[str] = None,
    ):
        """
        Decorator to register transformations.

        Args:
            from_survey: Source survey name
            to_survey: Target survey name
            description: Human-readable description
            requires_columns: Required columns for transformation
            accuracy_note: Note about transformation accuracy/limitations
        """

        def decorator(func: Callable) -> Callable:
            key = (from_survey.lower(), to_survey.lower())
            cls._transformations[key] = func
            cls._metadata[key] = {
                "description": description or f"Transform {from_survey} to {to_survey}",
                "requires_columns": requires_columns or [],
                "accuracy_note": accuracy_note,
                "function_name": func.__name__,
            }
            logger.info(f"Registered transformation: {from_survey} → {to_survey}")
            return func

        return decorator

    @classmethod
    def get_transformation(cls, from_survey: str, to_survey: str) -> Optional[Callable]:
        """
        Get transformation function.

        Args:
            from_survey: Source survey name
            to_survey: Target survey name

        Returns:
            Transformation function or None if not found
        """
        key = (from_survey.lower(), to_survey.lower())
        return cls._transformations.get(key)

    @classmethod
    def get_transformation_info(
        cls, from_survey: str, to_survey: str
    ) -> Optional[Dict[str, Any]]:
        """Get metadata about a transformation."""
        key = (from_survey.lower(), to_survey.lower())
        return cls._metadata.get(key)

    @classmethod
    def list_available_transformations(cls) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """List all available transformations with metadata."""
        return {
            key: {
                "transform_func": func,
                "metadata": cls._metadata.get(key, {}),
            }
            for key, func in cls._transformations.items()
        }

    @classmethod
    def get_supported_surveys(cls) -> Dict[str, Dict[str, list]]:
        """Get supported surveys and their transformation capabilities."""
        surveys = {}

        for from_survey, to_survey in cls._transformations.keys():
            if from_survey not in surveys:
                surveys[from_survey] = {
                    "can_transform_to": [],
                    "can_transform_from": [],
                }
            if to_survey not in surveys:
                surveys[to_survey] = {"can_transform_to": [], "can_transform_from": []}

            surveys[from_survey]["can_transform_to"].append(to_survey)
            surveys[to_survey]["can_transform_from"].append(from_survey)

        return surveys

    @classmethod
    def clear_registry(cls):
        """Clear all registered transformations (mainly for testing)."""
        cls._transformations.clear()
        cls._metadata.clear()

# =============================================================================
# Default transformations using the registry
# =============================================================================

@TransformationRegistry.register(
    "gaia",
    "sdss",
    description="Transform Gaia DR3 photometry to SDSS ugriz",
    requires_columns=["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
    accuracy_note="Based on Gaia DR3 transformations, ~0.05 mag accuracy",
)
def gaia_to_sdss(survey_tensor) -> torch.Tensor:
    """
    Transform Gaia photometry to SDSS using improved relations.

    Based on Gaia DR3 documentation and empirical relations.
    """
    try:
        # Get Gaia magnitudes
        g_mag = survey_tensor.get_column("phot_g_mean_mag")
        bp_mag = survey_tensor.get_column("phot_bp_mean_mag")
        rp_mag = survey_tensor.get_column("phot_rp_mean_mag")

        # Compute Gaia colors
        bp_rp = bp_mag - rp_mag
        g_rp = g_mag - rp_mag

        # Gaia DR3 to SDSS transformations (improved relations)
        # Based on Riello et al. 2021, A&A, 649, A3

        # SDSS g ≈ G + color corrections
        sdss_g = (
            g_mag + 0.13518 - 0.46245 * bp_rp - 0.25171 * bp_rp**2 + 0.021349 * bp_rp**3
        )

        # SDSS r ≈ G - (G-RP) corrections
        sdss_r = (
            g_mag
            - g_rp
            + 0.12879
            - 0.24662 * bp_rp
            - 0.027464 * bp_rp**2
            - 0.049465 * bp_rp**3
        )

        # SDSS i ≈ G - color corrections
        sdss_i = (
            g_mag - 0.29676 - 0.46833 * bp_rp - 0.015892 * bp_rp**2 - 0.10052 * bp_rp**3
        )

        # For u and z, use empirical relations (less accurate)
        sdss_u = sdss_g + 1.28 + 0.37 * (sdss_g - sdss_r)  # Rough estimate
        sdss_z = sdss_i - 0.38 - 0.054 * (sdss_g - sdss_r)  # Rough estimate

        # Stack into tensor [N, 5] for ugriz
        transformed = torch.stack([sdss_u, sdss_g, sdss_r, sdss_i, sdss_z], dim=1)

        logger.info(f"Transformed {len(survey_tensor)} objects from Gaia to SDSS")
        return transformed

    except KeyError as e:
        raise ValueError(f"Missing required Gaia columns for SDSS transformation: {e}")

@TransformationRegistry.register(
    "sdss",
    "gaia",
    description="Transform SDSS ugriz to Gaia GBpRp",
    requires_columns=["g", "r", "i"],
    accuracy_note="Reverse transformation, ~0.1 mag accuracy",
)
def sdss_to_gaia(survey_tensor) -> torch.Tensor:
    """Transform SDSS photometry to Gaia using reverse relations."""
    try:
        # Get SDSS magnitudes (at minimum g, r, i)
        sdss_g = survey_tensor.get_column("g")
        sdss_r = survey_tensor.get_column("r")
        sdss_i = survey_tensor.get_column("i")

        # Compute SDSS colors
        g_r = sdss_g - sdss_r
        g_i = sdss_g - sdss_i

        # Reverse transformations (approximate)
        # Based on inverse relations from Gaia DR3

        # Gaia G ≈ SDSS g with color corrections
        gaia_g = sdss_g - 0.0916 - 0.1069 * g_r + 0.0017 * g_r**2

        # Gaia BP ≈ SDSS g + color corrections
        gaia_bp = sdss_g + 0.2894 + 0.7560 * g_r - 0.0215 * g_r**2

        # Gaia RP ≈ SDSS r with color corrections
        gaia_rp = sdss_r - 0.0686 - 0.1131 * g_r + 0.0195 * g_r**2

        # Stack into tensor [N, 3] for GBpRp
        transformed = torch.stack([gaia_g, gaia_bp, gaia_rp], dim=1)

        logger.info(f"Transformed {len(survey_tensor)} objects from SDSS to Gaia")
        return transformed

    except KeyError as e:
        raise ValueError(f"Missing required SDSS columns for Gaia transformation: {e}")

@TransformationRegistry.register(
    "gaia",
    "lsst",
    description="Transform Gaia GBpRp to LSST ugrizy",
    requires_columns=["phot_g_mean_mag", "phot_bp_mean_mag", "phot_rp_mean_mag"],
    accuracy_note="Based on LSST Science Book transformations",
)
def gaia_to_lsst(survey_tensor) -> torch.Tensor:
    """Transform Gaia photometry to LSST using LSST Science Book relations."""
    try:
        # Get Gaia magnitudes
        g_mag = survey_tensor.get_column("phot_g_mean_mag")
        bp_mag = survey_tensor.get_column("phot_bp_mean_mag")
        rp_mag = survey_tensor.get_column("phot_rp_mean_mag")

        # Compute Gaia colors
        bp_rp = bp_mag - rp_mag

        # LSST transformations from Gaia (LSST Science Book relations)
        lsst_u = g_mag + 1.810 + 1.234 * bp_rp
        lsst_g = g_mag + 0.154 - 0.479 * bp_rp
        lsst_r = g_mag - 0.139 - 0.493 * bp_rp
        lsst_i = g_mag - 0.384 - 0.540 * bp_rp
        lsst_z = g_mag - 0.562 - 0.577 * bp_rp
        lsst_y = g_mag - 0.672 - 0.604 * bp_rp  # y-band estimate

        # Stack into tensor [N, 6] for ugrizy
        transformed = torch.stack(
            [lsst_u, lsst_g, lsst_r, lsst_i, lsst_z, lsst_y], dim=1
        )

        logger.info(f"Transformed {len(survey_tensor)} objects from Gaia to LSST")
        return transformed

    except KeyError as e:
        raise ValueError(f"Missing required Gaia columns for LSST transformation: {e}")

@TransformationRegistry.register(
    "2mass",
    "sdss",
    description="Transform 2MASS JHK to SDSS ugriz",
    requires_columns=["j", "h", "k"],
    accuracy_note="IR to optical transformation, limited accuracy for blue stars",
)
def twomass_to_sdss(survey_tensor) -> torch.Tensor:
    """Transform 2MASS near-IR photometry to SDSS optical."""
    try:
        # Get 2MASS magnitudes
        j_mag = survey_tensor.get_column("j")
        h_mag = survey_tensor.get_column("h")
        k_mag = survey_tensor.get_column("k")

        # Compute 2MASS colors
        j_k = j_mag - k_mag
        h_k = h_mag - k_mag

        # 2MASS to SDSS transformations (Bilir et al. 2008)
        # These are approximate and work best for late-type stars

        # Start with K-band and work outward
        sdss_z = k_mag + 2.54 + 0.05 * j_k  # K closely related to z
        sdss_i = sdss_z + 0.38 + 0.23 * j_k
        sdss_r = sdss_i + 0.42 + 0.11 * j_k
        sdss_g = sdss_r + 0.65 + 0.27 * j_k
        sdss_u = sdss_g + 1.39 + 0.18 * j_k  # Most uncertain

        # Stack into tensor
        transformed = torch.stack([sdss_u, sdss_g, sdss_r, sdss_i, sdss_z], dim=1)

        logger.warning(
            "2MASS→SDSS transformation is approximate, especially for blue stars"
        )
        logger.info(f"Transformed {len(survey_tensor)} objects from 2MASS to SDSS")
        return transformed

    except KeyError as e:
        raise ValueError(f"Missing required 2MASS columns for SDSS transformation: {e}")

@TransformationRegistry.register(
    "jwst",
    "sdss",
    description="Transform JWST NIRCam to SDSS ugriz",
    requires_columns=["f150w", "f200w", "f356w"],
    accuracy_note="Preliminary transformations, use with caution",
)
def jwst_to_sdss(survey_tensor) -> torch.Tensor:
    """Transform JWST NIRCam photometry to SDSS (preliminary)."""
    try:
        # Get key JWST filters (these overlap well with optical)
        f150w = survey_tensor.get_column("f150w")  # ~I band
        f200w = survey_tensor.get_column("f200w")  # ~H band
        f356w = survey_tensor.get_column("f356w")  # ~K band (roughly)

        # Very preliminary transformations (needs calibration with real data)
        # These are rough estimates based on filter curves

        sdss_i = f150w + 0.2  # F150W is roughly I-band
        sdss_z = sdss_i - 0.3 + 0.1 * (f150w - f200w)
        sdss_r = sdss_i + 0.4 + 0.2 * (f150w - f200w)
        sdss_g = sdss_r + 0.6 + 0.3 * (f150w - f356w)
        sdss_u = sdss_g + 1.2 + 0.4 * (f150w - f356w)  # Very uncertain

        transformed = torch.stack([sdss_u, sdss_g, sdss_r, sdss_i, sdss_z], dim=1)

        logger.warning(
            "JWST→SDSS transformations are PRELIMINARY - use with extreme caution"
        )
        logger.info(f"Transformed {len(survey_tensor)} objects from JWST to SDSS")
        return transformed

    except KeyError as e:
        raise ValueError(f"Missing required JWST columns for SDSS transformation: {e}")

# =============================================================================
# Utility functions
# =============================================================================

def apply_transformation(survey_tensor, target_survey: str) -> torch.Tensor:
    """
    Apply transformation to target survey.

    Args:
        survey_tensor: Source survey tensor
        target_survey: Target survey name

    Returns:
        Transformed photometry tensor

    Raises:
        ValueError: If transformation not available
    """
    source_survey = survey_tensor.survey_name
    transform_func = TransformationRegistry.get_transformation(
        source_survey, target_survey
    )

    if transform_func is None:
        available = TransformationRegistry.get_supported_surveys()
        source_options = available.get(source_survey.lower(), {}).get(
            "can_transform_to", []
        )

        raise ValueError(
            f"No transformation available from {source_survey} to {target_survey}. "
            f"Available transformations from {source_survey}: {source_options}"
        )

    # Get transformation info for logging
    info = TransformationRegistry.get_transformation_info(source_survey, target_survey)
    if info and info.get("accuracy_note"):
        logger.info(f"Transformation note: {info['accuracy_note']}")

    return transform_func(survey_tensor)

def get_transformation_chain(source: str, target: str) -> Optional[list]:
    """
    Find a chain of transformations from source to target survey.

    Args:
        source: Source survey name
        target: Target survey name

    Returns:
        List of transformation steps or None if no path exists
    """
    # Simple implementation - could be expanded to multi-step transformations
    direct_transform = TransformationRegistry.get_transformation(source, target)
    if direct_transform:
        return [(source, target)]

    # Could implement graph search for multi-step transformations here
    return None
