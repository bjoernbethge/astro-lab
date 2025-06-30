"""
Factory functions for creating TensorDict instances.

This module provides factory functions for creating various types of TensorDicts
with synthetic or example data.
"""

import math
from typing import Any, List, Optional

import torch

from .base import AstroTensorDict
from .crossmatch import CrossMatchTensorDict
from .photometric import PhotometricTensorDict
from .simulation import SimulationTensorDict
from .spatial import SpatialTensorDict


def create_nbody_simulation(
    n_particles: int = 100, system_type: str = "cluster"
) -> SimulationTensorDict:
    """
    Creates N-body simulation.

    Args:
        n_particles: Number of particles
        system_type: Type of system ("cluster", "galaxy", "solar_system")

    Returns:
        SimulationTensorDict
    """
    if system_type == "cluster":
        # Globular cluster
        positions = torch.randn(n_particles, 3) * 10  # kpc
        velocities = torch.randn(n_particles, 3) * 10  # km/s
        masses = torch.ones(n_particles)  # Solar masses

    elif system_type == "galaxy":
        # Disk galaxy
        r = torch.exp(torch.rand(n_particles)) * 5  # kpc
        theta = torch.rand(n_particles) * 2 * math.pi
        z = torch.normal(0, 0.5, (n_particles,))

        positions = torch.stack([r * torch.cos(theta), r * torch.sin(theta), z], dim=-1)

        # Rotation velocity
        v_rot = torch.sqrt(200 * r / (r + 1))  # km/s
        velocities = torch.stack(
            [
                -v_rot * torch.sin(theta),
                v_rot * torch.cos(theta),
                torch.zeros(n_particles),
            ],
            dim=-1,
        )

        masses = torch.ones(n_particles)

    elif system_type == "solar_system":
        # Simplified solar system
        if n_particles != 9:
            n_particles = 9  # Sun + 8 planets

        # Planet distances in AU
        distances = torch.tensor([0, 0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30.1])
        masses = torch.tensor(
            [1.0, 0.055, 0.815, 1.0, 0.107, 317.8, 95.2, 14.5, 17.1]
        )  # Earth masses

        positions = torch.zeros(n_particles, 3)
        velocities = torch.zeros(n_particles, 3)

        for i in range(1, n_particles):
            positions[i, 0] = distances[i] * 1.496e8  # Convert to km
            velocities[i, 1] = torch.sqrt(
                1.327e20 / (distances[i] * 1.496e8)
            )  # Orbital velocity

    else:
        raise ValueError(f"Unknown system type: {system_type}")

    return SimulationTensorDict(
        positions=positions,
        velocities=velocities,
        masses=masses,
        simulation_type=system_type,
    )


def create_crossmatch_example(
    n_objects1: int = 1000, n_objects2: int = 800, overlap_fraction: float = 0.7
) -> CrossMatchTensorDict:
    """
    Creates cross-match example.

    Args:
        n_objects1: Number of objects in catalog 1
        n_objects2: Number of objects in catalog 2
        overlap_fraction: Fraction of overlapping objects

    Returns:
        CrossMatchTensorDict with synthetic catalogs
    """
    # Create first catalog
    coords1 = torch.randn(n_objects1, 3) * 10
    mags1 = torch.randn(n_objects1, 3) + 15

    spatial1 = SpatialTensorDict(coordinates=coords1)
    phot1 = PhotometricTensorDict(magnitudes=mags1, bands=["g", "r", "i"])
    cat1 = AstroTensorDict(
        {"spatial": spatial1, "photometric": phot1, "survey_name": "Survey1"}
    )

    # Create second catalog with partial overlap
    n_overlap = int(n_objects2 * overlap_fraction)
    n_unique = n_objects2 - n_overlap

    # Overlapping objects (with small position errors)
    coords2_overlap = coords1[:n_overlap] + torch.randn(n_overlap, 3) * 0.1
    mags2_overlap = mags1[:n_overlap, :2] + torch.randn(n_overlap, 2) * 0.1

    # Unique objects
    coords2_unique = torch.randn(n_unique, 3) * 10
    mags2_unique = torch.randn(n_unique, 2) + 16

    coords2 = torch.cat([coords2_overlap, coords2_unique], dim=0)
    mags2 = torch.cat([mags2_overlap, mags2_unique], dim=0)

    spatial2 = SpatialTensorDict(coordinates=coords2)
    phot2 = PhotometricTensorDict(magnitudes=mags2, bands=["u", "g"])
    cat2 = AstroTensorDict(
        {"spatial": spatial2, "photometric": phot2, "survey_name": "Survey2"}
    )

    # Create cross-match
    return CrossMatchTensorDict(
        catalog1=cat1,
        catalog2=cat2,
        matches=torch.arange(n_overlap),
        distances=torch.rand(n_overlap) * 0.1,
    )


def create_generic_survey(
    coordinates: torch.Tensor,
    magnitudes: torch.Tensor,
    bands: List[str],
    survey_name: str,
    filter_system: str = "Generic",
    errors: Optional[torch.Tensor] = None,
    **kwargs,
) -> AstroTensorDict:
    """
    Creates generic survey TensorDict.

    Args:
        coordinates: [N, 3] RA, Dec, distance
        magnitudes: [N, n_bands] Magnitudes in each band
        bands: List of band names
        survey_name: Name of the survey
        filter_system: Filter system name
        errors: [N, n_bands] Magnitude errors (optional)

    Returns:
        AstroTensorDict with survey data
    """
    n_objects = coordinates.shape[0]

    data = {
        "coordinates": coordinates,
        "magnitudes": magnitudes,
        "survey_name": survey_name,
        "meta": {
            "filter_system": filter_system,
            "bands": bands,
        },
    }

    if errors is not None:
        data["errors"] = errors

    return AstroTensorDict(data, batch_size=(n_objects,), **kwargs)


def create_survey_from_pyg_data(data: Any, survey_name: str = "converted"):
    """
    Convert PyTorch Geometric Data to SurveyTensorDict.

    Args:
        data: PyG Data object
        survey_name: Name for the survey

    Returns:
        SurveyTensorDict
    """
    from .photometric import PhotometricTensorDict
    from .spatial import SpatialTensorDict
    from .survey import SurveyTensorDict

    # Extract coordinates
    if hasattr(data, "pos") and data.pos is not None:
        coords = data.pos
    elif hasattr(data, "x") and data.x.shape[-1] >= 3:
        coords = data.x[:, :3]  # Use first 3 dimensions as coordinates
    else:
        raise ValueError("Cannot extract coordinates from Data object")

    # Create spatial component
    spatial = SpatialTensorDict(coordinates=coords)

    # Create photometric component (dummy if not available)
    if hasattr(data, "x") and data.x.shape[-1] > 3:
        # Use remaining features as magnitudes
        magnitudes = data.x[:, 3:]
        n_bands = magnitudes.shape[1]
        bands = [f"band_{i}" for i in range(n_bands)]
    else:
        # No photometric data available
        raise ValueError(
            "No photometric data available in PyG Data object. "
            "Data must have more than 3 features to extract photometric information."
        )

    photometric = PhotometricTensorDict(magnitudes=magnitudes, bands=bands)

    # Create SurveyTensorDict
    return SurveyTensorDict(
        spatial=spatial,
        photometric=photometric,
        survey_name=survey_name,
    )
