"""
Astronomical Transforms
======================

Transforms specific to astronomical data processing using Astropy.
"""

import logging
from typing import List, Optional, Tuple

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import (
    SkyCoord,
)
from astropy.time import Time
from torch_geometric.data import Data
from torch_geometric.nn import fps
from torch_geometric.transforms import BaseTransform

logger = logging.getLogger(__name__)

# sklearn.cluster.KMeans was unused - removed


def spherical_to_cartesian(ra, dec, distance, degrees=True):
    """
    Convert spherical coordinates to cartesian coordinates.

    Args:
        ra: Right ascension
        dec: Declination
        distance: Distance
        degrees: Whether input angles are in degrees (default: True)

    Returns:
        x, y, z: Cartesian coordinates
    """
    if degrees:
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
    else:
        ra_rad = ra
        dec_rad = dec

    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)

    return x, y, z


class AstronomicalFeatures(BaseTransform):
    """Add astronomical features based on positions using Astropy."""

    def __init__(
        self,
        coord_columns: Tuple[str, str] = ("ra", "dec"),
        distance_column: Optional[str] = "distance",
        frame: str = "icrs",
        unit: str = "deg",
        distance_unit: str = "pc",
    ):
        """
        Initialize astronomical features transform.

        Args:
            coord_columns: Column names for coordinates (ra/dec or l/b)
            distance_column: Column name for distance (optional)
            frame: Coordinate frame ('icrs', 'galactic', 'fk5')
            unit: Unit for angles ('deg', 'rad', 'hourangle')
            distance_unit: Unit for distance ('pc', 'kpc', 'Mpc')
        """
        self.coord_columns = coord_columns
        self.distance_column = distance_column
        self.frame = frame
        self.unit = unit
        self.distance_unit = distance_unit

    def __call__(self, data: Data) -> Data:
        """Apply transform to add astronomical features."""
        # Try to find position data in various possible locations
        pos = None
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos.cpu().numpy()
        elif hasattr(data, "coordinates") and data.coordinates is not None:
            pos = data.coordinates.cpu().numpy()
        else:
            # Try to construct from individual coordinate columns
            coord_attrs = ["ra", "dec", "x", "y", "z"]
            available_coords = []
            for attr in coord_attrs:
                if hasattr(data, attr) and getattr(data, attr) is not None:
                    available_coords.append(getattr(data, attr).cpu().numpy())

            if len(available_coords) >= 2:
                pos = np.column_stack(available_coords)
            else:
                logger.warning(
                    "No position data found for AstronomicalFeatures transform"
                )
                return data

        # Create SkyCoord object
        if pos.shape[1] >= 2:
            coord1 = pos[:, 0]
            coord2 = pos[:, 1]

            # Validate coordinate ranges
            if self.frame in ["icrs", "fk5"]:
                # RA/Dec system - ensure valid ranges
                coord1 = np.clip(coord1, 0, 360)  # RA: 0-360 degrees
                coord2 = np.clip(coord2, -90, 90)  # Dec: -90 to +90 degrees
            elif self.frame == "galactic":
                # Galactic coordinates
                coord1 = coord1 % 360  # l: 0-360 degrees
                coord2 = np.clip(coord2, -90, 90)  # b: -90 to +90 degrees

            # Convert to astropy units after validation
            coord1_with_unit = coord1 * getattr(u, self.unit)
            coord2_with_unit = coord2 * getattr(u, self.unit)

            if pos.shape[1] >= 3 and self.distance_column:
                distance = np.abs(pos[:, 2]) * getattr(u, self.distance_unit)
                skycoord = SkyCoord(
                    coord1_with_unit,
                    coord2_with_unit,
                    distance=distance,
                    frame=self.frame,
                )
            else:
                skycoord = SkyCoord(
                    coord1_with_unit, coord2_with_unit, frame=self.frame
                )
        else:
            logger.warning("Position data must have at least 2 dimensions")
            return data

        # Convert to various coordinate systems
        galactic = skycoord.galactic
        icrs = skycoord.icrs

        # Extract features
        features = []

        # Galactic coordinates
        features.append(galactic.l.deg)
        features.append(galactic.b.deg)

        # ICRS coordinates
        features.append(icrs.ra.deg)
        features.append(icrs.dec.deg)

        # Distance features if available
        if hasattr(skycoord, "distance") and skycoord.distance is not None:
            features.append(skycoord.distance.pc)

            # Cartesian galactic coordinates
            cart = galactic.cartesian
            features.append(cart.x.value)
            features.append(cart.y.value)
            features.append(cart.z.value)

            # Galactocentric coordinates
            galcen = skycoord.galactocentric
            features.append(galcen.x.value)
            features.append(galcen.y.value)
            features.append(galcen.z.value)

            # Distance from galactic center
            galcen_dist = np.sqrt(
                galcen.x.value**2 + galcen.y.value**2 + galcen.z.value**2
            )
            features.append(galcen_dist)

        # Stack features
        astro_features = torch.tensor(np.column_stack(features), dtype=torch.float32)

        # Add to existing features or create new
        if hasattr(data, "x") and data.x is not None:
            x = data.x
            if x.dim() == 1:
                x = x.unsqueeze(1)
            data.x = torch.cat([x, astro_features], dim=1)
        else:
            if astro_features.dim() == 1:
                astro_features = astro_features.unsqueeze(1)
            data.x = astro_features

        logger.debug(f"Added {astro_features.shape[1]} astronomical features")
        return data


class GalacticCoordinateTransform(BaseTransform):
    """Transform coordinates to galactic system using Astropy."""

    def __init__(
        self,
        from_frame: str = "icrs",
        coord_unit: str = "deg",
        distance_unit: str = "pc",
    ):
        """
        Initialize coordinate transform.

        Args:
            from_frame: Source coordinate frame
            coord_unit: Unit for coordinates
            distance_unit: Unit for distance
        """
        self.from_frame = from_frame
        self.coord_unit = coord_unit
        self.distance_unit = distance_unit

    def __call__(self, data: Data) -> Data:
        """Transform positions to galactic coordinates."""
        # Try to find position data in various possible locations
        pos = None
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos.cpu().numpy()
        elif hasattr(data, "coordinates") and data.coordinates is not None:
            pos = data.coordinates.cpu().numpy()
        else:
            # Try to construct from individual coordinate columns
            coord_attrs = ["ra", "dec", "x", "y", "z"]
            available_coords = []
            for attr in coord_attrs:
                if hasattr(data, attr) and getattr(data, attr) is not None:
                    available_coords.append(getattr(data, attr).cpu().numpy())

            if len(available_coords) >= 2:
                pos = np.column_stack(available_coords)
            else:
                logger.warning("No position data found for GalacticCoordinateTransform")
                return data

        # Create SkyCoord
        if pos.shape[1] >= 2:
            coord1 = pos[:, 0] * getattr(u, self.coord_unit)
            coord2 = pos[:, 1] * getattr(u, self.coord_unit)

            if pos.shape[1] >= 3:
                distance = pos[:, 2] * getattr(u, self.distance_unit)
                skycoord = SkyCoord(
                    coord1, coord2, distance=distance, frame=self.from_frame
                )
            else:
                skycoord = SkyCoord(coord1, coord2, frame=self.from_frame)

            # Transform to galactic
            galactic = skycoord.galactic

            # Update positions
            new_pos = torch.zeros_like(data.pos)
            new_pos[:, 0] = torch.tensor(galactic.l.deg, dtype=torch.float32)
            new_pos[:, 1] = torch.tensor(galactic.b.deg, dtype=torch.float32)

            if pos.shape[1] >= 3 and hasattr(galactic, "distance"):
                new_pos[:, 2] = torch.tensor(
                    galactic.distance.value, dtype=torch.float32
                )

            data.pos = new_pos
            data.coord_frame = "galactic"

        return data


class ProperMotionCorrection(BaseTransform):
    """Apply proper motion corrections using Astropy."""

    def __init__(
        self,
        target_epoch: str = "J2000",
        pm_ra_column: str = "pmra",
        pm_dec_column: str = "pmdec",
        obstime_column: Optional[str] = None,
    ):
        """
        Initialize proper motion correction.

        Args:
            target_epoch: Target epoch for correction
            pm_ra_column: Column name for proper motion in RA
            pm_dec_column: Column name for proper motion in Dec
            obstime_column: Column name for observation time
        """
        self.target_epoch = Time(target_epoch)
        self.pm_ra_column = pm_ra_column
        self.pm_dec_column = pm_dec_column
        self.obstime_column = obstime_column

    def __call__(self, data: Data) -> Data:
        """Apply proper motion corrections."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        # This is a simplified example - real implementation would need
        # proper motion data in the input
        logger.debug("Applying proper motion corrections")

        return data


class ExtinctionCorrection(BaseTransform):
    """Apply extinction corrections using dust maps."""

    def __init__(
        self,
        dust_map: str = "sfd",
        r_v: float = 3.1,
    ):
        """
        Initialize extinction correction.

        Args:
            dust_map: Dust map to use ('sfd', 'planck', etc.)
            r_v: Total-to-selective extinction ratio
        """
        self.dust_map = dust_map
        self.r_v = r_v

    def __call__(self, data: Data) -> Data:
        """Apply extinction corrections to magnitudes."""
        # This would use dustmaps package in real implementation
        logger.debug(f"Applying extinction correction using {self.dust_map}")

        return data


class MultiScaleSampling(BaseTransform):
    """Multi-scale spatial sampling using FPS."""

    def __init__(self, scales: Optional[List[float]] = None):
        """
        Initialize multi-scale sampling.

        Args:
            scales: List of sampling scales (ratios). Default is [0.1, 0.25, 0.5]
        """
        self.scales = scales or [0.1, 0.25, 0.5]

    def __call__(self, data: Data) -> Data:
        """Apply multi-scale sampling."""
        if not hasattr(data, "pos") or data.pos is None:
            return data

        if not hasattr(data, "num_nodes") or data.num_nodes is None:
            data.num_nodes = data.pos.size(0)

        sampled_indices = []
        for scale in self.scales:
            n_samples = int(data.num_nodes * scale)
            if n_samples > 10:
                indices = fps(data.pos, ratio=scale, random_start=True)
                sampled_indices.append(indices)

        # Combine indices from different scales
        if sampled_indices:
            combined_indices = torch.unique(torch.cat(sampled_indices))
            logger.info(
                f"Multi-scale sampling: {data.num_nodes} → {len(combined_indices)} nodes"
            )

            # Create subgraph with sampled nodes
            for key, value in data:
                if isinstance(value, torch.Tensor) and value.size(0) == data.num_nodes:
                    setattr(data, key, value[combined_indices])

            data.num_nodes = len(combined_indices)

        return data
