"""
Astronomical Mixin for domain-specific operations.

This module provides astronomy-specific operations: coordinate transforms,
magnitude normalization, sigma clipping, proper motion, and
domain-specific feature extraction/validation.
"""

from typing import Tuple

import astropy.units as u
import numpy as np
import torch
from astropy.coordinates import SkyCoord


class AstronomicalMixin:
    """
    Mixin for astronomy-specific operations.
    Compatible with TensorDict-like containers (self[key] access).
    """

    def to_skycoord(self, frame="icrs"):
        """Convert coordinates to AstroPy SkyCoord object."""
        ra = self["coordinates"][:, 0].cpu().numpy()
        dec = self["coordinates"][:, 1].cpu().numpy()
        if self["coordinates"].shape[1] > 2:
            distance = self["coordinates"][:, 2].cpu().numpy() * u.pc
        else:
            distance = None
        return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, distance=distance, frame=frame)

    def to_galactic(self):
        """Convert to galactic coordinate system."""
        sky = self.to_skycoord()
        return np.vstack([sky.galactic.l.deg, sky.galactic.b.deg]).T

    def apply_proper_motion(self, pm_ra, pm_dec, epoch_from, epoch_to):
        """Apply proper motion correction between epochs."""
        sky = self.to_skycoord()
        sky_pm = SkyCoord(
            ra=sky.ra,
            dec=sky.dec,
            pm_ra_cosdec=pm_ra * u.mas / u.yr,
            pm_dec=pm_dec * u.mas / u.yr,
            frame="icrs",
            obstime=epoch_from,
        )
        sky_pm = sky_pm.apply_space_motion(new_obstime=epoch_to)
        return np.vstack([sky_pm.ra.deg, sky_pm.dec.deg]).T

    def magnitude_normalization(self, key="magnitudes"):
        """Apply astronomical magnitude normalization (subtract median)."""
        mags = self[key]
        median = torch.median(mags)
        self[key] = mags - median
        return self

    def sigma_clip(self, key="data", sigma=3.0):
        """Apply sigma-clipping for robust normalization."""
        data = self[key]
        median = torch.median(data)
        mad = torch.median(torch.abs(data - median))
        mask = torch.abs(data - median) < (sigma * mad)
        clipped = data[mask]
        self[key] = clipped
        return self

    def robust_normalize(
        self, data_key: str = "data", sigma: float = 3.0, method: str = "mad"
    ) -> "AstronomicalMixin":
        """
        Robust normalization using astronomical standards.

        Args:
            data_key: Key of the data tensor
            sigma: Sigma clipping threshold
            method: 'mad' for median absolute deviation, 'std' for standard

        Returns:
            Self with normalized data
        """
        if data_key not in self:
            raise ValueError(f"Data key '{data_key}' not found")

        data = self[data_key]

        # Iterative sigma clipping
        for _ in range(3):  # Maximum 3 iterations
            if method == "mad":
                median = torch.median(data)
                mad = torch.median(torch.abs(data - median))
                mask = torch.abs(data - median) < (sigma * mad)
                robust_mean = median
                robust_std = mad * 1.4826  # Convert MAD to std equivalent
            else:
                mean = torch.mean(data)
                std = torch.std(data)
                mask = torch.abs(data - mean) < (sigma * std)
                robust_mean = torch.mean(data[mask])
                robust_std = torch.std(data[mask])

            # Update data with masked values
            data = data[mask]
            if mask.sum() == len(mask):  # No more outliers
                break

        # Normalize
        normalized = (self[data_key] - robust_mean) / (robust_std + 1e-8)
        self[data_key] = normalized
        return self

    def angular_separation(
        self,
        other_coords: torch.Tensor,
        coords_key: str = "coordinates",
        method: str = "haversine",
    ) -> torch.Tensor:
        """
        Calculate angular separation between astronomical coordinates.

        Args:
            other_coords: [M, 2] or [M, 3] Other coordinates (RA, Dec[, distance])
            coords_key: Key of the coordinates tensor
            method: 'haversine' or 'cosine' formula

        Returns:
            [N, M] Angular separations in arcseconds
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]

        # Extract RA, Dec (assume first two dimensions)
        ra1 = coords[:, 0] * torch.pi / 180  # Convert to radians
        dec1 = coords[:, 1] * torch.pi / 180

        ra2 = other_coords[:, 0] * torch.pi / 180
        dec2 = other_coords[:, 1] * torch.pi / 180

        # Expand for broadcasting
        ra1 = ra1.unsqueeze(-1)
        dec1 = dec1.unsqueeze(-1)

        if method == "haversine":
            # Haversine formula (more numerically stable)
            dra = ra1 - ra2
            ddec = dec1 - dec2

            a = (
                torch.sin(ddec / 2) ** 2
                + torch.cos(dec1) * torch.cos(dec2) * torch.sin(dra / 2) ** 2
            )
            c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        else:
            # Standard spherical trigonometry
            cos_sep = torch.sin(dec1) * torch.sin(dec2) + torch.cos(dec1) * torch.cos(
                dec2
            ) * torch.cos(ra1 - ra2)
            c = torch.acos(torch.clamp(cos_sep, -1, 1))

        # Convert to arcseconds
        return c * 180 / torch.pi * 3600

    def extract_astronomical_features(
        self,
        coords_key: str = "coordinates",
        include_galactic: bool = True,
        include_magnitude_colors: bool = True,
        include_proper_motions: bool = True,
    ) -> torch.Tensor:
        """
        Extract astronomy-specific features.

        Args:
            coords_key: Key of the coordinates tensor
            include_galactic: Include galactic coordinates
            include_magnitude_colors: Include magnitude and color features
            include_proper_motions: Include proper motion features

        Returns:
            [N, F] Astronomical feature tensor
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]
        features = []

        # Basic coordinate features
        # ra = coords[:, 0]  # Removed unused variable
        dec = coords[:, 1]

        # Absolute galactic/ecliptic latitude (measure of distance from plane)
        features.append(torch.abs(dec))

        if coords.shape[1] > 2:
            distance = coords[:, 2]
            features.extend(
                [
                    distance,
                    torch.log10(distance + 1e-6),  # Log distance
                ]
            )

        # Galactic coordinates
        if include_galactic and hasattr(self, "to_galactic"):
            try:
                galactic_coords = self.to_galactic()
                galactic_l = torch.tensor(galactic_coords[:, 0], dtype=torch.float32)
                galactic_b = torch.tensor(galactic_coords[:, 1], dtype=torch.float32)
                features.extend(
                    [
                        galactic_l,
                        torch.abs(galactic_b),  # Distance from galactic plane
                    ]
                )
            except Exception:
                pass  # Skip if conversion fails

        # Magnitude and color features
        if include_magnitude_colors:
            if "g_mag" in self and "bp_mag" in self and "rp_mag" in self:
                g_mag = self["g_mag"]
                bp_mag = self["bp_mag"]
                rp_mag = self["rp_mag"]

                # Standard astronomical colors
                bp_rp = bp_mag - rp_mag
                g_rp = g_mag - rp_mag

                features.extend([g_mag, bp_rp, g_rp])
            elif "magnitudes" in self:
                mags = self["magnitudes"]
                if mags.ndim > 1:
                    # Multi-band magnitudes
                    features.append(mags.flatten())
                else:
                    features.append(mags)

        # Proper motion features
        if include_proper_motions:
            if "pmra" in self and "pmdec" in self:
                pmra = self["pmra"]
                pmdec = self["pmdec"]

                # Total proper motion
                pm_total = torch.sqrt(pmra**2 + pmdec**2)

                features.extend([pmra, pmdec, pm_total])

        # Parallax features
        if "parallax" in self:
            parallax = self["parallax"]
            features.extend(
                [
                    parallax,
                    torch.log10(torch.abs(parallax) + 1e-6),
                    1000.0 / (parallax + 1e-6),  # Distance estimate in pc
                ]
            )

        return (
            torch.stack(features, dim=-1)
            if features
            else torch.zeros(coords.shape[0], 0)
        )

    def validate_coordinates(self, key="coordinates"):
        """Validate astronomical coordinates."""
        if key not in self:
            return False

        coords = self[key]

        # Check basic tensor properties
        if not isinstance(coords, torch.Tensor):
            return False

        if coords.ndim != 2 or coords.shape[1] < 2:
            return False

        # ra = coords[:, 0]  # Removed unused variable
        dec = coords[:, 1]

        # Validate RA range [0, 360)
        if torch.any(ra < 0) or torch.any(ra >= 360):
            return False

        # Validate Dec range [-90, 90]
        if torch.any(dec < -90) or torch.any(dec > 90):
            return False

        # Check for finite values
        if not torch.isfinite(coords).all():
            return False

        return True

    def validate_magnitudes(self, mag_key: str = "magnitudes") -> bool:
        """Validate astronomical magnitude data."""
        if mag_key not in self:
            return False

        mags = self[mag_key]
        if not isinstance(mags, torch.Tensor):
            return False

        # Check for reasonable magnitude ranges (-30 to 50 mag)
        if torch.any(mags < -30) or torch.any(mags > 50):
            return False

        # Check for finite values
        return torch.isfinite(mags).all()

    def validate_proper_motions(self) -> bool:
        """Validate proper motion data."""
        if "pmra" not in self or "pmdec" not in self:
            return False

        pmra = self["pmra"]
        pmdec = self["pmdec"]

        # Check for reasonable proper motion ranges (±10000 mas/yr)
        if torch.any(torch.abs(pmra) > 10000) or torch.any(torch.abs(pmdec) > 10000):
            return False

        return torch.isfinite(pmra).all() and torch.isfinite(pmdec).all()

    def validate_parallax(self) -> bool:
        """Validate parallax data."""
        if "parallax" not in self:
            return False

        parallax = self["parallax"]

        # Check for reasonable parallax range (0.001 to 1000 mas)
        if torch.any(parallax < 0.001) or torch.any(parallax > 1000):
            return False

        return torch.isfinite(parallax).all()

    def minimum_spanning_tree(self, frame="galactocentric"):
        """
        Compute the Minimum Spanning Tree (MST) for astronomical coordinates.

        Args:
            frame: Coordinate frame for MST calculation

        Returns:
            MST edges as index pairs (N-1, 2)
        """
        try:
            sky = self.to_skycoord(frame=frame)
            xyz = sky.cartesian.xyz.to_value("pc")
            xyz = np.stack(xyz, axis=1)
        except Exception:
            # Fallback to direct coordinates
            coords = self["coordinates"]
            xyz = coords.cpu().numpy()

        from scipy.sparse.csgraph import minimum_spanning_tree
        from scipy.spatial import distance_matrix

        dist_mat = distance_matrix(xyz, xyz)
        mst = minimum_spanning_tree(dist_mat)
        edges = np.array(mst.nonzero()).T
        return edges

    def cross_match(
        self,
        other_coords: torch.Tensor,
        tolerance: float = 1.0,
        coords_key: str = "coordinates",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-match with another catalog.

        Args:
            other_coords: [M, 2] Other catalog coordinates (RA, Dec)
            tolerance: Matching tolerance in arcseconds
            coords_key: Key of the coordinates tensor

        Returns:
            Tuple of (matched_indices_self, matched_indices_other, distances)
        """
        # Calculate angular separations
        separations = self.angular_separation(other_coords, coords_key)

        # Find best matches within tolerance
        min_separations, best_matches = torch.min(separations, dim=1)

        # Filter by tolerance
        valid_matches = min_separations < tolerance

        self_indices = torch.where(valid_matches)[0]
        other_indices = best_matches[valid_matches]
        match_distances = min_separations[valid_matches]

        return self_indices, other_indices, match_distances

    def distance_modulus_correction(
        self, apparent_key: str, distance_key: str
    ) -> torch.Tensor:
        """
        Apply distance modulus correction to convert apparent to absolute magnitude.

        Args:
            apparent_key: Key for apparent magnitudes
            distance_key: Key for distances in parsecs

        Returns:
            Absolute magnitudes
        """
        if apparent_key not in self or distance_key not in self:
            raise ValueError("Required keys not found for distance modulus correction")

        apparent_mag = self[apparent_key]
        distance_pc = self[distance_key]

        # Distance modulus: μ = 5 * log10(d) - 5
        distance_modulus = 5 * torch.log10(distance_pc) - 5
        absolute_mag = apparent_mag - distance_modulus

        return absolute_mag
