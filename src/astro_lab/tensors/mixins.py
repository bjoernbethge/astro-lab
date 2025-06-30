"""
Mixins for common TensorDict functionality in AstroLab.

This module provides mixin classes that can be used to add common
functionality to TensorDict classes without code duplication.
"""

from typing import List, Optional, Tuple

import torch


class NormalizationMixin:
    """Mixin for astronomical data normalization."""

    def normalize(
        self,
        method: str = "standard",
        data_key: str = "data",
        dim: int = -1,
        robust_sigma: float = 3.0,
    ) -> "NormalizationMixin":
        """
        Normalize data using astronomical-appropriate methods.

        Args:
            method: 'standard', 'minmax', 'robust', 'magnitude', or 'log'
            data_key: Key of the data tensor to normalize
            dim: Dimension to normalize over
            robust_sigma: Sigma clipping threshold for robust normalization

        Returns:
            Self with normalized data
        """
        if data_key not in self:
            raise ValueError(f"Data key '{data_key}' not found in TensorDict")

        data = self[data_key]

        if method == "standard":
            mean = torch.mean(data, dim=dim, keepdim=True)
            std = torch.std(data, dim=dim, keepdim=True)
            normalized = (data - mean) / (std + 1e-8)
        elif method == "minmax":
            min_vals = torch.min(data, dim=dim, keepdim=True)[0]
            max_vals = torch.max(data, dim=dim, keepdim=True)[0]
            normalized = (data - min_vals) / (max_vals - min_vals + 1e-8)
        elif method == "robust":
            # Sigma clipping for astronomical data
            median = torch.median(data, dim=dim, keepdim=True)[0]
            mad = torch.median(torch.abs(data - median), dim=dim, keepdim=True)[0]
            mask = torch.abs(data - median) < (robust_sigma * mad)
            clipped_mean = torch.mean(data.where(mask, median), dim=dim, keepdim=True)
            clipped_std = torch.std(data.where(mask, median), dim=dim, keepdim=True)
            normalized = (data - clipped_mean) / (clipped_std + 1e-8)
        elif method == "magnitude":
            # Astronomical magnitude normalization: m - m_median
            median = torch.median(data, dim=dim, keepdim=True)[0]
            normalized = data - median
        elif method == "log":
            # Logarithmic normalization for flux-like quantities
            normalized = torch.log10(torch.clamp(data, min=1e-10))
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        self[data_key] = normalized
        self.add_history("normalize", method=method, data_key=data_key)
        return self


class CoordinateConversionMixin:
    """Mixin for astronomical coordinate conversions."""

    def to_spherical_coords(
        self, coords_key: str = "coordinates"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert Cartesian coordinates to spherical (RA, Dec, Distance).

        Args:
            coords_key: Key of the coordinates tensor [N, 3]

        Returns:
            Tuple of (RA [deg], Dec [deg], Distance) tensors
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]
        if coords.shape[-1] != 3:
            raise ValueError("Coordinates must be 3D Cartesian")

        x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]

        # Distance
        distance = torch.norm(coords, dim=-1)

        # RA (longitude) in degrees [0, 360)
        ra = torch.atan2(y, x) * 180 / torch.pi
        ra = torch.where(ra < 0, ra + 360, ra)

        # Dec (latitude) in degrees [-90, 90]
        dec = torch.asin(torch.clamp(z / (distance + 1e-10), -1, 1)) * 180 / torch.pi

        return ra, dec, distance

    def to_cartesian_coords(
        self, ra: torch.Tensor, dec: torch.Tensor, distance: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert spherical coordinates to Cartesian.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            distance: Distance

        Returns:
            [N, 3] Cartesian coordinates
        """
        # Convert to radians
        ra_rad = ra * torch.pi / 180
        dec_rad = dec * torch.pi / 180

        # Convert to Cartesian
        x = distance * torch.cos(dec_rad) * torch.cos(ra_rad)
        y = distance * torch.cos(dec_rad) * torch.sin(ra_rad)
        z = distance * torch.sin(dec_rad)

        return torch.stack([x, y, z], dim=-1)

    def angular_separation(
        self,
        other_coords: torch.Tensor,
        coords_key: str = "coordinates",
        method: str = "haversine",
    ) -> torch.Tensor:
        """
        Calculate angular separation between coordinates.

        Args:
            other_coords: [M, 3] Other coordinates
            coords_key: Key of the coordinates tensor
            method: 'haversine' or 'cosine' formula

        Returns:
            [N, M] Angular separations in arcseconds
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        # Convert both to spherical
        ra1, dec1, _ = self.to_spherical_coords(coords_key)

        # Create temporary spatial object for other coords
        temp_spatial = type(self)({coords_key: other_coords})
        ra2, dec2, _ = temp_spatial.to_spherical_coords(coords_key)

        # Convert to radians
        ra1_rad = ra1.unsqueeze(-1) * torch.pi / 180
        dec1_rad = dec1.unsqueeze(-1) * torch.pi / 180
        ra2_rad = ra2 * torch.pi / 180
        dec2_rad = dec2 * torch.pi / 180

        if method == "haversine":
            # Haversine formula (more numerically stable)
            dra = ra1_rad - ra2_rad
            ddec = dec1_rad - dec2_rad

            a = (
                torch.sin(ddec / 2) ** 2
                + torch.cos(dec1_rad) * torch.cos(dec2_rad) * torch.sin(dra / 2) ** 2
            )
            c = 2 * torch.asin(torch.sqrt(torch.clamp(a, 0, 1)))
        else:
            # Standard spherical trigonometry
            cos_sep = torch.sin(dec1_rad) * torch.sin(dec2_rad) + torch.cos(
                dec1_rad
            ) * torch.cos(dec2_rad) * torch.cos(ra1_rad - ra2_rad)
            c = torch.acos(torch.clamp(cos_sep, -1, 1))

        # Convert to arcseconds
        return c * 180 / torch.pi * 3600


class FeatureExtractionMixin:
    """Mixin for astronomical feature extraction."""

    def extract_spatial_features(
        self,
        coords_key: str = "coordinates",
        include_neighbors: bool = True,
        k_neighbors: int = 10,
    ) -> torch.Tensor:
        """
        Extract spatial features for astronomical objects.

        Args:
            coords_key: Key of the coordinates tensor
            include_neighbors: Include neighbor-based features
            k_neighbors: Number of neighbors for spatial features

        Returns:
            [N, F] Feature tensor
        """
        if coords_key not in self:
            raise ValueError(f"Coordinates key '{coords_key}' not found")

        coords = self[coords_key]
        features = []

        # Basic spatial features
        ra, dec, distance = self.to_spherical_coords(coords_key)
        features.extend(
            [
                distance,  # Distance from origin
                torch.abs(dec),  # Absolute galactic/ecliptic latitude
                torch.norm(coords, dim=-1),  # 3D distance
            ]
        )

        # Galactic coordinates if not already
        if hasattr(self, "skycoord"):
            galactic = self.skycoord.galactic
            features.extend(
                [
                    torch.tensor(
                        galactic.l.deg, dtype=torch.float32
                    ),  # Galactic longitude
                    torch.tensor(
                        galactic.b.deg, dtype=torch.float32
                    ),  # Galactic latitude
                ]
            )

        if include_neighbors and coords.shape[0] > k_neighbors:
            # Neighbor-based features
            from torch_geometric.nn import knn_graph

            edge_index = knn_graph(coords, k=k_neighbors, loop=False)

            # Local density (inverse of k-th nearest neighbor distance)
            row, col = edge_index
            distances = torch.norm(coords[row] - coords[col], dim=1)

            # Reshape to [N, k] and get k-th distance
            distances_reshaped = distances.view(-1, k_neighbors)
            kth_distance = distances_reshaped[:, -1]  # k-th nearest neighbor
            local_density = 1.0 / (kth_distance + 1e-10)
            features.append(local_density)

        return torch.stack(features, dim=-1)

    def extract_morphological_features(
        self, data_key: str = "data", method: str = "moments"
    ) -> torch.Tensor:
        """
        Extract morphological features for astronomical objects.

        Args:
            data_key: Key of the data tensor
            method: 'moments', 'shape', or 'texture'

        Returns:
            [N, F] Morphological feature tensor
        """
        if data_key not in self:
            raise ValueError(f"Data key '{data_key}' not found")

        data = self[data_key]
        features = []

        if method == "moments":
            # Statistical moments
            features.extend(
                [
                    torch.mean(data, dim=-1),  # First moment (mean)
                    torch.std(data, dim=-1),  # Second moment (std)
                    self._skewness(data),  # Third moment (skewness)
                    self._kurtosis(data),  # Fourth moment (kurtosis)
                ]
            )
        elif method == "shape":
            # Shape-based features for 2D/3D data
            if data.dim() >= 3:  # Image-like data
                features.extend(
                    [
                        torch.sum(data, dim=(-2, -1)),  # Total flux
                        torch.std(data, dim=(-2, -1)),  # Flux variation
                    ]
                )
        elif method == "texture":
            # Texture features (simplified)
            if data.dim() >= 2:
                # Local variation
                if data.dim() == 3:  # [N, H, W]
                    dx = torch.diff(data, dim=-1)
                    dy = torch.diff(data, dim=-2)
                    features.extend(
                        [
                            torch.mean(torch.abs(dx), dim=(-2, -1)),
                            torch.mean(torch.abs(dy), dim=(-2, -1)),
                        ]
                    )

        return (
            torch.stack(features, dim=-1) if features else torch.zeros(data.shape[0], 0)
        )

    def _skewness(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate skewness along given dimension."""
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        return torch.mean(normalized**3, dim=dim)

    def _kurtosis(self, data: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Calculate kurtosis along given dimension."""
        mean = torch.mean(data, dim=dim, keepdim=True)
        std = torch.std(data, dim=dim, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        return torch.mean(normalized**4, dim=dim) - 3.0  # Excess kurtosis


class ValidationMixin:
    """Mixin for astronomical data validation."""

    def validate_coordinates(self, coords_key: str = "coordinates") -> bool:
        """Validate coordinate data."""
        if coords_key not in self:
            return False

        coords = self[coords_key]
        if not isinstance(coords, torch.Tensor):
            return False

        # Check for valid 3D coordinates
        if coords.shape[-1] != 3:
            return False

        # Check for finite values
        if not torch.isfinite(coords).all():
            return False

        # Check reasonable coordinate ranges (e.g., not beyond observable universe)
        max_distance = torch.norm(coords, dim=-1).max()
        if max_distance > 1e6:  # 1 Mpc in pc, adjust as needed
            return False

        return True

    def validate_magnitudes(self, mag_key: str = "magnitudes") -> bool:
        """Validate magnitude data."""
        if mag_key not in self:
            return False

        mags = self[mag_key]
        if not isinstance(mags, torch.Tensor):
            return False

        # Check for reasonable magnitude ranges
        if torch.any(mags < -30) or torch.any(mags > 50):
            return False

        # Check for finite values
        return torch.isfinite(mags).all()

    def validate_required_keys(self, required_keys: List[str]) -> bool:
        """Validate that required keys are present."""
        return all(key in self for key in required_keys)

    def validate_astronomical_units(self) -> bool:
        """Validate that astronomical quantities have reasonable units."""
        # This would check metadata for proper units
        if hasattr(self, "_metadata"):
            unit_str = self._metadata.get("unit", "")
            valid_units = ["pc", "kpc", "Mpc", "AU", "km", "m"]
            return any(unit in unit_str for unit in valid_units)
        return True


class GraphConstructionMixin:
    """Mixin for graph construction from astronomical data."""

    def build_knn_graph(
        self,
        coords_key: str = "coordinates",
        k: int = 10,
        batch: Optional[torch.Tensor] = None,
        loop: bool = False,
    ) -> torch.Tensor:
        """Build k-nearest neighbor graph."""
        from torch_geometric.nn import knn_graph

        coords = self[coords_key]
        return knn_graph(coords, k=k, batch=batch, loop=loop)

    def build_radius_graph(
        self,
        coords_key: str = "coordinates",
        r: float = 10.0,
        batch: Optional[torch.Tensor] = None,
        loop: bool = False,
        max_num_neighbors: int = 64,
    ) -> torch.Tensor:
        """Build radius graph for spatial clustering."""
        from torch_geometric.nn import radius_graph

        coords = self[coords_key]
        return radius_graph(
            coords, r=r, batch=batch, loop=loop, max_num_neighbors=max_num_neighbors
        )

    def build_cosmic_web_graph(
        self,
        coords_key: str = "coordinates",
        method: str = "adaptive",
        base_radius: float = 10.0,
        k_neighbors: int = 10,
    ) -> torch.Tensor:
        """
        Build graph for cosmic web analysis.

        Args:
            method: 'knn', 'radius', or 'adaptive'
            base_radius: Base radius for radius graph (pc)
            k_neighbors: Number of neighbors for kNN
        """
        coords = self[coords_key]

        if method == "knn":
            return self.build_knn_graph(coords_key, k=k_neighbors)
        elif method == "radius":
            return self.build_radius_graph(coords_key, r=base_radius)
        elif method == "adaptive":
            # Adaptive radius based on local density
            knn_edges = self.build_knn_graph(coords_key, k=k_neighbors)
            row, col = knn_edges

            # Calculate k-th nearest neighbor distances
            distances = torch.norm(coords[row] - coords[col], dim=1)
            distances_reshaped = distances.view(-1, k_neighbors)
            kth_distances = distances_reshaped[:, -1]  # k-th nearest neighbor distance

            # Use 2 * k-th distance as adaptive radius
            adaptive_radii = 2.0 * kth_distances

            # Build radius graph with varying radii (approximation)
            median_radius = torch.median(adaptive_radii)
            return self.build_radius_graph(coords_key, r=median_radius.item())
        else:
            raise ValueError(f"Unknown graph method: {method}")
