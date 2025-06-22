"""
Astronomical Transform Classes for PyTorch Geometric
=================================================

Domain-specific transforms for astronomical data:
- Coordinate system conversions
- Magnitude and color computations
- Feature normalization and scaling
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform, Compose

class CoordinateSystemTransform(BaseTransform):
    """Transform between different astronomical coordinate systems."""

    def __init__(self, target_system: str = "galactic"):
        """
        Initialize coordinate system transform.

        Parameters
        ----------
        target_system : str, default "galactic"
            Target coordinate system ("galactic", "icrs", "fk5")
        """
        self.target_system = target_system

    def forward(self, data: Data) -> Data:
        """Apply coordinate system transformation."""
        if hasattr(data, "pos") and data.pos is not None:
            # Simple galactic coordinate conversion (simplified)
            if self.target_system == "galactic":
                # Convert RA/Dec to Galactic (simplified transformation)
                pos = data.pos.clone()
                # This is a placeholder - real implementation would use astropy
                data.pos = pos

        return data

class AddAstronomicalColors(BaseTransform):
    """Add astronomical color indices to node features."""

    def __init__(self, color_bands: List[str] = ["g", "r", "i"]):
        """
        Initialize color computation.

        Parameters
        ----------
        color_bands : List[str], default ["g", "r", "i"]
            Photometric bands to use for color computation
        """
        self.color_bands = color_bands

    def forward(self, data: Data) -> Data:
        """Compute and add color indices."""
        if hasattr(data, "x") and data.x is not None:
            # Assume magnitudes are in the first columns
            x = data.x
            n_features = x.shape[1]

            if n_features >= len(self.color_bands):
                colors = []
                for i in range(len(self.color_bands) - 1):
                    # Color = mag1 - mag2
                    color = x[:, i] - x[:, i + 1]
                    colors.append(color.unsqueeze(1))

                if colors:
                    color_tensor = torch.cat(colors, dim=1)
                    data.x = torch.cat([x, color_tensor], dim=1)

        return data

class AddDistanceFeatures(BaseTransform):
    """Add distance-based features for astronomical objects."""

    def __init__(self, add_distance_modulus: bool = True):
        """
        Initialize distance features.

        Parameters
        ----------
        add_distance_modulus : bool, default True
            Whether to add distance modulus (m - M)
        """
        self.add_distance_modulus = add_distance_modulus

    def forward(self, data: Data) -> Data:
        """Add distance-based features."""
        if hasattr(data, "pos") and data.pos is not None:
            pos = data.pos

            if pos.shape[1] >= 3:  # Has distance information
                distances = pos[:, 2]  # Assume 3rd column is distance

                if self.add_distance_modulus:
                    # Distance modulus: DM = 5 * log10(d) - 5
                    # Clamp to avoid log(0)
                    distances_clamped = torch.clamp(distances, min=1e-6)
                    distance_modulus = 5 * torch.log10(distances_clamped) - 5

                    if hasattr(data, "x") and data.x is not None:
                        data.x = torch.cat(
                            [data.x, distance_modulus.unsqueeze(1)], dim=1
                        )
                    else:
                        data.x = distance_modulus.unsqueeze(1)

        return data

class NormalizeAstronomicalFeatures(BaseTransform):
    """Normalize astronomical features (magnitudes, colors, etc.)."""

    def __init__(
        self, method: str = "zscore", mag_range: Tuple[float, float] = (10.0, 30.0)
    ):
        """
        Initialize feature normalization.

        Parameters
        ----------
        method : str, default "zscore"
            Normalization method ("zscore", "minmax", "robust")
        mag_range : Tuple[float, float], default (10.0, 30.0)
            Expected magnitude range for clipping
        """
        self.method = method
        self.mag_range = mag_range

    def forward(self, data: Data) -> Data:
        """Normalize astronomical features."""
        if hasattr(data, "x") and data.x is not None:
            x = data.x.clone()

            if self.method == "zscore":
                # Z-score normalization
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True)
                std = torch.clamp(std, min=1e-8)  # Avoid division by zero
                data.x = (x - mean) / std

            elif self.method == "minmax":
                # Min-max normalization
                min_vals = x.min(dim=0, keepdim=True)[0]
                max_vals = x.max(dim=0, keepdim=True)[0]
                range_vals = max_vals - min_vals
                range_vals = torch.clamp(range_vals, min=1e-8)
                data.x = (x - min_vals) / range_vals

            elif self.method == "robust":
                # Robust normalization using median and IQR
                median = x.median(dim=0, keepdim=True)[0]
                q75 = x.quantile(0.75, dim=0, keepdim=True)
                q25 = x.quantile(0.25, dim=0, keepdim=True)
                iqr = q75 - q25
                iqr = torch.clamp(iqr, min=1e-8)
                data.x = (x - median) / iqr

        return data

class AddRedshiftFeatures(BaseTransform):
    """Add redshift-derived features for extragalactic objects."""

    def __init__(self, cosmology_params: Optional[Dict[str, float]] = None):
        """
        Initialize redshift features.

        Parameters
        ----------
        cosmology_params : Dict[str, float], optional
            Cosmological parameters (H0, Omega_m, Omega_Lambda)
        """
        self.cosmology = cosmology_params or {
            "H0": 70.0,  # km/s/Mpc
            "Omega_m": 0.3,
            "Omega_Lambda": 0.7,
        }

    def forward(self, data: Data) -> Data:
        """Add redshift-derived features."""
        # This is a placeholder - real implementation would use astropy.cosmology
        if hasattr(data, "redshift") and data.redshift is not None:
            z = data.redshift

            # Luminosity distance (simplified)
            # Real implementation: astropy.cosmology.luminosity_distance()
            c = 299792.458  # km/s
            H0 = self.cosmology["H0"]
            d_L = c * z / H0  # Simplified for small z

            # Add as feature
            if hasattr(data, "x") and data.x is not None:
                data.x = torch.cat([data.x, d_L.unsqueeze(1)], dim=1)
            else:
                data.x = d_L.unsqueeze(1)

        return data

# === Convenience Functions ===

def get_default_astro_transforms(
    add_colors: bool = True,
    add_distances: bool = True,
    normalize: bool = True,
    coordinate_system: str = "icrs",
) -> Compose:
    """
    Get default astronomical transforms for most use cases.

    Parameters
    ----------
    add_colors : bool, default True
        Whether to add color indices
    add_distances : bool, default True
        Whether to add distance features
    normalize : bool, default True
        Whether to normalize features
    coordinate_system : str, default "icrs"
        Target coordinate system

    Returns
    -------
    Compose
        Composed transform pipeline
    """
    transforms = []

    # Coordinate system conversion
    if coordinate_system != "icrs":
        transforms.append(CoordinateSystemTransform(coordinate_system))

    # Add astronomical features
    if add_colors:
        transforms.append(AddAstronomicalColors())

    if add_distances:
        transforms.append(AddDistanceFeatures())

    # Normalization (should be last)
    if normalize:
        transforms.append(NormalizeAstronomicalFeatures())

    return Compose(transforms)

def get_galaxy_transforms() -> Compose:
    """Get transforms optimized for galaxy data."""
    return Compose(
        [
            AddAstronomicalColors(color_bands=["u", "g", "r", "i", "z"]),
            AddDistanceFeatures(add_distance_modulus=True),
            AddRedshiftFeatures(),
            NormalizeAstronomicalFeatures(method="robust"),
        ]
    )

def get_stellar_transforms() -> Compose:
    """Get transforms optimized for stellar data."""
    return Compose(
        [
            CoordinateSystemTransform("galactic"),
            AddAstronomicalColors(color_bands=["g", "r", "i"]),
            AddDistanceFeatures(add_distance_modulus=True),
            NormalizeAstronomicalFeatures(method="zscore"),
        ]
    )

def get_exoplanet_transforms() -> Compose:
    """Get transforms optimized for exoplanet data."""
    return Compose(
        [
            AddDistanceFeatures(add_distance_modulus=False),  # Most are nearby
            NormalizeAstronomicalFeatures(method="minmax"),
        ]
    )
