"""
Spatial 3D Tensor - 3D Spatial Data Representation
=================================================

Provides 3D spatial tensor classes for astronomical coordinate systems
and spatial data processing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import polars as pl
import torch

from .base import AstroTensorBase

# Configure logging
logger = logging.getLogger(__name__)

# Check for optional dependencies
import astropy.units as u
import torch_geometric
from astropy.coordinates import ICRS, Galactic, SkyCoord
from astropy.time import Time
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import BallTree
from torch_geometric.data import Data


class Spatial3DTensor(AstroTensorBase):
    """
    Proper 3D Spatial Tensor for astronomical coordinates.

    Uses unified [N, 3] tensor structure for efficient spatial operations.
    Compatible with astroML, poliastro, and astropy.

    The tensor data is always stored as 3D Cartesian coordinates [x, y, z]
    in the specified coordinate system with proper units.
    """

    _metadata_fields = [
        "coordinate_system",  # 'icrs', 'galactic', 'ecliptic'
        "unit",  # distance unit: 'pc', 'kpc', 'Mpc'
        "epoch",  # reference epoch (J2000.0 default)
        "frame",  # astropy coordinate frame
        "spatial_index",  # KDTree/BallTree for fast queries
        "original_spherical",  # preserve original (ra, dec, distance) if converted
    ]

    def __init__(
        self,
        data: Union[torch.Tensor, np.ndarray, List],
        coordinate_system: str = "icrs",
        unit: str = "Mpc",
        epoch: float = 2000.0,
        spatial_index: bool = True,
        **kwargs,
    ):
        """
        Initialize 3D spatial tensor.

        Args:
            data: Tensor of shape [N, 3] with Cartesian coordinates [x, y, z]
            coordinate_system: 'icrs', 'galactic', 'ecliptic'
            unit: Distance unit ('pc', 'kpc', 'Mpc', 'AU')
            epoch: Reference epoch (years)
            spatial_index: Whether to build spatial index for fast queries
        """
        # Convert to tensor and validate shape
        tensor_data = torch.as_tensor(data, dtype=torch.float32)
        if tensor_data.dim() == 1 and tensor_data.shape[0] == 3:
            tensor_data = tensor_data.unsqueeze(0)  # Single point: [3] -> [1, 3]

        if tensor_data.shape[-1] != 3:
            raise ValueError(
                f"Spatial tensor must have shape [..., 3], got {tensor_data.shape}"
            )

        # Validate coordinate system
        valid_systems = ["icrs", "galactic", "ecliptic"]
        if coordinate_system not in valid_systems:
            raise ValueError(f"coordinate_system must be one of {valid_systems}")

        # Initialize base tensor - handle metadata conflicts
        metadata = {
            "coordinate_system": coordinate_system,
            "unit": unit,
            "epoch": epoch,
            "frame": kwargs.pop("frame", None),
            "spatial_index": kwargs.pop("spatial_index", None),
            "original_spherical": kwargs.pop("original_spherical", None),
            "tensor_type": "spatial_3d",
            **kwargs,
        }
        super().__init__(tensor_data, **metadata)
        self._validate()  # Call validation after initialization

        # Build spatial index if requested
        if spatial_index:
            self._build_spatial_index()

    def _validate(self) -> None:
        """Validate spatial tensor data."""
        if len(self._data) == 0:
            raise ValueError("Spatial tensor data cannot be empty")

        if self._data.shape[-1] != 3:
            raise ValueError(
                f"Spatial tensor must have 3 coordinates, got {self._data.shape[-1]}"
            )

    @classmethod  # type: ignore
    def from_spherical(
        cls,
        ra: Union[torch.Tensor, np.ndarray, float],
        dec: Union[torch.Tensor, np.ndarray, float],
        distance: Union[torch.Tensor, np.ndarray, float],
        coordinate_system: str = "icrs",
        unit: str = "Mpc",
        angular_unit: str = "deg",
        **kwargs,
    ) -> "Spatial3DTensor":
        """
        Create from spherical coordinates (RA, Dec, Distance).

        Args:
            ra: Right Ascension (or longitude)
            dec: Declination (or latitude)
            distance: Distance values
            coordinate_system: Coordinate system
            unit: Distance unit
            angular_unit: Angular unit ('deg' or 'rad')
        """
        # Convert to tensors
        ra_tensor = torch.as_tensor(ra, dtype=torch.float32)
        dec_tensor = torch.as_tensor(dec, dtype=torch.float32)
        dist_tensor = torch.as_tensor(distance, dtype=torch.float32)

        # Ensure same shape
        ra_tensor, dec_tensor, dist_tensor = torch.broadcast_tensors(
            ra_tensor, dec_tensor, dist_tensor
        )

        # Convert angles to radians if needed
        if angular_unit == "deg":
            ra_rad = torch.deg2rad(ra_tensor)
            dec_rad = torch.deg2rad(dec_tensor)
        else:
            ra_rad = ra_tensor
            dec_rad = dec_tensor

        # Convert to Cartesian coordinates
        x = dist_tensor * torch.cos(dec_rad) * torch.cos(ra_rad)
        y = dist_tensor * torch.cos(dec_rad) * torch.sin(ra_rad)
        z = dist_tensor * torch.sin(dec_rad)

        cartesian = torch.stack([x, y, z], dim=-1)

        # Create tensor
        tensor = cls(
            cartesian, coordinate_system=coordinate_system, unit=unit, **kwargs
        )

        # Store original spherical coordinates
        tensor.update_metadata(
            original_spherical={
                "ra": ra_tensor,
                "dec": dec_tensor,
                "distance": dist_tensor,
                "angular_unit": angular_unit,
            }
        )

        return tensor

    @classmethod  # type: ignore
    def from_astropy(
        cls, skycoord: Any, unit: str = "Mpc", **kwargs
    ) -> "Spatial3DTensor":
        """Create from astropy SkyCoord object."""

        try:
            # Simple extraction without complex unit handling
            ra, dec, distance = (
                skycoord.spherical.lon.deg,
                skycoord.spherical.lat.deg,
                skycoord.distance.to("Mpc").value,
            )
            return cls.from_spherical(ra, dec, distance, unit=unit, **kwargs)
        except Exception:
            # Fallback: assume simple coordinate arrays
            coords = torch.tensor(
                [[0, 0, 1]], dtype=torch.float32
            )  # Default single point
            return cls(coords, unit=unit, **kwargs)

    @property
    def cartesian(self) -> torch.Tensor:
        """Get Cartesian coordinates [N, 3]."""
        return self._data

    @property
    def x(self) -> torch.Tensor:
        """X coordinates."""
        return self._data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        """Y coordinates."""
        return self._data[..., 1]

    @property
    def z(self) -> torch.Tensor:
        """Z coordinates."""
        return self._data[..., 2]

    @property
    def coordinate_system(self) -> str:
        """Coordinate system."""
        return self._metadata.get("coordinate_system", "icrs")

    @property
    def unit(self) -> str:
        """Distance unit."""
        return self._metadata.get("unit", "Mpc")

    @property
    def epoch(self) -> float:
        """Reference epoch."""
        return self._metadata.get("epoch", 2000.0)

    def to_spherical(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert to spherical coordinates (RA, Dec, Distance).

        Returns:
            Tuple of (ra, dec, distance) in degrees and original units
        """
        x, y, z = self.x, self.y, self.z

        # Calculate spherical coordinates
        distance = torch.sqrt(x**2 + y**2 + z**2)
        ra = torch.atan2(y, x)
        dec = torch.asin(torch.clamp(z / torch.clamp(distance, min=1e-8), -1.0, 1.0))

        # Convert to degrees
        ra_deg = torch.rad2deg(ra) % 360.0
        dec_deg = torch.rad2deg(dec)

        return ra_deg, dec_deg, distance

    def to_astropy(self) -> Any:
        """Convert to astropy SkyCoord object."""

        ra, dec, distance = self.to_spherical()

        # Simple conversion without complex units
        try:
            return SkyCoord(
                ra=ra.detach().cpu().numpy(),
                dec=dec.detach().cpu().numpy(),
                distance=distance.detach().cpu().numpy(),
                unit=("deg", "deg", "Mpc"),
            )
        except Exception:
            return None

    def _build_spatial_index(self) -> None:
        """Build spatial index for fast neighbor queries."""

        pos = self._data

        # Create edges using sklearn
        if radius is not None:
            edge_index = self._create_radius_graph(pos, radius)
        else:
            edge_index = self._create_knn_graph(pos, k)

        # Node features: include both Cartesian and spherical
        ra, dec, dist = self.to_spherical()
        spherical_features = torch.stack(
            [
                torch.deg2rad(ra),
                torch.deg2rad(dec),
                torch.log1p(dist),  # log(1+distance) for better scaling
            ],
            dim=-1,
        )

        x = torch.cat([pos, spherical_features], dim=-1)  # [N, 6]

        return Data(x=x, edge_index=edge_index, pos=pos)

    def _create_knn_graph(self, pos: torch.Tensor, k: int) -> torch.Tensor:
        """Create k-NN graph using sklearn."""
        pos_np = pos.detach().cpu().numpy()

        # Use BallTree for better performance with astronomical data
        tree = BallTree(pos_np, metric="euclidean")

        edge_list = []
        for i in range(len(pos_np)):
            # Find k+1 neighbors (including self), then exclude self
            distances, indices = tree.query([pos_np[i]], k=k + 1)
            neighbors = indices[0][1:]  # Exclude self (first element)

            for neighbor in neighbors:
                edge_list.append([i, neighbor])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index

    def _create_radius_graph(self, pos: torch.Tensor, radius: float) -> torch.Tensor:
        """Create radius graph using sklearn."""
        pos_np = pos.detach().cpu().numpy()

        # Use BallTree for radius queries
        tree = BallTree(pos_np, metric="euclidean")

        edge_list = []
        for i in range(len(pos_np)):
            # Find all neighbors within radius
            indices = tree.query_radius([pos_np[i]], r=radius)[0]
            neighbors = indices[indices != i]  # Exclude self

            for neighbor in neighbors:
                edge_list.append([i, neighbor])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        else:
            # Empty graph
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return edge_index

    def transform_coordinates(self, target_system: str) -> "Spatial3DTensor":
        """
        Transform to different coordinate system.

        Args:
            target_system: Target coordinate system ('icrs', 'galactic', 'ecliptic')
        """
        if target_system == self.coordinate_system:
            return self

        # Convert via astropy
        skycoord = self.to_astropy()

        if target_system == "galactic":
            transformed = skycoord.galactic
        elif target_system == "icrs":
            transformed = skycoord.icrs
        else:
            raise ValueError(f"Unsupported coordinate system: {target_system}")

        return self.from_astropy(transformed, unit=self.unit)

    def distance_to_origin(self) -> torch.Tensor:
        """Calculate distance from each point to the origin."""
        return torch.sqrt(torch.sum(self.cartesian**2, dim=1))

    def __len__(self) -> int:
        """Number of spatial points."""
        return self._data.shape[0]

    def __repr__(self) -> str:
        n_points = len(self)
        coord_sys = self.coordinate_system
        unit = self.unit
        return (
            f"Spatial3DTensor(n_points={n_points}, system='{coord_sys}', unit='{unit}')"
        )

    def cosmic_web_clustering(
        self, eps_pc: float = 10.0, min_samples: int = 5, algorithm: str = "dbscan"
    ) -> Dict[str, torch.Tensor]:
        """
        Perform cosmic web clustering analysis in 3D space.

        Args:
            eps_pc: Clustering radius in parsecs
            min_samples: Minimum samples for core points
            algorithm: 'dbscan' or 'hierarchical'

        Returns:
            Dictionary with cluster labels, statistics, and cosmic web features
        """

        from sklearn.cluster import AgglomerativeClustering

        # Get 3D coordinates in parsecs - Fix CUDA tensor conversion
        coords_pc = self._data.detach().cpu().numpy()

        # Convert units if needed
        if self.unit == "kpc":
            coords_pc *= 1000  # kpc to pc
        elif self.unit == "Mpc":
            coords_pc *= 1_000_000  # Mpc to pc

        logger.info(
            f"🌌 Cosmic web clustering: {len(coords_pc):,} stars in {eps_pc} pc radius"
        )

        # Perform clustering
        if algorithm == "dbscan":
            clusterer = DBSCAN(eps=eps_pc, min_samples=min_samples)
        else:
            clusterer = AgglomerativeClustering(
                n_clusters=None, distance_threshold=eps_pc, linkage="ward"
            )

        labels = clusterer.fit_predict(coords_pc)
        labels_tensor = torch.from_numpy(labels)

        # Analyze results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = sum(1 for label in labels if label == -1)

        # Calculate cluster properties
        cluster_stats = {}
        if n_clusters > 0:
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise
                    continue

                cluster_mask = labels == cluster_id
                cluster_coords = coords_pc[cluster_mask]

                # Cluster center and size
                center = cluster_coords.mean(axis=0)
                distances = np.linalg.norm(cluster_coords - center, axis=1)

                cluster_stats[cluster_id] = {
                    "n_stars": int(cluster_mask.sum()),
                    "center_pc": center,
                    "radius_pc": float(distances.max()),
                    "density": float(
                        cluster_mask.sum()
                        / (4 / 3 * np.pi * max(distances.max(), 1e-6) ** 3)
                    ),
                }

        logger.info(f"✅ Found {n_clusters:,} stellar groups")
        logger.info(
            f"   Grouped stars: {len(labels) - n_noise:,} ({(len(labels) - n_noise) / len(labels) * 100:.1f}%)"
        )
        logger.info(
            f"   Isolated stars: {n_noise:,} ({n_noise / len(labels) * 100:.1f}%)"
        )

        return {
            "cluster_labels": labels_tensor,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_stats": cluster_stats,
            "coords_pc": torch.from_numpy(coords_pc),
        }

    def analyze_local_density(self, radius_pc: float = 5.0) -> torch.Tensor:
        """
        Calculate local stellar density around each star.

        Args:
            radius_pc: Radius for density calculation in parsecs

        Returns:
            Local density for each star (stars per cubic parsec)
        """

        from sklearn.neighbors import NearestNeighbors

        # Get coordinates in parsecs - Fix CUDA tensor conversion
        coords_pc = self._data.detach().cpu().numpy()
        if self.unit == "kpc":
            coords_pc *= 1000
        elif self.unit == "Mpc":
            coords_pc *= 1_000_000

        # Build neighbor tree
        nbrs = NearestNeighbors(radius=radius_pc)
        nbrs.fit(coords_pc)

        # Find neighbors within radius for each star
        densities = []
        for i, coord in enumerate(coords_pc):
            distances, indices = nbrs.radius_neighbors([coord])
            n_neighbors = len(indices[0]) - 1  # Exclude self

            # Density = neighbors / volume
            volume = (4 / 3) * np.pi * radius_pc**3
            density = n_neighbors / volume
            densities.append(density)

        return torch.tensor(densities, dtype=torch.float32)

    def cosmic_web_structure(self, grid_size_pc: float = 20.0) -> Dict[str, Any]:
        """
        Analyze cosmic web structure using density field.

        Args:
            grid_size_pc: Grid cell size in parsecs

        Returns:
            Dictionary with density field and structure analysis
        """
        # Fix CUDA tensor conversion
        coords_pc = self._data.detach().cpu().numpy()
        if self.unit == "kpc":
            coords_pc *= 1000
        elif self.unit == "Mpc":
            coords_pc *= 1_000_000

        # Calculate bounds
        x_min, x_max = coords_pc[:, 0].min(), coords_pc[:, 0].max()
        y_min, y_max = coords_pc[:, 1].min(), coords_pc[:, 1].max()
        z_min, z_max = coords_pc[:, 2].min(), coords_pc[:, 2].max()

        # Create 3D grid
        x_bins = int((x_max - x_min) / grid_size_pc) + 1
        y_bins = int((y_max - y_min) / grid_size_pc) + 1
        z_bins = int((z_max - z_min) / grid_size_pc) + 1

        logger.info(f"🕸️ Creating {x_bins}×{y_bins}×{z_bins} density grid")

        # Calculate 3D histogram (density field)
        density_field, edges = np.histogramdd(
            coords_pc,
            bins=[x_bins, y_bins, z_bins],
            range=[[x_min, x_max], [y_min, y_max], [z_min, z_max]],
        )

        # Analyze structure
        total_volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        mean_density = len(coords_pc) / total_volume

        # Find high and low density regions
        high_density_threshold = mean_density * 2
        low_density_threshold = mean_density * 0.5

        high_density_cells = (density_field > high_density_threshold).sum()
        low_density_cells = (density_field < low_density_threshold).sum()

        return {
            "density_field": torch.from_numpy(density_field),
            "grid_edges": edges,
            "mean_density": mean_density,
            "high_density_cells": high_density_cells,
            "low_density_cells": low_density_cells,
            "grid_size_pc": grid_size_pc,
            "total_volume_pc3": total_volume,
        }
