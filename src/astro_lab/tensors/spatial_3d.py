"""
3D Spatial Coordinate Tensors for Astronomical Data

Proper astronomical spatial tensor with astroML and poliastro compatibility.
Uses unified [N, 3] tensor structure for efficient spatial operations.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import AstroTensorBase

try:
    import astropy.units as u
    from astropy.coordinates import ICRS, Galactic, SkyCoord
    from astropy.time import Time

    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False

try:
    from sklearn.neighbors import BallTree, KDTree

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch_geometric
    from torch_geometric.data import Data

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False


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

        # Build spatial index if requested
        if spatial_index and SKLEARN_AVAILABLE:
            self._build_spatial_index()

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
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy required for from_astropy")

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
        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy required for to_astropy")

        ra, dec, distance = self.to_spherical()

        # Simple conversion without complex units
        try:
            return SkyCoord(
                ra=ra.numpy(),
                dec=dec.numpy(),
                distance=distance.numpy(),
                unit=("deg", "deg", "Mpc"),
            )
        except Exception:
            return None

    def _build_spatial_index(self) -> None:
        """Build spatial index for fast neighbor queries."""
        if not SKLEARN_AVAILABLE:
            return

        try:
            coords = self._data.detach().cpu().numpy()
            # Use BallTree for spherical-like coordinates, KDTree for Cartesian
            if self.coordinate_system in ["icrs", "galactic"]:
                index = BallTree(coords, metric="euclidean")
            else:
                index = KDTree(coords)

            self.update_metadata(spatial_index=index)

        except Exception as e:
            print(f"⚠️  Failed to build spatial index: {e}")

    def query_neighbors(
        self,
        query_point: Union[torch.Tensor, np.ndarray],
        radius: float,
        max_neighbors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fast neighbor query using spatial index.

        Args:
            query_point: Query coordinates [3] or [1, 3]
            radius: Search radius in same units as tensor
            max_neighbors: Maximum number of neighbors

        Returns:
            Tuple of (distances, indices)
        """
        spatial_index = self._metadata.get("spatial_index")

        if spatial_index is None:
            # Fallback to brute force
            return self._brute_force_query(query_point, radius, max_neighbors)

        query = np.atleast_2d(query_point)

        # Query spatial index
        if hasattr(spatial_index, "query_radius"):
            # BallTree
            indices = spatial_index.query_radius(query, r=radius)[0]
            if max_neighbors and len(indices) > max_neighbors:
                distances = spatial_index.query(
                    query, k=min(len(indices), max_neighbors)
                )[0][0]
                indices = spatial_index.query(
                    query, k=min(len(indices), max_neighbors)
                )[1][0]
            else:
                distances = np.linalg.norm(self._data.numpy()[indices] - query, axis=1)
        else:
            # KDTree
            indices = spatial_index.query_radius(query, r=radius)[0]
            distances = np.linalg.norm(self._data.numpy()[indices] - query, axis=1)

            if max_neighbors and len(indices) > max_neighbors:
                sort_idx = np.argsort(distances)[:max_neighbors]
                indices = indices[sort_idx]
                distances = distances[sort_idx]

        return torch.tensor(distances), torch.tensor(indices, dtype=torch.long)

    def _brute_force_query(
        self,
        query_point: Union[torch.Tensor, np.ndarray],
        radius: float,
        max_neighbors: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback brute force neighbor query."""
        query = torch.as_tensor(query_point, dtype=torch.float32)
        if query.dim() == 1:
            query = query.unsqueeze(0)

        # Calculate distances
        distances = torch.norm(self._data - query, dim=-1)

        # Filter by radius
        mask = distances <= radius
        valid_indices = torch.where(mask)[0]
        valid_distances = distances[mask]

        # Sort and limit
        sort_idx = torch.argsort(valid_distances)
        if max_neighbors and len(sort_idx) > max_neighbors:
            sort_idx = sort_idx[:max_neighbors]

        return valid_distances[sort_idx], valid_indices[sort_idx]

    def angular_separation(self, other: "Spatial3DTensor") -> torch.Tensor:
        """
        Calculate angular separation using dot product.
        More efficient than haversine for 3D coordinates.
        """
        # Normalize to unit vectors
        pos1 = self._data / torch.norm(self._data, dim=-1, keepdim=True)
        pos2 = other._data / torch.norm(other._data, dim=-1, keepdim=True)

        # Dot product gives cos(angle)
        cos_angle = torch.sum(pos1 * pos2, dim=-1)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

        # Convert to degrees
        angle_rad = torch.acos(cos_angle)
        return torch.rad2deg(angle_rad)

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """
        Cone search around center position.

        Args:
            center: Center position [3] (Cartesian)
            radius_deg: Search radius in degrees

        Returns:
            Boolean mask of objects within cone
        """
        center_tensor = torch.as_tensor(center, dtype=torch.float32)
        if center_tensor.dim() == 1:
            center_tensor = center_tensor.unsqueeze(0)

        # Create temporary tensor for center
        center_spatial = Spatial3DTensor(
            center_tensor, coordinate_system=self.coordinate_system, unit=self.unit
        )

        separations = self.angular_separation(center_spatial)
        return separations <= radius_deg

    def cross_match(
        self,
        other: "Spatial3DTensor",
        radius_deg: float = 1.0 / 3600.0,  # 1 arcsec default
    ) -> Dict[str, torch.Tensor]:
        """
        Cross-match with another catalog.

        Args:
            other: Other spatial tensor to match against
            radius_deg: Matching radius in degrees

        Returns:
            Dictionary with match results
        """
        matches = []
        separations = []

        for i in range(len(self._data)):
            pos = self._data[i]
            distances, indices = other.query_neighbors(
                pos, radius_deg * np.pi / 180.0
            )  # Convert to radians for 3D

            if len(indices) > 0:
                # Take closest match
                best_idx = 0
                matches.append((i, int(indices[best_idx])))
                separations.append(float(distances[best_idx]))

        if matches:
            matches_array = torch.tensor(matches)
            return {
                "self_indices": matches_array[:, 0],
                "other_indices": matches_array[:, 1],
                "separations": torch.tensor(separations),
            }
        else:
            return {
                "self_indices": torch.tensor([], dtype=torch.long),
                "other_indices": torch.tensor([], dtype=torch.long),
                "separations": torch.tensor([]),
            }

    def to_torch_geometric(self, k: int = 8, radius: Optional[float] = None) -> "Data":
        """Convert to PyTorch Geometric Data object for GNN processing."""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch-geometric required")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for graph construction")

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
        tree = BallTree(pos_np, metric='euclidean')
        
        edge_list = []
        for i in range(len(pos_np)):
            # Find k+1 neighbors (including self), then exclude self
            distances, indices = tree.query([pos_np[i]], k=k+1)
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
        tree = BallTree(pos_np, metric='euclidean')
        
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

        if not ASTROPY_AVAILABLE:
            raise ImportError("astropy required for coordinate transformations")

        # Convert via astropy
        skycoord = self.to_astropy()

        if target_system == "galactic":
            transformed = skycoord.galactic
        elif target_system == "icrs":
            transformed = skycoord.icrs
        else:
            raise ValueError(f"Unsupported coordinate system: {target_system}")

        return self.from_astropy(transformed, unit=self.unit)

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
