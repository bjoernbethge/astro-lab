"""
This module defines the Spatial3DTensor for representing and manipulating
3D spatial coordinates within the Astro-Lab framework.
"""
from __future__ import annotations

import torch
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Self
from pydantic import Field

from .base import AstroTensorBase

class Spatial3DTensor(AstroTensorBase):
    """
    Represents 3D spatial coordinates (e.g., Cartesian x, y, z) for celestial
    objects, providing methods for coordinate transformations and calculations.
    """
    
    frame: str = Field(default="icrs", description="Coordinate frame")
    unit: str = Field(default="parsec", description="Distance unit")

    def __init__(
        self,
        data: Union[torch.Tensor, List, Any],
        frame: str = "icrs",
        unit: str = "parsec",
        **kwargs,
    ):
        """
        Initializes the Spatial3DTensor.

        Args:
            data: A tensor of 3D coordinates, shape [..., 3].
            frame: The coordinate frame (e.g., 'icrs', 'galactic').
            unit: The distance unit (e.g., 'parsec', 'au').
            **kwargs: Additional metadata.
        """
        super().__init__(data=data, frame=frame, unit=unit, **kwargs)
        self._validate()

    def _validate(self) -> None:
        """Validates the structure of the spatial data."""
        if self.data.dim() < 1:
            raise ValueError("Spatial data must be at least 1D.")
        if self.data.shape[-1] != 3:
            raise ValueError(
                f"Last dimension must be 3 for 3D spatial data, but got {self.data.shape[-1]}."
            )

    @property
    def x(self) -> torch.Tensor:
        return self.data[..., 0]

    @property
    def y(self) -> torch.Tensor:
        return self.data[..., 1]

    @property
    def z(self) -> torch.Tensor:
        return self.data[..., 2]

    def to_unit(self, new_unit: str) -> Self:
        """
        Converts the coordinates to a new distance unit.
        NOTE: This is a placeholder for a real conversion library.
        """
        # This should be implemented with a proper astro unit conversion library
        # For now, we'll just update the metadata.
        print(f"Placeholder: Converting units from {self.meta.get('unit')} to {new_unit}")
        new_meta = self.meta.copy()
        new_meta['unit'] = new_unit
        return self.__class__(data=self.data.clone(), **new_meta)

    def to_frame(self, new_frame: str) -> Self:
        """
        Transforms the coordinates to a new reference frame.
        NOTE: This is a placeholder for a real transformation library.
        """
        # This should be implemented with a library like astropy.coordinates
        print(f"Placeholder: Transforming frame from {self.meta.get('frame')} to {new_frame}")
        new_meta = self.meta.copy()
        new_meta['frame'] = new_frame
        # In a real implementation, the data would change.
        return self.__class__(data=self.data.clone(), **new_meta)

    @classmethod
    def from_cartesian(
        cls, 
        x: torch.Tensor, 
        y: torch.Tensor, 
        z: torch.Tensor, 
        **kwargs
    ) -> "Spatial3DTensor":
        """Creates a Spatial3DTensor from individual x, y, z tensors."""
        if not (x.shape == y.shape == z.shape):
            raise ValueError("x, y, and z tensors must have the same shape.")
        data = torch.stack([x, y, z], dim=1)
        return cls(data, **kwargs)

    @classmethod
    def from_spherical(
        cls,
        ra: torch.Tensor,
        dec: torch.Tensor,
        distance: torch.Tensor,
        angular_unit: str = "deg",
        **kwargs
    ) -> "Spatial3DTensor":
        """
        Creates a Spatial3DTensor from spherical coordinates (RA, Dec, distance).
        Uses astropy for robust conversion.
        """
        # Handle case where a single tensor is passed
        if isinstance(ra, torch.Tensor) and ra.ndim == 2 and ra.shape[1] == 3:
            # Single tensor with shape (N, 3) containing [ra, dec, distance]
            ra_coords = ra[:, 0]
            dec_coords = ra[:, 1]
            distance_coords = ra[:, 2]
        else:
            # Separate tensors
            ra_coords = ra
            dec_coords = dec
            distance_coords = distance

        if not (ra_coords.shape == dec_coords.shape == distance_coords.shape):
            raise ValueError("ra, dec, and distance tensors must have the same shape.")

        # Convert to numpy for astropy, ensuring they are detached from graph
        ra_np = ra_coords.detach().cpu().numpy()
        dec_np = dec_coords.detach().cpu().numpy()
        dist_np = distance_coords.detach().cpu().numpy()
        
        # Use astropy SkyCoord for conversion
        # We assume distance is given in parsecs, a common unit in Gaia data.
        coords = SkyCoord(
            ra=ra_np * u.Unit(angular_unit),
            dec=dec_np * u.Unit(angular_unit),
            distance=dist_np * u.parsec,
            frame='icrs'
        )
        
        cartesian = coords.cartesian
        x = torch.from_numpy(cartesian.x.value).to(ra_coords.device, dtype=ra_coords.dtype)
        y = torch.from_numpy(cartesian.y.value).to(ra_coords.device, dtype=ra_coords.dtype)
        z = torch.from_numpy(cartesian.z.value).to(ra_coords.device, dtype=ra_coords.dtype)
        
        return cls.from_cartesian(x, y, z, **kwargs)
        
    @property
    def cartesian(self) -> torch.Tensor:
        """Returns the cartesian coordinates."""
        return self.data

    @property
    def coordinate_system(self) -> str:
        """Returns the coordinate system."""
        return self.frame

    @property
    def epoch(self) -> float:
        """Returns the epoch."""
        return self.get_metadata("epoch", 2000.0)

    def distance_to_origin(self) -> torch.Tensor:
        """Calculate distance to origin for all points."""
        return torch.norm(self.data, dim=1)

    def query_neighbors(self, query_point: torch.Tensor, radius: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Query neighbors within a radius."""
        distances = torch.norm(self.data - query_point, dim=1)
        mask = distances <= radius
        neighbors = torch.where(mask)[0]
        neighbor_distances = distances[mask]
        return neighbors, neighbor_distances

    def to_spherical(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert to spherical coordinates (RA, Dec, distance)."""
        # Convert to numpy for astropy
        x_np = self.x.detach().cpu().numpy()
        y_np = self.y.detach().cpu().numpy()
        z_np = self.z.detach().cpu().numpy()
        
        # Use astropy for conversion
        coords = SkyCoord(
            x=x_np * u.kiloparsec,
            y=y_np * u.kiloparsec,
            z=z_np * u.kiloparsec,
            frame='icrs',
            representation_type='cartesian'
        )
        
        # Extract spherical coordinates using the correct representation
        spherical = coords.spherical
        ra = torch.from_numpy(spherical.lon.deg).to(self.device, dtype=self.dtype)
        dec = torch.from_numpy(spherical.lat.deg).to(self.device, dtype=self.dtype)
        distance = torch.from_numpy(spherical.distance.kiloparsec).to(self.device, dtype=self.dtype)
        
        return ra, dec, distance

    def angular_separation(self, other: "Spatial3DTensor") -> torch.Tensor:
        """Calculate angular separation between this and another tensor."""
        # Convert both to spherical coordinates
        ra1, dec1, _ = self.to_spherical()
        ra2, dec2, _ = other.to_spherical()
        
        # Calculate angular separation using spherical trigonometry
        cos_sep = torch.sin(dec1 * np.pi/180) * torch.sin(dec2 * np.pi/180) + \
                  torch.cos(dec1 * np.pi/180) * torch.cos(dec2 * np.pi/180) * \
                  torch.cos((ra1 - ra2) * np.pi/180)
        
        # Clamp to avoid numerical issues
        cos_sep = torch.clamp(cos_sep, -1.0, 1.0)
        separation = torch.acos(cos_sep) * 180 / np.pi
        
        return separation

    def to_torch_geometric(self, k: Optional[int] = None, add_self_loops: bool = True) -> "Data":
        """
        Convert to a torch_geometric.data.Data object, optionally building a k-NN graph.

        Args:
            k (int, optional): If provided, constructs a k-NN graph.
            add_self_loops (bool): If True, adds self-loops to the graph.

        Returns:
            torch_geometric.data.Data: A PyG Data object.
        """
        try:
            from torch_geometric.data import Data
            from torch_geometric.nn import knn_graph
        except ImportError:
            raise ImportError("torch_geometric is required. Please install it with 'uv pip install torch-geometric'.")

        edge_index = None
        if k is not None:
            edge_index = knn_graph(self.data, k=k, loop=add_self_loops)

        # Use positions as features if no other features are available
        return Data(
            pos=self.data,
            x=self.data,  # Using positions as node features
            edge_index=edge_index,
            **self.meta
        )

    def cone_search(self, center: torch.Tensor, radius_deg: float) -> torch.Tensor:
        """Perform cone search around a center point."""
        # Convert center to spherical coordinates
        center_tensor = Spatial3DTensor(center.unsqueeze(0))
        center_ra, center_dec, _ = center_tensor.to_spherical()
        
        # Convert all points to spherical
        ra, dec, _ = self.to_spherical()
        
        # Calculate angular separation from center
        cos_sep = torch.sin(center_dec * np.pi/180) * torch.sin(dec * np.pi/180) + \
                  torch.cos(center_dec * np.pi/180) * torch.cos(dec * np.pi/180) * \
                  torch.cos((center_ra - ra) * np.pi/180)
        
        cos_sep = torch.clamp(cos_sep, -1.0, 1.0)
        separation = torch.acos(cos_sep) * 180 / np.pi
        
        # Return indices of points within radius
        return torch.where(separation <= radius_deg)[0]

    def __repr__(self) -> str:
        return f"Spatial3DTensor(points={self.shape[0]}, device='{self.device}')"
