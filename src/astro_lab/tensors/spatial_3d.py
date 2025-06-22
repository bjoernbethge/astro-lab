"""
This module defines the Spatial3DTensor for representing and manipulating
3D spatial coordinates within the Astro-Lab framework.
"""
from __future__ import annotations

import torch
from astropy.coordinates import SkyCoord
import astropy.units as u

from .base import AstroTensorBase

class Spatial3DTensor(AstroTensorBase):
    """
    A specialized tensor for handling 3D spatial coordinates (x, y, z).
    This is a simplified, robust version for core functionality.
    """

    def __init__(self, data: torch.Tensor, **kwargs):
        """
        Initializes the Spatial3DTensor.
        Expects a tensor of shape (N, 3) for N points.
        """
        if not isinstance(data, torch.Tensor):
            raise TypeError("Spatial3DTensor data must be a torch.Tensor.")
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"Expected a tensor of shape (N, 3), but got {data.shape}")
        
        super().__init__(data, **kwargs)

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
        if not (ra.shape == dec.shape == distance.shape):
            raise ValueError("ra, dec, and distance tensors must have the same shape.")

        # Convert to numpy for astropy, ensuring they are detached from graph
        ra_np = ra.detach().cpu().numpy()
        dec_np = dec.detach().cpu().numpy()
        dist_np = distance.detach().cpu().numpy()
        
        # Use astropy SkyCoord for conversion
        # We assume distance is given in parsecs, a common unit in Gaia data.
        coords = SkyCoord(
            ra=ra_np * u.Unit(angular_unit),
            dec=dec_np * u.Unit(angular_unit),
            distance=dist_np * u.pc,
            frame='icrs'
        )
        
        cartesian = coords.cartesian
        x = torch.from_numpy(cartesian.x.value).to(ra.device, dtype=ra.dtype)
        y = torch.from_numpy(cartesian.y.value).to(ra.device, dtype=ra.dtype)
        z = torch.from_numpy(cartesian.z.value).to(ra.device, dtype=ra.dtype)
        
        return cls.from_cartesian(x, y, z, **kwargs)
        
    @property
    def x(self) -> torch.Tensor:
        """Returns the x-coordinates."""
        return self.data[:, 0]

    @property
    def y(self) -> torch.Tensor:
        """Returns the y-coordinates."""
        return self.data[:, 1]

    @property
    def z(self) -> torch.Tensor:
        """Returns the z-coordinates."""
        return self.data[:, 2]

    def __repr__(self) -> str:
        return f"Spatial3DTensor(points={self.shape[0]}, device='{self.device}')"
