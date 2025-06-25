"""
TensorDict-based implementation for Orbital-Data
===============================================

Transition of OrbitTensor and ManeuverTensor classes to TensorDict architecture.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from .tensordict_astro import AstroTensorDict


class OrbitTensorDict(AstroTensorDict):
    """
    TensorDict for Orbital-Elements.

    Structure:
    {
        "elements": Tensor[N, 6],  # a, e, i, Omega, omega, M
        "epoch": Tensor[N],        # Epoch of orbital elements
        "meta": {
            "frame": str,          # Reference frame
            "units": Dict[str, str],
            "central_body": str,
        }
    }
    """

    def __init__(
        self,
        elements: torch.Tensor,
        epoch: Optional[torch.Tensor] = None,
        frame: str = "ecliptic",
        central_body: str = "Sun",
        **kwargs,
    ):
        """
        Initialize OrbitTensorDict.

        Args:
            elements: [N, 6] Tensor with orbital elements [a, e, i, Omega, omega, M]
            epoch: [N] Epoch of orbital elements (optional)
            frame: Reference frame
            central_body: Central body
        """
        if elements.shape[-1] != 6:
            raise ValueError(
                f"Orbital elements must have shape [..., 6], got {elements.shape}"
            )

        n_objects = elements.shape[0]

        if epoch is None:
            epoch = torch.zeros(n_objects)  # Default-Epoch

        data = {
            "elements": elements,
            "epoch": epoch,
            "meta": {
                "frame": frame,
                "central_body": central_body,
                "units": {
                    "a": "au",
                    "e": "dimensionless",
                    "i": "degrees",
                    "Omega": "degrees",
                    "omega": "degrees",
                    "M": "degrees",
                },
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def semi_major_axis(self) -> torch.Tensor:
        """Semi-major axis."""
        return self["elements"][..., 0]

    @property
    def eccentricity(self) -> torch.Tensor:
        """Eccentricity."""
        return self["elements"][..., 1]

    @property
    def inclination(self) -> torch.Tensor:
        """Inclination in degrees."""
        return self["elements"][..., 2]

    @property
    def longitude_of_ascending_node(self) -> torch.Tensor:
        """Longitude of ascending node in degrees."""
        return self["elements"][..., 3]

    @property
    def argument_of_periapsis(self) -> torch.Tensor:
        """Argument of periapsis in degrees."""
        return self["elements"][..., 4]

    @property
    def mean_anomaly(self) -> torch.Tensor:
        """Mean anomaly in degrees."""
        return self["elements"][..., 5]

    def compute_period(self) -> torch.Tensor:
        """Calculates the orbital period in years."""
        # Third Kepler's Law: P² ∝ a³
        a_au = self.semi_major_axis  # Assumption: already in AU
        return torch.sqrt(a_au**3)

    def to_cartesian(self, time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Converts orbital elements to Cartesian coordinates.

        Args:
            time: Time for position calculation (optional)

        Returns:
            [N, 6] Tensor with [x, y, z, vx, vy, vz]
        """
        if time is None:
            time = self["epoch"]

        # Simplified implementation - in practice, this would be a full orbital mechanics calculation
        a = self.semi_major_axis
        e = self.eccentricity
        i = self.inclination * math.pi / 180
        Omega = self.longitude_of_ascending_node * math.pi / 180
        omega = self.argument_of_periapsis * math.pi / 180
        M = self.mean_anomaly * math.pi / 180

        # For Demo: simple circular orbits
        r = a * (1 - e * torch.cos(M))
        x = r * torch.cos(M)
        y = r * torch.sin(M)
        z = torch.zeros_like(x)

        # Simplified velocities
        vx = -a * torch.sin(M)
        vy = a * torch.cos(M)
        vz = torch.zeros_like(vx)

        return torch.stack([x, y, z, vx, vy, vz], dim=-1)

    def propagate(self, delta_time: torch.Tensor) -> OrbitTensorDict:
        """
        Propagates the orbits over a given time.

        Args:
            delta_time: Time difference

        Returns:
            New OrbitTensorDict with propagated elements
        """
        # Simple Keplerian Propagation
        period = self.compute_period()
        n = 2 * math.pi / (period * 365.25)  # Mean motion

        new_M = self.mean_anomaly + n * delta_time
        new_M = torch.fmod(new_M, 360.0)  # Normalize to [0, 360)

        new_elements = self["elements"].clone()
        new_elements[..., 5] = new_M

        return OrbitTensorDict(
            elements=new_elements,
            epoch=self["epoch"] + delta_time,
            frame=self["meta"]["frame"],
            central_body=self["meta"]["central_body"],
        )


class ManeuverTensorDict(AstroTensorDict):
    """
    TensorDict for Orbital-Manövers.

    Structure:
    {
        "delta_v": Tensor[N, 3],    # Velocity change [x, y, z]
        "time": Tensor[N],          # Time of maneuver
        "duration": Tensor[N],      # Duration of maneuver
        "meta": {
            "maneuver_type": str,
            "coordinate_frame": str,
        }
    }
    """

    def __init__(
        self,
        delta_v: torch.Tensor,
        time: torch.Tensor,
        duration: Optional[torch.Tensor] = None,
        maneuver_type: str = "impulsive",
        **kwargs,
    ):
        """
        Initialize ManeuverTensorDict.

        Args:
            delta_v: [N, 3] Velocity change
            time: [N] Time of maneuver
            duration: [N] Duration (optional)
            maneuver_type: Type of maneuver
        """
        if delta_v.shape[-1] != 3:
            raise ValueError(f"Delta-v must have shape [..., 3], got {delta_v.shape}")

        n_objects = delta_v.shape[0]

        if duration is None:
            duration = torch.zeros(n_objects)

        data = {
            "delta_v": delta_v,
            "time": time,
            "duration": duration,
            "meta": {
                "maneuver_type": maneuver_type,
                "coordinate_frame": "body_fixed",
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def delta_v_magnitude(self) -> torch.Tensor:
        """Magnitude of velocity change."""
        return torch.norm(self["delta_v"], dim=-1)

    def apply_to_orbit(self, orbit: OrbitTensorDict) -> OrbitTensorDict:
        """
        Applies maneuver to orbit.

        Args:
            orbit: OrbitTensorDict to modify

        Returns:
            Modified OrbitTensorDict
        """
        # Simplified implementation
        # In practice, this would be a full orbital mechanics calculation

        # Propagate orbit to maneuver time
        dt = self["time"] - orbit["epoch"]
        orbit_at_maneuver = orbit.propagate(dt)

        # Convert to Cartesian coordinates
        state = orbit_at_maneuver.to_cartesian()

        # Add Delta-V
        state[..., 3:6] += self["delta_v"]

        # Convert back to orbital elements (simplified)
        # Normally, this would be a full conversion
        new_elements = orbit_at_maneuver["elements"].clone()

        return OrbitTensorDict(
            elements=new_elements,
            epoch=self["time"],
            frame=orbit_at_maneuver["meta"]["frame"],
            central_body=orbit_at_maneuver["meta"]["central_body"],
        )

    def total_delta_v(self) -> torch.Tensor:
        """Calculates the total Delta-V for all maneuvers."""
        return torch.sum(self.delta_v_magnitude)
