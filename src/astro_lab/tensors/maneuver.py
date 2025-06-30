"""
Maneuver TensorDict for AstroLab
================================

TensorDict for orbital maneuvers and delta-v calculations.
"""

import math
from typing import Optional

import torch

from .base import AstroTensorDict
from .orbital import OrbitTensorDict


class ManeuverTensorDict(AstroTensorDict):
    """
    TensorDict for Orbital-Manövers.

    Structure:
    {
        "orbit": OrbitTensorDict,     # Base orbital elements
        "delta_v": Tensor[N, 3],      # Velocity change [x, y, z]
        "time": Tensor[N],            # Time of maneuver
        "duration": Tensor[N],        # Duration of maneuver
        "meta": {
            "maneuver_type": str,
            "coordinate_frame": str,
            "frame": str,             # Inherited from OrbitTensorDict
            "central_body": str,      # Inherited from OrbitTensorDict
        }
    }
    """

    def __init__(
        self,
        orbit: OrbitTensorDict,
        delta_v: torch.Tensor,
        time: torch.Tensor,
        duration: Optional[torch.Tensor] = None,
        maneuver_type: str = "impulsive",
        **kwargs,
    ):
        """
        Initialize ManeuverTensorDict.

        Args:
            orbit: Base orbital elements
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
            "orbit": orbit,
            "delta_v": delta_v,
            "time": time,
            "duration": duration,
            "meta": {
                "maneuver_type": maneuver_type,
                "coordinate_frame": "body_fixed",
                "frame": orbit["meta"]["frame"],
                "central_body": orbit["meta"]["central_body"],
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    @property
    def orbit(self) -> OrbitTensorDict:
        """Base orbital elements."""
        return self["orbit"]

    @property
    def delta_v_magnitude(self) -> torch.Tensor:
        """Magnitude of velocity change."""
        return torch.norm(self["delta_v"], dim=-1)

    @property
    def semi_major_axis(self) -> torch.Tensor:
        """Semi-major axis from orbit."""
        return self.orbit.semi_major_axis

    @property
    def eccentricity(self) -> torch.Tensor:
        """Eccentricity from orbit."""
        return self.orbit.eccentricity

    @property
    def inclination(self) -> torch.Tensor:
        """Inclination from orbit."""
        return self.orbit.inclination

    def apply_to_orbit(self) -> OrbitTensorDict:
        """
        Applies maneuver to orbit.

        Returns:
            Modified OrbitTensorDict
        """
        # Simplified implementation
        # In practice, this would be a full orbital mechanics calculation

        # Propagate orbit to maneuver time
        dt = self["time"] - self.orbit["epoch"]
        orbit_at_maneuver = self.orbit.propagate(dt)

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

    def compute_maneuver_cost(self) -> torch.Tensor:
        """Compute the cost of the maneuver (simplified)."""
        # cost model: proportional to delta-v magnitude
        return self.delta_v_magnitude * 1000  # Cost per km/s

    def get_maneuver_summary(self) -> dict:
        """Get a summary of the maneuver."""
        return {
            "maneuver_type": self["meta"]["maneuver_type"],
            "total_delta_v": self.total_delta_v().item(),
            "mean_delta_v": torch.mean(self.delta_v_magnitude).item(),
            "max_delta_v": torch.max(self.delta_v_magnitude).item(),
            "maneuver_time": self["time"].tolist(),
            "duration": self["duration"].tolist(),
            "cost": self.compute_maneuver_cost().tolist(),
        }


def create_hohmann_transfer(
    orbit1: OrbitTensorDict, orbit2: OrbitTensorDict
) -> ManeuverTensorDict:
    """
    Calculates Hohmann transfer between two orbits.

    Args:
        orbit1: Start orbit
        orbit2: Target orbit

    Returns:
        ManeuverTensorDict with transfer maneuvers
    """
    # Simplified Hohmann transfer calculation
    r1 = orbit1.semi_major_axis
    r2 = orbit2.semi_major_axis

    # Delta-V calculations
    mu = 1.327e11  # Solar gravitational parameter (km³/s²)

    v1 = torch.sqrt(mu / (r1 * 1.496e8))  # Convert AU to km
    v_transfer_1 = torch.sqrt(mu * (2 / (r1 * 1.496e8) - 2 / ((r1 + r2) * 1.496e8 / 2)))
    delta_v1 = v_transfer_1 - v1

    v2 = torch.sqrt(mu / (r2 * 1.496e8))
    v_transfer_2 = torch.sqrt(mu * (2 / (r2 * 1.496e8) - 2 / ((r1 + r2) * 1.496e8 / 2)))
    delta_v2 = v2 - v_transfer_2

    # Create maneuver sequence
    n_orbits = r1.shape[0]
    delta_v_tensor = torch.zeros(n_orbits * 2, 3)
    delta_v_tensor[::2, 0] = delta_v1  # First maneuver in x-direction
    delta_v_tensor[1::2, 0] = delta_v2  # Second maneuver in x-direction

    # Calculate transfer time
    transfer_time = (
        math.pi * torch.sqrt(((r1 + r2) * 1.496e8 / 2) ** 3 / mu) / 86400
    )  # Days
    times = torch.zeros(n_orbits * 2)
    times[1::2] = transfer_time.repeat(n_orbits)

    # Create combined orbit for the maneuver
    # This is a simplified approach - in practice, you'd need to handle the orbit properly
    # and propagate elements with full dynamics.
    combined_elements = torch.cat([orbit1["elements"], orbit2["elements"]], dim=0)

    return ManeuverTensorDict(
        orbit=OrbitTensorDict(
            elements=combined_elements,
            frame=orbit1["meta"]["frame"],
            central_body=orbit1["meta"]["central_body"],
        ),
        delta_v=delta_v_tensor,
        time=times,
        maneuver_type="hohmann_transfer",
    )
