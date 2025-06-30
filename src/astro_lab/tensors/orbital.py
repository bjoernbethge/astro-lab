"""
Orbital TensorDict for AstroLab
===============================

TensorDict for orbital elements and orbital mechanics calculations.
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from .base import AstroTensorDict


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

        a = self.semi_major_axis
        e = self.eccentricity
        i = self.inclination * math.pi / 180
        Omega = self.longitude_of_ascending_node * math.pi / 180
        omega = self.argument_of_periapsis * math.pi / 180
        M = self.mean_anomaly * math.pi / 180

        # Für eine realistische Transformation (vereinfachtes Beispiel):
        # Berechne Position im Perifokal-System
        r = a * (1 - e * torch.cos(M))
        x_pf = r * torch.cos(M)
        y_pf = r * torch.sin(M)
        torch.zeros_like(x_pf)

        # Rotationsmatrizen für die Transformation ins Inertialsystem
        cos_Omega = torch.cos(Omega)
        sin_Omega = torch.sin(Omega)
        cos_i = torch.cos(i)
        sin_i = torch.sin(i)
        cos_omega = torch.cos(omega)
        sin_omega = torch.sin(omega)

        # Rotation: perifokal -> inertial (vereinfachte Form)
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_pf + (
            -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        ) * y_pf
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_pf + (
            -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        ) * y_pf
        z = (sin_omega * sin_i) * x_pf + (cos_omega * sin_i) * y_pf

        # Geschwindigkeiten (echte Kepler-Geschwindigkeiten)
        # Berechne Geschwindigkeiten aus den Orbitalparametern
        mu = 1.327e20  # Gravitationsparameter der Sonne in km³/s²
        r_mag = torch.sqrt(x**2 + y**2 + z**2)

        # Orbitalgeschwindigkeit: v = sqrt(mu * (2/r - 1/a))
        v_mag = torch.sqrt(mu * (2.0 / r_mag - 1.0 / a))

        # Geschwindigkeitsrichtung (vereinfacht)
        vx = -v_mag * torch.sin(M)
        vy = v_mag * torch.cos(M)
        vz = torch.zeros_like(vx)

        return torch.stack([x, y, z, vx, vy, vz], dim=-1)

    def propagate(self, delta_time: torch.Tensor) -> "OrbitTensorDict":
        """
        Propagates the orbits over a given time.

        Args:
            delta_time: Time difference

        Returns:
            New OrbitTensorDict with propagated elements
        """
        # Keplerian Propagation
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


def from_kepler_elements(semi_major_axes, eccentricities, inclinations, **kwargs):
    """
    Creates Kepler orbits from basic parameters.

    Args:
        semi_major_axes: [N] Semi-major axes in AU
        eccentricities: [N] Eccentricities
        inclinations: [N] Inclinations in degrees

    Returns:
        OrbitTensorDict with Kepler elements
    """
    n_objects = semi_major_axes.shape[0]

    # Create complete orbital elements
    elements = torch.stack(
        [
            semi_major_axes,
            eccentricities,
            inclinations,
            torch.zeros(n_objects),  # Omega (RAAN)
            torch.zeros(n_objects),  # omega (Argument of Periapsis)
            torch.zeros(n_objects),  # M (mean anomaly - start at 0)
        ],
        dim=-1,
    )

    return OrbitTensorDict(elements, **kwargs)


def create_asteroid_population(n_asteroids=1000, belt_type="main", **kwargs):
    """
    Creates synthetic asteroid population.

    Args:
        n_asteroids: Number of asteroids
        belt_type: Type of belt ("main", "trojan", "neo")

    Returns:
        OrbitTensorDict with asteroid orbits
    """
    if belt_type == "main":
        # Main belt: 2.1 - 3.3 AU
        a = torch.rand(n_asteroids) * (3.3 - 2.1) + 2.1
        e = torch.exp(torch.rand(n_asteroids)) * 0.1
        e = torch.clamp(e, 0, 0.9)
        i = torch.abs(torch.randn(n_asteroids) * 5)
    elif belt_type == "trojan":
        # Jupiter Trojans at ~5.2 AU
        a = torch.randn(n_asteroids) * 0.1 + 5.2
        e = torch.exp(torch.rand(n_asteroids)) * 0.05
        e = torch.clamp(e, 0, 0.3)
        i = torch.abs(torch.randn(n_asteroids) * 10)
    elif belt_type == "neo":
        # Near-Earth Objects: a < 1.3 AU
        a = torch.rand(n_asteroids) * (1.3 - 0.8) + 0.8
        e = torch.rand(n_asteroids) * 0.8
        i = torch.abs(torch.randn(n_asteroids) * 15)
    else:
        raise ValueError(f"Unknown belt type: {belt_type}")

    return from_kepler_elements(a, e, i, central_body="Sun", **kwargs)
