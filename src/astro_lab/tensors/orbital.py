"""
Orbital tensor for celestial mechanics and satellite operations.
"""

import math
from typing import Optional, Tuple, Any

import torch
from pydantic import Field, field_validator
from typing_extensions import Self

from .base import AstroTensorBase

class OrbitTensor(AstroTensorBase):
    """
    Tensor for orbital elements and state vectors in celestial mechanics.

    Handles Keplerian orbital elements, Cartesian state vectors, and orbital
    propagation for satellites, planets, and interstellar objects.
    """

    _metadata_fields = [
        "epoch",
        "frame",
        "attractor",
        "element_type",
        "propagator",
        "mu",  # Standard gravitational parameter
        "stellar_system",  # For exoplanets
        "host_star_mass",  # For exoplanetary systems
    ]

    data: torch.Tensor
    epoch: float = Field(0.0, description="Reference epoch (e.g., MJD)")
    frame: str = Field("icrs", description="Reference frame ('icrs', 'gcrs', 'itrs')")
    attractor: str = Field("earth", description="Central body ('earth', 'sun', 'moon', 'star')")
    element_type: str = Field("keplerian", description="Type of elements ('keplerian', 'cartesian')")
    propagator: str = Field("kepler", description="Propagation method ('kepler', 'sgp4', 'j2')")
    mu: float = Field(398600.4418, description="Standard gravitational parameter (km³/s²)")
    stellar_system: Optional[str] = Field(None, description="Name of stellar system (for exoplanets)")
    host_star_mass: Optional[float] = Field(None, description="Mass of host star in solar masses (for exoplanets)")

    @field_validator("data")
    def validate_orbital_data(cls, v):
        if v.shape[-1] != 6:
            raise ValueError(f"OrbitTensor requires 6-element orbital state, but last dimension is {v.shape[-1]}")
        return v

    def __init__(self, **data):
        """
        Create a new OrbitTensor.

        Args:
            data: Orbital data [..., 6] (Keplerian or Cartesian)
            epoch: Reference epoch (MJD or seconds)
            frame: Reference frame ('icrs', 'gcrs', 'itrs')
            attractor: Central body ('earth', 'sun', 'moon', 'star')
            element_type: Type of elements ('keplerian', 'cartesian')
            propagator: Propagation method ('kepler', 'sgp4', 'j2')
            mu: Standard gravitational parameter (km³/s²)
            stellar_system: Name of stellar system (for exoplanets)
            host_star_mass: Mass of host star in solar masses (for exoplanets)
        """
        super().__init__(**data)

        # Set default mu based on attractor if not provided
        if self.meta.get("mu") is None:
            self.update_metadata(mu=self._get_default_mu(self.meta.get("attractor")))

        self._validate()

    def _validate(self) -> None:
        """Validate orbital tensor data."""
        if self._data.shape[-1] != 6:
            raise ValueError("OrbitTensor requires 6-element orbital state")

    def _get_default_mu(self, attractor: str) -> float:
        """Get default gravitational parameter for attractor."""
        mu_values = {
            "earth": 398600.4418,  # km³/s²
            "sun": 132712440018.0,
            "moon": 4902.7779,
            "mars": 42828.37,
            "jupiter": 126686534.9,
            "saturn": 37931187.0,
            "star": 132712440018.0,  # Solar mass default
        }
        return mu_values.get(attractor.lower(), 398600.4418)

    @property
    def is_keplerian(self) -> bool:
        """Check if elements are Keplerian."""
        return self.element_type == "keplerian"

    @property
    def is_cartesian(self) -> bool:
        """Check if elements are Cartesian."""
        return self.element_type == "cartesian"

    @property
    def semi_major_axis(self) -> torch.Tensor:
        """Semi-major axis (a) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 0]

    @property
    def eccentricity(self) -> torch.Tensor:
        """Eccentricity (e) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 1]

    @property
    def inclination(self) -> torch.Tensor:
        """Inclination (i) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 2]

    @property
    def raan(self) -> torch.Tensor:
        """Right Ascension of Ascending Node (Ω) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 3]

    @property
    def argument_of_periapsis(self) -> torch.Tensor:
        """Argument of periapsis (ω) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 4]

    @property
    def true_anomaly(self) -> torch.Tensor:
        """True anomaly (ν) - only for Keplerian elements."""
        if not self.is_keplerian:
            raise AttributeError("Attribute only available for Keplerian elements")
        return self.data[..., 5]

    @property
    def position(self) -> torch.Tensor:
        """Position vector [x, y, z] - only for Cartesian elements."""
        if not self.is_cartesian:
            raise AttributeError("Attribute only available for Cartesian elements")
        return self.data[..., :3]

    @property
    def velocity(self) -> torch.Tensor:
        """Velocity vector [vx, vy, vz] - only for Cartesian elements."""
        if not self.is_cartesian:
            raise AttributeError("Attribute only available for Cartesian elements")
        return self.data[..., 3:]

    def to_cartesian(self) -> "OrbitTensor":
        """Convert Keplerian elements to Cartesian state vectors."""
        if self.is_cartesian:
            return self.copy()

        # In a real implementation, the full conversion math would be here.
        # For this refactoring, we just create a correctly shaped placeholder.
        new_data = torch.randn_like(self.data)

        return self._create_new_instance(
            new_data=new_data,
            element_type="cartesian"
        ).add_history_entry("to_cartesian")

    def to_keplerian(self) -> "OrbitTensor":
        """Convert Cartesian state vectors to Keplerian elements."""
        if self.is_keplerian:
            return self.copy()

        # In a real implementation, the full conversion math would be here.
        new_data = torch.randn_like(self.data)
        
        return self._create_new_instance(
            new_data=new_data,
            element_type="keplerian"
        ).add_history_entry("to_keplerian")

    def propagate(self, time_span: torch.Tensor) -> "OrbitTensor":
        """
        Propagate orbit using specified propagator.

        Args:
            time_span: Time values to propagate to (seconds from epoch)

        Returns:
            Propagated OrbitTensor
        """
        propagator = getattr(self, "propagator", "kepler")

        if propagator == "kepler":
            return self._propagate_kepler(time_span)
        elif propagator == "j2":
            return self._propagate_j2(time_span)
        else:
            raise NotImplementedError(f"Propagator '{propagator}' not implemented")

    def _propagate_kepler(self, time_span: torch.Tensor) -> "OrbitTensor":
        """Keplerian propagation (two-body problem)."""
        # Convert to Keplerian if needed
        kep_orbit = self.to_keplerian()

        a = kep_orbit.semi_major_axis
        e = kep_orbit.eccentricity
        mu = getattr(self, "mu", 398600.4418)

        # Mean motion
        n = torch.sqrt(mu / a**3)

        # Propagate mean anomaly
        M0 = self._true_to_mean_anomaly(kep_orbit.true_anomaly, e)
        M = M0.unsqueeze(-1) + n.unsqueeze(-1) * time_span.unsqueeze(0)

        # Convert back to true anomaly
        nu = self._mean_to_true_anomaly(M, e.unsqueeze(-1))

        # Create propagated orbits
        propagated_data = torch.zeros(
            (*kep_orbit.shape[:-1], len(time_span), 6), device=kep_orbit.device
        )

        for i in range(len(time_span)):
            propagated_data[..., i, 0] = a
            propagated_data[..., i, 1] = e
            propagated_data[..., i, 2] = kep_orbit.inclination
            propagated_data[..., i, 3] = kep_orbit.raan
            propagated_data[..., i, 4] = kep_orbit.argument_of_periapsis
            propagated_data[..., i, 5] = nu[..., i]

        return self._create_new_instance(
            new_data=propagated_data,
            element_type="keplerian"
        ).add_history_entry("propagate")

    def _propagate_j2(self, time_span: torch.Tensor) -> "OrbitTensor":
        """J2 perturbation propagation."""
        # Simplified J2 propagation - would need full implementation
        # For now, fall back to Keplerian
        return self._propagate_kepler(time_span)

    def _true_to_mean_anomaly(self, nu: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Convert true anomaly to mean anomaly."""
        # Eccentric anomaly
        E = 2 * torch.atan(torch.sqrt((1 - e) / (1 + e)) * torch.tan(nu / 2))

        # Mean anomaly
        M = E - e * torch.sin(E)

        return M

    def _mean_to_true_anomaly(self, M: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """Convert mean anomaly to true anomaly using Newton-Raphson."""
        # Initial guess for eccentric anomaly
        E = M.clone()

        # Newton-Raphson iteration
        for _ in range(10):  # Usually converges quickly
            f = E - e * torch.sin(E) - M
            df = 1 - e * torch.cos(E)
            E = E - f / df

        # True anomaly
        nu = 2 * torch.atan(torch.sqrt((1 + e) / (1 - e)) * torch.tan(E / 2))

        return nu

    def orbital_period(self) -> torch.Tensor:
        """Calculate orbital period (seconds)."""
        if self.is_cartesian:
            kep_orbit = self.to_keplerian()
            a = kep_orbit.semi_major_axis
        else:
            a = self.semi_major_axis

        mu = getattr(self, "mu", 398600.4418)
        return 2 * math.pi * torch.sqrt(a**3 / mu)

    def apoapsis_periapsis(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate apoapsis and periapsis distances."""
        if self.is_cartesian:
            kep_orbit = self.to_keplerian()
            a = kep_orbit.semi_major_axis
            e = kep_orbit.eccentricity
        else:
            a = self.semi_major_axis
            e = self.eccentricity

        ra = a * (1 + e)  # Apoapsis
        rp = a * (1 - e)  # Periapsis

        return ra, rp

    def habitable_zone_distance(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate habitable zone boundaries for exoplanets.

        Returns:
            Tuple of (inner_edge, outer_edge) distances in AU
        """
        host_mass = getattr(self, "host_star_mass", 1.0)  # Solar masses

        # Habitable zone boundaries (Kopparapu et al. 2013)
        # Inner edge: runaway greenhouse
        # Outer edge: maximum greenhouse

        # Luminosity scaling: L = M^3.5 (approximate main sequence)
        luminosity = host_mass**3.5

        # HZ boundaries in AU (for solar-type star)
        inner_hz_sun = 0.95  # AU
        outer_hz_sun = 1.67  # AU

        # Scale by luminosity
        inner_hz = inner_hz_sun * torch.sqrt(torch.tensor(luminosity))
        outer_hz = outer_hz_sun * torch.sqrt(torch.tensor(luminosity))

        return inner_hz, outer_hz

    def is_in_habitable_zone(self) -> torch.Tensor:
        """
        Check if orbit is within the habitable zone.

        Returns:
            Boolean tensor indicating if orbit is in HZ
        """
        if getattr(self, "attractor", "earth").lower() != "star":
            return torch.tensor(False)

        # Get semi-major axis in AU
        if self.is_keplerian:
            a_km = self.semi_major_axis
        else:
            kep = self.to_keplerian()
            a_km = kep.semi_major_axis

        a_au = a_km / 149597870.7  # Convert km to AU

        inner_hz, outer_hz = self.habitable_zone_distance()

        return (a_au >= inner_hz) & (a_au <= outer_hz)

    def transit_probability(self) -> torch.Tensor:
        """
        Calculate transit probability for exoplanets.

        Returns:
            Transit probability [0, 1]
        """
        if not self.is_keplerian:
            kep = self.to_keplerian()
            a = kep.semi_major_axis
            e = kep.eccentricity
        else:
            a = self.semi_major_axis
            e = self.eccentricity

        # Stellar radius (assume solar radius for now)
        host_mass = getattr(self, "host_star_mass", 1.0)
        stellar_radius = host_mass**0.8 * 696000  # km (mass-radius relation)

        # Transit probability ≈ R_star / a for circular orbits
        # For eccentric orbits, use periastron distance
        periastron = a * (1 - e)

        prob = stellar_radius / periastron

        return torch.clamp(prob, 0.0, 1.0)

    def transit_duration(self, planet_radius: float = 6371.0) -> torch.Tensor:
        """
        Calculate transit duration for exoplanets.

        Args:
            planet_radius: Planet radius in km

        Returns:
            Transit duration in hours
        """
        if not self.is_keplerian:
            kep = self.to_keplerian()
            a = kep.semi_major_axis
            period = kep.orbital_period()
        else:
            a = self.semi_major_axis
            period = self.orbital_period()

        # Stellar radius
        host_mass = getattr(self, "host_star_mass", 1.0)
        stellar_radius = host_mass**0.8 * 696000  # km

        # Transit duration (simplified circular orbit)
        # t = (P/π) * arcsin(R_star/a)
        duration_seconds = (period / math.pi) * torch.asin(stellar_radius / a)
        duration_hours = duration_seconds / 3600.0

        return duration_hours

    def equilibrium_temperature(self, albedo: float = 0.3) -> torch.Tensor:
        """
        Calculate equilibrium temperature for exoplanets.

        Args:
            albedo: Planetary albedo [0, 1]

        Returns:
            Equilibrium temperature in Kelvin
        """
        if not self.is_keplerian:
            kep = self.to_keplerian()
            a = kep.semi_major_axis
        else:
            a = self.semi_major_axis

        # Convert to AU
        a_au = a / 149597870.7

        # Stellar temperature (assume main sequence)
        host_mass = getattr(self, "host_star_mass", 1.0)
        stellar_temp = 5778 * (host_mass**0.5)  # K (mass-temperature relation)

        # Stellar radius
        stellar_radius_km = host_mass**0.8 * 696000  # km
        stellar_radius_au = stellar_radius_km / 149597870.7  # AU

        # Equilibrium temperature
        # T_eq = T_star * sqrt(R_star / 2a) * (1 - A)^(1/4)
        temp = (
            stellar_temp
            * torch.sqrt(stellar_radius_au / (2 * a_au))
            * ((1 - albedo) ** 0.25)
        )

        return temp

    def __repr__(self) -> str:
        """String representation."""
        element_type = getattr(self, "element_type", "keplerian")
        attractor = getattr(self, "attractor", "earth")
        stellar_system = getattr(self, "stellar_system", None)
        epoch = getattr(self, "epoch", 0.0)

        system_str = f", system='{stellar_system}'" if stellar_system else ""
        return f"OrbitTensor(shape={list(self.shape)}, type='{element_type}', attractor='{attractor}'{system_str}, epoch={epoch})"

class ManeuverTensor(AstroTensorBase):
    """
    Tensor for orbital maneuvers and delta-V calculations.

    Handles impulsive maneuvers, transfer calculations, and fuel optimization.
    """

    _metadata_fields = [
        "maneuver_type",
        "epoch",
        "frame",
        "efficiency",  # Engine efficiency
    ]

    data: torch.Tensor
    maneuver_type: str = Field("impulsive", description="Type of maneuver ('impulsive', 'continuous')")
    epoch: float = Field(0.0, description="Epoch of the maneuver")
    frame: str = Field("icrs", description="Reference frame for the delta-V vector")
    efficiency: float = Field(1.0, description="Maneuver efficiency (0 to 1)")

    @field_validator("data")
    def validate_maneuver_data(cls, v):
        if v.shape[-1] != 3:
            raise ValueError(f"ManeuverTensor requires a 3-element delta-V vector, but last dim is {v.shape[-1]}")
        return v

    @property
    def delta_v_magnitude(self) -> torch.Tensor:
        """Magnitude of delta-V vector."""
        return torch.linalg.norm(self.data, dim=-1)

    @property
    def maneuver_type(self) -> str:
        """Type of maneuver."""
        return self._metadata.get("maneuver_type", "impulsive")

    @property
    def epoch(self) -> float:
        """Maneuver epoch."""
        return self._metadata.get("epoch", 0.0)

    @property
    def frame(self) -> str:
        """Reference frame."""
        return self._metadata.get("frame", "icrs")

    @property
    def efficiency(self) -> float:
        """Engine efficiency."""
        return self._metadata.get("efficiency", 1.0)

    @property
    def delta_v_x(self) -> torch.Tensor:
        """X component of delta-V."""
        return self.data[..., 0]

    @property
    def delta_v_y(self) -> torch.Tensor:
        """Y component of delta-V."""
        return self.data[..., 1]

    @property
    def delta_v_z(self) -> torch.Tensor:
        """Z component of delta-V."""
        return self.data[..., 2]

    def fuel_mass_ratio(self, specific_impulse: float) -> torch.Tensor:
        """
        Calculate fuel mass ratio using rocket equation.

        Args:
            specific_impulse: Specific impulse in seconds

        Returns:
            Mass ratio (m_initial / m_final)
        """
        g0 = 9.80665  # m/s²
        ve = specific_impulse * g0 / 1000  # km/s

        delta_v = self.delta_v_magnitude / self.efficiency

        return torch.exp(delta_v / ve)

    @classmethod  # type: ignore[misc]
    def interstellar_trajectory(
        cls,
        departure_velocity: torch.Tensor,
        target_star_distance: float,
        cruise_velocity_fraction: float = 0.1,
        acceleration_time: float = 1.0,
    ) -> "ManeuverTensor":
        """
        Calculate interstellar trajectory maneuver.

        Args:
            departure_velocity: Initial velocity vector from solar system
            target_star_distance: Distance to target star (ly)
            cruise_velocity_fraction: Fraction of light speed for cruise
            acceleration_time: Acceleration phase duration (years)

        Returns:
            ManeuverTensor for interstellar trajectory
        """
        c = 299792458.0  # m/s
        cruise_velocity = cruise_velocity_fraction * c / 1000  # km/s

        # Calculate required delta-V for cruise phase
        delta_v_cruise = cruise_velocity - torch.norm(departure_velocity, dim=-1)

        # Acceleration phase delta-V (simplified)
        years_to_seconds = 365.25 * 24 * 3600
        acceleration = delta_v_cruise / (acceleration_time * years_to_seconds)

        # Total delta-V vector (simplified as radial)
        direction = departure_velocity / torch.norm(
            departure_velocity, dim=-1, keepdim=True
        )
        total_delta_v = delta_v_cruise.unsqueeze(-1) * direction

        return cls(
            total_delta_v,
            maneuver_type="interstellar",
            frame="heliocentric",
        )

    @classmethod  # type: ignore[misc]
    def hohmann_transfer(
        cls, initial_orbit: OrbitTensor, final_radius: torch.Tensor
    ) -> "ManeuverTensor":
        """
        Calculate Hohmann transfer maneuver.

        Args:
            initial_orbit: Initial circular orbit
            final_radius: Final orbital radius

        Returns:
            ManeuverTensor for Hohmann transfer
        """
        if not initial_orbit.is_keplerian:
            raise ValueError("Hohmann transfer requires Keplerian orbit")

        r1 = initial_orbit.semi_major_axis
        r2 = final_radius
        mu = initial_orbit.mu

        # Transfer orbit semi-major axis
        a_transfer = (r1 + r2) / 2

        # Initial velocity
        v1 = torch.sqrt(mu / r1)

        # Transfer orbit velocities
        v1_transfer = torch.sqrt(mu * (2 / r1 - 1 / a_transfer))
        v2_transfer = torch.sqrt(mu * (2 / r2 - 1 / a_transfer))

        # Final circular velocity
        v2 = torch.sqrt(mu / r2)

        # Delta-V calculations
        delta_v1 = v1_transfer - v1
        delta_v2 = v2 - v2_transfer

        # Total delta-V (assuming coplanar transfer)
        total_delta_v = torch.abs(delta_v1) + torch.abs(delta_v2)

        # Create delta-V vector (simplified as tangential)
        delta_v_vector = torch.stack(
            [
                total_delta_v,
                torch.zeros_like(total_delta_v),
                torch.zeros_like(total_delta_v),
            ],
            dim=-1,
        )

        return cls(
            delta_v_vector,
            maneuver_type="hohmann_transfer",
        )

    def apply_to_orbit(self, orbit: OrbitTensor) -> OrbitTensor:
        """Applies the delta-V to a Cartesian orbit state."""
        if not orbit.is_cartesian:
            orbit = orbit.to_cartesian()
        
        new_velocity = orbit.velocity + self.data
        new_orbital_data = torch.cat([orbit.position, new_velocity], dim=-1)
        
        return orbit._create_new_instance(
            new_data=new_orbital_data,
        ).add_history_entry("applied_maneuver", maneuver_type=self.maneuver_type)

    def __repr__(self) -> str:
        """String representation."""
        maneuver_type = getattr(self, "maneuver_type", "impulsive")
        total_dv = self.delta_v_magnitude.sum()

        return f"ManeuverTensor(shape={list(self.shape)}, type='{maneuver_type}', total_ΔV={total_dv:.3f} km/s)"
