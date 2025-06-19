"""
Earth satellite tensor for Earth-specific satellite operations.
"""

import math
from typing import Dict, Optional

import torch

from .base import AstroTensorBase
from .orbital import OrbitTensor
from .spatial_3d import Spatial3DTensor


class EarthSatelliteTensor(AstroTensorBase):
    """
    Tensor for Earth satellites with Earth-specific operations using composition.

    Handles Earth-specific functionality like ground tracks,
    field of view calculations, and pass predictions.
    """

    _metadata_fields = [
        "satellite_name",
        "norad_id",
        "launch_date",
        "satellite_type",
        "orbit_tensor",
    ]

    def __init__(
        self,
        data,
        satellite_name: Optional[str] = None,
        norad_id: Optional[int] = None,
        launch_date: Optional[float] = None,
        satellite_type: str = "unknown",
        **kwargs,
    ):
        """
        Create a new EarthSatelliteTensor using composition.

        Args:
            data: Orbital data [..., 6]
            satellite_name: Satellite name
            norad_id: NORAD catalog number
            launch_date: Launch date (MJD)
            satellite_type: Type of satellite
            **kwargs: Additional arguments
        """
        # Create internal OrbitTensor
        orbit_tensor = OrbitTensor(data, attractor="earth", **kwargs)

        # Store satellite-specific metadata
        metadata = {
            "satellite_name": satellite_name,
            "norad_id": norad_id,
            "launch_date": launch_date,
            "satellite_type": satellite_type,
            "orbit_tensor": orbit_tensor,
        }

        super().__init__(data, **metadata, tensor_type="earth_satellite")

    def _validate(self) -> None:
        """Validate satellite tensor data."""
        if self._data.shape[-1] != 6:
            raise ValueError("EarthSatelliteTensor requires 6-element orbital state")

    @property
    def orbit_tensor(self) -> OrbitTensor:
        """Access to internal orbit tensor."""
        return self._metadata["orbit_tensor"]

    def ground_track(
        self, time_span: torch.Tensor, earth_rotation_rate: float = 7.2921159e-5
    ):
        """
        Calculate ground track coordinates.

        Args:
            time_span: Time values (seconds from epoch)
            earth_rotation_rate: Earth rotation rate (rad/s)

        Returns:
            Spatial3DTensor with ground track coordinates
        """
        # Create ground track using orbital propagation

        # Use orbit tensor for propagation
        propagated = self.orbit_tensor.propagate(time_span)

        # Convert to Cartesian if needed
        if propagated.is_keplerian:
            cart_orbit = propagated.to_cartesian()
        else:
            cart_orbit = propagated

        # Extract positions for each time step
        positions = cart_orbit.position  # [..., n_times, 3]

        # Account for Earth rotation
        lon_correction = earth_rotation_rate * time_span

        # Convert Cartesian to spherical coordinates
        x = positions[..., 0]
        y = positions[..., 1]
        z = positions[..., 2]

        # Calculate longitude and latitude
        lon = torch.atan2(y, x) * 180.0 / math.pi
        lat = torch.asin(z / torch.norm(positions, dim=-1)) * 180.0 / math.pi

        # Apply Earth rotation correction
        lon = lon - lon_correction.unsqueeze(0) * 180.0 / math.pi

        # Normalize longitude to [-180, 180]
        lon = ((lon + 180) % 360) - 180

        # Create spatial tensor with proper constructor
        return Spatial3DTensor(
            data=torch.stack([lon, lat, torch.full_like(lon, 6371.0)], dim=-1),
            coordinate_system="equatorial",
            unit="deg",
            distance_unit="km",
        )

    def field_of_view_footprint(
        self,
        altitude: float,
        fov_angle: float,
        time_span: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate field of view footprint on Earth surface.

        Args:
            altitude: Target altitude above Earth surface (km)
            fov_angle: Field of view half-angle (degrees)
            time_span: Time values for propagation

        Returns:
            Dictionary with footprint coordinates and area
        """
        earth_radius = 6371.0  # km

        if time_span is not None:
            # Use propagated positions
            propagated = self.orbit_tensor.propagate(time_span)
            if propagated.is_keplerian:
                positions = propagated.to_cartesian().position
            else:
                positions = propagated.position
        else:
            # Use current position
            if self.orbit_tensor.is_keplerian:
                positions = self.orbit_tensor.to_cartesian().position
            else:
                positions = self.orbit_tensor.position
            positions = positions.unsqueeze(-2)  # Add time dimension

        # Satellite altitude
        sat_altitude = torch.norm(positions, dim=-1) - earth_radius

        # Ground range calculation
        fov_rad = math.radians(fov_angle)

        # Maximum ground range for given FOV
        # Using spherical Earth approximation
        fov_tensor = torch.tensor(
            fov_rad, dtype=sat_altitude.dtype, device=sat_altitude.device
        )
        nadir_angle = torch.asin(
            (earth_radius + altitude)
            / (earth_radius + sat_altitude)
            * torch.sin(fov_tensor)
        )
        ground_range = earth_radius * (fov_rad - nadir_angle)

        # Footprint area (circular approximation)
        footprint_area = math.pi * ground_range**2

        # Calculate footprint boundary points (simplified circular footprint)
        n_points = 36  # Number of boundary points
        angles = torch.linspace(0, 2 * math.pi, n_points)

        # Get ground track center
        if time_span is not None:
            ground_track = self.ground_track(time_span)
            ra, dec, _ = ground_track.to_spherical()
            center_lon = ra
            center_lat = dec
        else:
            ground_track = self.ground_track(torch.tensor([0.0]))
            ra, dec, _ = ground_track.to_spherical()
            center_lon = ra[..., 0]
            center_lat = dec[..., 0]

        # Calculate boundary points (simplified)
        boundary_points = torch.zeros((*center_lon.shape, n_points, 2))

        for i, angle in enumerate(angles):
            # Simple circular approximation in lat/lon
            delta_lat = (
                (ground_range / earth_radius) * torch.cos(angle) * 180.0 / math.pi
            )
            delta_lon = (
                (ground_range / earth_radius)
                * torch.sin(angle)
                * 180.0
                / math.pi
                / torch.cos(torch.deg2rad(center_lat))
            )

            boundary_points[..., i, 0] = center_lon + delta_lon  # Longitude
            boundary_points[..., i, 1] = center_lat + delta_lat  # Latitude

        return {
            "center_coordinates": torch.stack([center_lon, center_lat], dim=-1),
            "boundary_points": boundary_points,
            "ground_range": ground_range,
            "footprint_area": footprint_area,
        }

    def ground_range(self, elevation_angle: float = 0.0) -> torch.Tensor:
        """
        Calculate maximum ground range for given elevation angle.

        Args:
            elevation_angle: Minimum elevation angle (degrees)

        Returns:
            Ground range (km)
        """
        earth_radius = 6371.0  # km

        # Get satellite altitude
        if self.orbit_tensor.is_keplerian:
            positions = self.orbit_tensor.to_cartesian().position
        else:
            positions = self.orbit_tensor.position

        sat_altitude = torch.norm(positions, dim=-1) - earth_radius

        # Convert elevation angle to radians
        elev_rad = math.radians(elevation_angle)

        # Calculate ground range using spherical Earth
        # Range = R * arccos(R / (R + h) * cos(elevation))
        elev_tensor = torch.tensor(
            elev_rad, dtype=sat_altitude.dtype, device=sat_altitude.device
        )
        cos_arg = (earth_radius / (earth_radius + sat_altitude)) * torch.cos(
            elev_tensor
        )
        cos_arg = torch.clamp(cos_arg, -1.0, 1.0)  # Ensure valid range

        ground_range = earth_radius * torch.acos(cos_arg)

        return ground_range

    def pass_prediction(
        self,
        ground_station_coords,  # Spatial3DTensor
        time_span: torch.Tensor,
        min_elevation: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict satellite passes over a ground station.

        Args:
            ground_station_coords: Ground station coordinates (Spatial3DTensor)
            time_span: Time values to check (seconds from epoch)
            min_elevation: Minimum elevation angle (degrees)

        Returns:
            Dictionary with pass information
        """
        earth_radius = 6371.0  # km

        # Ground station position in ECEF
        gs_ra, gs_dec, _ = ground_station_coords.to_spherical()
        gs_lon = gs_ra * math.pi / 180.0
        gs_lat = gs_dec * math.pi / 180.0

        gs_x = earth_radius * torch.cos(gs_lat) * torch.cos(gs_lon)
        gs_y = earth_radius * torch.cos(gs_lat) * torch.sin(gs_lon)
        gs_z = earth_radius * torch.sin(gs_lat)
        gs_pos = torch.stack([gs_x, gs_y, gs_z])

        # Propagate satellite using orbit tensor
        propagated = self.orbit_tensor.propagate(time_span)
        if propagated.is_keplerian:
            sat_positions = propagated.to_cartesian().position
        else:
            sat_positions = propagated.position

        # Calculate elevation angles
        sat_to_gs = sat_positions - gs_pos.unsqueeze(0).unsqueeze(0)
        up_vector = gs_pos / torch.norm(gs_pos)
        range_vector = torch.norm(sat_to_gs, dim=-1)
        elevation_sin = (
            torch.sum(sat_to_gs * up_vector.unsqueeze(0).unsqueeze(0), dim=-1)
            / range_vector
        )
        elevation_angles = (
            torch.asin(torch.clamp(elevation_sin, -1.0, 1.0)) * 180.0 / math.pi
        )

        # Find passes
        visible_mask = elevation_angles > min_elevation

        return {
            "elevation_angles": elevation_angles,
            "visible_mask": visible_mask,
            "range_distances": range_vector,
        }

    def sun_illumination(self, time_span: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate satellite illumination conditions.

        Args:
            time_span: Time values (seconds from epoch)

        Returns:
            Dictionary with illumination information
        """
        # Simplified sun position (would need proper ephemeris)
        # Assume sun at [1 AU, 0, 0] for simplicity
        au = 149597870.7  # km

        # Get device and dtype from satellite positions for consistency
        propagated = self.orbit_tensor.propagate(time_span)
        if propagated.is_keplerian:
            sat_positions = propagated.to_cartesian().position
        else:
            sat_positions = propagated.position

        sun_position = torch.tensor(
            [au, 0.0, 0.0], dtype=sat_positions.dtype, device=sat_positions.device
        )

        # Use already propagated satellite positions
        earth_radius = 6371.0  # km

        # Check if satellite is in Earth's shadow
        # Vector from Earth center to satellite
        sat_distance = torch.norm(sat_positions, dim=-1)

        # Vector from Earth to sun (normalized)
        sun_direction = sun_position / torch.norm(sun_position)

        # Project satellite position onto sun direction
        sat_sun_projection = torch.sum(
            sat_positions * sun_direction.unsqueeze(0).unsqueeze(0), dim=-1
        )

        # Distance from satellite to Earth-sun line
        sat_to_line = sat_positions - sat_sun_projection.unsqueeze(
            -1
        ) * sun_direction.unsqueeze(0).unsqueeze(0)
        distance_to_line = torch.norm(sat_to_line, dim=-1)

        # Satellite is in shadow if:
        # 1. Behind Earth relative to sun (sat_sun_projection < 0)
        # 2. Within Earth's shadow cylinder (distance_to_line < earth_radius)
        in_shadow = (sat_sun_projection < 0) & (distance_to_line < earth_radius)

        # Eclipse duration (simplified)
        eclipse_mask = in_shadow

        return {
            "in_shadow": in_shadow,
            "eclipse_mask": eclipse_mask,
            "sun_satellite_angle": torch.acos(
                torch.clamp(sat_sun_projection / sat_distance, -1.0, 1.0)
            )
            * 180.0
            / math.pi,
            "distance_to_shadow_line": distance_to_line,
        }

    def orbital_decay_prediction(
        self,
        atmospheric_density: float = 1e-12,  # kg/m³
        drag_coefficient: float = 2.2,
        satellite_area: float = 1.0,  # m²
        satellite_mass: float = 100.0,  # kg
    ) -> Dict[str, torch.Tensor]:
        """
        Predict orbital decay due to atmospheric drag.

        Args:
            atmospheric_density: Atmospheric density at satellite altitude
            drag_coefficient: Satellite drag coefficient
            satellite_area: Cross-sectional area
            satellite_mass: Satellite mass

        Returns:
            Dictionary with decay information
        """
        # Get current orbital parameters
        if self.orbit_tensor.is_cartesian:
            kep_orbit = self.orbit_tensor.to_keplerian()
        else:
            kep_orbit = self.orbit_tensor

        a = kep_orbit.semi_major_axis  # km
        e = kep_orbit.eccentricity

        # Convert to SI units
        a_m = a * 1000  # meters
        area_m2 = satellite_area
        mass_kg = satellite_mass

        # Ballistic coefficient
        ballistic_coeff = drag_coefficient * area_m2 / mass_kg

        # Proper orbital decay calculation with eccentricity
        mu = getattr(self.orbit_tensor, "mu", 398600.4418) * 1e9  # m³/s²

        # Mean motion
        n = torch.sqrt(mu / (a_m**3))  # rad/s

        # Atmospheric drag effects on orbital elements
        # King-Hele drag theory for elliptical orbits

        # Semi-major axis decay rate (includes eccentricity effects)
        # da/dt = -2πρB * a² * sqrt(μ/a³) * (1 + e²/4) for circular approximation
        # More accurate: includes eccentricity correction factor
        ecc_factor = 1 + 1.5 * e**2 + 0.125 * e**4  # Series expansion

        da_dt = (
            -2
            * math.pi
            * atmospheric_density
            * ballistic_coeff
            * a_m**2
            * n
            * ecc_factor
        )

        # Eccentricity decay rate
        # de/dt = -πρB * a * sqrt(μ/a³) * e * (1 + e²/8)
        de_dt = (
            -math.pi
            * atmospheric_density
            * ballistic_coeff
            * a_m
            * n
            * e
            * (1 + e**2 / 8)
        )

        # Perigee and apogee altitudes
        earth_radius = 6371.0e3  # m
        perigee_alt = a_m * (1 - e) - earth_radius
        apogee_alt = a_m * (1 + e) - earth_radius

        # Time to decay calculation
        # For circular orbits: t_decay ≈ a / |da/dt|
        # For elliptical orbits: more complex, but approximated here

        # Critical altitude (when perigee reaches ~100 km)
        critical_perigee = 100e3  # m
        altitude_to_decay = perigee_alt - critical_perigee

        # Time to decay (when perigee reaches critical altitude)
        if da_dt.item() != 0 and altitude_to_decay > 0:
            # Approximate time for perigee to reach critical altitude
            time_to_decay = altitude_to_decay / abs(da_dt.item() * (1 - e.item()))
        else:
            time_to_decay = float("inf")

        # Orbital lifetime in days
        lifetime_days = (
            time_to_decay / 86400.0 if time_to_decay != float("inf") else float("inf")
        )

        return {
            "semi_major_axis_decay_rate": da_dt / 1000,  # km/s
            "eccentricity_decay_rate": de_dt,  # 1/s
            "time_to_decay": torch.tensor(time_to_decay),  # seconds
            "lifetime_days": torch.tensor(lifetime_days),  # days
            "ballistic_coefficient": torch.tensor(ballistic_coeff),
            "current_altitude": a - 6371.0,  # km above Earth
            "perigee_altitude": perigee_alt / 1000,  # km above Earth
            "apogee_altitude": apogee_alt / 1000,  # km above Earth
            "eccentricity_factor": ecc_factor,
        }

    def __repr__(self) -> str:
        """String representation."""
        satellite_name = getattr(self, "satellite_name", "Unknown")
        norad_id = getattr(self, "norad_id", None)

        id_str = f", NORAD {norad_id}" if norad_id else ""

        return f"EarthSatelliteTensor(name='{satellite_name}'{id_str}, shape={list(self.shape)})"


class AttitudeTensor(AstroTensorBase):
    """
    Tensor for satellite attitude and orientation.

    Handles quaternions, Euler angles, and attitude propagation.
    """

    _metadata_fields = [
        "representation",
        "frame",
        "epoch",
        "angular_velocity",
    ]

    def __init__(
        self,
        data,
        representation: str = "quaternion",
        frame: str = "body",
        epoch: float = 0.0,
        angular_velocity: Optional[torch.Tensor] = None,
    ):
        """
        Create a new AttitudeTensor.

        Args:
            data: Attitude data [..., 4] (quaternions) or [..., 3] (Euler angles)
            representation: Attitude representation ('quaternion', 'euler', 'rotation_matrix')
            frame: Reference frame ('body', 'inertial', 'orbital')
            epoch: Reference epoch
            angular_velocity: Angular velocity vector (rad/s)
        """
        metadata = {
            "representation": representation,
            "frame": frame,
            "epoch": epoch,
            "angular_velocity": angular_velocity,
        }

        super().__init__(data, **metadata, tensor_type="attitude")

        # Validate dimensions
        if representation == "quaternion" and self._data.shape[-1] != 4:
            raise ValueError("Quaternion representation requires 4 components")
        elif representation == "euler" and self._data.shape[-1] != 3:
            raise ValueError("Euler angle representation requires 3 components")
        elif representation == "rotation_matrix" and self._data.shape[-2:] != (3, 3):
            raise ValueError("Rotation matrix representation requires 3x3 matrix")

    @property
    def is_quaternion(self) -> bool:
        """Check if representation is quaternion."""
        return getattr(self, "representation", "quaternion") == "quaternion"

    @property
    def is_euler(self) -> bool:
        """Check if representation is Euler angles."""
        return getattr(self, "representation", "quaternion") == "euler"

    @property
    def q_w(self) -> torch.Tensor:
        """Quaternion scalar component (w)."""
        if not self.is_quaternion:
            raise ValueError("q_w only available for quaternion representation")
        return self[..., 0]

    @property
    def q_x(self) -> torch.Tensor:
        """Quaternion x component."""
        if not self.is_quaternion:
            raise ValueError("q_x only available for quaternion representation")
        return self[..., 1]

    @property
    def q_y(self) -> torch.Tensor:
        """Quaternion y component."""
        if not self.is_quaternion:
            raise ValueError("q_y only available for quaternion representation")
        return self[..., 2]

    @property
    def q_z(self) -> torch.Tensor:
        """Quaternion z component."""
        if not self.is_quaternion:
            raise ValueError("q_z only available for quaternion representation")
        return self[..., 3]

    def to_rotation_matrix(self) -> torch.Tensor:
        """Convert to rotation matrix."""
        if self.is_quaternion:
            # Quaternion to rotation matrix
            w, x, y, z = self.q_w, self.q_x, self.q_y, self.q_z

            # Rotation matrix elements
            r11 = 1 - 2 * (y**2 + z**2)
            r12 = 2 * (x * y - w * z)
            r13 = 2 * (x * z + w * y)
            r21 = 2 * (x * y + w * z)
            r22 = 1 - 2 * (x**2 + z**2)
            r23 = 2 * (y * z - w * x)
            r31 = 2 * (x * z - w * y)
            r32 = 2 * (y * z + w * x)
            r33 = 1 - 2 * (x**2 + y**2)

            # Stack into rotation matrix
            row1 = torch.stack([r11, r12, r13], dim=-1)
            row2 = torch.stack([r21, r22, r23], dim=-1)
            row3 = torch.stack([r31, r32, r33], dim=-1)

            return torch.stack([row1, row2, row3], dim=-2)

        elif self.is_euler:
            # Euler angles to rotation matrix (ZYX convention)
            phi, theta, psi = self[..., 0], self[..., 1], self[..., 2]

            # Individual rotation matrices
            cos_phi, sin_phi = torch.cos(phi), torch.sin(phi)
            cos_theta, sin_theta = torch.cos(theta), torch.sin(theta)
            cos_psi, sin_psi = torch.cos(psi), torch.sin(psi)

            # Combined rotation matrix
            r11 = cos_theta * cos_psi
            r12 = -cos_theta * sin_psi
            r13 = sin_theta
            r21 = sin_phi * sin_theta * cos_psi + cos_phi * sin_psi
            r22 = -sin_phi * sin_theta * sin_psi + cos_phi * cos_psi
            r23 = -sin_phi * cos_theta
            r31 = -cos_phi * sin_theta * cos_psi + sin_phi * sin_psi
            r32 = cos_phi * sin_theta * sin_psi + sin_phi * cos_psi
            r33 = cos_phi * cos_theta

            row1 = torch.stack([r11, r12, r13], dim=-1)
            row2 = torch.stack([r21, r22, r23], dim=-1)
            row3 = torch.stack([r31, r32, r33], dim=-1)

            return torch.stack([row1, row2, row3], dim=-2)
        else:
            raise NotImplementedError(
                f"Conversion from {getattr(self, 'representation')} not implemented"
            )

    def pointing_vector(
        self, body_vector: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate pointing vector in inertial frame.

        Args:
            body_vector: Vector in body frame (default: [0, 0, 1] - z-axis)

        Returns:
            Pointing vector in inertial frame
        """
        if body_vector is None:
            body_vector = torch.tensor([0.0, 0.0, 1.0])

        # Get rotation matrix
        R = self.to_rotation_matrix()

        # Transform body vector to inertial frame
        pointing = torch.matmul(R, body_vector.unsqueeze(-1)).squeeze(-1)

        return pointing

    def __repr__(self) -> str:
        """String representation."""
        representation = getattr(self, "representation", "quaternion")
        frame = getattr(self, "frame", "body")

        return f"AttitudeTensor(shape={list(self.shape)}, repr='{representation}', frame='{frame}')"
