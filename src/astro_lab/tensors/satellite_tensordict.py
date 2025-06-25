"""
TensorDict-basierte Implementierung für Satelliten-Daten
======================================================

Umstellung der EarthSatelliteTensor und verwandter Klassen auf TensorDict-Architektur.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from tensordict import TensorDict

from .orbital_tensordict import OrbitTensorDict
from .tensordict_astro import AstroTensorDict


class EarthSatelliteTensorDict(AstroTensorDict):
    """
    TensorDict for Earth satellite data.

    Structure:
    {
        "tle_data": Tensor[N, 8],     # TLE parameters
        "orbit": OrbitTensorDict,     # Orbital elements
        "state": Tensor[N, 6],        # Position and velocity
        "meta": {
            "satellite_names": List[str],
            "catalog_numbers": List[int],
            "launch_dates": List[str],
            "reference_epoch": float,
        }
    }
    """

    def __init__(
        self,
        tle_data: torch.Tensor,
        satellite_names: Optional[List[str]] = None,
        catalog_numbers: Optional[List[int]] = None,
        reference_epoch: float = 2000.0,
        **kwargs,
    ):
        """
        Initialize EarthSatelliteTensorDict.

        Args:
            tle_data: [N, 8] TLE-parameter [inclination, raan, eccentricity,
                     arg_perigee, mean_anomaly, mean_motion, bstar, epoch]
            satellite_names: Names of the satellites
            catalog_numbers: Catalog numbers
            reference_epoch: Reference epoch
        """
        if tle_data.shape[-1] != 8:
            raise ValueError(f"TLE data must have shape [..., 8], got {tle_data.shape}")

        n_objects = tle_data.shape[0]

        if satellite_names is None:
            satellite_names = [f"SAT_{i:04d}" for i in range(n_objects)]

        if catalog_numbers is None:
            catalog_numbers = list(range(n_objects))

        # Convert TLE to orbital elements
        orbital_elements = self._tle_to_orbital_elements(tle_data)
        orbit = OrbitTensorDict(orbital_elements, frame="teme", central_body="Earth")

        # Calculate initial states
        initial_state = orbit.to_cartesian()

        data = {
            "tle_data": tle_data,
            "orbit": orbit,
            "state": initial_state,
            "meta": {
                "satellite_names": satellite_names,
                "catalog_numbers": catalog_numbers,
                "launch_dates": ["unknown"] * n_objects,
                "reference_epoch": reference_epoch,
            },
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def _tle_to_orbital_elements(self, tle_data: torch.Tensor) -> torch.Tensor:
        """
        Converts TLE data to standard orbital elements.

        Args:
            tle_data: [N, 8] TLE parameters

        Returns:
            [N, 6] Orbital elements [a, e, i, Omega, omega, M]
        """
        # TLE format: [inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, bstar, epoch]
        inclination = tle_data[..., 0]
        raan = tle_data[..., 1]  # Right Ascension of Ascending Node (Omega)
        eccentricity = tle_data[..., 2]
        arg_perigee = tle_data[..., 3]  # Argument of Perigee (omega)
        mean_anomaly = tle_data[..., 4]
        mean_motion = tle_data[..., 5]  # Revolutions per day

        # Calculate semi-major axis from mean motion
        # n = sqrt(GM/a³) -> a = (GM/n²)^(1/3)
        GM_earth = 398600.4418  # km³/s²
        n_rad_per_sec = mean_motion * 2 * math.pi / 86400  # Convert to rad/s
        semi_major_axis = (GM_earth / (n_rad_per_sec**2)) ** (1 / 3)

        # Create orbital elements tensor
        orbital_elements = torch.stack(
            [
                semi_major_axis,
                eccentricity,
                inclination,
                raan,
                arg_perigee,
                mean_anomaly,
            ],
            dim=-1,
        )

        return orbital_elements

    @property
    def satellite_names(self) -> List[str]:
        """Names of the satellites."""
        return self["meta", "satellite_names"]

    @property
    def catalog_numbers(self) -> List[int]:
        """Catalog numbers of the satellites."""
        return self["meta", "catalog_numbers"]

    def propagate_sgp4(self, time_minutes: torch.Tensor) -> EarthSatelliteTensorDict:
        """
        Propagates satellite orbits with simplified SGP4.

        Args:
            time_minutes: [N] Time in minutes since epoch

        Returns:
            New EarthSatelliteTensorDict with propagated states
        """
        # Simplified SGP4 implementation
        tle_data = self["tle_data"]
        mean_motion = tle_data[..., 5]
        mean_anomaly = tle_data[..., 4]

        # New mean anomaly
        delta_M = mean_motion * 2 * math.pi * time_minutes / 1440  # Convert to radians
        new_mean_anomaly = torch.fmod(mean_anomaly + delta_M, 2 * math.pi)

        # Update TLE data
        new_tle_data = tle_data.clone()
        new_tle_data[..., 4] = new_mean_anomaly
        new_tle_data[..., 7] += time_minutes / 1440  # Update epoch

        return EarthSatelliteTensorDict(
            tle_data=new_tle_data,
            satellite_names=self.satellite_names,
            catalog_numbers=self.catalog_numbers,
            reference_epoch=self["meta", "reference_epoch"],
        )

    def compute_ground_track(
        self, n_points: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes satellite ground track.

        Args:
            n_points: Number of points for the track

        Returns:
            Tuple of (longitude, latitude) tensors
        """
        # Calculate positions over one period
        periods = self["orbit"].compute_period()
        time_steps = torch.linspace(0, periods.max(), n_points)

        longitudes = []
        latitudes = []

        for t in time_steps:
            # Propagate to this time
            sat_at_time = self.propagate_sgp4(t * 1440)  # Convert to minutes

            # Convert to geographic coordinates
            state = sat_at_time["state"]
            positions = state[..., :3]  # x, y, z

            # Simplified conversion to Lat/Lon
            longitude = (
                torch.atan2(positions[..., 1], positions[..., 0]) * 180 / math.pi
            )
            latitude = (
                torch.asin(
                    torch.clamp(
                        positions[..., 2] / torch.norm(positions, dim=-1), -1, 1
                    )
                )
                * 180
                / math.pi
            )

            longitudes.append(longitude)
            latitudes.append(latitude)

        return torch.stack(longitudes, dim=-1), torch.stack(latitudes, dim=-1)

    def compute_visibility(
        self, ground_position: torch.Tensor, min_elevation: float = 10.0
    ) -> torch.Tensor:
        """
        Computes visibility from ground stations.

        Args:
            ground_position: [3] Ground station position [x, y, z]
            min_elevation: Minimum elevation in degrees

        Returns:
            [N] Boolean tensor for visibility
        """
        # Current satellite positions
        sat_positions = self["state"][..., :3]  # [N, 3]

        # Vector from ground station to satellite
        sat_vectors = sat_positions - ground_position.unsqueeze(0)

        # Calculate elevation (simplified)
        distances = torch.norm(sat_vectors, dim=-1)
        heights = (
            torch.norm(sat_positions, dim=-1) - 6371.0
        )  # Height above Earth surface
        elevations = torch.asin(torch.clamp(heights / distances, -1, 1)) * 180 / math.pi

        return elevations >= min_elevation

    def get_satellite_by_name(self, name: str) -> EarthSatelliteTensorDict:
        """
        Gets a specific satellite by name.

        Args:
            name: Name of the satellite

        Returns:
            EarthSatelliteTensorDict with one satellite
        """
        if name not in self.satellite_names:
            raise ValueError(f"Satellite '{name}' not found")

        idx = self.satellite_names.index(name)

        # Extract data for this satellite
        return EarthSatelliteTensorDict(
            tle_data=self["tle_data"][idx : idx + 1],
            satellite_names=[name],
            catalog_numbers=[self.catalog_numbers[idx]],
            reference_epoch=self["meta", "reference_epoch"],
        )

    def filter_by_altitude(
        self, min_altitude: float, max_altitude: float
    ) -> EarthSatelliteTensorDict:
        """
        Filters satellites by altitude.

        Args:
            min_altitude: Minimum altitude in km
            max_altitude: Maximum altitude in km

        Returns:
            Filtered EarthSatelliteTensorDict
        """
        # Calculate altitudes from semi-major axis
        altitudes = self["orbit"].semi_major_axis - 6371.0  # Earth radius

        mask = (altitudes >= min_altitude) & (altitudes <= max_altitude)
        indices = torch.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(
                f"No satellites found in altitude range {min_altitude}-{max_altitude} km"
            )

        # Filter all data
        filtered_names = [self.satellite_names[i] for i in indices]
        filtered_catalog = [self.catalog_numbers[i] for i in indices]

        return EarthSatelliteTensorDict(
            tle_data=self["tle_data"][indices],
            satellite_names=filtered_names,
            catalog_numbers=filtered_catalog,
            reference_epoch=self["meta", "reference_epoch"],
        )
