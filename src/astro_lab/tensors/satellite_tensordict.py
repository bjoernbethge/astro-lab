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

from .tensordict_astro import AstroTensorDict
from .orbital_tensordict import OrbitTensorDict


class EarthSatelliteTensorDict(AstroTensorDict):
    """
    TensorDict für Erdsatelliten-Daten.

    Struktur:
    {
        "tle_data": Tensor[N, 8],     # TLE Parameter
        "orbit": OrbitTensorDict,     # Orbital-Elemente
        "state": Tensor[N, 6],        # Position und Geschwindigkeit
        "meta": {
            "satellite_names": List[str],
            "catalog_numbers": List[int],
            "launch_dates": List[str],
            "reference_epoch": float,
        }
    }
    """

    def __init__(self, tle_data: torch.Tensor, 
                 satellite_names: Optional[List[str]] = None,
                 catalog_numbers: Optional[List[int]] = None,
                 reference_epoch: float = 2000.0, **kwargs):
        """
        Initialisiert EarthSatelliteTensorDict.

        Args:
            tle_data: [N, 8] TLE-Parameter [inclination, raan, eccentricity, 
                     arg_perigee, mean_anomaly, mean_motion, bstar, epoch]
            satellite_names: Namen der Satelliten
            catalog_numbers: Katalog-Nummern
            reference_epoch: Referenz-Epoche
        """
        if tle_data.shape[-1] != 8:
            raise ValueError(f"TLE data must have shape [..., 8], got {tle_data.shape}")

        n_objects = tle_data.shape[0]

        if satellite_names is None:
            satellite_names = [f"SAT_{i:04d}" for i in range(n_objects)]

        if catalog_numbers is None:
            catalog_numbers = list(range(n_objects))

        # Konvertiere TLE zu Orbital-Elementen
        orbital_elements = self._tle_to_orbital_elements(tle_data)
        orbit = OrbitTensorDict(orbital_elements, frame="teme", central_body="Earth")

        # Berechne initiale Zustände
        initial_state = orbit.to_cartesian()

        data = {
            "tle_data": tle_data,
            "orbit": orbit,
            "state": initial_state,
            "meta": TensorDict({
                "satellite_names": satellite_names,
                "catalog_numbers": catalog_numbers,
                "launch_dates": ["unknown"] * n_objects,
                "reference_epoch": reference_epoch,
            }, batch_size=(n_objects,))
        }

        super().__init__(data, batch_size=(n_objects,), **kwargs)

    def _tle_to_orbital_elements(self, tle_data: torch.Tensor) -> torch.Tensor:
        """
        Konvertiert TLE-Daten zu Standard-Orbital-Elementen.

        Args:
            tle_data: [N, 8] TLE-Parameter

        Returns:
            [N, 6] Orbital-Elemente [a, e, i, Omega, omega, M]
        """
        # TLE Format: [inclination, raan, eccentricity, arg_perigee, mean_anomaly, mean_motion, bstar, epoch]
        inclination = tle_data[..., 0]
        raan = tle_data[..., 1]  # Right Ascension of Ascending Node (Omega)
        eccentricity = tle_data[..., 2]
        arg_perigee = tle_data[..., 3]  # Argument of Perigee (omega)
        mean_anomaly = tle_data[..., 4]
        mean_motion = tle_data[..., 5]  # Revolutions per day

        # Berechne große Halbachse aus mittlerer Bewegung
        # n = sqrt(GM/a³) -> a = (GM/n²)^(1/3)
        GM_earth = 398600.4418  # km³/s²
        n_rad_per_sec = mean_motion * 2 * math.pi / 86400  # Convert to rad/s
        semi_major_axis = (GM_earth / (n_rad_per_sec ** 2)) ** (1/3)

        # Erstelle Orbital-Elemente-Tensor
        orbital_elements = torch.stack([
            semi_major_axis,
            eccentricity,
            inclination,
            raan,
            arg_perigee,
            mean_anomaly
        ], dim=-1)

        return orbital_elements

    @property
    def satellite_names(self) -> List[str]:
        """Namen der Satelliten."""
        return self["meta", "satellite_names"]

    @property
    def catalog_numbers(self) -> List[int]:
        """Katalog-Nummern der Satelliten."""
        return self["meta", "catalog_numbers"]

    def propagate_sgp4(self, time_minutes: torch.Tensor) -> EarthSatelliteTensorDict:
        """
        Propagiert Satelliten-Orbits mit vereinfachtem SGP4.

        Args:
            time_minutes: [N] Zeit in Minuten seit Epoche

        Returns:
            Neuer EarthSatelliteTensorDict mit propagierten Zuständen
        """
        # Vereinfachte SGP4-Implementierung
        tle_data = self["tle_data"]
        mean_motion = tle_data[..., 5]
        mean_anomaly = tle_data[..., 4]

        # Neue mittlere Anomalie
        delta_M = mean_motion * 2 * math.pi * time_minutes / 1440  # Convert to radians
        new_mean_anomaly = torch.fmod(mean_anomaly + delta_M, 2 * math.pi)

        # Aktualisiere TLE-Daten
        new_tle_data = tle_data.clone()
        new_tle_data[..., 4] = new_mean_anomaly
        new_tle_data[..., 7] += time_minutes / 1440  # Update epoch

        return EarthSatelliteTensorDict(
            tle_data=new_tle_data,
            satellite_names=self.satellite_names,
            catalog_numbers=self.catalog_numbers,
            reference_epoch=self["meta", "reference_epoch"]
        )

    def compute_ground_track(self, n_points: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Berechnet Bodenspur der Satelliten.

        Args:
            n_points: Anzahl der Punkte für die Spur

        Returns:
            Tuple von (longitude, latitude) Tensoren
        """
        # Berechne Positionen über eine Periode
        periods = self["orbit"].compute_period()
        time_steps = torch.linspace(0, periods.max(), n_points)

        longitudes = []
        latitudes = []

        for t in time_steps:
            # Propagiere zu diesem Zeitpunkt
            sat_at_time = self.propagate_sgp4(t * 1440)  # Convert to minutes

            # Konvertiere zu geographischen Koordinaten
            state = sat_at_time["state"]
            positions = state[..., :3]  # x, y, z

            # Vereinfachte Konvertierung zu Lat/Lon
            longitude = torch.atan2(positions[..., 1], positions[..., 0]) * 180 / math.pi
            latitude = torch.asin(torch.clamp(
                positions[..., 2] / torch.norm(positions, dim=-1), -1, 1
            )) * 180 / math.pi

            longitudes.append(longitude)
            latitudes.append(latitude)

        return torch.stack(longitudes, dim=-1), torch.stack(latitudes, dim=-1)

    def compute_visibility(self, ground_position: torch.Tensor, 
                          min_elevation: float = 10.0) -> torch.Tensor:
        """
        Berechnet Sichtbarkeit von Bodenstationen.

        Args:
            ground_position: [3] Bodenstation-Position [x, y, z]
            min_elevation: Minimale Elevation in Grad

        Returns:
            [N] Boolean-Tensor für Sichtbarkeit
        """
        # Aktuelle Satelliten-Positionen
        sat_positions = self["state"][..., :3]  # [N, 3]

        # Vektor von Bodenstation zu Satellit
        sat_vectors = sat_positions - ground_position.unsqueeze(0)

        # Berechne Elevation (vereinfacht)
        distances = torch.norm(sat_vectors, dim=-1)
        heights = torch.norm(sat_positions, dim=-1) - 6371.0  # Höhe über Erdoberfläche
        elevations = torch.asin(torch.clamp(heights / distances, -1, 1)) * 180 / math.pi

        return elevations >= min_elevation

    def get_satellite_by_name(self, name: str) -> EarthSatelliteTensorDict:
        """
        Holt einen spezifischen Satelliten nach Name.

        Args:
            name: Name des Satelliten

        Returns:
            EarthSatelliteTensorDict mit einem Satelliten
        """
        if name not in self.satellite_names:
            raise ValueError(f"Satellite '{name}' not found")

        idx = self.satellite_names.index(name)

        # Extrahiere Daten für diesen Satelliten
        return EarthSatelliteTensorDict(
            tle_data=self["tle_data"][idx:idx+1],
            satellite_names=[name],
            catalog_numbers=[self.catalog_numbers[idx]],
            reference_epoch=self["meta", "reference_epoch"]
        )

    def filter_by_altitude(self, min_altitude: float, max_altitude: float) -> EarthSatelliteTensorDict:
        """
        Filtert Satelliten nach Höhe.

        Args:
            min_altitude: Minimale Höhe in km
            max_altitude: Maximale Höhe in km

        Returns:
            Gefilterter EarthSatelliteTensorDict
        """
        # Berechne Höhen aus Semi-Major-Axis
        altitudes = self["orbit"].semi_major_axis - 6371.0  # Erdradius

        mask = (altitudes >= min_altitude) & (altitudes <= max_altitude)
        indices = torch.where(mask)[0]

        if len(indices) == 0:
            raise ValueError(f"No satellites found in altitude range {min_altitude}-{max_altitude} km")

        # Filtere alle Daten
        filtered_names = [self.satellite_names[i] for i in indices]
        filtered_catalog = [self.catalog_numbers[i] for i in indices]

        return EarthSatelliteTensorDict(
            tle_data=self["tle_data"][indices],
            satellite_names=filtered_names,
            catalog_numbers=filtered_catalog,
            reference_epoch=self["meta", "reference_epoch"]
        )
