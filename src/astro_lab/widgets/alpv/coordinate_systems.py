"""
Astronomically correct coordinate system converter for PyVista.

Provides coordinate system conversions with Astropy integration
for accurate astronomical visualization.
"""

from typing import Any, Dict, Optional, Tuple

# Astropy imports for astronomical correctness
import astropy.units as u
import numpy as np
import pyvista as pv
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time
from astropy.visualization import quantity_support

# Enable quantity support
quantity_support()


class AstronomicalCoordinateConverter:
    """
    Astronomically correct coordinate system converter for PyVista.

    Provides coordinate system conversions with Astropy integration
    for accurate astronomical visualization.
    """

    def __init__(self, default_unit: str = "pc"):
        """
        Initialize astronomical coordinate converter.

        Args:
            default_unit: Default distance unit
        """
        self.default_unit = default_unit
        self.supported_systems = ["icrs", "galactic", "ecliptic", "altaz", "cartesian"]

    def convert_coordinates(
        self,
        coordinates: np.ndarray,
        from_system: str,
        to_system: str,
        distance: Optional[np.ndarray] = None,
        time: Optional[Time] = None,
        location: Optional[EarthLocation] = None,
    ) -> np.ndarray:
        """
        Convert between astronomical coordinate systems.

        Args:
            coordinates: Input coordinates [N, 2] or [N, 3]
            from_system: Source coordinate system
            to_system: Target coordinate system
            distance: Distance array (for 3D conversion)
            time: Time for time-dependent conversions
            location: Observer location for alt-az conversion

        Returns:
            Converted coordinates
        """
        if from_system == to_system:
            return coordinates

        # Validate coordinate systems
        if from_system not in self.supported_systems:
            raise ValueError(f"Unsupported source system: {from_system}")
        if to_system not in self.supported_systems:
            raise ValueError(f"Unsupported target system: {to_system}")

        # Convert based on system types
        if from_system == "icrs" and to_system == "galactic":
            return self._icrs_to_galactic(coordinates, distance)
        elif from_system == "galactic" and to_system == "icrs":
            return self._galactic_to_icrs(coordinates, distance)
        elif from_system == "icrs" and to_system == "cartesian":
            return self._icrs_to_cartesian(coordinates, distance)
        elif from_system == "galactic" and to_system == "cartesian":
            return self._galactic_to_cartesian(coordinates, distance)
        elif from_system == "cartesian" and to_system == "icrs":
            return self._cartesian_to_icrs(coordinates)
        elif from_system == "cartesian" and to_system == "galactic":
            return self._cartesian_to_galactic(coordinates)
        elif from_system == "icrs" and to_system == "altaz":
            return self._icrs_to_altaz(coordinates, time, location)
        elif from_system == "altaz" and to_system == "icrs":
            return self._altaz_to_icrs(coordinates, time, location)
        else:
            # For other conversions, return as-is for now
            return coordinates

    def _icrs_to_galactic(
        self, coordinates: np.ndarray, distance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert ICRS to Galactic coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (RA, Dec)
            ra, dec = coordinates[:, 0], coordinates[:, 1]

            # Create SkyCoord objects
            coords = SkyCoord(
                ra=ra * u.Unit("deg"), dec=dec * u.Unit("deg"), frame="icrs"
            )

            # Convert to Galactic
            galactic = coords.galactic

            # Extract l, b
            galactic_galactic_l = galactic.l.deg
            b = galactic.b.deg

            result = np.column_stack([galactic_galactic_l, b])

            # Add distance if provided
            if distance is not None:
                result = np.column_stack([result, distance])

            return result
        else:
            return coordinates

    def _galactic_to_icrs(
        self, coordinates: np.ndarray, distance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert Galactic to ICRS coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (l, b)
            galactic_l, b = coordinates[:, 0], coordinates[:, 1]

            # Create SkyCoord objects
            coords = SkyCoord(
                l=galactic_l * u.Unit("deg"), b=b * u.Unit("deg"), frame="galactic"
            )

            # Convert to ICRS
            icrs = coords.icrs

            # Extract RA, Dec
            ra = icrs.ra.deg
            dec = icrs.dec.deg

            result = np.column_stack([ra, dec])

            # Add distance if provided
            if distance is not None:
                result = np.column_stack([result, distance])

            return result
        else:
            return coordinates

    def _icrs_to_cartesian(
        self, coordinates: np.ndarray, distance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert ICRS to Cartesian coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (RA, Dec)
            ra, dec = coordinates[:, 0], coordinates[:, 1]

            # Convert to radians
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)

            # Convert to unit sphere coordinates
            x = np.cos(dec_rad) * np.cos(ra_rad)
            y = np.cos(dec_rad) * np.sin(ra_rad)
            z = np.sin(dec_rad)

            # Scale by distance if provided
            if distance is not None:
                x *= distance
                y *= distance
                z *= distance

            return np.column_stack([x, y, z])
        else:
            return coordinates

    def _galactic_to_cartesian(
        self, coordinates: np.ndarray, distance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Convert Galactic to Cartesian coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (l, b)
            galactic_l, b = coordinates[:, 0], coordinates[:, 1]

            # Convert to radians
            l_rad = np.radians(galactic_l)
            b_rad = np.radians(b)

            # Convert to unit sphere coordinates
            x = np.cos(b_rad) * np.cos(l_rad)
            y = np.cos(b_rad) * np.sin(l_rad)
            z = np.sin(b_rad)

            # Scale by distance if provided
            if distance is not None:
                x *= distance
                y *= distance
                z *= distance

            return np.column_stack([x, y, z])
        else:
            return coordinates

    def _cartesian_to_icrs(self, coordinates: np.ndarray) -> np.ndarray:
        """Convert Cartesian to ICRS coordinates."""
        if coordinates.shape[1] == 3:
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

            # Calculate distance
            distance = np.sqrt(x**2 + y**2 + z**2)

            # Normalize to unit sphere
            x_norm = x / distance
            y_norm = y / distance
            z_norm = z / distance

            # Convert to spherical coordinates
            ra = np.arctan2(y_norm, x_norm)
            dec = np.arcsin(z_norm)

            # Convert to degrees
            ra_deg = np.degrees(ra)
            dec_deg = np.degrees(dec)

            # Handle negative RA
            ra_deg[ra_deg < 0] += 360

            return np.column_stack([ra_deg, dec_deg, distance])
        else:
            return coordinates

    def _cartesian_to_galactic(self, coordinates: np.ndarray) -> np.ndarray:
        """Convert Cartesian to Galactic coordinates."""
        if coordinates.shape[1] == 3:
            x, y, z = coordinates[:, 0], coordinates[:, 1], coordinates[:, 2]

            # Calculate distance
            distance = np.sqrt(x**2 + y**2 + z**2)

            # Normalize to unit sphere
            x_norm = x / distance
            y_norm = y / distance
            z_norm = z / distance

            # Convert to spherical coordinates
            galactic_l = np.arctan2(y_norm, x_norm)
            b = np.arcsin(z_norm)

            # Convert to degrees
            l_deg = np.degrees(galactic_l)
            b_deg = np.degrees(b)

            # Handle negative l
            l_deg[l_deg < 0] += 360

            return np.column_stack([l_deg, b_deg, distance])
        else:
            return coordinates

    def _icrs_to_altaz(
        self,
        coordinates: np.ndarray,
        time: Optional[Time] = None,
        location: Optional[EarthLocation] = None,
    ) -> np.ndarray:
        """Convert ICRS to Alt-Az coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (RA, Dec)
            ra, dec = coordinates[:, 0], coordinates[:, 1]

            # Create SkyCoord objects
            coords = SkyCoord(
                ra=ra * u.Unit("deg"), dec=dec * u.Unit("deg"), frame="icrs"
            )

            # Default time and location if not provided
            if time is None:
                time = Time.now()
            if location is None:
                location = EarthLocation.of_site("greenwich")

            # Convert to Alt-Az
            altaz = coords.transform_to(AltAz(obstime=time, location=location))

            # Extract alt, az
            alt = altaz.alt.deg
            az = altaz.az.deg

            return np.column_stack([az, alt])
        else:
            return coordinates

    def _altaz_to_icrs(
        self,
        coordinates: np.ndarray,
        time: Optional[Time] = None,
        location: Optional[EarthLocation] = None,
    ) -> np.ndarray:
        """Convert Alt-Az to ICRS coordinates."""
        if coordinates.shape[1] == 2:
            # 2D coordinates (az, alt)
            az, alt = coordinates[:, 0], coordinates[:, 1]

            # Create SkyCoord objects
            coords = SkyCoord(
                az=az * u.Unit("deg"), alt=alt * u.Unit("deg"), frame="altaz"
            )

            # Default time and location if not provided
            if time is None:
                time = Time.now()
            if location is None:
                location = EarthLocation.of_site("greenwich")

            # Convert to ICRS
            icrs = coords.transform_to("icrs")

            # Extract RA, Dec
            ra = icrs.ra.deg
            dec = icrs.dec.deg

            return np.column_stack([ra, dec])
        else:
            return coordinates

    def create_astronomical_grid(
        self,
        coordinate_system: str = "icrs",
        ra_range: Tuple[float, float] = (0, 360),
        dec_range: Tuple[float, float] = (-90, 90),
        resolution: int = 100,
        distance: float = 1.0,
    ) -> np.ndarray:
        """
        Create astronomical coordinate grid.

        Args:
            coordinate_system: Coordinate system
            ra_range: RA range in degrees
            dec_range: Dec range in degrees
            resolution: Grid resolution
            distance: Distance from origin

        Returns:
            Grid coordinates
        """
        # Create RA, Dec grid
        ra = np.linspace(ra_range[0], ra_range[1], resolution)
        dec = np.linspace(dec_range[0], dec_range[1], resolution)

        ra_grid, dec_grid = np.meshgrid(ra, dec)

        # Convert to Cartesian
        coordinates = np.column_stack([ra_grid.flatten(), dec_grid.flatten()])
        cartesian = self.convert_coordinates(
            coordinates,
            coordinate_system,
            "cartesian",
            distance=np.full(len(coordinates), distance),
        )

        return cartesian.reshape(resolution, resolution, 3)

    def create_astronomical_sphere(
        self, radius: float = 1.0, resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create astronomical sphere mesh.

        Args:
            radius: Sphere radius
            resolution: Mesh resolution

        Returns:
            Vertices and faces
        """
        # Create sphere using PyVista
        sphere = pv.Sphere(
            radius=radius, phi_resolution=resolution, theta_resolution=resolution
        )

        return sphere.points, sphere.faces

    def create_astronomical_cylinder(
        self, radius: float = 1.0, height: float = 2.0, resolution: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create astronomical cylinder mesh.

        Args:
            radius: Cylinder radius
            height: Cylinder height
            resolution: Mesh resolution

        Returns:
            Vertices and faces
        """
        # Create cylinder using PyVista
        cylinder = pv.Cylinder(radius=radius, height=height, resolution=resolution)

        return cylinder.points, cylinder.faces

    def get_coordinate_system_info(self, system: str) -> Dict[str, Any]:
        """
        Get information about coordinate system.

        Args:
            system: Coordinate system name

        Returns:
            System information
        """
        info = {
            "icrs": {
                "name": "International Celestial Reference System",
                "axes": ["RA (Right Ascension)", "Dec (Declination)"],
                "units": ["degrees", "degrees"],
                "ranges": [(0, 360), (-90, 90)],
                "description": "Standard astronomical coordinate system",
            },
            "galactic": {
                "name": "Galactic Coordinate System",
                "axes": ["l (Galactic Longitude)", "b (Galactic Latitude)"],
                "units": ["degrees", "degrees"],
                "ranges": [(0, 360), (-90, 90)],
                "description": "Galaxy-centered coordinate system",
            },
            "cartesian": {
                "name": "Cartesian Coordinate System",
                "axes": ["X", "Y", "Z"],
                "units": [self.default_unit, self.default_unit, self.default_unit],
                "ranges": [(-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf)],
                "description": "3D Cartesian coordinates",
            },
            "altaz": {
                "name": "Alt-Az Coordinate System",
                "axes": ["Az (Azimuth)", "Alt (Altitude)"],
                "units": ["degrees", "degrees"],
                "ranges": [(0, 360), (-90, 90)],
                "description": "Observer-centered coordinate system",
            },
        }

        return info.get(
            system, {"name": "Unknown", "description": "Unknown coordinate system"}
        )


__all__ = [
    "AstronomicalCoordinateConverter",
]
