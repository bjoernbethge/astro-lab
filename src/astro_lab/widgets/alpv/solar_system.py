"""
Solar System Visualization with PyVista
=====================================

Astronomically accurate solar system visualization using PyVista's planet functions.
"""

import logging

import numpy as np
import pyvista as pv
from pyvista import examples

logger = logging.getLogger(__name__)


class SolarSystemVisualizer:
    """
    Solar system visualizer using PyVista's planet functions.

    Provides astronomically accurate visualization of the solar system
    with proper relative sizes, distances, and textures.
    """

    def __init__(self, scale_factor: float = 1.0, show_orbits: bool = True):
        """
        Initialize solar system visualizer.

        Args:
            scale_factor: Scale factor for distances (1.0 = realistic, smaller = compressed)
            show_orbits: Whether to show planetary orbits
        """
        self.scale_factor = scale_factor
        self.show_orbits = show_orbits
        self.plotter = None
        self.planets = {}

        # Astronomical units in meters
        self.AU = 149597870700.0

        # Planet data (distance from sun in AU, radius in km)
        self.planet_data = {
            "sun": {"distance": 0.0, "radius": 696340.0, "color": "yellow"},
            "mercury": {"distance": 0.387, "radius": 2439.7, "color": "gray"},
            "venus": {"distance": 0.723, "radius": 6051.8, "color": "orange"},
            "earth": {"distance": 1.0, "radius": 6371.0, "color": "blue"},
            "mars": {"distance": 1.524, "radius": 3389.5, "color": "red"},
            "jupiter": {"distance": 5.203, "radius": 69911.0, "color": "brown"},
            "saturn": {"distance": 9.537, "radius": 58232.0, "color": "gold"},
            "uranus": {"distance": 19.191, "radius": 25362.0, "color": "cyan"},
            "neptune": {"distance": 30.069, "radius": 24622.0, "color": "blue"},
        }

    def create_solar_system_scene(self, **kwargs) -> pv.Plotter:
        """
        Create a complete solar system scene.

        Args:
            **kwargs: Additional plotter parameters

        Returns:
            PyVista plotter with solar system
        """
        # Create plotter
        self.plotter = pv.Plotter(**kwargs)
        self.plotter.set_background("black")

        # Add stars background
        try:
            stars = examples.planets.download_stars_sky_background()
            self.plotter.add_mesh(stars, name="stars")
        except Exception as e:
            logger.warning(f"Could not load stars background: {e}")

        # Add planets
        for planet_name in self.planet_data.keys():
            self._add_planet(planet_name)

        # Add orbits if requested
        if self.show_orbits:
            self._add_orbits()

        # Add coordinate axes
        self.plotter.add_axes(
            xlabel="X (AU)",
            ylabel="Y (AU)",
            zlabel="Z (AU)",
            line_width=2,
            labels_off=False,
            color="white",
        )

        return self.plotter

    def _add_planet(self, planet_name: str) -> None:
        """Add a planet to the scene."""
        try:
            # Load planet using PyVista's planet functions
            if planet_name == "sun":
                planet = examples.planets.load_sun(radius=0.1)  # Scaled down
            elif planet_name == "earth":
                planet = examples.planets.load_earth(radius=0.05)
            elif planet_name == "mars":
                planet = examples.planets.load_mars(radius=0.03)
            elif planet_name == "jupiter":
                planet = examples.planets.load_jupiter(radius=0.2)
            elif planet_name == "saturn":
                planet = examples.planets.load_saturn(radius=0.15)
                # Add Saturn's rings
                rings = examples.planets.load_saturn_rings()
                rings.translate([0, 0, 0])
                self.plotter.add_mesh(rings, name="saturn_rings")
            elif planet_name == "uranus":
                planet = examples.planets.load_uranus(radius=0.08)
            elif planet_name == "neptune":
                planet = examples.planets.load_neptune(radius=0.08)
            elif planet_name == "mercury":
                planet = examples.planets.load_mercury(radius=0.02)
            elif planet_name == "venus":
                planet = examples.planets.load_venus(radius=0.04)
            else:
                # Fallback: create simple sphere
                radius = (
                    self.planet_data[planet_name]["radius"] / 1000000.0
                )  # Scale down
                planet = pv.Sphere(radius=radius)
                planet.point_data["colors"] = [
                    self.planet_data[planet_name]["color"]
                ] * len(planet.points)

            # Position planet
            distance = self.planet_data[planet_name]["distance"] * self.scale_factor
            planet.translate([distance, 0, 0])

            # Store planet reference
            self.planets[planet_name] = planet

            # Add to plotter
            self.plotter.add_mesh(planet, name=planet_name)

            # Add planet label
            self.plotter.add_point_labels(
                [distance, 0, 0.1],
                [planet_name.title()],
                font_size=12,
                text_color="white",
            )

        except Exception as e:
            logger.error(f"Error adding planet {planet_name}: {e}")

    def _add_orbits(self) -> None:
        """Add planetary orbits to the scene."""
        for planet_name, data in self.planet_data.items():
            if planet_name == "sun":
                continue

            distance = data["distance"] * self.scale_factor
            radius = distance

            # Create orbit circle
            theta = np.linspace(0, 2 * np.pi, 100)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            z = np.zeros_like(theta)

            orbit_points = np.column_stack([x, y, z])
            orbit = pv.lines_from_points(orbit_points, close=True)

            self.plotter.add_mesh(
                orbit,
                color="white",
                line_width=1,
                opacity=0.3,
                name=f"{planet_name}_orbit",
            )

    def create_earth_moon_system(self, **kwargs) -> pv.Plotter:
        """
        Create Earth-Moon system visualization.

        Args:
            **kwargs: Additional plotter parameters

        Returns:
            PyVista plotter with Earth-Moon system
        """
        self.plotter = pv.Plotter(**kwargs)
        self.plotter.set_background("black")

        try:
            # Load Earth and Moon
            earth = examples.planets.load_earth(radius=1.0)
            moon = examples.planets.load_moon(radius=0.27)  # Relative to Earth

            # Position Moon (distance ~384,400 km, scaled down)
            moon.translate([2.0, 0, 0])

            # Add to scene
            self.plotter.add_mesh(earth, name="earth")
            self.plotter.add_mesh(moon, name="moon")

            # Add labels
            self.plotter.add_point_labels(
                [0, 0, 1.5], ["Earth"], font_size=14, text_color="white"
            )
            self.plotter.add_point_labels(
                [2.0, 0, 1.5], ["Moon"], font_size=12, text_color="white"
            )

        except Exception as e:
            logger.error(f"Error creating Earth-Moon system: {e}")

        return self.plotter

    def create_planetary_comparison(self, **kwargs) -> pv.Plotter:
        """
        Create planetary size comparison visualization.

        Args:
            **kwargs: Additional plotter parameters

        Returns:
            PyVista plotter with planetary comparison
        """
        self.plotter = pv.Plotter(**kwargs)
        self.plotter.set_background("black")

        planets_to_show = ["mercury", "venus", "earth", "mars", "jupiter", "saturn"]
        x_positions = np.linspace(-5, 5, len(planets_to_show))

        for i, planet_name in enumerate(planets_to_show):
            try:
                # Load planet with relative sizes
                if planet_name == "earth":
                    planet = examples.planets.load_earth(radius=0.5)
                elif planet_name == "mars":
                    planet = examples.planets.load_mars(radius=0.3)
                elif planet_name == "jupiter":
                    planet = examples.planets.load_jupiter(radius=1.5)
                elif planet_name == "saturn":
                    planet = examples.planets.load_saturn(radius=1.2)
                elif planet_name == "mercury":
                    planet = examples.planets.load_mercury(radius=0.2)
                elif planet_name == "venus":
                    planet = examples.planets.load_venus(radius=0.4)
                else:
                    continue

                # Position planet
                planet.translate([x_positions[i], 0, 0])

                # Add to scene
                self.plotter.add_mesh(planet, name=planet_name)

                # Add label
                self.plotter.add_point_labels(
                    [x_positions[i], 0, 1.0],
                    [planet_name.title()],
                    font_size=12,
                    text_color="white",
                )

            except Exception as e:
                logger.error(f"Error adding planet {planet_name}: {e}")

        return self.plotter

    def show(self, **kwargs) -> None:
        """Show the solar system scene."""
        if self.plotter is None:
            self.create_solar_system_scene()
        self.plotter.show(**kwargs)

    def save_screenshot(self, filename: str, **kwargs) -> None:
        """Save screenshot of the solar system scene."""
        if self.plotter is None:
            self.create_solar_system_scene()
        self.plotter.screenshot(filename, **kwargs)


# Convenience functions
def create_solar_system_scene(scale_factor: float = 1.0, **kwargs) -> pv.Plotter:
    """
    Create solar system scene.

    Args:
        scale_factor: Scale factor for distances
        **kwargs: Additional parameters

    Returns:
        PyVista plotter with solar system
    """
    visualizer = SolarSystemVisualizer(scale_factor=scale_factor)
    return visualizer.create_solar_system_scene(**kwargs)


def create_earth_moon_system(**kwargs) -> pv.Plotter:
    """
    Create Earth-Moon system.

    Args:
        **kwargs: Additional parameters

    Returns:
        PyVista plotter with Earth-Moon system
    """
    visualizer = SolarSystemVisualizer()
    return visualizer.create_earth_moon_system(**kwargs)


def create_planetary_comparison(**kwargs) -> pv.Plotter:
    """
    Create planetary size comparison.

    Args:
        **kwargs: Additional parameters

    Returns:
        PyVista plotter with planetary comparison
    """
    visualizer = SolarSystemVisualizer()
    return visualizer.create_planetary_comparison(**kwargs)


__all__ = [
    "SolarSystemVisualizer",
    "create_solar_system_scene",
    "create_earth_moon_system",
    "create_planetary_comparison",
]
