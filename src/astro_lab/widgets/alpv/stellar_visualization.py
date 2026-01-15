"""
Astronomically correct stellar visualization with PyVista.

Provides specialized visualization functions for stellar data with proper
astronomical parameters, HR diagrams, and stellar evolution.
"""

from typing import Any, Dict, Union

# Astropy imports for astronomical correctness
import numpy as np
import polars as pl
import pyvista as pv
from astropy.visualization import quantity_support

# Enable quantity support
quantity_support()


class StellarPyVistaVisualizer:
    """
    Astronomically correct stellar visualization with PyVista.

    Provides specialized visualization functions for stellar data with proper
    astronomical parameters, HR diagrams, and stellar evolution.
    """

    def __init__(self, coordinate_system: str = "icrs", distance_unit: str = "pc"):
        """
        Initialize stellar PyVista visualizer.

        Args:
            coordinate_system: Astronomical coordinate system
            distance_unit: Distance unit
        """
        self.coordinate_system = coordinate_system
        self.distance_unit = distance_unit
        self.plotter = None

    def plot_stellar_data(
        self,
        stellar_data: Union[pl.DataFrame, Dict[str, Any]],
        visualization_type: str = "hr_diagram_3d",
        **kwargs,
    ) -> Any:
        """
        Plot stellar data with astronomical accuracy.

        Args:
            stellar_data: Stellar data
            visualization_type: Type of visualization
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        if visualization_type == "hr_diagram_3d":
            return self.plot_hr_diagram_3d(stellar_data, **kwargs)
        elif visualization_type == "stellar_positions":
            return self.plot_stellar_positions(stellar_data, **kwargs)
        elif visualization_type == "stellar_evolution":
            return self.plot_stellar_evolution(stellar_data, **kwargs)
        elif visualization_type == "binary_system":
            return self.plot_binary_system(stellar_data, **kwargs)
        elif visualization_type == "stellar_cluster":
            return self.plot_stellar_cluster(stellar_data, **kwargs)
        else:
            raise ValueError(f"Unknown visualization type: {visualization_type}")

    def plot_hr_diagram_3d(
        self, stellar_data: Union[pl.DataFrame, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Plot 3D Hertzsprung-Russell diagram with astronomical accuracy.

        Args:
            stellar_data: Stellar data with temperature, luminosity, mass
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        # Convert data to numpy arrays
        if isinstance(stellar_data, pl.DataFrame):
            temperature = stellar_data.get_column("temperature").to_numpy()
            luminosity = stellar_data.get_column("luminosity").to_numpy()
            mass = stellar_data.get_column("mass").to_numpy()
        else:
            temperature = np.array(stellar_data["temperature"])
            luminosity = np.array(stellar_data["luminosity"])
            mass = np.array(stellar_data["mass"])

        # Create 3D coordinates for HR diagram
        # X: log(Temperature) - inverted for standard HR diagram
        # Y: log(Luminosity)
        # Z: Mass
        x = np.log10(temperature)
        y = np.log10(luminosity)
        z = mass

        # Create point cloud
        points = np.column_stack([x, y, z])
        cloud = pv.PolyData(points)

        # Add stellar properties as point data
        cloud.point_data["temperature"] = temperature
        cloud.point_data["luminosity"] = luminosity
        cloud.point_data["mass"] = mass

        # Calculate stellar colors based on temperature
        colors = self._temperature_to_rgb(temperature)
        cloud.point_data["colors"] = colors

        # Calculate stellar sizes based on mass
        sizes = self._mass_to_size(mass)
        cloud.point_data["sizes"] = sizes

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")

        # Add axes
        self.plotter.add_axes(
            xlabel="log(Temperature) [K]",
            ylabel="log(Luminosity) [L☉]",
            zlabel="Mass [M☉]",
            line_width=2,
            labels_off=False,
            color="white",
        )

        # Plot points with stellar colors
        self.plotter.add_points(
            cloud,
            point_size=10,
            render_points_as_spheres=True,
            scalars="temperature",
            cmap="plasma",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Temperature [K]",
                "title_color": "white",
                "label_color": "white",
            },
        )

        # Add main sequence line
        self._add_main_sequence_line()

        # Add stellar type annotations
        self._add_stellar_type_annotations()

        return self.plotter

    def plot_stellar_positions(
        self, stellar_data: Union[pl.DataFrame, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Plot stellar positions in 3D space.

        Args:
            stellar_data: Stellar data with positions
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        # Convert data to numpy arrays
        if isinstance(stellar_data, pl.DataFrame):
            positions = stellar_data.select(["x", "y", "z"]).to_numpy()
            magnitudes = stellar_data.get_column("magnitude").to_numpy()
            temperatures = stellar_data.get_column("temperature").to_numpy()
        else:
            positions = np.array(stellar_data["positions"])
            magnitudes = np.array(stellar_data["magnitudes"])
            temperatures = np.array(stellar_data["temperatures"])

        # Create point cloud
        cloud = pv.PolyData(positions)

        # Add stellar properties
        cloud.point_data["magnitude"] = magnitudes
        cloud.point_data["temperature"] = temperatures

        # Calculate stellar colors and sizes
        colors = self._temperature_to_rgb(temperatures)
        sizes = self._magnitude_to_size(magnitudes)

        cloud.point_data["colors"] = colors
        cloud.point_data["sizes"] = sizes

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")

        # Add axes
        self.plotter.add_axes(
            xlabel=f"X ({self.distance_unit})",
            ylabel=f"Y ({self.distance_unit})",
            zlabel=f"Z ({self.distance_unit})",
            line_width=2,
            labels_off=False,
            color="white",
        )

        # Plot stars
        self.plotter.add_points(
            cloud,
            point_size=8,
            render_points_as_spheres=True,
            scalars="temperature",
            cmap="plasma",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Temperature [K]",
                "title_color": "white",
                "label_color": "white",
            },
        )

        return self.plotter

    def plot_stellar_evolution(
        self, stellar_data: Union[pl.DataFrame, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Plot stellar evolution tracks.

        Args:
            stellar_data: Stellar evolution data
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        # Convert data to numpy arrays
        if isinstance(stellar_data, pl.DataFrame):
            time = stellar_data.get_column("time").to_numpy()
            temperature = stellar_data.get_column("temperature").to_numpy()
            luminosity = stellar_data.get_column("luminosity").to_numpy()
            radius = stellar_data.get_column("radius").to_numpy()
        else:
            time = np.array(stellar_data["time"])
            temperature = np.array(stellar_data["temperature"])
            luminosity = np.array(stellar_data["luminosity"])
            radius = np.array(stellar_data["radius"])

        # Create 3D evolution track
        x = np.log10(temperature)
        y = np.log10(luminosity)
        z = time

        # Create line
        points = np.column_stack([x, y, z])
        line = pv.lines_from_points(points)

        # Add evolution properties
        line.point_data["time"] = time
        line.point_data["radius"] = radius

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")

        # Add axes
        self.plotter.add_axes(
            xlabel="log(Temperature) [K]",
            ylabel="log(Luminosity) [L☉]",
            zlabel="Time [Myr]",
            line_width=2,
            labels_off=False,
            color="white",
        )

        # Plot evolution track
        self.plotter.add_mesh(
            line,
            line_width=3,
            scalars="time",
            cmap="viridis",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Time [Myr]",
                "title_color": "white",
                "label_color": "white",
            },
        )

        return self.plotter

    def plot_binary_system(
        self, binary_data: Union[pl.DataFrame, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Plot binary star system.

        Args:
            binary_data: Binary system data
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        # Extract binary parameters
        if isinstance(binary_data, pl.DataFrame):
            primary_mass = binary_data.get_column("primary_mass").to_numpy()[0]
            secondary_mass = binary_data.get_column("secondary_mass").to_numpy()[0]
            separation = binary_data.get_column("separation").to_numpy()[0]
            primary_temp = binary_data.get_column("primary_temperature").to_numpy()[0]
            secondary_temp = binary_data.get_column("secondary_temperature").to_numpy()[
                0
            ]
        else:
            primary_mass = binary_data["primary_mass"]
            secondary_mass = binary_data["secondary_mass"]
            separation = binary_data["separation"]
            primary_temp = binary_data["primary_temperature"]
            secondary_temp = binary_data["secondary_temperature"]

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")

        # Add axes
        self.plotter.add_axes(
            xlabel=f"X ({self.distance_unit})",
            ylabel=f"Y ({self.distance_unit})",
            zlabel=f"Z ({self.distance_unit})",
            line_width=2,
            labels_off=False,
            color="white",
        )

        # Create primary star
        primary_radius = self._mass_to_radius(primary_mass)
        primary_sphere = pv.Sphere(
            radius=primary_radius, center=(-separation / 2, 0, 0)
        )
        primary_color = self._temperature_to_rgb([primary_temp])[0]

        self.plotter.add_mesh(primary_sphere, color=primary_color, show_edges=False)

        # Create secondary star
        secondary_radius = self._mass_to_radius(secondary_mass)
        secondary_sphere = pv.Sphere(
            radius=secondary_radius, center=(separation / 2, 0, 0)
        )
        secondary_color = self._temperature_to_rgb([secondary_temp])[0]

        self.plotter.add_mesh(secondary_sphere, color=secondary_color, show_edges=False)

        # Add orbital path
        orbit_points = self._create_orbital_path(separation)
        orbit_line = pv.lines_from_points(orbit_points)

        self.plotter.add_mesh(orbit_line, color="white", line_width=2, opacity=0.5)

        # Add labels
        self.plotter.add_point_labels(
            [-separation / 2, 0, 0],
            [f"Primary ({primary_mass:.1f}M☉)"],
            color="white",
            font_size=12,
        )

        self.plotter.add_point_labels(
            [separation / 2, 0, 0],
            [f"Secondary ({secondary_mass:.1f}M☉)"],
            color="white",
            font_size=12,
        )

        return self.plotter

    def plot_stellar_cluster(
        self, cluster_data: Union[pl.DataFrame, Dict[str, Any]], **kwargs
    ) -> Any:
        """
        Plot stellar cluster.

        Args:
            cluster_data: Cluster data
            **kwargs: Additional parameters

        Returns:
            PyVista visualization
        """
        # Convert data to numpy arrays
        if isinstance(cluster_data, pl.DataFrame):
            positions = cluster_data.select(["x", "y", "z"]).to_numpy()
            masses = cluster_data.get_column("mass").to_numpy()
            temperatures = cluster_data.get_column("temperature").to_numpy()
        else:
            positions = np.array(cluster_data["positions"])
            masses = np.array(cluster_data["masses"])
            temperatures = np.array(cluster_data["temperatures"])

        # Create point cloud
        cloud = pv.PolyData(positions)

        # Add stellar properties
        cloud.point_data["mass"] = masses
        cloud.point_data["temperature"] = temperatures

        # Calculate stellar colors and sizes
        colors = self._temperature_to_rgb(temperatures)
        sizes = self._mass_to_size(masses)

        cloud.point_data["colors"] = colors
        cloud.point_data["sizes"] = sizes

        # Create plotter
        self.plotter = pv.Plotter()
        self.plotter.set_background("black")

        # Add axes
        self.plotter.add_axes(
            xlabel=f"X ({self.distance_unit})",
            ylabel=f"Y ({self.distance_unit})",
            zlabel=f"Z ({self.distance_unit})",
            line_width=2,
            labels_off=False,
            color="white",
        )

        # Plot cluster stars
        self.plotter.add_points(
            cloud,
            point_size=6,
            render_points_as_spheres=True,
            scalars="temperature",
            cmap="plasma",
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Temperature [K]",
                "title_color": "white",
                "label_color": "white",
            },
        )

        return self.plotter

    def _temperature_to_rgb(self, temperatures: np.ndarray) -> np.ndarray:
        """
        Convert stellar temperatures to RGB colors.

        Args:
            temperatures: Stellar temperatures in Kelvin

        Returns:
            RGB colors [N, 3]
        """
        colors = []

        for temp in temperatures:
            # Clamp to reasonable stellar temperature range
            temp = max(1000, min(temp, 40000))

            # Wien's displacement law approximation
            if temp < 3700:
                # Red stars (M-type)
                color = [1.0, (temp - 1000) / 2700, 0.0]
            elif temp < 5200:
                # Orange stars (K-type)
                color = [
                    1.0,
                    0.8 + (temp - 3700) / 1500 * 0.2,
                    (temp - 3700) / 1500 * 0.3,
                ]
            elif temp < 6000:
                # Yellow stars (G-type)
                color = [
                    1.0,
                    0.9 + (temp - 5200) / 800 * 0.1,
                    0.3 + (temp - 5200) / 800 * 0.4,
                ]
            elif temp < 7500:
                # White stars (F-type)
                color = [
                    0.9 + (temp - 6000) / 1500 * 0.1,
                    0.9 + (temp - 6000) / 1500 * 0.1,
                    0.7 + (temp - 6000) / 1500 * 0.3,
                ]
            elif temp < 10000:
                # Blue-white stars (A-type)
                color = [
                    0.7 + (temp - 7500) / 2500 * 0.3,
                    0.8 + (temp - 7500) / 2500 * 0.2,
                    1.0,
                ]
            else:
                # Blue stars (B/O-type)
                color = [
                    0.5 + (temp - 10000) / 30000 * 0.5,
                    0.6 + (temp - 10000) / 30000 * 0.4,
                    1.0,
                ]

            colors.append(color)

        return np.array(colors)

    def _mass_to_size(self, masses: np.ndarray) -> np.ndarray:
        """
        Convert stellar masses to visualization sizes.

        Args:
            masses: Stellar masses in solar masses

        Returns:
            Sizes for visualization
        """
        # Scale masses to reasonable point sizes
        return np.clip(masses * 5, 2, 20)

    def _mass_to_radius(self, mass: float) -> float:
        """
        Convert stellar mass to radius.

        Args:
            mass: Stellar mass in solar masses

        Returns:
            Stellar radius in solar radii
        """
        # Simplified mass-radius relation
        return mass**0.8

    def _magnitude_to_size(self, magnitudes: np.ndarray) -> np.ndarray:
        """
        Convert apparent magnitudes to visualization sizes.

        Args:
            magnitudes: Apparent magnitudes

        Returns:
            Sizes for visualization
        """
        # Brighter stars (lower magnitude) = larger size
        return np.clip(10 - magnitudes, 1, 15)

    def _add_main_sequence_line(self) -> None:
        """Add main sequence line to HR diagram."""
        # Simplified main sequence
        log_temps = np.linspace(3.5, 4.7, 100)  # log10(Temperature)
        log_lums = -0.5 + 2.5 * (log_temps - 3.8)  # Approximate main sequence

        # Create main sequence line
        ms_points = np.column_stack([log_temps, log_lums, np.ones(100)])
        ms_line = pv.lines_from_points(ms_points)

        self.plotter.add_mesh(ms_line, color="yellow", line_width=2, opacity=0.7)

    def _add_stellar_type_annotations(self) -> None:
        """Add stellar type annotations to HR diagram."""
        # Stellar type positions
        types = {
            "O": [4.7, 5.0],
            "B": [4.3, 3.0],
            "A": [3.9, 1.5],
            "F": [3.7, 0.5],
            "G": [3.6, 0.0],
            "K": [3.5, -0.5],
            "M": [3.4, -1.0],
        }

        for stellar_type, (log_temp, log_lum) in types.items():
            self.plotter.add_point_labels(
                [log_temp, log_lum, 1.0],
                [stellar_type],
                color="white",
                font_size=14,
                bold=True,
            )

    def _create_orbital_path(
        self, separation: float, num_points: int = 100
    ) -> np.ndarray:
        """
        Create orbital path for binary system.

        Args:
            separation: Binary separation
            num_points: Number of points in orbit

        Returns:
            Orbital path points
        """
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = separation / 2 * np.cos(angles)
        y = separation / 2 * np.sin(angles)
        z = np.zeros(num_points)

        return np.column_stack([x, y, z])

    def show(self, **kwargs) -> None:
        """Show the stellar visualization."""
        if self.plotter is not None:
            self.plotter.show(**kwargs)

    def save_screenshot(self, filename: str, **kwargs) -> None:
        """
        Save screenshot of stellar visualization.

        Args:
            filename: Output filename
            **kwargs: Additional parameters
        """
        if self.plotter is not None:
            self.plotter.screenshot(filename, **kwargs)


__all__ = [
    "StellarPyVistaVisualizer",
]
