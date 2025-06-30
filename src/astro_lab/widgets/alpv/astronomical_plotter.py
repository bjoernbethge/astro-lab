"""
Astronomically correct PyVista plotter for astronomical data visualization.

Provides specialized plotting functions for astronomical data with proper
coordinate systems, units, and scientific standards.
"""

from typing import Any, Dict, Optional

# Astropy imports for astronomical correctness
import numpy as np
from astropy.visualization import quantity_support

# Enable quantity support
quantity_support()

# PyVista imports
import pyvista as pv


class AstronomicalPyVistaPlotter:
    """
    Astronomically correct PyVista plotter for scientific visualization.

    Provides specialized plotting functions with proper astronomical
    coordinate systems, units, and scientific standards.
    """

    def __init__(self, coordinate_system: str = "icrs", distance_unit: str = "pc"):
        """
        Initialize astronomical PyVista plotter.

        Args:
            coordinate_system: Astronomical coordinate system
            distance_unit: Distance unit
        """
        self.coordinate_system = coordinate_system
        self.distance_unit = distance_unit
        self.plotter = None

    def create_astronomical_plotter(self, **kwargs) -> Any:
        """
        Create astronomically configured PyVista plotter.

        Args:
            **kwargs: PyVista plotter arguments

        Returns:
            Configured PyVista plotter
        """
        # Default astronomical settings
        default_kwargs = {
            "background": "black",
            "show_axes": True,
            "axes_color": "white",
            "window_size": [1200, 800],
            "anti_aliasing": "fxaa",
        }
        default_kwargs.update(kwargs)

        self.plotter = pv.Plotter(**default_kwargs)

        # Setup astronomical axes
        self._setup_astronomical_axes()

        return self.plotter

    def _setup_astronomical_axes(self) -> None:
        """Setup astronomically correct axes."""
        if self.plotter is None:
            return

        # Add astronomical axes
        self.plotter.add_axes(
            xlabel=f"X ({self.distance_unit})",
            ylabel=f"Y ({self.distance_unit})",
            zlabel=f"Z ({self.distance_unit})",
            line_width=2,
            labels_off=False,
            color="white",
        )

    def plot_astronomical_points(
        self,
        positions: np.ndarray,
        values: Optional[np.ndarray] = None,
        colors: Optional[np.ndarray] = None,
        sizes: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Plot astronomical points with proper styling.

        Args:
            positions: Point positions [N, 3]
            values: Scalar values for coloring
            colors: RGB colors [N, 3]
            sizes: Point sizes
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        # Create point cloud
        cloud = pv.PolyData(positions)

        # Add scalar data
        if values is not None:
            cloud.point_data["values"] = values

        # Add colors
        if colors is not None:
            cloud.point_data["colors"] = colors

        # Add sizes
        if sizes is not None:
            cloud.point_data["sizes"] = sizes

        # Default astronomical styling
        default_kwargs = {
            "point_size": 5,
            "render_points_as_spheres": True,
            "cmap": "viridis",
            "show_scalar_bar": True,
            "scalar_bar_args": {"title": "Astronomical Values"},
        }
        default_kwargs.update(kwargs)

        if self.plotter is None:
            self.create_astronomical_plotter()

        return self.plotter.add_points(cloud, **default_kwargs)

    def plot_astronomical_lines(
        self, points: np.ndarray, values: Optional[np.ndarray] = None, **kwargs
    ) -> Any:
        """
        Plot astronomical lines (e.g., orbits, trajectories).

        Args:
            points: Line points [N, 3]
            values: Scalar values for coloring
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        # Create line
        line = pv.lines_from_points(points)

        # Add scalar data
        if values is not None:
            line.point_data["values"] = values

        # Default astronomical styling
        default_kwargs = {
            "line_width": 3,
            "cmap": "plasma",
            "show_scalar_bar": True,
            "scalar_bar_args": {"title": "Astronomical Values"},
        }
        default_kwargs.update(kwargs)

        if self.plotter is None:
            self.create_astronomical_plotter()

        return self.plotter.add_mesh(line, **default_kwargs)

    def plot_astronomical_surface(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        values: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Plot astronomical surface (e.g., galaxy disk, nebula shell).

        Args:
            vertices: Surface vertices [N, 3]
            faces: Face indices
            values: Scalar values for coloring
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        # Create surface mesh
        mesh = pv.PolyData(vertices, faces)

        # Add scalar data
        if values is not None:
            mesh.point_data["values"] = values

        # Default astronomical styling
        default_kwargs = {
            "cmap": "viridis",
            "show_scalar_bar": True,
            "scalar_bar_args": {"title": "Astronomical Values"},
            "opacity": 0.8,
        }
        default_kwargs.update(kwargs)

        if self.plotter is None:
            self.create_astronomical_plotter()

        return self.plotter.add_mesh(mesh, **default_kwargs)

    def plot_astronomical_volume(
        self, grid: np.ndarray, values: np.ndarray, **kwargs
    ) -> Any:
        """
        Plot astronomical volume data (e.g., nebula density, cosmic web).

        Args:
            grid: 3D grid coordinates
            values: Volume values
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Create volume grid
        volume = pv.StructuredGrid()
        volume.points = grid.reshape(-1, 3)
        volume.dimensions = values.shape

        # Add scalar data
        volume.point_data["values"] = values.flatten()

        # Default astronomical styling
        default_kwargs = {
            "cmap": "plasma",
            "opacity": 0.3,
            "show_scalar_bar": True,
            "scalar_bar_args": {"title": "Astronomical Values"},
        }
        default_kwargs.update(kwargs)

        return self.plotter.add_volume(volume, **default_kwargs)

    def plot_astronomical_vector_field(
        self, positions: np.ndarray, vectors: np.ndarray, **kwargs
    ) -> Any:
        """
        Plot astronomical vector field (e.g., velocity field, magnetic field).

        Args:
            positions: Field positions [N, 3]
            vectors: Vector values [N, 3]
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Create vector field
        field = pv.PolyData(positions)
        field.point_data["vectors"] = vectors

        # Default astronomical styling
        default_kwargs = {"scale": 1.0, "color": "white", "line_width": 2}
        default_kwargs.update(kwargs)

        return self.plotter.add_mesh(field.glyph(orient="vectors"), **default_kwargs)

    def plot_astronomical_sphere(
        self,
        center: np.ndarray,
        radius: float,
        values: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Plot astronomical sphere (e.g., star, planet).

        Args:
            center: Sphere center [3]
            radius: Sphere radius
            values: Scalar values for coloring
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Create sphere
        sphere = pv.Sphere(radius=radius, center=center)

        # Add scalar data
        if values is not None:
            sphere.point_data["values"] = values

        # Default astronomical styling
        default_kwargs = {
            "show_edges": False,
            "cmap": "viridis",
            "show_scalar_bar": True,
            "scalar_bar_args": {"title": "Astronomical Values"},
        }
        default_kwargs.update(kwargs)

        return self.plotter.add_mesh(sphere, **default_kwargs)

    def plot_astronomical_ellipsoid(
        self,
        center: np.ndarray,
        axes: np.ndarray,
        rotation: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Any:
        """
        Plot astronomical ellipsoid (e.g., galaxy, nebula).

        Args:
            center: Ellipsoid center [3]
            axes: Ellipsoid axes [3]
            rotation: Rotation matrix [3, 3]
            **kwargs: Additional plotting parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Create ellipsoid
        ellipsoid = pv.Ellipsoid(
            center=center, x_radius=axes[0], y_radius=axes[1], z_radius=axes[2]
        )

        # Apply rotation if provided
        if rotation is not None:
            ellipsoid.transform(rotation)

        # Default astronomical styling
        default_kwargs = {"show_edges": False, "opacity": 0.7, "color": "white"}
        default_kwargs.update(kwargs)

        return self.plotter.add_mesh(ellipsoid, **default_kwargs)

    def add_astronomical_annotation(
        self, position: np.ndarray, text: str, **kwargs
    ) -> Any:
        """
        Add astronomical annotation.

        Args:
            position: Annotation position [3]
            text: Annotation text
            **kwargs: Additional parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Default astronomical styling
        default_kwargs = {"color": "white", "font_size": 12, "bold": True}
        default_kwargs.update(kwargs)

        return self.plotter.add_point_labels(position, [text], **default_kwargs)

    def add_astronomical_colorbar(
        self, title: str = "Astronomical Values", **kwargs
    ) -> Any:
        """
        Add astronomically styled colorbar.

        Args:
            title: Colorbar title
            **kwargs: Additional parameters

        Returns:
            PyVista actor
        """
        if self.plotter is None:
            self.create_astronomical_plotter()

        # Default astronomical styling
        default_kwargs = {
            "title": title,
            "title_color": "white",
            "label_color": "white",
            "outline_color": "white",
            "position_x": 0.85,
            "position_y": 0.1,
            "width": 0.1,
            "height": 0.8,
        }
        default_kwargs.update(kwargs)

        return self.plotter.add_scalar_bar(**default_kwargs)

    def show(self, **kwargs) -> None:
        """Show the astronomical visualization."""
        if self.plotter is not None:
            self.plotter.show(**kwargs)

    def save_screenshot(self, filename: str, **kwargs) -> None:
        """
        Save screenshot of astronomical visualization.

        Args:
            filename: Output filename
            **kwargs: Additional parameters
        """
        if self.plotter is not None:
            self.plotter.screenshot(filename, **kwargs)

    def get_astronomical_statistics(self) -> Dict[str, Any]:
        """
        Get astronomical statistics from the plotter.

        Returns:
            Statistics dictionary
        """
        stats = {
            "coordinate_system": self.coordinate_system,
            "distance_unit": self.distance_unit,
            "plotter_created": self.plotter is not None,
        }

        return stats


__all__ = [
    "AstronomicalPyVistaPlotter",
]
