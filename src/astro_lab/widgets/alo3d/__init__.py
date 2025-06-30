"""
Open3D Widgets for AstroLab - Point Cloud and Real-time 3D Visualization
========================================================================

High-performance point cloud visualization and real-time 3D interaction.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import open3d as o3d

# Core components
from .astronomical_visualizer import (
    AstronomicalVisualizer,
    create_astronomical_scene,
    setup_render_options,
)
from .stellar_visualization import (
    StellarVisualizer,
    create_hr_diagram_3d,
    create_stellar_point_cloud,
    visualize_proper_motion,
)
from .tensor_bridge import (
    AstronomicalOpen3DZeroCopyBridge,
    create_open3d_from_tensordict,
)
from .utilities import (
    # Color utilities
    apply_colormap,
    compute_convex_hull,
    compute_point_cloud_distance,
    create_gradient_colors,
    # Clustering utilities
    dbscan_clustering,
    # Geometry utilities
    estimate_normals,
    export_to_ply,
    load_point_cloud,
    normalize_colors,
    remove_statistical_outliers,
    # I/O utilities
    save_point_cloud,
    voxel_downsample,
)

logger = logging.getLogger(__name__)


def create_open3d_visualization(
    tensordict: Any, visualization_type: str = "auto", **kwargs
):
    """
    Create Open3D visualization from TensorDict.

    Args:
        tensordict: AstroLab TensorDict
        visualization_type: Type of visualization ('auto', 'points', 'mesh', 'voxel')
        **kwargs: Open3D-specific parameters

    Returns:
        Open3D geometry or visualizer
    """
    bridge = AstronomicalOpen3DZeroCopyBridge()

    # Auto-detect visualization type
    if visualization_type == "auto":
        if hasattr(tensordict, "n_objects"):
            n_objects = tensordict.n_objects
            if n_objects > 1000000:
                visualization_type = "voxel"  # Use voxels for very large datasets
            elif n_objects > 10000:
                visualization_type = "points"
            else:
                visualization_type = "mesh"  # Can afford mesh for small datasets
        else:
            visualization_type = "points"

    # Route based on TensorDict type
    from astro_lab.tensors import (
        AnalysisTensorDict,
        PhotometricTensorDict,
        SpatialTensorDict,
    )

    if isinstance(tensordict, SpatialTensorDict):
        return bridge.spatial_to_open3d(
            tensordict, visualization_type=visualization_type, **kwargs
        )
    elif isinstance(tensordict, PhotometricTensorDict):
        return bridge.photometric_to_open3d(
            tensordict, visualization_type=visualization_type, **kwargs
        )
    elif isinstance(tensordict, AnalysisTensorDict):
        return bridge.analysis_to_open3d(
            tensordict, visualization_type=visualization_type, **kwargs
        )
    else:
        # Generic coordinate conversion
        coords = (
            tensordict["coordinates"] if "coordinates" in tensordict else tensordict
        )
        return bridge.coordinates_to_open3d(coords, **kwargs)


def create_point_cloud_visualization(
    spatial_tensor: Any, color_by: str = "distance", point_size: float = 1.0, **kwargs
):
    """
    Create optimized point cloud visualization with various coloring schemes.

    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        color_by: Coloring scheme ('distance', 'density', 'cluster', 'height', 'custom')
        point_size: Size of points
        **kwargs: Additional visualization parameters

    Returns:
        Open3D point cloud
    """
    if hasattr(spatial_tensor, "coordinates"):
        coords = spatial_tensor["coordinates"].cpu().numpy()
    else:
        coords = spatial_tensor.cpu().numpy()

    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)

    # Apply coloring scheme
    if color_by == "distance":
        distances = np.linalg.norm(coords, axis=1)
        colors = apply_colormap(distances, colormap="viridis")
    elif color_by == "density":
        # Local density estimation
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        densities = []
        k = kwargs.get("density_neighbors", 20)

        for i in range(len(coords)):
            _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], k)
            if len(idx) > 1:
                distances = np.array(
                    [np.linalg.norm(coords[i] - coords[j]) for j in idx[1:]]
                )
                density = 1.0 / (np.mean(distances) + 1e-8)
            else:
                density = 0.0
            densities.append(density)

        colors = apply_colormap(np.array(densities), colormap="plasma")
    elif color_by == "height":
        heights = coords[:, 2]  # Z-coordinate
        colors = apply_colormap(heights, colormap="coolwarm")
    elif color_by == "cluster" and "cluster_labels" in kwargs:
        labels = kwargs["cluster_labels"]
        colors = create_gradient_colors(labels, colormap="tab10")
    elif color_by == "custom" and "colors" in kwargs:
        colors = kwargs["colors"]
    else:
        # Default: blue gradient based on distance
        distances = np.linalg.norm(coords, axis=1)
        colors = apply_colormap(distances, colormap="blues")

    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Estimate normals for better rendering
    if kwargs.get("compute_normals", True):
        pcd = estimate_normals(pcd, **kwargs)

    # Apply point size (stored as metadata for visualization)
    pcd.point_size = point_size

    return pcd


def create_interactive_viewer(
    geometries: Union[Dict[str, Any], List[Any]],
    window_name: str = "AstroLab Open3D Viewer",
    **kwargs,
):
    """
    Create interactive viewer with controls for multiple geometries.

    Args:
        geometries: Dictionary of {name: geometry} or list of geometries
        window_name: Window title
        **kwargs: Viewer parameters

    Returns:
        Open3D visualizer
    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name=window_name,
        width=kwargs.get("width", 1280),
        height=kwargs.get("height", 720),
    )

    # Add geometries
    if isinstance(geometries, dict):
        for name, geom in geometries.items():
            vis.add_geometry(geom)
    else:
        for geom in geometries:
            vis.add_geometry(geom)

    # Configure rendering
    render_option = vis.get_render_option()
    render_option.point_size = kwargs.get("point_size", 2.0)
    render_option.line_width = kwargs.get("line_width", 1.0)
    render_option.background_color = np.array(kwargs.get("background_color", [0, 0, 0]))
    render_option.show_coordinate_frame = kwargs.get("show_axes", True)

    # Advanced rendering options
    if kwargs.get("use_sun_light", False):
        render_option.light_on = True

    # Add key callbacks for interactivity
    def toggle_normals(vis):
        render_option.point_show_normal = not render_option.point_show_normal
        return False

    def increase_point_size(vis):
        render_option.point_size = min(render_option.point_size + 1.0, 10.0)
        return False

    def decrease_point_size(vis):
        render_option.point_size = max(render_option.point_size - 1.0, 1.0)
        return False

    vis.register_key_callback(ord("N"), toggle_normals)
    vis.register_key_callback(ord("+"), increase_point_size)
    vis.register_key_callback(ord("-"), decrease_point_size)

    return vis


def create_cosmic_web_visualization(
    spatial_tensor: Any, analysis_results: Optional[Dict] = None, **kwargs
):
    """
    Create cosmic web visualization with clusters and filaments.

    Args:
        spatial_tensor: SpatialTensorDict with coordinates
        analysis_results: Optional analysis results
        **kwargs: Visualization parameters

    Returns:
        List of Open3D geometries
    """
    geometries = {}

    # Main point cloud
    pcd = create_point_cloud_visualization(
        spatial_tensor,
        color_by="cluster"
        if analysis_results and "cluster_labels" in analysis_results
        else "density",
        cluster_labels=analysis_results.get("cluster_labels")
        if analysis_results
        else None,
        **kwargs,
    )
    geometries["points"] = pcd

    # Add cluster centers if available
    if analysis_results and "cluster_centers" in analysis_results:
        centers = analysis_results["cluster_centers"]
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(centers)
        center_pcd.paint_uniform_color([1, 0, 0])  # Red centers
        geometries["cluster_centers"] = center_pcd

    # Add filaments if available
    if analysis_results and "filaments" in analysis_results:
        filaments = analysis_results["filaments"]
        lines = o3d.geometry.LineSet()

        # Convert filament data to line set
        points = []
        lines_idx = []

        for filament in filaments:
            start_idx = len(points)
            points.extend(filament["points"])

            # Create line segments
            for i in range(len(filament["points"]) - 1):
                lines_idx.append([start_idx + i, start_idx + i + 1])

        lines.points = o3d.utility.Vector3dVector(np.array(points))
        lines.lines = o3d.utility.Vector2iVector(np.array(lines_idx))
        lines.paint_uniform_color([0, 1, 0])  # Green filaments
        geometries["filaments"] = lines

    # Add bounding box
    if kwargs.get("show_bounds", True):
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (0.5, 0.5, 0.5)
        geometries["bounds"] = bbox

    return geometries


def create_octree_visualization(
    point_cloud: o3d.geometry.PointCloud, max_depth: int = 5, **kwargs
):
    """
    Create octree visualization for hierarchical data structure.

    Args:
        point_cloud: Input point cloud
        max_depth: Maximum octree depth
        **kwargs: Octree parameters

    Returns:
        Octree geometry
    """
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(
        point_cloud, size_expand=kwargs.get("size_expand", 0.01)
    )
    return octree


def create_mesh_from_points(
    point_cloud: o3d.geometry.PointCloud, method: str = "poisson", **kwargs
):
    """
    Create mesh from point cloud using various reconstruction methods.

    Args:
        point_cloud: Input point cloud (must have normals)
        method: Reconstruction method ('poisson', 'ball_pivoting', 'alpha')
        **kwargs: Method-specific parameters

    Returns:
        Triangle mesh
    """
    if not point_cloud.has_normals():
        point_cloud = estimate_normals(point_cloud)

    if method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            point_cloud,
            depth=kwargs.get("depth", 9),
            width=kwargs.get("width", 0),
            scale=kwargs.get("scale", 1.1),
            linear_fit=kwargs.get("linear_fit", False),
        )
    elif method == "ball_pivoting":
        radii = kwargs.get("radii", [0.005, 0.01, 0.02, 0.04])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud, o3d.utility.DoubleVector(radii)
        )
    elif method == "alpha":
        alpha = kwargs.get("alpha", 0.03)
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            point_cloud, alpha
        )
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")

    # Post-process mesh
    if kwargs.get("remove_degenerate", True):
        mesh.remove_degenerate_triangles()
    if kwargs.get("remove_duplicated", True):
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()

    return mesh


def visualize_with_animation(
    geometries: List[Any], trajectory: Optional[np.ndarray] = None, **kwargs
):
    """
    Create animated visualization with camera trajectory.

    Args:
        geometries: List of geometries to visualize
        trajectory: Camera trajectory points
        **kwargs: Animation parameters
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for geom in geometries:
        vis.add_geometry(geom)

    # Setup camera trajectory
    if trajectory is not None:
        ctr = vis.get_view_control()
        for point in trajectory:
            ctr.set_lookat(point)
            vis.poll_events()
            vis.update_renderer()

    return vis


# Convenience functions
def visualize(tensordict: Any, **kwargs):
    """Quick visualization of astronomical data."""
    geom = create_open3d_visualization(tensordict, **kwargs)
    o3d.visualization.draw_geometries(
        [geom] if not isinstance(geom, list) else geom,
        window_name="AstroLab Quick View",
        **kwargs,
    )


def save_visualization(
    geometries: Union[Any, List[Any], Dict[str, Any]], output_path: str, **kwargs
):
    """Save visualization to file."""
    # Convert to list
    if isinstance(geometries, dict):
        geom_list = list(geometries.values())
    elif not isinstance(geometries, list):
        geom_list = [geometries]
    else:
        geom_list = geometries

    # Save based on file extension
    if output_path.endswith(".ply"):
        for i, geom in enumerate(geom_list):
            if isinstance(geom, o3d.geometry.PointCloud):
                o3d.io.write_point_cloud(output_path.replace(".ply", f"_{i}.ply"), geom)
    elif output_path.endswith(".png"):
        # Render to image
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        for geom in geom_list:
            vis.add_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(output_path)
        vis.destroy_window()


__all__ = [
    # Main functions
    "create_open3d_visualization",
    "create_point_cloud_visualization",
    "create_interactive_viewer",
    "create_cosmic_web_visualization",
    # Advanced features
    "create_octree_visualization",
    "create_mesh_from_points",
    "visualize_with_animation",
    # Core components
    "AstronomicalVisualizer",
    "StellarVisualizer",
    "AstronomicalOpen3DZeroCopyBridge",
    # Utilities
    "estimate_normals",
    "compute_convex_hull",
    "remove_statistical_outliers",
    "voxel_downsample",
    "apply_colormap",
    "normalize_colors",
    "create_gradient_colors",
    "dbscan_clustering",
    "compute_point_cloud_distance",
    "save_point_cloud",
    "load_point_cloud",
    "export_to_ply",
    # Convenience
    "visualize",
    "save_visualization",
]
