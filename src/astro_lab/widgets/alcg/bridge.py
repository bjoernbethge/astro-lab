"""
Astronomical Cosmograph Bridge
======================================

Deep integration between AstroLab TensorDicts and Cosmograph visualization
with proper astronomical data handling and GPU acceleration.
"""

import colorsys
import logging
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, TypedDict, Union

import numpy as np

# TensorDict imports
from astro_lab.tensors import (
    AnalysisTensorDict,
    PhotometricTensorDict,
    SpatialTensorDict,
)

logger = logging.getLogger(__name__)


# Type definitions matching Cosmograph API
class CosmographNodeData(TypedDict, total=False):
    id: str
    x: float
    y: float
    z: Optional[float]
    color: Optional[str]
    size: Optional[float]
    label: Optional[str]


class CosmographLinkData(TypedDict, total=False):
    source: str
    target: str
    color: Optional[str]
    width: Optional[float]
    value: Optional[float]


@dataclass
class CosmographConfig:
    """Configuration for Cosmograph visualization with astronomical defaults."""

    # Display settings
    width: int = 1000
    height: int = 800
    background_color: str = "#000011"  # Deep space background

    # Node settings
    node_size: Union[float, str] = 4.0
    node_size_range: Optional[List[float]] = None
    node_color: Union[str, Callable] = "#ffffff"
    node_color_by: Optional[str] = None
    node_size_by: Optional[str] = None

    # Link settings
    link_width: float = 1.0
    link_color: str = "#333333"
    link_opacity: float = 0.7
    render_links: bool = True

    # Simulation settings
    simulation_repulsion: float = 0.5
    simulation_gravity: float = 0.0
    simulation_friction: float = 0.85
    simulation_center: float = 0.0
    simulation_decay: int = 1000
    disable_simulation: Optional[bool] = None

    # Space and rendering
    space_size: int = 4096
    fit_view_on_init: bool = True
    fit_view_delay: int = 1000
    fit_view_padding: float = 0.1

    # Interactive features
    enable_zoom: bool = True
    enable_pan: bool = True
    enable_hover: bool = True
    show_fps: bool = False

    # Astronomical-specific settings
    coordinate_system: str = "icrs"
    distance_unit: str = "pc"
    survey_type: str = "unknown"

    def __post_init__(self):
        """Set defaults after initialization."""
        if self.node_size_range is None:
            self.node_size_range = [2.0, 8.0]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Cosmograph."""
        config_dict = asdict(self)

        # Remove astronomical-specific fields that Cosmograph doesn't understand
        astronomical_fields = ["coordinate_system", "distance_unit", "survey_type"]
        for field in astronomical_fields:
            config_dict.pop(field, None)

        return config_dict


class CosmographBridge:
    """
    bridge for AstroLab TensorDict to Cosmograph visualization.

    Features:
    - Automatic survey detection and styling
    - GPU-accelerated graph construction
    - Proper astronomical coordinate handling
    - Interactive analysis integration
    - Real-time clustering visualization
    """

    def __init__(self, use_gpu: bool = True, device: Optional[str] = None):
        """
        Initialize enhanced bridge.

        Args:
            use_gpu: Whether to use GPU acceleration
            device: Specific device to use ('cuda', 'cpu', etc.)
        """
        self.device = device or ("cuda" if use_gpu else "cpu")
        self.use_gpu = use_gpu and self.device == "cuda"

        # Cache for expensive computations
        self._graph_cache = {}
        self._style_cache = {}

        # Survey-specific styling presets
        self.survey_presets = self._initialize_survey_presets()

        logger.info(f"Cosmograph bridge initialized (GPU: {self.use_gpu})")

    def _initialize_survey_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize survey-specific styling presets."""
        return {
            "gaia": {
                "node_color": "#FFD700",  # Gold for stars
                "link_color": "#FFA500",  # Orange links
                "node_size_range": [2, 8],
                "background_color": "#000011",
                "description": "Gaia stellar astrometry",
            },
            "sdss": {
                "node_color": "#4A90E2",  # Blue for galaxies
                "link_color": "#2E5A87",
                "node_size_range": [3, 10],
                "background_color": "#000511",
                "description": "SDSS galaxy survey",
            },
            "nsa": {
                "node_color": "#E24A4A",  # Red for NSA
                "link_color": "#B83838",
                "node_size_range": [4, 12],
                "background_color": "#110505",
                "description": "NASA-Sloan Atlas",
            },
            "tng50": {
                "node_color": "#00FF88",  # Green for simulation
                "link_color": "#00CC66",
                "node_size_range": [2, 6],
                "background_color": "#001105",
                "description": "TNG50 cosmological simulation",
            },
            "exoplanet": {
                "node_color": "#FF00FF",  # Magenta for exoplanets
                "link_color": "#CC00CC",
                "node_size_range": [3, 9],
                "background_color": "#110011",
                "description": "Exoplanet host stars",
            },
            "linear": {
                "node_color": "#FF8800",  # Orange for asteroids
                "link_color": "#CC6600",
                "node_size_range": [1, 4],
                "background_color": "#110500",
                "description": "LINEAR asteroid survey",
            },
            "analysis": {
                "node_color": "#FFFFFF",  # White for analysis results
                "link_color": "#888888",
                "node_size_range": [2, 8],
                "background_color": "#000000",
                "description": "Analysis results",
            },
        }

    def tensordict_to_cosmograph(
        self,
        tensordict: Any,
        survey: str = "unknown",
        config: Optional[CosmographConfig] = None,
        build_graph: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert any TensorDict to Cosmograph format with automatic type detection.

        Args:
            tensordict: Any AstroLab TensorDict
            survey: Survey name for styling
            config: Optional Cosmograph configuration
            build_graph: Whether to build connectivity graph
            **kwargs: Additional parameters

        Returns:
            Dictionary with nodes, links, and config for Cosmograph
        """
        # Determine tensordict type and route accordingly
        if isinstance(tensordict, SpatialTensorDict):
            return self.spatial_tensordict_to_cosmograph(
                tensordict, survey, config, build_graph, **kwargs
            )
        elif isinstance(tensordict, AnalysisTensorDict):
            return self.analysis_tensordict_to_cosmograph(tensordict, config, **kwargs)
        elif isinstance(tensordict, PhotometricTensorDict):
            # Photometric data alone cannot create spatial visualization
            raise ValueError(
                "PhotometricTensorDict requires spatial coordinates for "
                "Cosmograph visualization"
            )
        else:
            # Generic tensordict with coordinates
            if "coordinates" in tensordict:
                # Create temporary SpatialTensorDict
                temp_spatial = SpatialTensorDict(
                    coordinates=tensordict["coordinates"],
                    coordinate_system=tensordict.meta.get(
                        "coordinate_system", "unknown"
                    ),
                    unit=tensordict.meta.get("unit", "pc"),
                )
                return self.spatial_tensordict_to_cosmograph(
                    temp_spatial, survey, config, build_graph, **kwargs
                )
            else:
                raise ValueError(
                    f"Cannot visualize TensorDict type: {type(tensordict)}"
                )

    def spatial_tensordict_to_cosmograph(
        self,
        spatial_tensordict: SpatialTensorDict,
        survey: str = "unknown",
        config: Optional[CosmographConfig] = None,
        build_graph: bool = True,
        max_edges: int = 100000,
        k_neighbors: int = 8,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert SpatialTensorDict to Cosmograph with enhanced features.

        Args:
            spatial_tensordict: SpatialTensorDict with coordinates
            survey: Survey name for styling
            config: Optional configuration
            build_graph: Whether to build neighbor graph
            max_edges: Maximum number of edges for performance
            k_neighbors: Number of neighbors for graph construction
            **kwargs: Additional parameters

        Returns:
            Cosmograph data structure
        """
        start_time = time.time()

        # Extract coordinates and metadata
        coords = spatial_tensordict["coordinates"]
        coordinate_system = spatial_tensordict.meta.get("coordinate_system", "unknown")
        unit = spatial_tensordict.meta.get("unit", "pc")
        n_objects = spatial_tensordict.n_objects

        logger.info(
            f"Converting SpatialTensorDict to Cosmograph: {n_objects} objects "
            f"({coordinate_system}, {unit})"
        )

        # Move to CPU for data export
        coords_cpu = coords.cpu().numpy()

        # Create configuration
        if config is None:
            config = self._create_default_config(survey, n_objects)

        # Apply survey-specific styling
        self._apply_survey_styling(config, survey)

        # Create nodes
        nodes = self._create_nodes_from_coordinates(
            coords_cpu, survey, spatial_tensordict.meta, **kwargs
        )

        # Create links if requested
        links = []
        if build_graph:
            links = self._create_links_from_spatial(
                spatial_tensordict, k_neighbors, max_edges, **kwargs
            )

        # Create final result
        result = {
            "nodes": nodes,
            "links": links,
            "config": config.to_dict(),
            "metadata": {
                "survey": survey,
                "coordinate_system": coordinate_system,
                "unit": unit,
                "n_objects": n_objects,
                "n_links": len(links),
                "processing_time": time.time() - start_time,
            },
        }

        logger.info(
            f"Cosmograph conversion complete: {len(nodes)} nodes, {len(links)} links "
            f"({time.time() - start_time:.2f}s)"
        )
        return result

    def analysis_tensordict_to_cosmograph(
        self,
        analysis_tensordict: AnalysisTensorDict,
        config: Optional[CosmographConfig] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Convert AnalysisTensorDict to Cosmograph with analysis overlays.

        Args:
            analysis_tensordict: AnalysisTensorDict with analysis results
            config: Optional configuration
            **kwargs: Additional parameters

        Returns:
            Cosmograph data with analysis visualization
        """
        # Extract base spatial tensor
        base_tensors = getattr(analysis_tensordict, "base_tensors", {})
        if "spatial" not in base_tensors:
            raise ValueError("AnalysisTensorDict must contain spatial base tensor")

        spatial_tensordict = base_tensors["spatial"]
        coords = spatial_tensordict["coordinates"].cpu().numpy()

        # Get survey information
        survey = analysis_tensordict.get("survey", "analysis")
        coordinate_system = spatial_tensordict.meta.get("coordinate_system", "unknown")

        # Create configuration
        if config is None:
            config = self._create_default_config(survey, len(coords))

        # Extract clustering results for coloring
        clustering_results = analysis_tensordict.get("clustering_results", {})
        cluster_labels = None

        if clustering_results:
            # Use first scale's clustering
            first_scale = list(clustering_results.keys())[0]
            scale_results = clustering_results[first_scale]
            if "labels" in scale_results:
                cluster_labels = scale_results["labels"]
                if hasattr(cluster_labels, "cpu"):
                    cluster_labels = cluster_labels.cpu().numpy()

        # Create nodes with cluster coloring
        nodes = self._create_nodes_with_clustering(
            coords, cluster_labels, survey, analysis_tensordict, **kwargs
        )

        # Create links from analysis or spatial tensor
        links = []
        # Note: SpatialTensorDict doesn't have build_graph method in current implementation
        # Links would need to be created using other methods or external graph building

        # Add analysis metadata to config
        config.survey_type = "analysis"
        if clustering_results:
            config.node_color_by = "cluster"

        return {
            "nodes": nodes,
            "links": links,
            "config": config.to_dict(),
            "metadata": {
                "survey": survey,
                "analysis_type": "cosmic_web",
                "coordinate_system": coordinate_system,
                "n_objects": len(coords),
                "n_clusters": len(set(cluster_labels))
                if cluster_labels is not None
                else 0,
                "clustering_scales": list(clustering_results.keys())
                if clustering_results
                else [],
            },
        }

    def multimodal_tensordict_to_cosmograph(
        self,
        spatial_tensordict: SpatialTensorDict,
        photometric_tensordict: Optional[PhotometricTensorDict] = None,
        analysis_tensordict: Optional[AnalysisTensorDict] = None,
        survey: str = "unknown",
        config: Optional[CosmographConfig] = None,
        color_by: str = "survey",
        size_by: str = "uniform",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create multi-modal Cosmograph visualization from multiple TensorDicts.

        Args:
            spatial_tensordict: Primary spatial data
            photometric_tensordict: Optional photometric data
            analysis_tensordict: Optional analysis results
            survey: Survey name
            config: Optional configuration
            color_by: Coloring strategy ("survey", "magnitude", "color", "cluster")
            size_by: Sizing strategy ("uniform", "magnitude", "distance")
            **kwargs: Additional parameters

        Returns:
            Multi-modal Cosmograph visualization
        """
        coords = spatial_tensordict["coordinates"].cpu().numpy()
        n_objects = len(coords)

        # Create base configuration
        if config is None:
            config = self._create_default_config(survey, n_objects)

        # Create nodes with multi-modal data
        nodes = []
        for i in range(n_objects):
            node = CosmographNodeData(
                id=f"obj_{i}",
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if coords.shape[1] > 2 else 0.0,
            )

            # Add photometric information
            if photometric_tensordict is not None:
                photometric_data = self._extract_photometric_node_data(
                    photometric_tensordict, i, color_by, size_by
                )
                # Update node with photometric data
                for key, value in photometric_data.items():
                    node[key] = value

            # Add analysis information
            if analysis_tensordict is not None:
                analysis_data = self._extract_analysis_node_data(analysis_tensordict, i)
                # Update node with analysis data
                for key, value in analysis_data.items():
                    node[key] = value

            # Apply default styling if no specific data
            if "color" not in node:
                survey_style = self.survey_presets.get(
                    survey, self.survey_presets["analysis"]
                )
                node["color"] = survey_style["node_color"]

            if "size" not in node:
                node["size"] = (
                    config.node_size
                    if isinstance(config.node_size, (int, float))
                    else 4.0
                )

            nodes.append(node)

        # Create links
        links = []
        # Note: SpatialTensorDict doesn't have build_graph method in current implementation
        # Links would need to be created using other methods or external graph building

        return {
            "nodes": nodes,
            "links": links,
            "config": config.to_dict(),
            "metadata": {
                "survey": survey,
                "type": "multimodal",
                "n_objects": n_objects,
                "has_photometry": photometric_tensordict is not None,
                "has_analysis": analysis_tensordict is not None,
                "color_by": color_by,
                "size_by": size_by,
            },
        }

    def _create_default_config(self, survey: str, n_objects: int) -> CosmographConfig:
        """Create default configuration based on survey and object count."""

        # Adjust settings based on dataset size
        if n_objects > 100000:
            # Large dataset - optimize for performance
            config = CosmographConfig(
                simulation_repulsion=0.3,
                simulation_friction=0.9,
                render_links=n_objects
                < 500000,  # Disable links for very large datasets
                fit_view_delay=2000,
                space_size=8192,
            )
        elif n_objects > 10000:
            # Medium dataset - balanced settings
            config = CosmographConfig(
                simulation_repulsion=0.4,
                simulation_friction=0.85,
                render_links=True,
                fit_view_delay=1500,
            )
        else:
            # Small dataset - full features
            config = CosmographConfig(
                simulation_repulsion=0.5,
                simulation_friction=0.8,
                render_links=True,
                fit_view_delay=1000,
            )

        # Set survey-specific defaults
        config.survey_type = survey

        return config

    def _apply_survey_styling(self, config: CosmographConfig, survey: str):
        """Apply survey-specific styling to configuration."""

        if survey in self.survey_presets:
            preset = self.survey_presets[survey]
            config.node_color = preset["node_color"]
            config.link_color = preset["link_color"]
            config.node_size_range = preset["node_size_range"]
            config.background_color = preset["background_color"]

    def _create_nodes_from_coordinates(
        self, coords: np.ndarray, survey: str, metadata: Dict[str, Any], **kwargs
    ) -> List[CosmographNodeData]:
        """Create node list from coordinate array."""

        nodes = []
        survey_style = self.survey_presets.get(survey, self.survey_presets["analysis"])

        for i in range(len(coords)):
            node = CosmographNodeData(
                id=f"obj_{i}",
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if coords.shape[1] > 2 else 0.0,
                color=survey_style["node_color"],
                size=4.0,
                label=f"{survey}_{i}",
            )
            nodes.append(node)

        return nodes

    def _create_nodes_with_clustering(
        self,
        coords: np.ndarray,
        cluster_labels: Optional[np.ndarray],
        survey: str,
        analysis_tensordict: AnalysisTensorDict,
        **kwargs,
    ) -> List[CosmographNodeData]:
        """Create nodes with cluster-based coloring."""

        nodes = []

        # Generate cluster colors
        if cluster_labels is not None:
            unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
            cluster_colors = self._generate_cluster_colors(len(unique_clusters))

            # Create color mapping
            color_map = {}
            for i, cluster_id in enumerate(unique_clusters):
                color_map[cluster_id] = cluster_colors[i]
            color_map[-1] = "#666666"  # Gray for noise

        for i in range(len(coords)):
            # Determine node color
            if cluster_labels is not None:
                cluster_id = cluster_labels[i]
                node_color = color_map.get(cluster_id, "#666666")
            else:
                survey_style = self.survey_presets.get(
                    survey, self.survey_presets["analysis"]
                )
                node_color = survey_style["node_color"]

            node = CosmographNodeData(
                id=f"obj_{i}",
                x=float(coords[i, 0]),
                y=float(coords[i, 1]),
                z=float(coords[i, 2]) if coords.shape[1] > 2 else 0.0,
                color=node_color,
                size=4.0,
                label=f"{survey}_{i}",
            )

            # Add cluster information
            if cluster_labels is not None:
                node["cluster"] = int(cluster_labels[i])

            nodes.append(node)

        return nodes

    def _create_links_from_spatial(
        self,
        spatial_tensordict: SpatialTensorDict,
        k_neighbors: int,
        max_edges: int,
        **kwargs,
    ) -> List[CosmographLinkData]:
        """Create links using SpatialTensorDict capabilities."""

        links = []

        try:
            # Use SpatialTensorDict's graph building if available
            # Note: build_graph method not available in current implementation
            # Links would need to be created using other methods
            pass

            # Limit edges for performance
            if len(links) > max_edges:
                links = links[:max_edges]
                logger.info(f"Limited links to {max_edges} for performance")

        except Exception as e:
            logger.debug(f"Link creation failed: {e}")

        return links

    def _convert_graph_to_links(
        self, graph_data: Any, nodes: Optional[List[CosmographNodeData]] = None
    ) -> List[CosmographLinkData]:
        """Convert PyTorch Geometric graph to Cosmograph links."""

        links = []
        edge_index = graph_data.edge_index.cpu().numpy()

        for i in range(edge_index.shape[1]):
            source_idx = int(edge_index[0, i])
            target_idx = int(edge_index[1, i])

            link = CosmographLinkData(
                source=f"obj_{source_idx}",
                target=f"obj_{target_idx}",
                color="#333333",
                width=1.0,
            )
            links.append(link)

        return links

    def _extract_photometric_node_data(
        self,
        photometric_tensordict: PhotometricTensorDict,
        node_index: int,
        color_by: str,
        size_by: str,
    ) -> Dict[str, Any]:
        """Extract photometric data for a single node."""

        node_data = {}
        magnitudes = photometric_tensordict["magnitudes"]

        # Color based on photometric data
        if color_by == "magnitude" and magnitudes.shape[1] > 0:
            # Use first band magnitude for coloring
            mag = magnitudes[node_index, 0].item()
            node_data["color"] = self._magnitude_to_color(mag)
            node_data["magnitude"] = mag

        elif color_by == "color" and photometric_tensordict.n_bands >= 2:
            # Use color index for coloring
            try:
                colors = photometric_tensordict.compute_colors()
                if colors:
                    color_names = [k for k in colors.keys()]
                    if color_names:
                        color_name = color_names[0]
                        color_value = colors[color_name][node_index].item()
                        node_data["color"] = self._color_index_to_color(color_value)
                        node_data["color_index"] = color_value
            except Exception:
                pass

        # Size based on magnitude
        if size_by == "magnitude" and magnitudes.shape[1] > 0:
            mag = magnitudes[node_index, 0].item()
            # Brighter stars (lower magnitude) are larger
            size = max(2.0, 12.0 - mag)
            node_data["size"] = min(size, 15.0)  # Cap maximum size

        return node_data

    def _extract_analysis_node_data(
        self, analysis_tensordict: AnalysisTensorDict, node_index: int
    ) -> Dict[str, Any]:
        """Extract analysis data for a single node."""

        node_data = {}

        # Extract cluster information
        clustering_results = analysis_tensordict.get("clustering_results", {})
        if clustering_results:
            first_scale = list(clustering_results.keys())[0]
            if "labels" in clustering_results[first_scale]:
                cluster_labels = clustering_results[first_scale]["labels"]
                if hasattr(cluster_labels, "__getitem__"):
                    cluster_id = cluster_labels[node_index]
                    node_data["cluster"] = (
                        int(cluster_id)
                        if hasattr(cluster_id, "item")
                        else int(cluster_id)
                    )

        return node_data

    def _generate_cluster_colors(self, n_clusters: int) -> List[str]:
        """Generate distinct colors for clusters."""

        if n_clusters <= 1:
            return ["#FF0000"]

        # Generate colors using HSV color space for maximum distinction
        colors = []
        for i in range(n_clusters):
            hue = (i * 360 / n_clusters) % 360
            # Use high saturation and moderate value for visibility
            saturation = 80 + (i % 3) * 10  # 80-100% saturation
            value = 70 + (i % 2) * 20  # 70-90% value

            # Convert HSV to RGB
            color_hex = self._hsv_to_hex(hue, saturation, value)
            colors.append(color_hex)

        return colors

    def _hsv_to_hex(self, h: float, s: float, v: float) -> str:
        """Convert HSV to hex color."""
        h_norm = h / 360.0
        s_norm = s / 100.0
        v_norm = v / 100.0

        r, g, b = colorsys.hsv_to_rgb(h_norm, s_norm, v_norm)

        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _magnitude_to_color(self, magnitude: float) -> str:
        """Convert magnitude to color (blue=bright, red=faint)."""

        # Normalize magnitude to 0-1 (assuming range 0-20)
        normalized = np.clip((magnitude - 0) / 20, 0, 1)

        # Blue for bright, red for faint
        red = int(255 * normalized)
        blue = int(255 * (1 - normalized))
        green = int(128 * (1 - abs(0.5 - normalized)))

        return f"#{red:02x}{green:02x}{blue:02x}"

    def _color_index_to_color(self, color_index: float) -> str:
        """Convert color index to RGB color."""

        # Normalize color index (-0.5 to 2.0 typical range)
        normalized = np.clip((color_index + 0.5) / 2.5, 0, 1)

        # Blue for negative (hot), red for positive (cool)
        red = int(255 * normalized)
        blue = int(255 * (1 - normalized))
        green = int(128)

        return f"#{red:02x}{green:02x}{blue:02x}"
