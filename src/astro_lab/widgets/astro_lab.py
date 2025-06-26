"""
AstroLab Widget - Main interface for astronomical data visualization and analysis
===============================================================================

Thin wrapper that delegates to specialized modules:
- astro_lab.data.graphs for graph creation
- Local widget modules for visualization
- astro_lab.data for data processing
"""

import logging
from contextlib import contextmanager
from typing import Any, Optional

import torch
from torch_geometric.data import Data

from astro_lab.data.graphs import (
    create_astronomical_graph,
    create_knn_graph,
    create_radius_graph,
)
from astro_lab.tensors import SurveyTensorDict

# Use local widget modules
from .graph import cluster_and_analyze
from .plotly_bridge import create_plotly_visualization

# Try to import Blender API
try:
    from . import albpy
    from .albpy import AstroLabApi, blender_memory_context, bpy_object_context
except ImportError:
    albpy = None
    AstroLabApi = None
    blender_memory_context = None
    bpy_object_context = None

logger = logging.getLogger(__name__)


@contextmanager
def memory_management():
    """Context manager for proper memory management."""
    try:
        yield
    finally:
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Force garbage collection
        import gc

        gc.collect()


class AstroLabWidget:
    """
    Main widget for astronomical data visualization and analysis.

    Uses local visualization modules that were moved from utils.viz.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the widget."""
        self._setup_blender_api()
        self.backend = kwargs.get("backend", "auto")
        logger.info("AstroLabWidget initialized with local visualization modules.")

    def _setup_blender_api(self) -> None:
        """Set up Blender API access if available."""
        if albpy is None or AstroLabApi is None:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.warning("Blender (albpy) not available - API access disabled.")
            return

        try:
            with memory_management():
                self.al = AstroLabApi()
                self.ops = getattr(albpy, "ops", None)
                self.data = getattr(albpy, "data", None)
                self.context = getattr(albpy, "context", None)
                self.scene = (
                    getattr(self.context, "scene", None) if self.context else None
                )
                logger.info("Blender (albpy) API connected.")
        except Exception as e:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.error(f"Failed to connect Blender (albpy) API: {e}")

    def plot(
        self,
        data: Any,
        plot_type: str = "scatter_3d",
        backend: str = "auto",
        max_points: int = 100_000,
        **config: Any,
    ) -> Any:
        """
        Visualize astronomical data using local visualization modules.

        Args:
            data: SurveyTensorDict or Data object
            plot_type: Type of plot
            backend: Visualization backend ('auto', 'plotly')
            max_points: Maximum points to plot
            **config: Additional configuration

        Returns:
            Visualization object
        """
        with memory_management():
            survey_tensor = self._extract_survey_tensor(data)

            if survey_tensor is None:
                raise ValueError("Failed to extract survey tensor from data")

            # Use local plotly_bridge module
            if backend in ["auto", "plotly"]:
                return create_plotly_visualization(
                    survey_tensor, plot_type=plot_type, **config
                )
            else:
                # For other backends, just log for now
                coords = survey_tensor["spatial"]["coordinates"]
                logger.info(
                    f"Would visualize {len(coords)} points with {backend} backend"
                )
                return survey_tensor

    def create_graph(
        self,
        data: Any,
        method: str = "knn",
        k: int = 10,
        radius: Optional[float] = None,
        **kwargs: Any,
    ) -> Data:
        """
        Create PyTorch Geometric graph - delegates to data.graphs module.

        Args:
            data: SurveyTensorDict or compatible data
            method: Graph creation method ('knn', 'radius', 'astronomical')
            k: Number of neighbors for KNN
            radius: Radius for radius graphs
            **kwargs: Additional parameters

        Returns:
            PyTorch Geometric Data object
        """
        with memory_management():
            survey_tensor = self._extract_survey_tensor(data)

            if survey_tensor is None:
                raise ValueError("Failed to extract survey tensor from data")

            # Delegate to data.graphs module
            if method == "knn":
                return create_knn_graph(survey_tensor, k_neighbors=k, **kwargs)
            elif method == "radius":
                if radius is None:
                    raise ValueError("Radius must be specified for radius graphs")
                return create_radius_graph(survey_tensor, radius=radius, **kwargs)
            elif method == "astronomical":
                return create_astronomical_graph(survey_tensor, k_neighbors=k, **kwargs)
            else:
                raise ValueError(f"Unknown graph method: {method}")

    def cluster_data(
        self,
        data: Any,
        algorithm: str = "dbscan",
        eps: float = 10.0,
        min_samples: int = 5,
        **kwargs: Any,
    ) -> Any:
        """
        Cluster data using local graph module.

        Args:
            data: SurveyTensorDict or compatible data
            algorithm: Clustering algorithm
            eps: DBSCAN epsilon parameter
            min_samples: DBSCAN minimum samples
            **kwargs: Additional parameters

        Returns:
            Clustering results
        """
        with memory_management():
            survey_tensor = self._extract_survey_tensor(data)

            if survey_tensor is None:
                raise ValueError("Failed to extract survey tensor from data")

            coords = survey_tensor["spatial"]["coordinates"]

            # Use local graph module
            return cluster_and_analyze(
                coords, algorithm=algorithm, eps=eps, min_samples=min_samples, **kwargs
            )

    def _extract_survey_tensor(self, data: Any) -> SurveyTensorDict:
        """Extract SurveyTensorDict from various input formats."""
        if isinstance(data, SurveyTensorDict):
            return data
        elif isinstance(data, Data):
            # Convert PyG Data to SurveyTensorDict
            from astro_lab.tensors.factories import create_generic_survey

            if hasattr(data, "pos") and data.pos is not None:
                coords = data.pos
            elif hasattr(data, "x") and data.x.shape[-1] >= 3:
                coords = data.x[:, :3]  # Use first 3 dimensions as coordinates
            else:
                raise ValueError("Cannot extract coordinates from Data object")

            # Create dummy magnitudes if not available
            magnitudes = torch.zeros(coords.shape[0], 1)

            return create_generic_survey(
                coordinates=coords,
                magnitudes=magnitudes,
                bands=["dummy"],
                survey_name="converted",
            )
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
