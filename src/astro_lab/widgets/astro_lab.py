"""
AstroLab Widget - Main interface for astronomical data visualization and analysis
===============================================================================

Provides a unified interface to all AstroLab functionality including:
- Data visualization with multiple backends (Open3D, PyVista, Blender)
- Graph creation and analysis
- Clustering and density analysis
- Model preparation and training

## Quick Start

```python
from astro_lab.widgets import AstroLabWidget
from astro_lab.data import load_survey_data

# Load data
data = load_survey_data("gaia")

# Create widget
widget = AstroLabWidget()

# Visualize data
widget.plot(data, backend="open3d")

# Create graph for ML
graph = widget.create_graph(data, k=10)

# Analyze structure
analysis = widget.analyze_structure(data, k=10)
```

## Features

- **Multi-backend visualization**: Open3D, PyVista, Blender
- **GPU acceleration**: CUDA support for large datasets
- **Graph analysis**: PyTorch Geometric integration
- **Clustering**: DBSCAN and other algorithms
- **Blender integration**: 3D visualization and rendering
"""

import logging
from typing import Any, Optional

import torch
from tensordict import TensorDict

from ..tensors import SurveyTensorDict
from .analysis import AnalysisModule
from .clustering import ClusteringModule
from .graph import GraphModule
from .visualization import VisualizationModule

# Try to import Blender API
try:
    import bpy

    from ..utils.bpy import AstroLabApi
except ImportError:
    bpy = None
    AstroLabApi = None

logger = logging.getLogger(__name__)


class AstroLabWidget:
    """
    Main widget for astronomical data visualization and analysis.

    Provides a unified interface to all AstroLab functionality including
    visualization, graph analysis, clustering, and model preparation.
    Delegates to specialized modules for specific functionality.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the widget and set up Blender API attributes if available.

        Args:
            **kwargs: Additional configuration parameters
        """
        self._setup_blender_api()

        # Initialize specialized modules
        self.graph = GraphModule()
        self.clustering = ClusteringModule()
        self.analysis = AnalysisModule()
        self.visualization = VisualizationModule()

        # Set default backend
        self.backend = kwargs.get("backend", "auto")

        logger.info("AstroLabWidget initialized with all modules.")

    def _setup_blender_api(self) -> None:
        """
        Set up direct API access to Blender (al, ops, data, context, scene).

        If Blender is not available, attributes are set to None.
        """
        if bpy is None or AstroLabApi is None:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.warning("Blender not available - API access disabled.")
            return

        try:
            self.al = AstroLabApi()
            self.ops = getattr(bpy, "ops", None)
            self.data = getattr(bpy, "data", None)
            self.context = getattr(bpy, "context", None)
            self.scene = getattr(self.context, "scene", None) if self.context else None
            logger.info("Blender API connected. Access via widget.al, widget.ops, ...")
        except Exception as e:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.error(f"Failed to connect Blender API: {e}")

    def plot(
        self,
        data: Any,
        plot_type: str = "scatter_3d",
        backend: str = "auto",
        max_points: int = 100_000,
        **config: Any,
    ) -> Any:
        """
        Visualize astronomical data using the optimal backend.

        Args:
            data: AstroDataset or SurveyTensor
            plot_type: Type of plot (default: 'scatter_3d')
            backend: 'auto', 'open3d', 'pyvista', or 'blender'
            max_points: Maximum number of points to visualize
            **config: Additional backend-specific configuration

        Returns:
            The visualization object from the chosen backend

        Raises:
            ValueError: If backend is not supported
        """
        logger.info(f"Plotting with backend: {backend}")

        # Extract SurveyTensor
        survey_tensor = self._extract_survey_tensor(data)

        # Subsample if needed
        if len(survey_tensor.data) > max_points:
            logger.warning(
                f"Subsampling {len(survey_tensor.data)} to {max_points} points for performance."
            )
            indices = torch.randperm(len(survey_tensor.data))[:max_points]
            survey_tensor = survey_tensor.apply_mask(indices)

        # Select backend
        if backend == "auto":
            backend = self.visualization.select_backend(survey_tensor)

        # Delegate to visualization module
        if backend == "open3d":
            return self.visualization.plot_to_open3d(survey_tensor, plot_type, **config)
        elif backend == "pyvista":
            return self.visualization.plot_to_pyvista(
                survey_tensor, plot_type, **config
            )
        elif backend == "blender":
            return self.visualization.plot_to_blender(
                survey_tensor, plot_type, **config
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def show(self, *args: Any, **kwargs: Any) -> None:
        """
        Show the last created interactive visualization.

        Args:
            *args: Positional arguments passed to visualization backend
            **kwargs: Keyword arguments passed to visualization backend
        """
        self.visualization.show(*args, **kwargs)

    # Delegate to specialized modules
    def create_graph(
        self,
        data: Any,
        k: int = 10,
        radius: Optional[float] = None,
        use_gpu: bool = True,
    ) -> Any:
        """
        Create PyTorch Geometric Data object for model training.

        Args:
            data: AstroDataset or SurveyTensor
            k: Number of nearest neighbors
            radius: Radius for neighbor search
            use_gpu: Whether to use GPU acceleration

        Returns:
            PyTorch Geometric Data object
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.create_graph(survey_tensor, k, radius, use_gpu)

    def find_neighbors(
        self,
        data: Any,
        k: int = 10,
        radius: Optional[float] = None,
        use_gpu: bool = True,
    ) -> Any:
        """
        GPU-accelerated neighbor finding.

        Args:
            data: AstroDataset or SurveyTensor
            k: Number of nearest neighbors
            radius: Radius for neighbor search
            use_gpu: Whether to use GPU acceleration

        Returns:
            Dictionary with edge_index and distances
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.find_neighbors(survey_tensor, k, radius, use_gpu)

    def prepare_for_model(
        self, data: Any, model_type: str = "gnn", **kwargs: Any
    ) -> Any:
        """
        Prepare data for specific model types.

        Args:
            data: AstroDataset or SurveyTensor
            model_type: Type of model ('gnn', 'point_cloud', 'spatial')
            **kwargs: Additional model-specific parameters

        Returns:
            Prepared data for the specified model type
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.prepare_for_model(survey_tensor, model_type, **kwargs)

    def get_model_input_features(self, data: Any) -> Any:
        """
        Extract all available features for model input.

        Args:
            data: AstroDataset or SurveyTensor

        Returns:
            Dictionary with different feature types
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.get_model_input_features(survey_tensor)

    def cluster_data(
        self,
        data: Any,
        eps: float = 10.0,
        min_samples: int = 5,
        algorithm: str = "dbscan",
        use_gpu: bool = True,
    ) -> Any:
        """
        GPU-accelerated clustering analysis.

        Args:
            data: AstroDataset or SurveyTensor
            eps: Epsilon parameter for DBSCAN
            min_samples: Minimum samples for DBSCAN
            algorithm: Clustering algorithm ('dbscan', 'kmeans')
            use_gpu: Whether to use GPU acceleration

        Returns:
            Clustering results dictionary
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.clustering.cluster_data(
            survey_tensor, eps, min_samples, algorithm, use_gpu
        )

    def analyze_density(
        self, data: Any, radius: float = 5.0, use_gpu: bool = True
    ) -> Any:
        """
        GPU-accelerated local density analysis.

        Args:
            data: AstroDataset or SurveyTensor
            radius: Radius for density calculation
            use_gpu: Whether to use GPU acceleration

        Returns:
            Density analysis results
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.analysis.analyze_density(survey_tensor, radius, use_gpu)

    def analyze_structure(self, data: Any, k: int = 10, use_gpu: bool = True) -> Any:
        """
        GPU-accelerated structure analysis.

        Args:
            data: AstroDataset or SurveyTensor
            k: Number of neighbors for structure analysis
            use_gpu: Whether to use GPU acceleration

        Returns:
            Structure analysis results
        """
        survey_tensor = self._extract_survey_tensor(data)
        return self.analysis.analyze_structure(survey_tensor, k, use_gpu)

    def _extract_survey_tensor(self, data: Any) -> Any:
        """
        Extract or create a SurveyTensorDict from various input data types.

        Args:
            data: Input data (AstroDataset, SurveyTensorDict, etc.)

        Returns:
            SurveyTensorDict object

        Raises:
            ValueError: If data is empty or invalid
            TypeError: If data type is not supported
        """
        from ..data.core import AstroDataset

        if isinstance(data, SurveyTensorDict):
            return data

        if isinstance(data, AstroDataset):
            if not data:
                raise ValueError("Cannot process an empty AstroDataset.")

            pyg_data = data[0]
            survey_name = getattr(data, "survey", "unknown")

            # Check if pyg_data has attribute x
            if hasattr(pyg_data, "x"):
                logger.info(
                    f"Creating SurveyTensorDict for '{survey_name}' from AstroDataset."
                )
                return SurveyTensorDict(
                    data={"features": getattr(pyg_data, "x")}, survey_name=survey_name
                )
            else:
                raise AttributeError("AstroDataset[0] has no attribute 'x'.")

        raise TypeError(
            f"Unsupported data type: {type(data).__name__}. Please provide an AstroDataset or SurveyTensorDict."
        )
