"""
AstroLab Widget - Main Interactive Widget for Astronomical Data Analysis
======================================================================

Provides a unified interface for astronomical data visualization, analysis,
and interactive exploration using various backends (Open3D, PyVista, Blender).
"""

import logging
from typing import Any, Optional

import torch

from ..tensors.survey import SurveyTensor
from .analysis import AnalysisModule
from .clustering import ClusteringModule
from .graph import GraphModule
from .visualization import VisualizationModule

logger = logging.getLogger(__name__)

# Blender API imports (optional)
try:
    from ..utils.bpy import AstroLabApi, bpy
except ImportError:
    AstroLabApi = None
    bpy = None


class AstroLabWidget:
    """
    Main widget for astronomical data visualization and analysis.
    Provides Blender API attributes and delegates to specialized modules.
    """
    
    def __init__(self, **kwargs: Any):
        """
        Initialize the widget and set up Blender API attributes if available.
        """
        self._setup_blender_api()

        # Initialize specialized modules
        self.graph = GraphModule()
        self.clustering = ClusteringModule()
        self.analysis = AnalysisModule()
        self.visualization = VisualizationModule()
        
        logger.info("AstroLabWidget initialized with all modules.")

    def _setup_blender_api(self):
        """
        Sets up the direct API to Blender (al, ops, data, context, scene).
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
            self.ops = getattr(bpy, 'ops', None)
            self.data = getattr(bpy, 'data', None)
            self.context = getattr(bpy, 'context', None)
            self.scene = getattr(self.context, 'scene', None) if self.context else None
            logger.info("Blender API connected. Access via widget.al, widget.ops, ...")
        except Exception as e:
            self.al = None
            self.ops = None
            self.data = None
            self.context = None
            self.scene = None
            logger.error(f"Failed to connect Blender API: {e}")

    def plot(self, data: Any, plot_type: str = "scatter_3d", backend: str = "auto", max_points: int = 100_000, **config: Any) -> Any:
        """
        Visualize astronomical data using the optimal backend.

        Args:
            data: AstroDataset or SurveyTensor
            plot_type: Type of plot (default: 'scatter_3d')
            backend: 'auto', 'open3d', 'pyvista', or 'blender'
            max_points: Maximum number of points to visualize
            config: Additional backend-specific config

        Returns:
            The visualization object from the chosen backend.
        """
        logger.info(f"Plotting with backend: {backend}")
        
        # Extract SurveyTensor
        survey_tensor = self._extract_survey_tensor(data)
        
        # Subsample if needed
        if len(survey_tensor.data) > max_points:
            logger.warning(f"Subsampling {len(survey_tensor.data)} to {max_points} points for performance.")
            indices = torch.randperm(len(survey_tensor.data))[:max_points]
            survey_tensor = survey_tensor.apply_mask(indices)
        
        # Select backend
        if backend == "auto":
            backend = self.visualization.select_backend(survey_tensor)
        
        # Delegate to visualization module
        if backend == "open3d":
            return self.visualization.plot_to_open3d(survey_tensor, plot_type, **config)
        elif backend == "pyvista":
            return self.visualization.plot_to_pyvista(survey_tensor, plot_type, **config)
        elif backend == "blender":
            return self.visualization.plot_to_blender(survey_tensor, plot_type, **config)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    def show(self, *args, **kwargs):
        """
        Show the last created interactive visualization.
        """
        self.visualization.show(*args, **kwargs)

    # Graph methods (delegated to GraphModule)
    def create_graph(self, data: Any, k: int = 10, radius: Optional[float] = None, use_gpu: bool = True):
        """Create PyTorch Geometric Data object for model training."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.create_graph(survey_tensor, k, radius, use_gpu)

    def find_neighbors(self, data: Any, k: int = 10, radius: Optional[float] = None, use_gpu: bool = True):
        """GPU-accelerated neighbor finding."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.find_neighbors(survey_tensor, k, radius, use_gpu)

    def prepare_for_model(self, data: Any, model_type: str = "gnn", **kwargs):
        """Prepare data for specific model types."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.prepare_for_model(survey_tensor, model_type, **kwargs)

    def get_model_input_features(self, data: Any):
        """Extract all available features for model input."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.graph.get_model_input_features(survey_tensor)

    # Clustering methods (delegated to ClusteringModule)
    def cluster_data(self, data: Any, eps: float = 10.0, min_samples: int = 5, algorithm: str = "dbscan", use_gpu: bool = True):
        """GPU-accelerated clustering analysis."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.clustering.cluster_data(survey_tensor, eps, min_samples, algorithm, use_gpu)

    # Analysis methods (delegated to AnalysisModule)
    def analyze_density(self, data: Any, radius: float = 5.0, use_gpu: bool = True):
        """GPU-accelerated local density analysis."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.analysis.analyze_density(survey_tensor, radius, use_gpu)

    def analyze_structure(self, data: Any, k: int = 10, use_gpu: bool = True):
        """GPU-accelerated structure analysis."""
        survey_tensor = self._extract_survey_tensor(data)
        return self.analysis.analyze_structure(survey_tensor, k, use_gpu)

    def _extract_survey_tensor(self, data: Any) -> SurveyTensor:
        """
        Extracts or creates a SurveyTensor from various input data types.
        """
        from ..data.core import AstroDataset
        
        if isinstance(data, SurveyTensor):
            return data
        
        if isinstance(data, AstroDataset):
            if not data:
                raise ValueError("Cannot process an empty AstroDataset.")
            
            pyg_data = data[0]
            survey_name = getattr(data, 'survey', 'unknown')
            
            # Check if pyg_data has attribute x
            if hasattr(pyg_data, 'x'):
                logger.info(f"Creating SurveyTensor for '{survey_name}' from AstroDataset.")
                return SurveyTensor(data=pyg_data.x, survey_name=survey_name)
            else:
                raise AttributeError("AstroDataset[0] has no attribute 'x'.")
        
        raise TypeError(f"Unsupported data type: {type(data).__name__}. Please provide an AstroDataset or SurveyTensor.")
