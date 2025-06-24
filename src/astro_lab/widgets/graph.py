"""
Graph Module - PyTorch Geometric integration and model preparation
================================================================

Provides methods for creating PyTorch Geometric Data objects
and integrating with models from the models module.
"""

import logging
from typing import Any, Optional, Dict

import torch
import torch_cluster
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

from ..tensors.survey import SurveyTensor

logger = logging.getLogger(__name__)


class GraphModule:
    """
    PyTorch Geometric integration and model preparation.
    """
    
    def create_graph(self, survey_tensor: SurveyTensor, k: int = 10, radius: Optional[float] = None, use_gpu: bool = True, **kwargs: Any) -> Data:
        """
        Create PyTorch Geometric Data object for model training.
        
        Args:
            survey_tensor: SurveyTensor with spatial and photometric data
            k: Number of nearest neighbors
            radius: Radius for neighbor search
            use_gpu: Whether to use GPU acceleration
            **kwargs: Additional parameters
            
        Returns:
            PyTorch Geometric Data object
        """
        spatial_tensor = survey_tensor.get_spatial_tensor()
        
        # Get node features (coordinates + photometric data)
        node_features = []
        node_features.append(spatial_tensor.data)  # 3D coordinates
        
        try:
            phot_tensor = survey_tensor.get_photometric_tensor()
            node_features.append(phot_tensor.data)
        except Exception as e:
            logger.warning(f"Could not add photometric features: {e}")
        
        # Combine features
        x = torch.cat(node_features, dim=1)
        
        # Get edge index from neighbor finding
        neighbor_result = self.find_neighbors(survey_tensor, k=k, radius=radius, use_gpu=use_gpu)
        edge_index = neighbor_result["edge_index"]
        
        # Create PyG Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            pos=spatial_tensor.data,  # Node positions for spatial models
        )
        
        logger.info(f"Created graph with {len(x)} nodes and {edge_index.shape[1]} edges")
        return graph_data
    
    def find_neighbors(self, survey_tensor: SurveyTensor, k: int = 10, radius: Optional[float] = None, use_gpu: bool = True) -> Dict[str, torch.Tensor]:
        """
        GPU-accelerated neighbor finding using torch_cluster and torch_geometric.
        
        Args:
            survey_tensor: SurveyTensor with spatial data
            k: Number of nearest neighbors (if radius is None)
            radius: Radius for neighbor search (if provided, overrides k)
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Dictionary with 'edge_index' and 'distances'
        """
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data
        
        device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        coords_device = coords.to(device)
        
        logger.info(f"Finding neighbors on {device} for {len(coords)} points...")
        
        if radius is not None:
            # Radius-based search using PyTorch Geometric
            edge_index = radius_graph(
                x=coords_device,
                r=radius,
                loop=False,
                flow='source_to_target'
            )
        else:
            # k-NN search using torch_cluster
            edge_index = torch_cluster.knn_graph(
                x=coords_device,
                k=k,
                loop=False,
                flow='source_to_target'
            )
        
        # Calculate distances
        distances = torch.norm(
            coords_device[edge_index[0]] - coords_device[edge_index[1]], 
            dim=1
        )
        
        return {
            "edge_index": edge_index.cpu(),
            "distances": distances.cpu()
        }

    def prepare_for_model(self, survey_tensor: SurveyTensor, model_type: str = "gnn", **kwargs: Any) -> Any:
        """
        Prepare data for specific model types from the models module.
        
        Args:
            survey_tensor: SurveyTensor with data
            model_type: Type of model ('gnn', 'point_cloud', 'spatial')
            **kwargs: Additional model-specific parameters
            
        Returns:
            Prepared data for the specified model type
        """
        if model_type == "gnn":
            return self.create_graph(survey_tensor, **kwargs)
        elif model_type == "point_cloud":
            spatial_tensor = survey_tensor.get_spatial_tensor()
            return spatial_tensor.data
        elif model_type == "spatial":
            return survey_tensor
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_model_input_features(self, survey_tensor: SurveyTensor) -> Dict[str, torch.Tensor]:
        """
        Extract all available features for model input.
        
        Args:
            survey_tensor: SurveyTensor with data
            
        Returns:
            Dictionary with different feature types
        """
        features = {}
        
        # Spatial features
        spatial_tensor = survey_tensor.get_spatial_tensor()
        features["spatial"] = spatial_tensor.data
        features["x"] = spatial_tensor.x
        features["y"] = spatial_tensor.y
        features["z"] = spatial_tensor.z
        
        # Photometric features
        try:
            phot_tensor = survey_tensor.get_photometric_tensor()
            features["photometric"] = phot_tensor.data
            features["bands"] = phot_tensor.bands
        except Exception as e:
            logger.warning(f"Could not extract photometric features: {e}")
        
        # Raw survey data
        features["raw"] = survey_tensor.data
        
        return features 