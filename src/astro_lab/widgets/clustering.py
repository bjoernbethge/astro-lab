"""
Clustering Module - GPU-accelerated clustering analysis
======================================================

Provides GPU-accelerated clustering methods using PyTorch Geometric
and torch_cluster with simple CPU fallbacks.
"""

import logging
from typing import Any, Dict

import numpy as np
import torch
import torch_cluster
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import BallTree

from ..tensors import SurveyTensorDict
from ..utils.viz.graph import cluster_and_analyze

logger = logging.getLogger(__name__)


class ClusteringModule:
    """
    GPU-accelerated clustering analysis.
    """

    def cluster_data(
        self,
        survey_tensor: SurveyTensorDict,
        eps: float = 10.0,
        min_samples: int = 5,
        algorithm: str = "dbscan",
        use_gpu: bool = True,
    ) -> Dict[str, Any]:
        """
        GPU-accelerated clustering analysis.

        Args:
            survey_tensor: SurveyTensorDict with spatial data
            eps: Clustering radius
            min_samples: Minimum samples for core points
            algorithm: 'dbscan' or 'hierarchical'
            use_gpu: Whether to use GPU acceleration

        Returns:
            Dictionary with cluster labels, statistics, and analysis results
        """
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data
        return cluster_and_analyze(
            coords,
            algorithm=algorithm,
            eps=eps,
            min_samples=min_samples,
            use_gpu=use_gpu,
        )
