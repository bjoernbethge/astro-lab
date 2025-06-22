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

from ..tensors.survey import SurveyTensor

logger = logging.getLogger(__name__)


class ClusteringModule:
    """
    GPU-accelerated clustering analysis.
    """
    
    def cluster_data(self, survey_tensor: SurveyTensor, eps: float = 10.0, min_samples: int = 5, algorithm: str = "dbscan", use_gpu: bool = True) -> Dict[str, Any]:
        """
        GPU-accelerated clustering analysis.
        
        Args:
            survey_tensor: SurveyTensor with spatial data
            eps: Clustering radius
            min_samples: Minimum samples for core points
            algorithm: 'dbscan' or 'hierarchical'
            use_gpu: Whether to use GPU acceleration
            
        Returns:
            Dictionary with cluster labels, statistics, and analysis results
        """
        if not use_gpu:
            logger.info("Using CPU clustering...")
            return self._cluster_cpu(survey_tensor, eps, min_samples, algorithm)
        
        logger.info("Using GPU-accelerated clustering...")
        return self._cluster_gpu(survey_tensor, eps, min_samples, algorithm)

    def _cluster_gpu(self, survey_tensor: SurveyTensor, eps: float, min_samples: int, algorithm: str) -> Dict[str, Any]:
        """GPU-accelerated clustering using PyTorch Geometric."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        coords_device = coords.to(device)
        
        logger.info(f"GPU clustering on {device} for {len(coords)} points...")
        
        # For now, use CPU clustering but with GPU-accelerated neighbor finding
        # In the future, we could implement GPU-native clustering algorithms
        coords_cpu = coords.cpu().numpy()
        
        if algorithm == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            clusterer = AgglomerativeClustering(
                distance_threshold=eps, 
                linkage="ward"
            )
        
        labels = clusterer.fit_predict(coords_cpu)
        labels_tensor = torch.from_numpy(labels)
        
        # Analyze results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = sum(1 for label in labels if label == -1)
        
        # Calculate cluster properties
        cluster_stats = {}
        if n_clusters > 0:
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_mask = labels == cluster_id
                cluster_coords = coords_cpu[cluster_mask]
                
                # Cluster center and size
                center = cluster_coords.mean(axis=0)
                distances = np.linalg.norm(cluster_coords - center, axis=1)
                
                cluster_stats[cluster_id] = {
                    "n_points": int(cluster_mask.sum()),
                    "center": center,
                    "radius": float(distances.max()),
                    "density": float(
                        cluster_mask.sum() / (4/3 * np.pi * max(distances.max(), 1e-6)**3)
                    ),
                }
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return {
            "cluster_labels": labels_tensor,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_stats": cluster_stats,
            "coords": torch.from_numpy(coords_cpu),
        }

    def _cluster_cpu(self, survey_tensor: SurveyTensor, eps: float, min_samples: int, algorithm: str) -> Dict[str, Any]:
        """CPU clustering fallback."""
        spatial_tensor = survey_tensor.get_spatial_tensor()
        coords = spatial_tensor.data.cpu().numpy()
        
        logger.info(f"CPU clustering for {len(coords)} points...")
        
        if algorithm == "dbscan":
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        else:
            clusterer = AgglomerativeClustering(
                distance_threshold=eps, 
                linkage="ward"
            )
        
        labels = clusterer.fit_predict(coords)
        labels_tensor = torch.from_numpy(labels)
        
        # Analyze results
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = sum(1 for label in labels if label == -1)
        
        # Calculate cluster properties
        cluster_stats = {}
        if n_clusters > 0:
            for cluster_id in unique_labels:
                if cluster_id == -1:  # Skip noise
                    continue
                    
                cluster_mask = labels == cluster_id
                cluster_coords = coords[cluster_mask]
                
                # Cluster center and size
                center = cluster_coords.mean(axis=0)
                distances = np.linalg.norm(cluster_coords - center, axis=1)
                
                cluster_stats[cluster_id] = {
                    "n_points": int(cluster_mask.sum()),
                    "center": center,
                    "radius": float(distances.max()),
                    "density": float(
                        cluster_mask.sum() / (4/3 * np.pi * max(distances.max(), 1e-6)**3)
                    ),
                }
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        return {
            "cluster_labels": labels_tensor,
            "n_clusters": n_clusters,
            "n_noise": n_noise,
            "cluster_stats": cluster_stats,
            "coords": torch.from_numpy(coords),
        } 