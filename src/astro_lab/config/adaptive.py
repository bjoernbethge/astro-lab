"""
Adaptive Configuration System
=============================

Automatically determines optimal configuration based on hardware and data.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import psutil
from pathlib import Path

logger = logging.getLogger(__name__)


class AdaptiveConfig:
    """
    Automatically adapts configuration to hardware and dataset characteristics.
    
    Features:
    - Hardware detection (GPU memory, CPU cores)
    - Dataset size analysis
    - Optimal hyperparameter selection
    - Memory-efficient settings
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize adaptive configuration system."""
        self.hardware_info = self._detect_hardware()
        self.dataset_info = {}
        
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect hardware capabilities."""
        info = {
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total / 1e9,
            "has_cuda": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_compute_capability": torch.cuda.get_device_capability(0),
                "cuda_version": torch.version.cuda,
            })
        
        logger.info(f"Detected hardware: {info}")
        return info
    
    def analyze_dataset(
        self, 
        num_nodes: int, 
        num_edges: int, 
        num_features: int,
        num_graphs: int = 1,
        avg_degree: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze dataset characteristics.
        
        Args:
            num_nodes: Total number of nodes
            num_edges: Total number of edges
            num_features: Number of node features
            num_graphs: Number of graphs in dataset
            avg_degree: Average node degree
            
        Returns:
            Dataset analysis results
        """
        self.dataset_info = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_features": num_features,
            "num_graphs": num_graphs,
            "avg_degree": avg_degree or (2 * num_edges / num_nodes),
            "density": 2 * num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            "size_category": self._categorize_dataset_size(num_nodes, num_edges),
        }
        
        logger.info(f"Dataset analysis: {self.dataset_info}")
        return self.dataset_info
    
    def _categorize_dataset_size(self, num_nodes: int, num_edges: int) -> str:
        """Categorize dataset size."""
        if num_nodes < 10_000:
            return "small"
        elif num_nodes < 100_000:
            return "medium"
        elif num_nodes < 1_000_000:
            return "large"
        elif num_nodes < 10_000_000:
            return "xlarge"
        else:
            return "xxlarge"
    
    def get_optimal_config(
        self,
        task: str = "node_classification",
        model_type: str = "gnn",
        survey: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get optimal configuration based on hardware and dataset.
        
        Args:
            task: Task type
            model_type: Model type
            survey: Survey name for specific optimizations
            
        Returns:
            Optimal configuration dictionary
        """
        config = {}
        
        # Hardware-based configuration
        hardware_config = self._get_hardware_config()
        config.update(hardware_config)
        
        # Dataset-based configuration
        if self.dataset_info:
            dataset_config = self._get_dataset_config()
            config.update(dataset_config)
        
        # Task-specific configuration
        task_config = self._get_task_config(task)
        config.update(task_config)
        
        # Model-specific configuration
        model_config = self._get_model_config(model_type)
        config.update(model_config)
        
        # Survey-specific optimizations
        if survey:
            survey_config = self._get_survey_config(survey)
            config.update(survey_config)
        
        # Validation and adjustments
        config = self._validate_and_adjust_config(config)
        
        logger.info(f"Generated optimal config: {config}")
        return config
    
    def _get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware-based configuration."""
        config = {}
        
        # CPU configuration
        cpu_count = self.hardware_info["cpu_count"]
        config["num_workers"] = min(cpu_count, 8)  # Cap at 8 for stability
        
        # GPU configuration
        if self.hardware_info["has_cuda"]:
            gpu_memory = self.hardware_info["gpu_memory_gb"]
            
            # Batch size based on GPU memory (optimized for RTX 4070 and similar)
            if gpu_memory < 8:
                config["batch_size"] = 16
                config["accumulate_grad_batches"] = 4
            elif gpu_memory < 12:  # RTX 3060 Ti, 3070
                config["batch_size"] = 32
                config["accumulate_grad_batches"] = 2
            elif gpu_memory < 16:  # RTX 4070, 3080
                config["batch_size"] = 64
                config["accumulate_grad_batches"] = 1
                # RTX 4070 specific optimizations
                if "4070" in self.hardware_info.get("gpu_name", ""):
                    config["batch_size"] = 48  # Sweet spot for 12GB
                    config["gradient_checkpointing"] = True
            elif gpu_memory < 24:  # RTX 4080, 3090
                config["batch_size"] = 96
                config["accumulate_grad_batches"] = 1
            else:  # RTX 4090, A100
                config["batch_size"] = 128
                config["accumulate_grad_batches"] = 1
            
            # Precision based on GPU
            compute_capability = self.hardware_info["gpu_compute_capability"]
            if compute_capability >= (7, 0):  # Volta and newer
                config["precision"] = "16-mixed"
            else:
                config["precision"] = "32-true"
            
            # Multi-GPU settings
            if self.hardware_info["gpu_count"] > 1:
                config["devices"] = self.hardware_info["gpu_count"]
                config["strategy"] = "ddp"
            else:
                config["devices"] = 1
                
            # Compilation settings (for newer GPUs)
            if compute_capability >= (8, 0):  # Ampere and newer
                config["compile_model"] = True
                config["compile_mode"] = "default"
            else:
                config["compile_model"] = False
        else:
            # CPU-only configuration
            config["batch_size"] = 32
            config["devices"] = 1
            config["accelerator"] = "cpu"
            config["precision"] = "32-true"
            config["compile_model"] = False
        
        return config
    
    def _get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset-based configuration."""
        config = {}
        size_category = self.dataset_info["size_category"]
        
        # Sampling strategy
        if size_category == "small":
            config["sampling_strategy"] = "none"
        elif size_category == "medium":
            config["sampling_strategy"] = "none"
            config["enable_dynamic_batching"] = True
        elif size_category == "large":
            config["sampling_strategy"] = "neighbor"
            config["neighbor_sizes"] = [25, 10]
        elif size_category == "xlarge":
            config["sampling_strategy"] = "neighbor"
            config["neighbor_sizes"] = [15, 10, 5]
        else:  # xxlarge
            config["sampling_strategy"] = "cluster"
            config["num_clusters"] = 2000
        
        # Model size based on dataset
        num_features = self.dataset_info["num_features"]
        if num_features < 32:
            config["hidden_dim"] = 64
        elif num_features < 128:
            config["hidden_dim"] = 128
        else:
            config["hidden_dim"] = 256
        
        # Number of layers based on graph properties
        avg_degree = self.dataset_info["avg_degree"]
        if avg_degree < 5:
            config["num_layers"] = 2
        elif avg_degree < 20:
            config["num_layers"] = 3
        else:
            config["num_layers"] = 4
        
        return config
    
    def _get_task_config(self, task: str) -> Dict[str, Any]:
        """Get task-specific configuration."""
        config = {
            "task": task,
        }
        
        if task in ["node_classification", "node_regression"]:
            config["model_type"] = "node"
            config["split_method"] = "spatial"  # Better for astronomical data
        elif task in ["graph_classification", "graph_regression"]:
            config["model_type"] = "graph"
            config["pooling"] = "mean"
        elif task == "link_prediction":
            config["model_type"] = "link"
            config["negative_sampling_ratio"] = 1.0
        
        # Loss function
        if "classification" in task:
            config["loss"] = "cross_entropy"
            config["label_smoothing"] = 0.1
        else:
            config["loss"] = "smooth_l1"
        
        return config
    
    def _get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        config = {}
        
        # GNN architecture selection based on dataset
        if self.dataset_info:
            density = self.dataset_info["density"]
            avg_degree = self.dataset_info["avg_degree"]
            
            if density > 0.1 or avg_degree > 50:
                # Dense graphs: use GAT with fewer heads
                config["conv_type"] = "gat"
                config["heads"] = 2
            elif avg_degree < 5:
                # Sparse graphs: use SAGE
                config["conv_type"] = "sage"
                config["aggr"] = "mean"
            else:
                # Default: GCN for good balance
                config["conv_type"] = "gcn"
                config["improved"] = True
        
        # Regularization
        config["dropout"] = 0.1
        config["weight_decay"] = 1e-4
        
        return config
    
    def _get_survey_config(self, survey: str) -> Dict[str, Any]:
        """Get survey-specific optimizations."""
        survey_configs = {
            "gaia": {
                "k_neighbors": 10,
                "learning_rate": 1e-3,
                "warmup_steps": 1000,
            },
            "sdss": {
                "k_neighbors": 15,
                "learning_rate": 5e-4,
                "scheduler": "cosine",
            },
            "nsa": {
                "k_neighbors": 20,
                "learning_rate": 5e-4,
                "gradient_clip_val": 0.5,
            },
            "tng50": {
                "precision": "32-true",  # Simulations need full precision
                "learning_rate": 1e-4,
            },
        }
        
        return survey_configs.get(survey, {})
    
    def _validate_and_adjust_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and adjust configuration for consistency."""
        # Adjust batch size if using gradient accumulation
        if "accumulate_grad_batches" in config and config["accumulate_grad_batches"] > 1:
            effective_batch_size = config["batch_size"] * config["accumulate_grad_batches"]
            logger.info(f"Effective batch size with accumulation: {effective_batch_size}")
        
        # Ensure sampling is disabled for small datasets
        if self.dataset_info and self.dataset_info["num_nodes"] < 10000:
            config["sampling_strategy"] = "none"
        
        # Adjust workers based on batch size
        if config.get("batch_size", 32) < 16:
            config["num_workers"] = min(config.get("num_workers", 4), 2)
        
        # Set defaults for missing values
        defaults = {
            "max_epochs": 100,
            "early_stopping_patience": 10,
            "optimizer": "adamw",
            "scheduler": "reduce_on_plateau",
            "gradient_clip_val": 1.0,
        }
        
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config


# Global instance for easy access
adaptive_config = AdaptiveConfig()


def get_adaptive_config(
    dataset_stats: Optional[Dict[str, Any]] = None,
    task: str = "node_classification",
    model_type: str = "gnn",
    survey: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get adaptive configuration based on current environment.
    
    Args:
        dataset_stats: Dataset statistics (num_nodes, num_edges, etc.)
        task: Task type
        model_type: Model type
        survey: Survey name
        
    Returns:
        Optimal configuration
    """
    if dataset_stats:
        adaptive_config.analyze_dataset(**dataset_stats)
    
    return adaptive_config.get_optimal_config(task, model_type, survey)
