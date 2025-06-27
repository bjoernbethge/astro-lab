"""
Training Optimization Utilities (2025 Best Practices)
====================================================

Central utilities for optimal training configuration based on 2025 best practices.
Provides automatic detection and configuration for GPUs, DataLoaders, and memory.
"""

import logging
import multiprocessing
import torch
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class TrainingOptimizer:
    """Automatic optimization utilities for training."""
    
    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get comprehensive device information."""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": None,
            "device_count": 0,
            "device_name": None,
            "device_memory_gb": 0,
            "cpu_count": multiprocessing.cpu_count(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "device_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
            
        return info
    
    @staticmethod
    def get_optimal_dataloader_config(
        use_gpu: Optional[bool] = None,
        batch_size: int = 32,
        dataset_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get optimal DataLoader configuration based on hardware.
        
        Args:
            use_gpu: Whether to use GPU (auto-detect if None)
            batch_size: Batch size for training
            dataset_size: Size of dataset (for worker calculation)
            
        Returns:
            Optimal DataLoader configuration
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        config = {}
        
        if use_gpu:
            # GPU-optimized settings (2025 best practices)
            cpu_count = multiprocessing.cpu_count()
            
            # Optimal workers: 4-8 for consumer GPUs, 8-16 for workstation
            if torch.cuda.get_device_properties(0).total_memory > 16e9:  # >16GB VRAM
                optimal_workers = min(16, cpu_count)
            else:
                optimal_workers = min(8, cpu_count)
            
            config = {
                "num_workers": optimal_workers,
                "pin_memory": True,
                "persistent_workers": True,
                "prefetch_factor": 4 if optimal_workers >= 4 else 2,
                "drop_last": True,  # Better batch consistency
            }
            
            # Adjust based on batch size
            if batch_size > 128:
                config["prefetch_factor"] = 2  # Reduce prefetch for large batches
            elif batch_size < 32:
                config["prefetch_factor"] = 8  # Increase prefetch for small batches
                
        else:
            # CPU-optimized settings
            config = {
                "num_workers": 0,  # Avoid multiprocessing overhead on CPU
                "pin_memory": False,
                "persistent_workers": False,
                "prefetch_factor": None,
                "drop_last": False,  # Keep all samples on CPU
            }
        
        return config
    
    @staticmethod
    def get_optimal_training_config(
        model_size: str = "medium",  # small, medium, large
        use_gpu: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Get optimal training configuration based on model size and hardware.
        
        Args:
            model_size: Model size category
            use_gpu: Whether to use GPU (auto-detect if None)
            
        Returns:
            Optimal training configuration
        """
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()
        
        # Base configuration
        config = {
            "gradient_clip_val": 1.0,
            "accumulate_grad_batches": 1,
            "deterministic": True,
            "benchmark": True,
        }
        
        if use_gpu:
            # GPU-specific optimizations
            config.update({
                "precision": "16-mixed",  # Automatic mixed precision
                "sync_batchnorm": True,
                "detect_anomaly": False,  # Disable for performance
                "gpu_memory_fraction": 0.95,
            })
            
            # Adjust based on model size
            if model_size == "large":
                config.update({
                    "accumulate_grad_batches": 4,  # Simulate larger batches
                    "gradient_clip_val": 0.5,  # More aggressive clipping
                })
            elif model_size == "small":
                config.update({
                    "precision": "32",  # Full precision for small models
                })
        else:
            # CPU configuration
            config.update({
                "precision": "32",
                "sync_batchnorm": False,
            })
        
        return config
    
    @staticmethod
    def get_optimal_batch_size(
        model_type: str,
        num_features: int,
        gpu_memory_gb: Optional[float] = None,
    ) -> int:
        """
        Estimate optimal batch size based on model and GPU memory.
        
        Args:
            model_type: Type of model (node, graph, temporal, point)
            num_features: Number of input features
            gpu_memory_gb: GPU memory in GB (auto-detect if None)
            
        Returns:
            Recommended batch size
        """
        if gpu_memory_gb is None and torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        elif gpu_memory_gb is None:
            return 32  # Default CPU batch size
        
        # Base batch sizes for different model types (empirical)
        base_batch_sizes = {
            "node": 1024,      # Node models can handle larger batches
            "graph": 32,       # Graph models need smaller batches
            "temporal": 64,    # Temporal models moderate
            "point": 128,      # Point cloud models
        }
        
        base_size = base_batch_sizes.get(model_type, 64)
        
        # Adjust based on GPU memory
        memory_factor = gpu_memory_gb / 8.0  # Normalized to 8GB baseline
        
        # Adjust based on feature size
        feature_factor = 64 / max(num_features, 1)  # Normalized to 64 features
        
        # Calculate optimal batch size
        optimal_size = int(base_size * memory_factor * feature_factor)
        
        # Clamp to reasonable ranges
        min_sizes = {"node": 32, "graph": 8, "temporal": 16, "point": 16}
        max_sizes = {"node": 4096, "graph": 128, "temporal": 256, "point": 512}
        
        min_size = min_sizes.get(model_type, 16)
        max_size = max_sizes.get(model_type, 256)
        
        # Round to nearest power of 2 for efficiency
        optimal_size = max(min_size, min(optimal_size, max_size))
        optimal_size = 2 ** round(torch.log2(torch.tensor(float(optimal_size))).item())
        
        return int(optimal_size)
    
    @staticmethod
    def setup_gpu_optimization():
        """Setup GPU optimizations for training."""
        if not torch.cuda.is_available():
            logger.info("No GPU available, skipping GPU optimizations")
            return
        
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        
        # Set memory fraction to avoid OOM
        torch.cuda.set_per_process_memory_fraction(0.95)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Log GPU info
        device_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"🚀 GPU optimizations enabled:")
        logger.info(f"   Device: {device_name}")
        logger.info(f"   Memory: {memory_gb:.2f} GB")
        logger.info(f"   cuDNN benchmark: True