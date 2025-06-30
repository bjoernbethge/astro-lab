"""
GPU-Optimized Training for RTX 4070 and Modern GPUs
===================================================

Implements 2025 best practices for efficient GPU training:
- Flash Attention 2.0 for transformers
- torch.compile with mode='max-autotune' for Ada Lovelace
- Optimal memory management for 12GB VRAM
- CUDA Graphs for reduced kernel launch overhead
- Tensor Cores utilization with automatic mixed precision
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
from contextlib import contextmanager
import warnings

logger = logging.getLogger(__name__)


@dataclass
class RTX4070Config:
    """Optimized settings for RTX 4070 (Ada Lovelace)."""
    
    # Memory: 12GB GDDR6X
    vram_gb: float = 12.0
    
    # Architecture specifics
    compute_capability: tuple = (8, 9)  # SM 8.9
    tensor_cores_gen: int = 4  # 4th gen Tensor Cores
    rt_cores_gen: int = 3  # 3rd gen RT Cores
    
    # Optimal settings for Ada Lovelace
    optimal_batch_size: Dict[str, int] = None
    use_flash_attention: bool = True
    use_cuda_graphs: bool = True
    compile_mode: str = "max-autotune"  # Best for RTX 4070
    
    # Memory efficiency
    gradient_checkpointing_threshold: int = 10000  # nodes
    tf32_mode: bool = True  # Use TF32 for better perf
    channels_last: bool = True  # Better memory layout
    
    def __post_init__(self):
        if self.optimal_batch_size is None:
            self.optimal_batch_size = {
                "small": 256,    # < 10k nodes
                "medium": 128,   # 10k-100k nodes  
                "large": 64,     # 100k-1M nodes
                "xlarge": 32,    # > 1M nodes
            }


class GPUOptimizer:
    """
    GPU optimization manager for efficient training.
    
    Automatically configures optimal settings based on GPU type.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize GPU optimizer."""
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.device.type == "cuda":
            self.gpu_properties = torch.cuda.get_device_properties(self.device)
            self.gpu_name = self.gpu_properties.name
            self.vram_gb = self.gpu_properties.total_memory / 1e9
            
            # Detect GPU type and load optimal config
            self.config = self._get_gpu_config()
            self._setup_gpu_optimizations()
        else:
            self.config = None
            logger.warning("No GPU detected, GPU optimizations disabled")
    
    def _get_gpu_config(self) -> RTX4070Config:
        """Get GPU-specific configuration."""
        # RTX 4070 detection
        if "4070" in self.gpu_name:
            logger.info("Detected RTX 4070 - Loading optimized configuration")
            return RTX4070Config()
        
        # RTX 4090
        elif "4090" in self.gpu_name:
            config = RTX4070Config()
            config.vram_gb = 24.0
            config.optimal_batch_size = {
                "small": 512, "medium": 256, "large": 128, "xlarge": 64
            }
            return config
        
        # RTX 3090/3080
        elif "3090" in self.gpu_name or "3080" in self.gpu_name:
            config = RTX4070Config()
            config.vram_gb = 24.0 if "3090" in self.gpu_name else 10.0
            config.compute_capability = (8, 6)  # Ampere
            config.use_flash_attention = False  # Not optimal on Ampere
            config.compile_mode = "reduce-overhead"
            return config
        
        # Default modern GPU
        else:
            logger.info(f"Using default config for {self.gpu_name}")
            config = RTX4070Config()
            config.vram_gb = self.vram_gb
            return config
    
    def _setup_gpu_optimizations(self):
        