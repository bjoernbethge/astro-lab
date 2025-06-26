"""
Training Utilities
==================

Common utilities for training astronomical ML models.
"""

import random
import logging
import sys
from typing import Optional

import numpy as np
import torch


def set_random_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior on CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(
    level: str = "INFO",
    format: Optional[str] = None,
    filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for training.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Custom log format string
        filename: Optional log file path
        
    Returns:
        Configured logger instance
    """
    if format is None:
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename:
        handlers.append(logging.FileHandler(filename))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format,
        handlers=handlers,
        force=True  # Override any existing configuration
    )
    
    # Set specific loggers to appropriate levels
    logging.getLogger("lightning").setLevel(logging.INFO)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    return logging.getLogger("astro_lab.training")
