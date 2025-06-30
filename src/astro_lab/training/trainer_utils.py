"""
Trainer Utilities Module for AstroLab
====================================

Handles common training operations, hardware detection, and optimization utilities.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from astro_lab.memory import clear_cuda_cache

logger = logging.getLogger(__name__)


def detect_hardware() -> Dict[str, Any]:
    """
    Detect available hardware and return configuration.

    Returns:
        Dictionary with hardware information and recommended settings
    """
    hardware_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_device_name": None,
        "cuda_memory_gb": None,
        "cpu_count": os.cpu_count(),
        "recommended_batch_size": 32,
        "recommended_workers": 4,
        "use_mixed_precision": False,
        "use_cuda_graphs": False,
        "use_torch_compile": False,
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        hardware_info.update(
            {
                "cuda_device_name": props.name,
                "cuda_memory_gb": props.total_memory / 1e9,
                "cuda_compute_capability": f"{props.major}.{props.minor}",
            }
        )

        # Adaptive settings based on GPU memory - 2025 optimized
        memory_gb = hardware_info["cuda_memory_gb"]
        if memory_gb >= 24:  # RTX 4090, A100, etc.
            hardware_info.update(
                {
                    "recommended_batch_size": 64,  # Reduced for safety
                    "recommended_workers": 4,      # Reduced from 8
                    "use_mixed_precision": True,
                    "use_cuda_graphs": True,
                    "use_torch_compile": True,
                }
            )
        elif memory_gb >= 12:  # RTX 3080, 4080, etc.
            hardware_info.update(
                {
                    "recommended_batch_size": 32,  # Reduced from 64
                    "recommended_workers": 4,      # Reduced from 6
                    "use_mixed_precision": True,
                    "use_cuda_graphs": False,      # Safer default
                    "use_torch_compile": True,
                }
            )
        elif memory_gb >= 8:  # RTX 3070, 4070, etc.
            hardware_info.update(
                {
                    "recommended_batch_size": 8,   # Reduced from 16 for 8GB
                    "recommended_workers": 2,      # Conservative for 8GB
                    "use_mixed_precision": True,   # Keep enabled for memory savings
                    "use_cuda_graphs": False,
                    "use_torch_compile": True,
                }
            )
        elif memory_gb >= 4:  # GTX 1060, RTX 2060, etc.
            hardware_info.update(
                {
                    "recommended_batch_size": 4,   # Very small for 4GB
                    "recommended_workers": 1,      
                    "use_mixed_precision": False,  # May cause instability on older GPUs
                    "use_cuda_graphs": False,
                    "use_torch_compile": False,
                }
            )

    logger.info(f"Hardware detected: {hardware_info}")
    return hardware_info


def setup_device(device: Optional[str] = None) -> torch.device:
    """
    Setup and return the best available device.

    Args:
        device: Device specification ('cuda', 'cpu', or specific device)

    Returns:
        PyTorch device
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = "cpu"

    torch_device = torch.device(device)

    if torch_device.type == "cuda":
        # Set PyTorch CUDA allocator configuration for better memory management
        # Enable expandable segments to reduce fragmentation (PyTorch 2.7+)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        
        # Set conservative memory fraction to avoid OOM
        try:
            torch.cuda.set_per_process_memory_fraction(0.85)  # Reduced from 0.9
        except Exception as e:
            logger.warning(f"Could not set memory fraction: {e}")
            
        # Log device info
        props = torch.cuda.get_device_properties(torch_device)
        memory_gb = props.total_memory / 1e9
        logger.info(f"Using CUDA device: {props.name} ({memory_gb:.1f}GB)")
        
        # Enable TensorFloat-32 for Ampere+ GPUs if available
        if props.major >= 8:  # Ampere or newer
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TensorFloat-32 enabled for faster training")
    else:
        logger.info("Using CPU device")

    return torch_device


def optimize_dataloader(
    dataloader: DataLoader,
    device: torch.device,
    pin_memory: Optional[bool] = None,
    num_workers: Optional[int] = None,
) -> DataLoader:
    """
    Optimize dataloader settings for the given device.

    Args:
        dataloader: DataLoader to optimize
        device: Target device
        pin_memory: Whether to pin memory (auto-detect if None)
        num_workers: Number of workers (auto-detect if None)

    Returns:
        Optimized DataLoader
    """
    hardware_info = detect_hardware()

    # Auto-detect settings if not provided
    if pin_memory is None:
        pin_memory = device.type == "cuda"

    if num_workers is None:
        num_workers = hardware_info["recommended_workers"]

    # Update dataloader settings
    dataloader.pin_memory = pin_memory
    dataloader.num_workers = num_workers

    logger.info(f"Optimized DataLoader: pin_memory={pin_memory}, workers={num_workers}")
    return dataloader


def setup_mixed_precision(
    model: nn.Module,
    device: torch.device,
    use_mixed_precision: Optional[bool] = None,
) -> Tuple[bool, Optional[torch.amp.GradScaler]]:
    """
    Setup mixed precision training if supported.

    Args:
        model: Model to train
        device: Training device
        use_mixed_precision: Whether to use mixed precision (auto-detect if None)

    Returns:
        Tuple of (use_mixed_precision, grad_scaler)
    """
    hardware_info = detect_hardware()

    if use_mixed_precision is None:
        use_mixed_precision = hardware_info["use_mixed_precision"]

    if not use_mixed_precision or device.type != "cuda":
        logger.info("Mixed precision disabled (not CUDA or explicitly disabled)")
        return False, None

    # Check GPU compute capability for Tensor Cores
    try:
        props = torch.cuda.get_device_properties(device)
        if props.major < 7:  # Volta or newer for efficient mixed precision
            logger.warning(f"GPU compute capability {props.major}.{props.minor} < 7.0, mixed precision may not be efficient")
            return False, None
    except Exception:
        logger.warning("Could not determine GPU compute capability")
        return False, None

    # Test if model supports mixed precision with proper PyG Data object
    try:
        model.eval()
        
        # Create realistic PyG Data object for testing
        from torch_geometric.data import Data, Batch
        
        # Single graph for testing - use proper feature dimensions
        test_data = Data(
            x=torch.randn(5, 7, dtype=torch.float32).to(device),  # Match expected input features
            edge_index=torch.tensor([[0, 1, 2, 3], 
                                   [1, 2, 3, 4]], dtype=torch.long).to(device),
            batch=torch.zeros(5, dtype=torch.long).to(device),
            y=torch.tensor([0], dtype=torch.long).to(device)
        )
        
        # Test autocast with proper device type and dtype specification
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            with torch.no_grad():
                _ = model(test_data)

        # Create gradient scaler for CUDA
        grad_scaler = torch.amp.GradScaler()
        logger.info("Mixed precision training enabled with autocast and GradScaler")
        return True, grad_scaler

    except Exception as e:
        logger.warning(f"Mixed precision not supported: {type(e).__name__}: {e}")
        return False, None


def setup_torch_compile(
    model: nn.Module,
    device: torch.device,
    use_torch_compile: Optional[bool] = None,
) -> nn.Module:
    """
    Setup torch.compile optimization if supported.

    Args:
        model: Model to optimize
        device: Training device
        use_torch_compile: Whether to use torch.compile (auto-detect if None)

    Returns:
        Compiled model or original model
    """
    hardware_info = detect_hardware()

    if use_torch_compile is None:
        use_torch_compile = hardware_info["use_torch_compile"]

    if not use_torch_compile or device.type != "cuda":
        return model

    try:
        # Check if torch.compile is available
        if hasattr(torch, "compile"):
            compiled_model = torch.compile(model, mode="reduce-overhead")
            logger.info("Torch.compile optimization enabled")
            return compiled_model
        else:
            logger.warning("torch.compile not available (PyTorch < 2.0)")
            return model

    except Exception as e:
        logger.warning(f"Torch.compile failed: {e}")
        return model


def setup_cuda_graphs(
    model: nn.Module,
    device: torch.device,
    use_cuda_graphs: Optional[bool] = None,
) -> bool:
    """
    Setup CUDA graphs optimization if supported.

    Args:
        model: Model to optimize
        device: Training device
        use_cuda_graphs: Whether to use CUDA graphs (auto-detect if None)

    Returns:
        Whether CUDA graphs are enabled
    """
    hardware_info = detect_hardware()

    if use_cuda_graphs is None:
        use_cuda_graphs = hardware_info["use_cuda_graphs"]

    if not use_cuda_graphs or device.type != "cuda":
        return False

    try:
        # Check CUDA compute capability
        props = torch.cuda.get_device_properties(device)
        if props.major < 7:  # Volta or newer required
            logger.warning("CUDA graphs require Volta or newer GPU")
            return False

        logger.info("CUDA graphs optimization enabled")
        return True

    except Exception as e:
        logger.warning(f"CUDA graphs setup failed: {e}")
        return False


def get_optimal_batch_size(
    model: nn.Module,
    device: torch.device,
    sample_input: torch.Tensor,
    target_memory_usage: float = 0.8,
) -> int:
    """
    Find optimal batch size for the model and device.

    Args:
        model: Model to test
        device: Target device
        sample_input: Sample input tensor
        target_memory_usage: Target GPU memory usage (0.0-1.0)

    Returns:
        Optimal batch size
    """
    if device.type != "cuda":
        return 32  # Default for CPU

    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_usage

    # Start with small batch size
    batch_size = 1
    max_batch_size = 1024

    while batch_size <= max_batch_size:
        try:
            # Clear cache
            clear_cuda_cache()

            # Test forward pass
            batch_input = sample_input.repeat(batch_size, 1)
            with torch.no_grad():
                _ = model(batch_input)

            # Check memory usage
            current_memory = torch.cuda.memory_allocated(device)
            if current_memory > target_memory:
                break

            batch_size *= 2

        except torch.cuda.OutOfMemoryError:
            break
        except Exception as e:
            logger.warning(f"Error testing batch size {batch_size}: {e}")
            break

    # Return previous successful batch size
    optimal_batch_size = max(1, batch_size // 2)
    logger.info(f"Optimal batch size: {optimal_batch_size}")

    return optimal_batch_size


def validate_model_inputs(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_samples: int = 2,  # Reduziert auf nur 2 Samples für 8GB GPU
) -> bool:
    """
    Validate that model can process inputs from dataloader.

    Args:
        model: Model to validate
        dataloader: DataLoader to test
        device: Target device
        max_samples: Maximum samples to test

    Returns:
        True if validation passes
    """
    logger.info(f"Validating model inputs with {max_samples} samples...")

    try:
        # Clear all caches before validation
        if device.type == "cuda":
            torch.cuda.empty_cache()
            # Reset peak memory if available (PyTorch 2.0+)
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated(device)
            logger.info(f"Initial GPU memory: {initial_memory / 1e6:.1f}MB")
            
        model.eval()
        sample_count = 0

        for batch_idx, batch in enumerate(dataloader):
            if sample_count >= max_samples:
                break

            try:
                # Clear cache between batches
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    
                # Move batch to device
                if hasattr(batch, "to"):
                    batch = batch.to(device)
                elif isinstance(batch, (list, tuple)):
                    batch = [b.to(device) if hasattr(b, "to") else b for b in batch]
                elif isinstance(batch, dict):
                    batch = {
                        k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
                    }

                # Test forward pass with memory monitoring
                with torch.no_grad():
                    if isinstance(batch, (list, tuple)):
                        output = model(*batch)
                    elif isinstance(batch, dict):
                        output = model(**batch)
                    else:
                        output = model(batch)
                
                # Log memory usage
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated(device)
                    # Use available memory stats function
                    if hasattr(torch.cuda, 'max_memory_allocated'):
                        peak_memory = torch.cuda.max_memory_allocated(device)
                        logger.info(f"Batch {batch_idx}: current={current_memory/1e6:.1f}MB, peak={peak_memory/1e6:.1f}MB")
                    else:
                        logger.info(f"Batch {batch_idx}: current={current_memory/1e6:.1f}MB")
                
                # Validate output shape
                if hasattr(output, 'shape'):
                    logger.info(f"Output shape: {output.shape}")
                
                sample_count += 1
                
                # Clean up immediately
                del output
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA out of memory during validation batch {batch_idx}: {e}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    # Log memory summary with error details if available
                    if hasattr(torch.cuda, 'memory_summary'):
                        logger.error(f"Memory summary:\n{torch.cuda.memory_summary()}")
                    else:
                        logger.error(f"Memory allocated: {torch.cuda.memory_allocated(device)/1e6:.1f}MB")
                return False
            except Exception as e:
                logger.error(f"Model input validation failed for batch {batch_idx}: {type(e).__name__}: {e}")
                return False

        logger.info(f"✅ Model input validation passed ({sample_count} samples)")
        return True

    except Exception as e:
        logger.error(f"Model input validation failed: {type(e).__name__}: {e}")
        return False


def log_training_info(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> None:
    """
    Log comprehensive training information.

    Args:
        model: Model being trained
        dataloader: Training dataloader
        device: Training device
        config: Training configuration
    """
    logger.info("=" * 60)
    logger.info("TRAINING SETUP")
    logger.info("=" * 60)

    # Hardware info
    hardware_info = detect_hardware()
    logger.info(f"Device: {device}")
    logger.info(
        f"GPU Memory: {hardware_info['cuda_memory_gb']:.1f}GB"
        if hardware_info["cuda_memory_gb"]
        else "CPU"
    )

    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable"
    )

    # Dataset info
    dataset_size = (
        len(dataloader.dataset) if hasattr(dataloader, "dataset") else "Unknown"
    )
    batch_size = (
        dataloader.batch_size if hasattr(dataloader, "batch_size") else "Unknown"
    )
    logger.info(f"Dataset Size: {dataset_size}")
    logger.info(f"Batch Size: {batch_size}")

    # Configuration highlights
    logger.info(f"Max Epochs: {config.get('max_epochs', 'Unknown')}")
    logger.info(f"Learning Rate: {config.get('learning_rate', 'Unknown')}")
    logger.info(f"Mixed Precision: {config.get('use_mixed_precision', False)}")
    logger.info(f"Torch Compile: {config.get('use_torch_compile', False)}")
    logger.info(f"CUDA Graphs: {config.get('use_cuda_graphs', False)}")

    logger.info("=" * 60)
