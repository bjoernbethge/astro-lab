"""
Advanced Memory Management Utilities using contextlib
====================================================

This module provides comprehensive memory management utilities using Python's
contextlib for astronomical data processing applications.
"""

import gc
import logging
import sys
import threading
import time
import warnings
from contextlib import ExitStack, closing, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

logger = logging.getLogger(__name__)


# =========================================================================
# Memory Statistics and Monitoring
# =========================================================================


@dataclass
class MemoryStats:
    """Memory statistics container."""

    initial_objects: int = 0
    final_objects: int = 0
    initial_memory: float = 0.0  # MB
    final_memory: float = 0.0  # MB
    peak_memory: float = 0.0  # MB
    cuda_allocated: float = 0.0  # MB
    cuda_reserved: float = 0.0  # MB
    operation_name: str = "Unknown"
    duration: float = 0.0  # seconds

    @property
    def object_diff(self) -> int:
        """Number of objects created during operation."""
        return self.final_objects - self.initial_objects

    @property
    def memory_diff(self) -> float:
        """Memory difference in MB."""
        return self.final_memory - self.initial_memory

    def __str__(self) -> str:
        """Human-readable memory statistics."""
        return (
            f"Memory Stats [{self.operation_name}]: "
            f"Objects: {self.object_diff:+d}, "
            f"Memory: {self.memory_diff:+.1f}MB, "
            f"Peak: {self.peak_memory:.1f}MB, "
            f"Duration: {self.duration:.2f}s"
        )


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def get_cuda_memory_stats() -> Dict[str, float]:
    """Get CUDA memory statistics in MB."""
    stats = {"allocated": 0.0, "reserved": 0.0, "peak": 0.0}
    try:
        import torch

        if torch.cuda.is_available():
            stats["allocated"] = torch.cuda.memory_allocated() / 1024 / 1024
            stats["reserved"] = torch.cuda.memory_reserved() / 1024 / 1024
            stats["peak"] = torch.cuda.max_memory_allocated() / 1024 / 1024
    except ImportError:
        pass
    return stats


# =========================================================================
# Core Context Managers
# =========================================================================


@contextmanager
def memory_tracking_context(
    operation_name: str = "Operation",
) -> Generator[MemoryStats, None, None]:
    """
    Track memory usage during an operation.

    Args:
        operation_name: Name of the operation for logging

    Yields:
        MemoryStats: Memory statistics object that gets updated
    """
    start_time = time.time()

    # Initialize statistics
    stats = MemoryStats(
        operation_name=operation_name,
        initial_objects=len(gc.get_objects()),
        initial_memory=get_memory_usage(),
    )

    cuda_stats = get_cuda_memory_stats()
    stats.cuda_allocated = cuda_stats["allocated"]
    stats.cuda_reserved = cuda_stats["reserved"]

    # Reset CUDA peak memory stats
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    try:
        yield stats
    finally:
        # Update final statistics
        stats.duration = time.time() - start_time
        stats.final_objects = len(gc.get_objects())
        stats.final_memory = get_memory_usage()

        cuda_stats = get_cuda_memory_stats()
        stats.peak_memory = max(stats.peak_memory, cuda_stats["peak"])

        # Log statistics
        if stats.object_diff > 1000 or stats.memory_diff > 10:
            logger.warning(str(stats))
        else:
            logger.debug(str(stats))


@contextmanager
def resource_cleanup_context(
    cleanup_functions: Optional[List[Callable]] = None, suppress_errors: bool = True
) -> Generator[List[Callable], None, None]:
    """
    Context manager for registering and executing cleanup functions.

    Args:
        cleanup_functions: Initial list of cleanup functions
        suppress_errors: Whether to suppress errors during cleanup

    Yields:
        List of cleanup functions that can be extended
    """
    cleanup_list = cleanup_functions or []

    try:
        yield cleanup_list
    finally:
        # Execute cleanup functions in reverse order (LIFO)
        for cleanup_func in reversed(cleanup_list):
            try:
                if callable(cleanup_func):
                    cleanup_func()
                    logger.debug(
                        f"✅ Cleanup function {cleanup_func.__name__} executed"
                    )
            except Exception as e:
                if suppress_errors:
                    logger.debug(
                        f"⚠️ Cleanup function {cleanup_func.__name__} failed: {e}"
                    )
                else:
                    raise


@contextmanager
def pytorch_memory_context(
    description: str = "PyTorch operation",
    clear_cache: bool = True,
    reset_stats: bool = True,
) -> Generator[Dict[str, float], None, None]:
    """
    Advanced PyTorch memory management context.

    Args:
        description: Description for logging
        clear_cache: Whether to clear CUDA cache before/after
        reset_stats: Whether to reset memory statistics

    Yields:
        Dict with initial memory statistics
    """
    initial_stats = {}

    try:
        import torch

        if torch.cuda.is_available():
            if clear_cache:
                torch.cuda.empty_cache()

            if reset_stats:
                torch.cuda.reset_peak_memory_stats()

            initial_stats = get_cuda_memory_stats()
            logger.debug(
                f"PyTorch context [{description}]: Initial CUDA memory {initial_stats['allocated']:.1f}MB"
            )
    except ImportError:
        logger.debug(f"PyTorch not available for {description}")

    try:
        yield initial_stats
    finally:
        try:
            import torch

            if torch.cuda.is_available():
                final_stats = get_cuda_memory_stats()
                memory_diff = final_stats["allocated"] - initial_stats.get(
                    "allocated", 0
                )

                if clear_cache:
                    torch.cuda.empty_cache()

                logger.debug(
                    f"PyTorch context [{description}]: "
                    f"Memory change: {memory_diff:+.1f}MB, "
                    f"Peak: {final_stats['peak']:.1f}MB"
                )
        except ImportError:
            pass


@contextmanager
def comprehensive_cleanup_context(
    operation_name: str = "Operation",
    enable_monitoring: bool = True,
    cleanup_pytorch: bool = True,
    cleanup_matplotlib: bool = True,
    cleanup_blender: bool = True,
    force_gc: bool = True,
) -> Generator[MemoryStats, None, None]:
    """
    Comprehensive cleanup context combining all memory management features.

    Args:
        operation_name: Name of the operation
        enable_monitoring: Whether to enable memory monitoring
        cleanup_pytorch: Whether to cleanup PyTorch resources
        cleanup_matplotlib: Whether to cleanup matplotlib resources
        cleanup_blender: Whether to cleanup Blender resources
        force_gc: Whether to force garbage collection

    Yields:
        MemoryStats: Memory statistics (if monitoring enabled)
    """
    with ExitStack() as stack:
        # Add memory monitoring if enabled
        if enable_monitoring:
            stats = stack.enter_context(memory_tracking_context(operation_name))
        else:
            stats = MemoryStats(operation_name=operation_name)

        # Add PyTorch memory management
        if cleanup_pytorch:
            stack.enter_context(pytorch_memory_context(f"{operation_name} - PyTorch"))

        # Register cleanup functions
        cleanup_functions = []

        if cleanup_matplotlib:
            cleanup_functions.append(_cleanup_matplotlib)

        if cleanup_blender:
            cleanup_functions.append(_cleanup_blender)

        if force_gc:
            cleanup_functions.append(_force_garbage_collection)

        # Add system cleanup
        cleanup_functions.append(_cleanup_system_caches)

        stack.enter_context(resource_cleanup_context(cleanup_functions))

        try:
            yield stats
        except Exception as e:
            logger.error(f"Error in {operation_name}: {e}")
            raise


# =========================================================================
# Specialized Context Managers
# =========================================================================


@contextmanager
def file_processing_context(
    file_path: Union[str, Path],
    chunk_size: Optional[int] = None,
    memory_limit_mb: float = 1000.0,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for processing large files with memory management.

    Args:
        file_path: Path to the file being processed
        chunk_size: Chunk size for processing
        memory_limit_mb: Memory limit in MB

    Yields:
        Dict with file processing parameters
    """
    file_path = Path(file_path)
    operation_name = f"File processing: {file_path.name}"

    with comprehensive_cleanup_context(operation_name) as stats:
        # Calculate optimal chunk size if not provided
        if chunk_size is None and file_path.exists():
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            chunk_size = max(1000, int(memory_limit_mb * 1000 / max(file_size_mb, 1)))

        processing_params = {
            "file_path": file_path,
            "chunk_size": chunk_size or 10000,
            "memory_limit_mb": memory_limit_mb,
            "stats": stats,
        }

        logger.info(f"📂 Processing file: {file_path} (chunk_size: {chunk_size})")

        yield processing_params


@contextmanager
def model_training_context(
    model_name: str = "model",
    enable_mixed_precision: bool = False,
    gradient_checkpointing: bool = False,
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for model training with memory optimization.

    Args:
        model_name: Name of the model being trained
        enable_mixed_precision: Whether to enable mixed precision training
        gradient_checkpointing: Whether to enable gradient checkpointing

    Yields:
        Dict with training configuration
    """
    operation_name = f"Training: {model_name}"

    with comprehensive_cleanup_context(operation_name) as stats:
        training_config = {
            "model_name": model_name,
            "mixed_precision": enable_mixed_precision,
            "gradient_checkpointing": gradient_checkpointing,
            "stats": stats,
        }

        # Set up training optimizations
        try:
            import torch

            if torch.cuda.is_available():
                # Enable memory-efficient attention if available
                if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                    torch.backends.cuda.enable_flash_sdp(True)

                # Enable memory format optimization
                if hasattr(torch.backends.cudnn, "allow_tf32"):
                    torch.backends.cudnn.allow_tf32 = True

                logger.info(f"🧠 Training setup: {model_name} with CUDA optimizations")

        except ImportError:
            logger.info(f"🧠 Training setup: {model_name} (CPU only)")

        yield training_config


@contextmanager
def batch_processing_context(
    total_items: int, batch_size: int = 1000, memory_threshold_mb: float = 500.0
) -> Generator[Dict[str, Any], None, None]:
    """
    Context manager for batch processing with adaptive memory management.

    Args:
        total_items: Total number of items to process
        batch_size: Initial batch size
        memory_threshold_mb: Memory threshold for batch size adjustment

    Yields:
        Dict with batch processing parameters
    """
    operation_name = f"Batch processing: {total_items} items"

    with comprehensive_cleanup_context(operation_name) as stats:
        num_batches = (total_items + batch_size - 1) // batch_size

        batch_config = {
            "total_items": total_items,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "memory_threshold_mb": memory_threshold_mb,
            "stats": stats,
            "adaptive_batch_size": True,
        }

        logger.info(
            f"📊 Batch processing: {total_items} items in {num_batches} batches"
        )

        yield batch_config


# =========================================================================
# Cleanup Functions
# =========================================================================


def _cleanup_matplotlib():
    """Clean up matplotlib resources."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")

        # Clear matplotlib font cache
        import matplotlib.font_manager

        matplotlib.font_manager._rebuild()

        logger.debug("✅ Matplotlib cleanup completed")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"⚠️ Matplotlib cleanup failed: {e}")


def _cleanup_blender():
    """Clean up Blender resources."""
    try:
        import bpy

        if hasattr(bpy, "context") and bpy.context is not None:
            # Purge orphaned data
            bpy.ops.outliner.orphans_purge(
                do_local_ids=True, do_linked_ids=True, do_recursive=True
            )

            # Clear undo history
            bpy.ops.ed.undo_history_clear()

        logger.debug("✅ Blender cleanup completed")
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"⚠️ Blender cleanup failed: {e}")


def _force_garbage_collection():
    """Force comprehensive garbage collection."""
    # Multiple GC passes for thorough cleanup
    collected_total = 0
    for i in range(3):
        collected = gc.collect()
        collected_total += collected
        if collected == 0:
            break

    # Clear generation statistics
    gc.set_debug(0)

    logger.debug(f"✅ Garbage collection: {collected_total} objects collected")


def _cleanup_system_caches():
    """Clean up system-level caches."""
    # Clear type cache
    if hasattr(sys, "_clear_type_cache"):
        sys._clear_type_cache()

    # Clear import cache for astro_lab modules
    modules_to_clear = [
        mod for mod in sys.modules.keys() if mod.startswith("astro_lab")
    ]
    cache_cleared = 0

    for mod_name in modules_to_clear:
        module = sys.modules.get(mod_name)
        if module and hasattr(module, "__dict__"):
            mod_dict = module.__dict__
            for key in list(mod_dict.keys()):
                if key.startswith("_cache") or key.endswith("_cache"):
                    mod_dict.pop(key, None)
                    cache_cleared += 1

    logger.debug(f"✅ System cache cleanup: {cache_cleared} cache entries cleared")


# =========================================================================
# Memory Monitoring Utilities
# =========================================================================


class MemoryMonitor:
    """Continuous memory monitoring utility."""

    def __init__(self, interval: float = 1.0, threshold_mb: float = 1000.0):
        """
        Initialize memory monitor.

        Args:
            interval: Monitoring interval in seconds
            threshold_mb: Memory threshold for warnings
        """
        self.interval = interval
        self.threshold_mb = threshold_mb
        self.monitoring = False
        self.thread: Optional[threading.Thread] = None
        self.max_memory = 0.0

    def start_monitoring(self):
        """Start memory monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(
            f"📊 Memory monitoring started (threshold: {self.threshold_mb:.1f}MB)"
        )

    def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        logger.info(f"📊 Memory monitoring stopped (peak: {self.max_memory:.1f}MB)")

    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.monitoring:
            try:
                current_memory = get_memory_usage()
                self.max_memory = max(self.max_memory, current_memory)

                if current_memory > self.threshold_mb:
                    logger.warning(f"⚠️ High memory usage: {current_memory:.1f}MB")

                time.sleep(self.interval)
            except Exception as e:
                logger.debug(f"Memory monitoring error: {e}")
                break

    @contextmanager
    def monitoring_context(self):
        """Context manager for automatic monitoring."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()


# =========================================================================
# Convenience Functions
# =========================================================================


def create_memory_efficient_context(
    operation_name: str,
    enable_pytorch: bool = True,
    enable_monitoring: bool = True,
    memory_threshold_mb: float = 500.0,
):
    """
    Create a memory-efficient context manager with sensible defaults.

    Args:
        operation_name: Name of the operation
        enable_pytorch: Whether to enable PyTorch optimizations
        enable_monitoring: Whether to enable memory monitoring
        memory_threshold_mb: Memory threshold for warnings

    Returns:
        Context manager for memory-efficient operations
    """
    return comprehensive_cleanup_context(
        operation_name=operation_name,
        enable_monitoring=enable_monitoring,
        cleanup_pytorch=enable_pytorch,
        cleanup_matplotlib=True,
        cleanup_blender=True,
        force_gc=True,
    )


# Export main utilities
__all__ = [
    "MemoryStats",
    "memory_tracking_context",
    "resource_cleanup_context",
    "pytorch_memory_context",
    "comprehensive_cleanup_context",
    "file_processing_context",
    "model_training_context",
    "batch_processing_context",
    "MemoryMonitor",
    "create_memory_efficient_context",
    "get_memory_usage",
    "get_cuda_memory_stats",
]
