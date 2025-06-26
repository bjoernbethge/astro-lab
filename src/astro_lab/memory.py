"""
AstroLab Memory Management
==========================

Centralized memory management for AstroLab applications.
Provides robust memory tracking, cleanup, and optimization.
"""

import atexit
import gc
import logging
import tracemalloc
import weakref
from contextlib import contextmanager
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)

# Memory tracking registry
_memory_registry = weakref.WeakSet()

# Start memory tracking
tracemalloc.start()


def _cleanup():
    """Perform comprehensive memory cleanup."""
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()

    # Clear memory registry
    _memory_registry.clear()


def _log_memory_usage(snapshot_before):
    """Log memory usage changes."""
    try:
        snapshot_after = tracemalloc.take_snapshot()
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
        if top_stats:
            logger.debug(
                f"Memory change: {top_stats[0] if top_stats else 'No significant change'}"
            )
    except Exception as e:
        logger.debug(f"Could not log memory usage: {e}")


@contextmanager
def memory_management():
    """Context manager for memory-efficient operations."""
    snapshot_before = tracemalloc.take_snapshot()
    try:
        yield
    finally:
        _cleanup()
        _log_memory_usage(snapshot_before)


def register_for_cleanup(obj: Any) -> None:
    """Register object for memory tracking, only if weakref is supported."""
    try:
        # Skip if object is None or already a weak reference
        if obj is None or isinstance(obj, weakref.ReferenceType):
            return

        # Test if the object supports weak references
        test_ref = weakref.ref(obj)
        del test_ref

        # Add to registry if test passed
        _memory_registry.add(weakref.ref(obj))

    except TypeError:
        # Object does not support weak references (e.g. C-extensions, widgets)
        logger.debug(f"Object {type(obj)} does not support weak references, skipping")
        return
    except Exception as e:
        logger.debug(f"Could not register object for cleanup: {e}")
        return


def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    try:
        current, peak = tracemalloc.get_traced_memory()
        return {
            "current_mb": current / 1024 / 1024,
            "peak_mb": peak / 1024 / 1024,
            "cuda_available": torch.cuda.is_available(),
            "cuda_memory_allocated": torch.cuda.memory_allocated() / 1024 / 1024
            if torch.cuda.is_available()
            else 0,
            "cuda_memory_reserved": torch.cuda.memory_reserved() / 1024 / 1024
            if torch.cuda.is_available()
            else 0,
        }
    except Exception as e:
        logger.warning(f"Error getting memory stats: {e}")
        return {"error": str(e)}


def diagnose_memory_leaks() -> Dict[str, Any]:
    """Diagnose potential memory issues."""
    try:
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")

        # Get top consumers
        top_consumers = [
            {
                "file": stat.traceback.format()[-1],
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count,
            }
            for stat in top_stats[:5]
        ]

        # Generate recommendations
        recommendations = []
        memory_info = get_memory_stats()

        if any(stat.count > 1000 for stat in top_stats):
            recommendations.append("High object count - consider generators")

        cuda_memory = memory_info.get("cuda_memory_allocated", 0)
        if isinstance(cuda_memory, (int, float)) and cuda_memory > 1000:
            recommendations.append("High CUDA usage - consider batch processing")

        current_memory = memory_info.get("current_mb", 0)
        if isinstance(current_memory, (int, float)) and current_memory > 500:
            recommendations.append("High memory usage - check for circular references")

        return {
            "top_consumers": top_consumers,
            "memory_info": memory_info,
            "potential_issues": len([s for s in top_stats if s.count > 1000]),
            "recommendations": recommendations,
        }
    except Exception as e:
        logger.error(f"Error diagnosing memory: {e}")
        return {"error": str(e)}


def force_comprehensive_cleanup() -> Dict[str, Any]:
    """Force comprehensive memory cleanup."""
    try:
        logger.info("🧹 Starting comprehensive memory cleanup...")

        # Multiple cleanup passes
        for i in range(5):
            collected = gc.collect()
            logger.debug(f"GC pass {i + 1}: collected {collected} objects")

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Get final stats
        stats = get_memory_stats()
        current_mb = stats.get("current_mb", 0)
        if isinstance(current_mb, (int, float)):
            logger.info(f"🧹 Cleanup completed. Current: {current_mb:.1f}MB")

        return {"success": True, **stats}
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return {"success": False, "error": str(e)}


def final_cleanup():
    """Final cleanup function registered with atexit."""
    try:
        tracemalloc.stop()
        _cleanup()
        logger.info("🧹 Final memory cleanup completed")
    except Exception as e:
        logger.warning(f"Error during final cleanup: {e}")


# Register final cleanup
atexit.register(final_cleanup)


# Export public API
__all__ = [
    "memory_management",
    "get_memory_stats",
    "diagnose_memory_leaks",
    "force_comprehensive_cleanup",
    "register_for_cleanup",
]
