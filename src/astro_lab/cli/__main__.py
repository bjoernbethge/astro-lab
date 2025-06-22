#!/usr/bin/env python3
"""
AstroLab CLI - Unified Command Line Interface
===========================================

Provides a unified interface for all AstroLab operations with proper resource management.
"""

import gc
import logging
import sys
import warnings
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

# Configure logging early
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =========================================================================
# Advanced Memory Management with contextlib
# =========================================================================

@contextmanager
def cuda_memory_context(description: str = "CUDA operation"):
    """
    Advanced CUDA memory management context manager.

    Args:
        description: Description for logging
    """
    initial_memory = None
    try:
        # Import torch only when needed
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated()
            logger.debug(
                f"CUDA context [{description}]: Initial memory {initial_memory / 1024**2:.1f}MB"
            )
    except ImportError:
        pass

    try:
        yield
    finally:
        try:
            import torch

            if torch.cuda.is_available() and initial_memory is not None:
                final_memory = torch.cuda.memory_allocated()
                peak_memory = torch.cuda.max_memory_allocated()
                torch.cuda.empty_cache()
                logger.debug(
                    f"CUDA context [{description}]: "
                    f"Final: {final_memory / 1024**2:.1f}MB, "
                    f"Peak: {peak_memory / 1024**2:.1f}MB"
                )
        except ImportError:
            pass

@contextmanager
def comprehensive_cleanup_context(description: str = "Operation"):
    """
    Comprehensive resource cleanup context manager using contextlib.ExitStack.

    Args:
        description: Description for logging
    """
    with ExitStack() as stack:
        # Add CUDA memory management
        stack.enter_context(cuda_memory_context(description))

        # Track initial memory state
        initial_objects = len(gc.get_objects())

        try:
            yield stack
        finally:
            # Comprehensive cleanup sequence
            _perform_comprehensive_cleanup(description, initial_objects)

def _perform_comprehensive_cleanup(description: str, initial_objects: int):
    """Perform comprehensive cleanup with proper error handling."""
    cleanup_steps = [
        ("PyTorch cleanup", _cleanup_pytorch),
        ("Matplotlib cleanup", _cleanup_matplotlib),
        ("Blender cleanup", _cleanup_blender),
        ("Garbage collection", _cleanup_garbage),
        ("System caches", _cleanup_system_caches),
    ]

    for step_name, cleanup_func in cleanup_steps:
        with suppress(Exception):
            cleanup_func()
            logger.debug(f"✅ {step_name} completed for {description}")

    # Final memory report
    final_objects = len(gc.get_objects())
    object_diff = final_objects - initial_objects
    if object_diff > 1000:
        logger.warning(
            f"Memory context [{description}]: {object_diff} objects not freed"
        )
    else:
        logger.debug(f"Memory context [{description}]: {object_diff} object difference")

def _cleanup_pytorch():
    """Clean up PyTorch resources."""
    try:
        import torch

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Reset memory stats
            with suppress(AttributeError):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_accumulated_memory_stats()

        # Clear autograd graphs
        torch.autograd.set_grad_enabled(False)
        torch.autograd.set_grad_enabled(True)

    except ImportError:
        pass

def _cleanup_matplotlib():
    """Clean up matplotlib resources."""
    try:
        import matplotlib.pyplot as plt

        plt.close("all")

        # Clear matplotlib caches
        import matplotlib

        with suppress(AttributeError):
            matplotlib.font_manager._rebuild()

    except ImportError:
        pass

def _cleanup_blender():
    """Clean up Blender resources."""
    try:
        import bpy

        if hasattr(bpy, "context") and bpy.context is not None:
            with suppress(Exception):
                bpy.ops.outliner.orphans_purge(
                    do_local_ids=True, do_linked_ids=True, do_recursive=True
                )

    except ImportError:
        pass

def _cleanup_garbage():
    """Perform garbage collection."""
    # Multiple GC passes for thorough cleanup
    for _ in range(3):
        collected = gc.collect()
        if collected == 0:
            break

    # Clear generational garbage collection
    gc.set_debug(0)

def _cleanup_system_caches():
    """Clean up system-level caches."""
    import sys

    # Clear type cache
    if hasattr(sys, "_clear_type_cache"):
        sys._clear_type_cache()

    # Clear import cache for modules that might be reloaded
    modules_to_clear = [
        mod for mod in sys.modules.keys() if mod.startswith("astro_lab")
    ]
    for mod in modules_to_clear:
        if hasattr(sys.modules[mod], "__dict__"):
            with suppress(Exception):
                # Clear module-level caches
                mod_dict = sys.modules[mod].__dict__
                for key in list(mod_dict.keys()):
                    if key.startswith("_cache") or key.endswith("_cache"):
                        mod_dict.pop(key, None)

@contextmanager
def error_handling_context(operation_name: str):
    """
    Comprehensive error handling context manager.

    Args:
        operation_name: Name of the operation for error reporting
    """
    try:
        yield
    except KeyboardInterrupt:
        logger.info(f"\n⚠️ {operation_name} interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except ImportError as e:
        logger.error(f"❌ Missing dependency for {operation_name}: {e}")
        logger.info("💡 Try: uv sync --all-extras")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"❌ File not found in {operation_name}: {e}")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"❌ Permission denied in {operation_name}: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ {operation_name} failed: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)

# =========================================================================
# CLI Commands with Resource Management
# =========================================================================

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """🚀 AstroLab - Advanced Astronomical Data Processing & Machine Learning"""
    if ctx.invoked_subcommand is None:
        with comprehensive_cleanup_context("CLI welcome"):
            click.echo("🌟 Welcome to AstroLab!")
            click.echo()
            click.echo("Available commands:")
            click.echo("  preprocess  - Process astronomical survey data")
            click.echo("  train       - Train machine learning models")
            click.echo("  download    - Download survey data")
            click.echo()
            click.echo("Use --help with any command for detailed information.")
            click.echo("Example: uv run python -m astro_lab.cli preprocess --help")
        sys.exit(0)

@cli.command()
@click.argument("input_file", required=False, type=click.Path(exists=True))
@click.option(
    "--config", "-c", help="Configuration file or survey name (gaia, nsa, sdss)"
)
@click.option("--surveys", "-s", multiple=True, help="Survey names to process")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option(
    "--k-neighbors", "-k", default=8, help="Number of neighbors for graph construction"
)
@click.option(
    "--max-samples", "-n", type=int, help="Maximum number of samples to process"
)
@click.option(
    "--mode",
    type=click.Choice(["batch", "single", "stats", "tng50"]),
    default="batch",
    help="Processing mode",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def preprocess(
    input_file: Optional[str],
    config: Optional[str],
    surveys: List[str],
    output_dir: Optional[str],
    k_neighbors: int,
    max_samples: Optional[int],
    mode: str,
    verbose: bool,
):
    """
    🔄 Preprocess astronomical survey data with advanced resource management.

    This unified command handles all preprocessing tasks with automatic cleanup.

    Examples:
        uv run python -m astro_lab.cli preprocess
        uv run python -m astro_lab.cli preprocess data/gaia.parquet --config gaia
        uv run python -m astro_lab.cli preprocess --surveys gaia nsa
    """
    operation_name = "Preprocessing"

    with (
        comprehensive_cleanup_context(operation_name),
        error_handling_context(operation_name),
    ):
        # Set up logging level
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Lazy import to reduce memory footprint
        from astro_lab.data.config import DataConfig
        from astro_lab.data.manager import AstroDataManager

        logger.info(f"🚀 Starting {operation_name.lower()}...")

        # Configure processing parameters
        config_params = {
            "k_neighbors": k_neighbors,
            "max_samples": max_samples,
            "output_dir": output_dir or "data/processed",
            "surveys": list(surveys) if surveys else ["gaia", "nsa", "sdss"],
            "mode": mode,
        }

        # Load configuration
        if config:
            config_obj = DataConfig.from_name_or_path(config)
            config_params.update(config_obj.dict())

        # Initialize data manager with resource management
        with cuda_memory_context("Data manager initialization"):
            manager = AstroDataManager(**config_params)

        # Execute processing
        if input_file:
            logger.info(f"📂 Processing single file: {input_file}")
            with cuda_memory_context("Single file processing"):
                result = manager.process_file(Path(input_file))
        else:
            logger.info(f"📊 Processing surveys: {config_params['surveys']}")
            with cuda_memory_context("Batch processing"):
                result = manager.process_surveys()

        logger.info("✅ Preprocessing completed successfully!")

        # Memory usage summary
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"📊 Final memory usage: {memory_mb:.1f} MB")
        except ImportError:
            pass

@cli.command()
@click.option("--config", "-c", help="Training configuration file")
@click.option("--model", "-m", help="Model architecture")
@click.option("--epochs", "-e", type=int, default=100, help="Number of epochs")
@click.option("--batch-size", "-b", type=int, default=32, help="Batch size")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def train(
    config: Optional[str],
    model: Optional[str],
    epochs: int,
    batch_size: int,
    verbose: bool,
):
    """🧠 Train machine learning models with memory management."""
    operation_name = "Training"

    with (
        comprehensive_cleanup_context(operation_name),
        error_handling_context(operation_name),
    ):
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Lazy import
        from astro_lab.training.trainer import AstroTrainer

        logger.info("🧠 Starting model training...")

        with cuda_memory_context("Model training"):
            trainer = AstroTrainer(
                config=config, model=model, epochs=epochs, batch_size=batch_size
            )
            trainer.train()

        logger.info("✅ Training completed successfully!")

@cli.command()
@click.option("--survey", "-s", help="Survey to download (gaia, nsa, sdss)")
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def download(survey: Optional[str], output_dir: Optional[str], verbose: bool):
    """📥 Download survey data with resource management."""
    operation_name = "Download"

    with (
        comprehensive_cleanup_context(operation_name),
        error_handling_context(operation_name),
    ):
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Lazy import
        from astro_lab.cli.download import download_survey_data

        logger.info(f"📥 Downloading {survey or 'default'} survey data...")

        with cuda_memory_context("Data download"):
            download_survey_data(
                survey=survey or "gaia", output_dir=output_dir or "data/raw"
            )

        logger.info("✅ Download completed successfully!")

def main():
    """
    Main entry point with comprehensive resource management.

    This function ensures proper cleanup even if the CLI exits unexpectedly.
    """
    try:
        with comprehensive_cleanup_context("CLI main"):
            cli()
    except SystemExit as e:
        # Handle normal exits gracefully
        if e.code != 0:
            logger.debug(f"CLI exited with code: {e.code}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in CLI: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)
    finally:
        # Final cleanup message (only visible in debug mode)
        logger.debug("🧹 CLI cleanup completed")

if __name__ == "__main__":
    main()
