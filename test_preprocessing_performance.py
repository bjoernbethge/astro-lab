"""
Test Preprocessing Performance
=============================

Script to test and benchmark the optimized preprocessing for all surveys.
"""

import logging
import time
from pathlib import Path

import psutil
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage."""
    # CPU Memory
    cpu_percent = psutil.virtual_memory().percent
    cpu_used_gb = psutil.virtual_memory().used / (1024**3)

    # GPU memory (if available)
    gpu_percent = 0
    gpu_used_gb = 0

    if torch.cuda.is_available():
        try:
            gpu_memory = torch.cuda.mem_get_info()
            gpu_used = gpu_memory[1] - gpu_memory[0]
            gpu_used_gb = gpu_used / (1024**3)
            gpu_percent = (gpu_used / gpu_memory[1]) * 100
        except Exception:
            pass

    return {
        "cpu_percent": cpu_percent,
        "cpu_used_gb": cpu_used_gb,
        "gpu_percent": gpu_percent,
        "gpu_used_gb": gpu_used_gb,
    }


def test_survey_preprocessing(survey_name, max_samples=None, create_hetero=False):
    """Test preprocessing for a specific survey."""
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing {survey_name.upper()} preprocessing")
    logger.info(f"{'=' * 60}")

    # Import preprocessor
    try:
        module_name = f"astro_lab.data.preprocessors.{survey_name}"
        import importlib

        module = importlib.import_module(module_name)

        # Get preprocessor class
        if survey_name == "nsa":
            PreprocessorClass = getattr(module, "NSAPreprocessor")
        elif survey_name == "gaia":
            PreprocessorClass = getattr(module, "GaiaPreprocessor")
        elif survey_name == "sdss":
            PreprocessorClass = getattr(module, "SDSSPreprocessor")
        elif survey_name == "des":
            PreprocessorClass = getattr(module, "DesPreprocessor")
        else:
            class_name = f"{survey_name.capitalize()}Preprocessor"
            PreprocessorClass = getattr(module, class_name)

    except Exception as e:
        logger.error(f"Could not load preprocessor for {survey_name}: {e}")
        return None

    # Initialize preprocessor
    preprocessor = PreprocessorClass(
        config={
            "survey_name": survey_name,
            "data_dir": str(Path("data")),
            "device": "auto",
            "gpu_batch_size": 10000,
            "edge_batch_size": 50000,
        }
    )

    # Memory before
    mem_before = get_memory_usage()
    logger.info(
        f"Memory before: CPU {mem_before['cpu_used_gb']:.2f}GB ({mem_before['cpu_percent']:.1f}%), "
        f"GPU {mem_before['gpu_used_gb']:.2f}GB ({mem_before['gpu_percent']:.1f}%)"
    )

    # Start timing
    start_time = time.time()

    try:
        # Load and preprocess data
        df = preprocessor.preprocess()

        # End timing
        process_time = time.time() - start_time

        # Memory after
        mem_after = get_memory_usage()
        logger.info(
            f"Memory after: CPU {mem_after['cpu_used_gb']:.2f}GB ({mem_after['cpu_percent']:.1f}%), "
            f"GPU {mem_after['gpu_used_gb']:.2f}GB ({mem_after['gpu_percent']:.1f}%)"
        )

        # Memory usage
        cpu_mem_used = mem_after["cpu_used_gb"] - mem_before["cpu_used_gb"]
        gpu_mem_used = mem_after["gpu_used_gb"] - mem_before["gpu_used_gb"]

        # Results
        results = {
            "survey": survey_name,
            "n_objects": len(df),
            "process_time": process_time,
            "cpu_memory_gb": cpu_mem_used,
            "gpu_memory_gb": gpu_mem_used,
            "graph_type": "homogeneous",
        }

        # Log results
        logger.info(f"\nResults for {survey_name}:")
        logger.info(f"  Objects: {results['n_objects']:,}")
        logger.info(f"  Processing time: {results['process_time']:.2f}s")
        logger.info(f"  CPU memory used: {results['cpu_memory_gb']:.2f}GB")
        logger.info(f"  GPU memory used: {results['gpu_memory_gb']:.2f}GB")

        # Clean up
        del df
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    except Exception as e:
        logger.error(f"Error processing {survey_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_dataset_integration(survey_name, max_samples=None):
    """Test the dataset integration."""
    logger.info(f"\nTesting dataset integration for {survey_name}")

    try:
        from astro_lab.data.dataset import SurveyGraphDataset

        start_time = time.time()

        dataset = SurveyGraphDataset(
            survey=survey_name,
            max_samples=max_samples,
            add_astronomical_features=True,
            add_cosmic_web=False,  # Skip for performance test
            force_reload=True,
        )

        # Get first graph
        graph = dataset[0]

        # Get info
        info = dataset.get_info()
        stats = dataset.get_statistics()

        process_time = time.time() - start_time

        logger.info(f"Dataset created in {process_time:.2f}s")
        logger.info(f"Info: {info}")
        logger.info(f"Stats: {stats}")

        return True

    except Exception as e:
        logger.error(f"Dataset integration failed for {survey_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run preprocessing performance tests."""

    # Test surveys
    surveys = [
        "gaia",
        # "nsa",  # Skip if not available
        # "des",  # Skip if not available
    ]

    results = {}

    # First test just the preprocessors
    logger.info("\n" + "=" * 60)
    logger.info("TESTING PREPROCESSORS")
    logger.info("=" * 60)

    for survey in surveys:
        try:
            # Test with small sample size first
            result = test_survey_preprocessing(survey, max_samples=1000)
            if result:
                results[survey] = result
        except Exception as e:
            logger.error(f"Error processing {survey}: {e}")
            continue

    # Test dataset integration
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DATASET INTEGRATION")
    logger.info("=" * 60)

    for survey in surveys:
        if survey in results:  # Only test if preprocessing worked
            test_dataset_integration(survey, max_samples=100)

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info(f"{'=' * 60}")

    for survey, result in results.items():
        if result:
            logger.info(
                f"{survey}: "
                f"{result['n_objects']} objects, "
                f"{result['process_time']:.1f}s, "
                f"CPU {result['cpu_memory_gb']:.1f}GB, "
                f"GPU {result['gpu_memory_gb']:.1f}GB"
            )


if __name__ == "__main__":
    main()
