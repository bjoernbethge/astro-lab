"""
Test Configuration - Pytest Configuration and Fixtures
=====================================================

Provides pytest configuration and common fixtures for testing
the AstroLab framework.
"""

import gc
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import pytest
import torch

# Performance optimization: Only disable CUDA if explicitly requested
# Use CUDA_VISIBLE_DEVICES="" to disable CUDA for specific tests if needed
# Default behavior: Keep CUDA available if present

# Optimize PyTorch settings for testing
torch.set_num_threads(1)  # Reduce thread contention in parallel tests
torch.set_grad_enabled(False)  # Disable gradients by default for faster tests

# Add src to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import AstroDataset for fixtures
# from astro_lab.data.core import AstroDataset

# Suppress common warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ResourceWarning)

# Memory management (gc already imported above)

# Test performance helpers
def is_ci_environment():
    """Check if we're running in a CI environment."""
    ci_vars = ["CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS", "TRAVIS"]
    return any(os.environ.get(var) for var in ci_vars)

def should_skip_slow_tests():
    """Check if slow tests should be skipped."""
    return os.environ.get("SKIP_SLOW_TESTS", "").lower() in ("1", "true", "yes")

# Pytest hooks for performance optimization
def pytest_runtest_setup(item):
    """Setup for each test - apply skip conditions."""
    # Skip slow tests if requested
    if should_skip_slow_tests() and "slow" in item.keywords:
        pytest.skip("Skipping slow test (SKIP_SLOW_TESTS=1)")
    
    # Skip tests that require data if in CI
    if is_ci_environment() and "requires_data" in item.keywords:
        pytest.skip("Skipping data-dependent test in CI")
    
    # Skip CUDA tests if no GPU available
    if "cuda" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture(autouse=True)
def cleanup_torch_memory():
    """Automatically clean up PyTorch memory after each test following PyTorch best practices."""
    yield

    # Clear PyTorch cache (most important step)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Reset memory stats to prevent accumulation
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except AttributeError:
            # Older PyTorch versions don't have these functions
            pass

    # Clear autograd computational graphs (main source of memory leaks)
    # This is the most important part according to PyTorch forums
    torch.autograd.set_grad_enabled(False)
    torch.autograd.set_grad_enabled(True)

    # Blender cleanup removed - no longer needed due to lazy loading

    # Force garbage collection multiple times (PyTorch recommendation from forums)
    for _ in range(3):
        gc.collect()

    # Clear Python's internal caches
    import sys

    if hasattr(sys, "_clear_type_cache"):
        sys._clear_type_cache()


@pytest.fixture(autouse=True)
def set_torch_deterministic():
    """Set PyTorch to deterministic mode for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)

    # Set deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    yield

    # Reset after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


@pytest.fixture(scope="session")
def project_root_dir() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root_dir: Path) -> Path:
    """Get the data directory."""
    return project_root_dir / "data"


# Removed test_data_dir - use system data paths instead


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Real data fixtures - use AstroDataset system instead of manual paths


# AstroDataset fixtures - simplified using the existing system with proper root
@pytest.fixture(scope="session")
def gaia_dataset(data_dir: Path):
    """Create Gaia AstroDataset using the existing data system."""
    from astro_lab.data.core import AstroDataset
    try:
        return AstroDataset(root=str(data_dir), survey="gaia", max_samples=1000)
    except Exception as e:
        pytest.skip(f"Could not create Gaia dataset: {e}")


@pytest.fixture(scope="session")
def nsa_dataset(data_dir: Path):
    """Create NSA AstroDataset using the existing data system."""
    from astro_lab.data.core import AstroDataset
    try:
        return AstroDataset(root=str(data_dir), survey="nsa", max_samples=1000)
    except Exception as e:
        pytest.skip(f"Could not create NSA dataset: {e}")


@pytest.fixture(scope="session")
def exoplanet_dataset(data_dir: Path):
    """Create exoplanet AstroDataset using the existing data system."""
    from astro_lab.data.core import AstroDataset
    try:
        return AstroDataset(root=str(data_dir), survey="exoplanet", max_samples=1000)
    except Exception as e:
        pytest.skip(f"Could not create Exoplanet dataset: {e}")


@pytest.fixture(scope="session")
def linear_dataset(data_dir: Path):
    """Create LINEAR AstroDataset using the existing data system."""
    from astro_lab.data.core import AstroDataset
    try:
        return AstroDataset(root=str(data_dir), survey="linear", max_samples=1000)
    except Exception as e:
        pytest.skip(f"Could not create LINEAR dataset: {e}")


@pytest.fixture(scope="session")
def rrlyrae_dataset(data_dir: Path):
    """Create RR Lyrae AstroDataset using the existing data system."""
    from astro_lab.data.core import AstroDataset
    try:
        return AstroDataset(root=str(data_dir), survey="rrlyrae", max_samples=100, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create RR Lyrae dataset: {e}")


# Legacy fixtures for backwards compatibility
@pytest.fixture(scope="session")
def test_data_dir(project_root_dir: Path):
    """Temporary test data directory for backwards compatibility."""
    import tempfile
    with tempfile.TemporaryDirectory(prefix="astro_lab_test_") as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture(scope="session")  
def gaia_data_path(data_dir: Path):
    """Legacy fixture - returns path to gaia data if available."""
    gaia_files = [
        data_dir / "raw" / "gaia" / "gaia_dr3_bright_all_sky_mag12.0.parquet",
        data_dir / "processed" / "gaia" / "gaia_dr3_bright_all_sky_mag12.0_processed.parquet"
    ]
    for gaia_file in gaia_files:
        if gaia_file.exists():
            return gaia_file
    return None


@pytest.fixture(scope="session")  
def nsa_data_path(data_dir: Path):
    """Legacy fixture - returns path to nsa data if available."""
    nsa_files = [
        data_dir / "raw" / "nsa" / "nsa.parquet",
        data_dir / "raw" / "nsa" / "nsa_v1_0_1.parquet",
        data_dir / "processed" / "nsa" / "nsa_v1_0_1_processed.parquet"
    ]
    for nsa_file in nsa_files:
        if nsa_file.exists():
            return nsa_file
    return None


@pytest.fixture(scope="session")  
def linear_data_path(data_dir: Path):
    """Legacy fixture - returns path to linear data if available."""
    linear_files = [
        data_dir / "raw" / "linear" / "linear_raw.parquet",
        data_dir / "processed" / "linear" / "linear_raw_processed.parquet"
    ]
    for linear_file in linear_files:
        if linear_file.exists():
            return linear_file
    return None


# Polars DataFrame fixtures - using AstroDataManager for consistency
@pytest.fixture(scope="session")
def gaia_df() -> Optional[pl.DataFrame]:
    """Load Gaia data as Polars DataFrame using the data manager system."""
    try:
        from astro_lab.data.manager import AstroDataManager
        manager = AstroDataManager()
        # Load using existing catalog system if available
        catalogs = manager.list_catalogs()
        gaia_catalogs = catalogs.filter(pl.col("name").str.contains("gaia"))
        if len(gaia_catalogs) > 0:
            catalog_path = gaia_catalogs[0]["path"]
            return manager.load_catalog(catalog_path).head(1000)  # Limit for tests
        return None
    except Exception:
        return None


@pytest.fixture(scope="session")
def nsa_df() -> Optional[pl.DataFrame]:
    """Load NSA data as Polars DataFrame using the data manager system."""
    try:
        from astro_lab.data.manager import AstroDataManager
        manager = AstroDataManager()
        catalogs = manager.list_catalogs()
        nsa_catalogs = catalogs.filter(pl.col("name").str.contains("nsa"))
        if len(nsa_catalogs) > 0:
            catalog_path = nsa_catalogs[0]["path"]
            return manager.load_catalog(catalog_path).head(1000)
        return None
    except Exception:
        return None


@pytest.fixture(scope="session")
def linear_df() -> Optional[pl.DataFrame]:
    """Load LINEAR data as Polars DataFrame using the data manager system."""
    try:
        from astro_lab.data.manager import AstroDataManager
        manager = AstroDataManager()
        catalogs = manager.list_catalogs()
        linear_catalogs = catalogs.filter(pl.col("name").str.contains("linear"))
        if len(linear_catalogs) > 0:
            catalog_path = linear_catalogs[0]["path"]
            return manager.load_catalog(catalog_path).head(1000)
        return None
    except Exception:
        return None


# Utility fixtures
@pytest.fixture
def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def skip_if_no_astropy():
    """Skip test if astropy is not available."""
    try:
        import astropy
    except ImportError:
        pytest.skip("astropy not available")


@pytest.fixture
def skip_if_no_torch_geometric():
    """Skip test if PyTorch Geometric is not available."""
    try:
        import torch_geometric
    except ImportError:
        pytest.skip("PyTorch Geometric not available")


@pytest.fixture
def skip_if_no_mlflow():
    """Skip test if MLflow is not available."""
    try:
        import mlflow
    except ImportError:
        pytest.skip("MLflow not available")


@pytest.fixture
def cleanup_mlflow():
    """Clean up MLflow experiments after testing."""
    yield
    try:
        import mlflow

        # Clean up any test experiments
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        for exp in experiments:
            if "test" in exp.name.lower():
                try:
                    client.delete_experiment(exp.experiment_id)
                except Exception:
                    pass
    except ImportError:
        pass


# TNG fixtures removed - we use real data only


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "cuda: marks tests that require CUDA")
    config.addinivalue_line("markers", "data: marks tests that require real data")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests that require CUDA
        if "cuda" in item.name.lower():
            item.add_marker(pytest.mark.cuda)

        # Mark tests that require real data
        if any(
            keyword in item.name.lower()
            for keyword in ["gaia", "nsa", "exoplanet", "data"]
        ):
            item.add_marker(pytest.mark.data)

        # Mark integration tests
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)

        # Mark slow tests
        if any(keyword in item.name.lower() for keyword in ["slow", "large", "heavy"]):
            item.add_marker(pytest.mark.slow)


def pytest_sessionfinish(session, exitstatus):
    """Clean up after test session."""
    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection
    gc.collect()
