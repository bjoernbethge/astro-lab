"""
Test Configuration - Pytest Configuration and Fixtures
=====================================================

Provides pytest configuration and common fixtures for testing
the AstroLab framework.
"""

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
import pytest
import torch
import polars as pl

# Add src to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

# Import AstroDataset for fixtures
from astro_lab.data.core import AstroDataset

# Suppress common warnings during testing
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Memory management
import gc


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
    
    # Clean up Blender memory if available (safely)
    try:
        import bpy
        if hasattr(bpy, 'context') and bpy.context is not None:
            bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except (ImportError, AttributeError, RuntimeError):
        pass
    
    # Force garbage collection multiple times (PyTorch recommendation from forums)
    for _ in range(3):
        gc.collect()
    
    # Clear Python's internal caches
    import sys
    if hasattr(sys, '_clear_type_cache'):
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


@pytest.fixture(scope="session")
def test_data_dir(project_root_dir: Path) -> Generator[Path, None, None]:
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory(prefix="astro_lab_test_") as tmp_dir:
        test_dir = Path(tmp_dir)
        yield test_dir


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Real data fixtures - direct path access
@pytest.fixture(scope="session")
def gaia_data_path(data_dir: Path) -> Optional[Path]:
    """Get path to real Gaia data file."""
    gaia_dir = data_dir / "raw" / "gaia"
    if not gaia_dir.exists():
        pytest.skip("Gaia data directory not found")
    
    # Prefer the smaller mag10.0 file for testing
    mag10_file = gaia_dir / "gaia_dr3_bright_all_sky_mag10.0.parquet"
    if mag10_file.exists():
        return mag10_file
    
    mag12_file = gaia_dir / "gaia_dr3_bright_all_sky_mag12.0.parquet"
    if mag12_file.exists():
        return mag12_file
    
    pytest.skip("No Gaia data files found")


@pytest.fixture(scope="session")
def nsa_data_path(data_dir: Path) -> Optional[Path]:
    """Get path to real NSA data file."""
    nsa_dir = data_dir / "raw" / "nsa"
    if not nsa_dir.exists():
        pytest.skip("NSA data directory not found")
    
    # Prefer processed parquet files for testing
    processed_file = data_dir / "nsa_processed.parquet"
    if processed_file.exists():
        return processed_file
    
    # Check for raw FITS files
    nsa_v0_file = nsa_dir / "nsa_v0_1_2.fits"
    if nsa_v0_file.exists():
        return nsa_v0_file
    
    nsa_v1_file = nsa_dir / "nsa_v1_0_1.fits"
    if nsa_v1_file.exists():
        return nsa_v1_file
    
    pytest.skip("No NSA data files found")


@pytest.fixture(scope="session")
def exoplanet_data_path(data_dir: Path) -> Optional[Path]:
    """Get path to real exoplanet data file."""
    exo_dir = data_dir / "raw" / "exoplanet"
    if not exo_dir.exists():
        pytest.skip("Exoplanet data directory not found")
    
    exo_file = exo_dir / "confirmed_exoplanets.parquet"
    if exo_file.exists():
        return exo_file
    
    pytest.skip("No exoplanet data files found")


@pytest.fixture(scope="session")
def linear_data_path(data_dir: Path) -> Optional[Path]:
    """Get path to real LINEAR data file."""
    linear_dir = data_dir / "raw" / "linear"
    if not linear_dir.exists():
        pytest.skip("LINEAR data directory not found")
    
    # Check for compressed data files
    linear_tar = data_dir / "raw" / "allLINEARfinal_dat.tar.gz"
    if linear_tar.exists():
        return linear_tar
    
    linear_targets = data_dir / "raw" / "allLINEARfinal_targets.dat.gz"
    if linear_targets.exists():
        return linear_targets
    
    pytest.skip("No LINEAR data files found")


@pytest.fixture(scope="session")
def rrlyrae_data_path(data_dir: Path) -> Optional[Path]:
    """Get path to real RR Lyrae data file."""
    rr_files = [
        data_dir / "RRLyrae.fit",
        data_dir / "processed" / "rrlyrae_real_data_cleaned.parquet"
    ]
    
    for rr_file in rr_files:
        if rr_file.exists():
            return rr_file
    
    pytest.skip("No RR Lyrae data files found")


# AstroDataset fixtures - direct dataset creation
@pytest.fixture(scope="session")
def gaia_dataset():
    """Create Gaia AstroDataset fixture using processed data."""
    try:
        # Use a smaller processed file for testing (n100 exists)
        return AstroDataset(survey="gaia", max_samples=100, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create Gaia dataset: {e}")


@pytest.fixture(scope="session")
def nsa_dataset():
    """Create NSA AstroDataset fixture using processed data."""
    try:
        return AstroDataset(survey="nsa", max_samples=50, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create NSA dataset: {e}")


@pytest.fixture(scope="session")
def exoplanet_dataset():
    """Create Exoplanet AstroDataset fixture using processed data."""
    try:
        return AstroDataset(survey="exoplanet", max_samples=30, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create Exoplanet dataset: {e}")


@pytest.fixture(scope="session")
def linear_dataset():
    """Create LINEAR AstroDataset fixture using processed data."""
    try:
        return AstroDataset(survey="linear", max_samples=50, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create LINEAR dataset: {e}")


@pytest.fixture(scope="session")
def rrlyrae_dataset():
    """Create RR Lyrae AstroDataset fixture using processed data."""
    try:
        return AstroDataset(survey="rrlyrae", max_samples=20, k_neighbors=8)
    except Exception as e:
        pytest.skip(f"Could not create RR Lyrae dataset: {e}")


@pytest.fixture(scope="session")
def multiple_datasets_available(
    gaia_dataset, 
    nsa_dataset, 
    exoplanet_dataset
) -> bool:
    """Check if multiple datasets are available."""
    return all([gaia_dataset, nsa_dataset, exoplanet_dataset])


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


@pytest.fixture
def tng50_test_data(tmp_path):
    """Create realistic TNG50 test data for testing."""
    import polars as pl
    import numpy as np
    
    # Create TNG50 data directory
    tng50_dir = tmp_path / "data" / "raw" / "tng50"
    tng50_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate realistic TNG50 particle data
    n_particles = 1000
    
    # Create realistic particle data
    particle_data = {
        "x": np.random.uniform(-50, 50, n_particles),  # Mpc
        "y": np.random.uniform(-50, 50, n_particles),  # Mpc
        "z": np.random.uniform(-50, 50, n_particles),  # Mpc
        "vx": np.random.normal(0, 200, n_particles),   # km/s
        "vy": np.random.normal(0, 200, n_particles),   # km/s
        "vz": np.random.normal(0, 200, n_particles),   # km/s
        "mass": np.random.lognormal(10, 0.5, n_particles),  # M☉
        "stellar_mass": np.random.lognormal(9, 0.3, n_particles),  # M☉
        "gas_mass": np.random.lognormal(9.5, 0.4, n_particles),    # M☉
        "metallicity": np.random.uniform(0.001, 0.02, n_particles),  # Z☉
        "age": np.random.uniform(0, 13.8, n_particles),  # Gyr
        "particle_type": np.random.choice([0, 1, 2, 3, 4, 5], n_particles),  # Gas, DM, etc.
    }
    
    # Create DataFrame and save
    df = pl.DataFrame(particle_data)
    parquet_file = tng50_dir / "tng50_particles.parquet"
    df.write_parquet(parquet_file)
    
    # Create metadata file
    metadata = {
        "simulation": "TNG50-1",
        "snapshot": 99,
        "redshift": 0.0,
        "box_size": 35.0,  # Mpc
        "particle_count": n_particles,
        "units": {
            "length": "Mpc",
            "velocity": "km/s", 
            "mass": "M☉",
            "time": "Gyr"
        }
    }
    
    return {
        "data_file": parquet_file,
        "metadata": metadata,
        "data_dir": tng50_dir,
        "particle_data": particle_data
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests that require CUDA"
    )
    config.addinivalue_line(
        "markers", "data: marks tests that require real data"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Mark tests that require CUDA
        if "cuda" in item.name.lower():
            item.add_marker(pytest.mark.cuda)
        
        # Mark tests that require real data
        if any(keyword in item.name.lower() for keyword in ["gaia", "nsa", "exoplanet", "data"]):
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
