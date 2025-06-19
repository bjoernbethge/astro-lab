"""
Pytest configuration and fixtures for astro-lab tests.
"""

import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import numpy as np
import pytest
import torch

# Add src to Python path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

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


# Real data fixtures
@pytest.fixture(scope="session")
def gaia_data_available(data_dir: Path) -> bool:
    """Check if Gaia data is available."""
    gaia_files = [
        data_dir / "raw" / "gaia" / "gaia_dr3_bright_all_sky_mag10.0.parquet",
        data_dir / "raw" / "gaia" / "gaia_dr3_bright_all_sky_mag12.0.parquet"
    ]
    return any(f.exists() for f in gaia_files)


@pytest.fixture(scope="session")
def nsa_data_available(data_dir: Path) -> bool:
    """Check if NSA data is available."""
    nsa_files = [
        data_dir / "nsa_processed.parquet",
        data_dir / "nsa_v0_1_2.fits",
        data_dir / "datasets" / "nsa" / "catalog_sample_50.parquet",
        data_dir / "processed" / "nsa" / "nsa_catalog.parquet"
    ]
    return any(f.exists() for f in nsa_files)


@pytest.fixture(scope="session")
def exoplanet_data_available(data_dir: Path) -> bool:
    """Check if exoplanet data is available."""
    exo_files = [
        data_dir / "processed" / "exoplanet_graphs" / "raw" / "confirmed_exoplanets.parquet"
    ]
    return any(f.exists() for f in exo_files)


@pytest.fixture(scope="session")
def linear_data_available(data_dir: Path) -> bool:
    """Check if LINEAR data is available."""
    linear_files = [
        data_dir / "allLINEARfinal_dat.tar.gz",
        data_dir / "allLINEARfinal_targets.dat.gz",
        data_dir / "datasets" / "astroml" / "linear_lightcurves.parquet"
    ]
    return any(f.exists() for f in linear_files)


@pytest.fixture(scope="session")
def rrlyrae_data_available(data_dir: Path) -> bool:
    """Check if RR Lyrae data is available."""
    rr_files = [
        data_dir / "RRLyrae.fit",
        data_dir / "processed" / "rrlyrae_real_data_cleaned.parquet"
    ]
    return any(f.exists() for f in rr_files)


@pytest.fixture(scope="session")
def gaia_data_path(data_dir: Path, gaia_data_available: bool) -> Optional[Path]:
    """Get path to Gaia data file."""
    if not gaia_data_available:
        return None
    
    # Prefer the smaller mag10.0 file for testing
    mag10_file = data_dir / "raw" / "gaia" / "gaia_dr3_bright_all_sky_mag10.0.parquet"
    if mag10_file.exists():
        return mag10_file
    
    mag12_file = data_dir / "raw" / "gaia" / "gaia_dr3_bright_all_sky_mag12.0.parquet"
    if mag12_file.exists():
        return mag12_file
    
    return None


@pytest.fixture(scope="session")
def nsa_data_path(data_dir: Path, nsa_data_available: bool) -> Optional[Path]:
    """Get path to NSA data file."""
    if not nsa_data_available:
        return None
    
    # Prefer processed parquet files for testing
    processed_file = data_dir / "nsa_processed.parquet"
    if processed_file.exists():
        return processed_file
    
    catalog_file = data_dir / "datasets" / "nsa" / "catalog_sample_50.parquet"
    if catalog_file.exists():
        return catalog_file
    
    nsa_catalog = data_dir / "processed" / "nsa" / "nsa_catalog.parquet"
    if nsa_catalog.exists():
        return nsa_catalog
    
    return None


@pytest.fixture(scope="session")
def exoplanet_data_path(data_dir: Path, exoplanet_data_available: bool) -> Optional[Path]:
    """Get path to exoplanet data file."""
    if not exoplanet_data_available:
        return None
    
    exo_file = data_dir / "processed" / "exoplanet_graphs" / "raw" / "confirmed_exoplanets.parquet"
    if exo_file.exists():
        return exo_file
    
    return None


@pytest.fixture(scope="session")
def multiple_datasets_available(
    gaia_data_available: bool, 
    nsa_data_available: bool, 
    exoplanet_data_available: bool
) -> bool:
    """Check if multiple datasets are available for cross-survey operations."""
    available_count = sum([gaia_data_available, nsa_data_available, exoplanet_data_available])
    return available_count >= 2


# Skip conditions based on data availability
@pytest.fixture
def skip_if_no_gaia_data(gaia_data_available: bool):
    """Skip test if Gaia data is not available."""
    if not gaia_data_available:
        pytest.skip("Gaia data not available")


@pytest.fixture
def skip_if_no_nsa_data(nsa_data_available: bool):
    """Skip test if NSA data is not available."""
    if not nsa_data_available:
        pytest.skip("NSA data not available")


@pytest.fixture
def skip_if_no_exoplanet_data(exoplanet_data_available: bool):
    """Skip test if exoplanet data is not available."""
    if not exoplanet_data_available:
        pytest.skip("Exoplanet data not available")


@pytest.fixture
def skip_if_no_multiple_datasets(multiple_datasets_available: bool):
    """Skip test if multiple datasets are not available for cross-survey operations."""
    if not multiple_datasets_available:
        pytest.skip("Cross-survey operations require multiple datasets")


@pytest.fixture
def sample_tensor_data() -> Dict[str, torch.Tensor]:
    """Generate sample tensor data for testing."""
    return {
        "small_1d": torch.randn(10),
        "small_2d": torch.randn(5, 3),
        "medium_3d": torch.randn(10, 4, 3),
        "coordinates": torch.randn(100, 3),  # 3D spatial coordinates
        "magnitudes": torch.randn(50, 5),  # 5-band photometry
        "spectrum": torch.randn(200),  # Spectral data
        "lightcurve": torch.randn(100, 2),  # Time, flux
    }


@pytest.fixture
def sample_astronomical_data() -> Dict[str, Any]:
    """Generate sample astronomical data structures."""
    n_objects = 100
    n_bands = 5
    n_wavelengths = 200

    return {
        # Spatial coordinates (RA, Dec, Distance)
        "ra": np.random.uniform(0, 360, n_objects),
        "dec": np.random.uniform(-90, 90, n_objects),
        "distance": np.random.exponential(100, n_objects),  # Mpc
        # Photometric data
        "magnitudes": np.random.normal(20, 2, (n_objects, n_bands)),
        "mag_errors": np.random.exponential(0.1, (n_objects, n_bands)),
        "bands": ["u", "g", "r", "i", "z"],
        # Spectroscopic data
        "wavelengths": np.linspace(3000, 9000, n_wavelengths),  # Angstrom
        "flux": np.random.exponential(1e-17, (n_objects, n_wavelengths)),
        "flux_errors": np.random.exponential(1e-18, (n_objects, n_wavelengths)),
        # Time series data
        "time": np.sort(np.random.uniform(0, 1000, 50)),  # Days
        "lightcurve_flux": np.random.normal(1.0, 0.1, 50),
        # Metadata
        "redshift": np.random.exponential(0.1, n_objects),
        "stellar_mass": np.random.normal(10.5, 0.5, n_objects),  # log(M*/M☉)
        "object_type": np.random.choice(["galaxy", "star", "quasar"], n_objects),
    }


@pytest.fixture
def mock_fits_file(test_data_dir: Path) -> Path:
    """Create a mock FITS file for testing."""
    try:
        from astropy.io import fits
        from astropy.table import Table

        # Create sample astronomical table
        n_objects = 50
        table = Table(
            {
                "ra": np.random.uniform(0, 360, n_objects),
                "dec": np.random.uniform(-90, 90, n_objects),
                "mag_g": np.random.normal(20, 2, n_objects),
                "mag_r": np.random.normal(19.5, 2, n_objects),
                "mag_i": np.random.normal(19, 2, n_objects),
                "redshift": np.random.exponential(0.1, n_objects),
            }
        )

        fits_file = test_data_dir / "test_catalog.fits"
        table.write(fits_file, format="fits", overwrite=True)
        return fits_file

    except ImportError:
        pytest.skip("astropy not available for FITS testing")


@pytest.fixture
def mock_parquet_file(
    test_data_dir: Path, sample_astronomical_data: Dict[str, Any]
) -> Path:
    """Create a mock Parquet file for testing."""
    try:
        import polars as pl

        # Create DataFrame from sample data
        df_data = {
            "ra": sample_astronomical_data["ra"],
            "dec": sample_astronomical_data["dec"],
            "distance": sample_astronomical_data["distance"],
            "redshift": sample_astronomical_data["redshift"],
            "stellar_mass": sample_astronomical_data["stellar_mass"],
            "object_type": sample_astronomical_data["object_type"],
        }

        # Add photometric data
        for i, band in enumerate(sample_astronomical_data["bands"]):
            df_data[f"mag_{band}"] = sample_astronomical_data["magnitudes"][:, i]
            df_data[f"magerr_{band}"] = sample_astronomical_data["mag_errors"][:, i]

        df = pl.DataFrame(df_data)
        parquet_file = test_data_dir / "test_catalog.parquet"
        df.write_parquet(parquet_file)
        return parquet_file

    except ImportError:
        pytest.skip("polars not available for Parquet testing")


@pytest.fixture
def sample_graph_data() -> Dict[str, torch.Tensor]:
    """Generate sample graph data for PyTorch Geometric testing."""
    n_nodes = 100
    n_edges = 200

    # Node features (astronomical properties)
    x = torch.randn(n_nodes, 10)  # 10 features per node

    # Edge indices
    edge_index = torch.randint(0, n_nodes, (2, n_edges))

    # Edge attributes (distances, etc.)
    edge_attr = torch.randn(n_edges, 3)

    # Node positions (3D coordinates)
    pos = torch.randn(n_nodes, 3)

    # Graph-level target
    y = torch.randn(1)

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "pos": pos,
        "y": y,
        "batch": torch.zeros(n_nodes, dtype=torch.long),
    }


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


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and PyTorch memory settings."""
    # Set PyTorch memory management environment variables (based on community best practices)
    import os
    # Most important: PyTorch CUDA memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'
    # For debugging only - makes tests slower but helps identify memory issues
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Uncomment only for debugging
    os.environ['PYTHONHASHSEED'] = '0'  # Deterministic hash seed
    # Suppress known fake-bpy-module memory leak warnings (harmless)
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'  # Reduce file I/O overhead
    
    config.addinivalue_line("markers", "cuda: tests that require CUDA")
    config.addinivalue_line("markers", "slow: tests that take a long time to run")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "data: tests that require real data")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark CUDA tests
        if "cuda" in item.nodeid.lower() or any(
            "cuda" in str(mark) for mark in item.iter_markers()
        ):
            item.add_marker(pytest.mark.cuda)

        # Mark slow tests
        if any(
            slow_keyword in item.nodeid.lower()
            for slow_keyword in ["benchmark", "performance", "training"]
        ):
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid.lower() or "test_pipeline" in item.nodeid:
            item.add_marker(pytest.mark.integration)

        # Mark data tests
        if any(
            data_keyword in item.nodeid.lower()
            for data_keyword in ["gaia", "nsa", "exoplanet", "survey", "dataset"]
        ):
            item.add_marker(pytest.mark.data)

        # Mark unit tests (default)
        if not any(item.iter_markers(name) for name in ["integration", "slow", "data"]):
            item.add_marker(pytest.mark.unit)


def pytest_sessionfinish(session, exitstatus):
    """Clean up after all tests are finished."""
    # Final memory cleanup following PyTorch community best practices
    
    if torch.cuda.is_available():
        # Reset all CUDA memory statistics (PyTorch forums recommendation)
        try:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
        except AttributeError:
            pass
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Print CUDA memory stats for debugging
        print(f"\nCUDA Memory: Allocated={torch.cuda.memory_allocated() / 1024**2:.1f} MB, "
              f"Cached={torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    # Force multiple garbage collections (PyTorch community recommendation)
    collected_total = 0
    for i in range(5):
        collected = gc.collect()
        collected_total += collected
        if collected == 0:
            break
    
    if collected_total > 0:
        print(f"Total GC: Collected {collected_total} objects")
    
    # Clear Python's internal caches
    import sys
    if hasattr(sys, '_clear_type_cache'):
        sys._clear_type_cache()
    
    # Print final memory statistics
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"\nFinal Memory Usage: RSS={memory_info.rss / 1024 / 1024:.1f} MB")
        
        # Calculate memory efficiency
        if torch.cuda.is_available():
            gpu_mem_used = torch.cuda.memory_allocated() / 1024**2
            if gpu_mem_used > 0:
                print(f"GPU Memory Used: {gpu_mem_used:.1f} MB")
            
    except ImportError:
        pass
