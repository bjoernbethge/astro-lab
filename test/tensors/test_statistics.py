"""
Tests for StatisticsTensor
=========================

Test suite for astronomical statistical operations.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import StatisticsTensor


class TestStatisticsTensor:
    """Test StatisticsTensor functionality."""

    @pytest.fixture
    def sample_photometric_data(self):
        """Create sample photometric data for testing."""
        np.random.seed(42)
        n_objects = 200

        # Create realistic magnitude data
        # Simulate main sequence stars with scatter
        g_mag = np.random.normal(18.0, 3.0, n_objects)
        r_mag = g_mag - np.random.normal(0.7, 0.3, n_objects)  # g-r color
        i_mag = r_mag - np.random.normal(0.3, 0.2, n_objects)  # r-i color

        # Add some bright and faint outliers
        g_mag[:10] = np.random.uniform(12.0, 14.0, 10)  # Bright stars
        g_mag[-10:] = np.random.uniform(24.0, 26.0, 10)  # Faint stars

        data = np.column_stack([g_mag, r_mag, i_mag])

        # Create spatial coordinates
        ra = np.random.uniform(0, 360, n_objects)
        dec = np.random.uniform(-30, 30, n_objects)
        coordinates = np.column_stack([ra, dec])

        # Create weights (completeness)
        weights = np.ones(n_objects)
        # Reduce weights for faint objects
        faint_mask = g_mag > 22.0
        weights[faint_mask] = 0.5

        return data, coordinates, weights

    @pytest.fixture
    def stats_tensor(self, sample_photometric_data):
        """Create StatisticsTensor instance."""
        data, coords, weights = sample_photometric_data
        return StatisticsTensor(data=data, coordinates=coords, weights=weights)

    def test_initialization(self, sample_photometric_data):
        """Test StatisticsTensor initialization."""
        data, coords, weights = sample_photometric_data

        # Test with all parameters - use keyword arguments for Pydantic compatibility
        tensor = StatisticsTensor(data=data, coordinates=coords, weights=weights)
        assert tensor.n_objects == 200
        assert tensor.n_observables == 3
        assert tensor.coordinates.shape == (200, 2)
        assert tensor.weights.shape == (200,)

        # Test without coordinates and weights
        tensor_simple = StatisticsTensor(data=data)
        assert tensor_simple.n_objects == 200
        assert tensor_simple.coordinates is None
        assert tensor_simple.weights.shape == (200,)  # Should default to ones

        # Test 1D data
        data_1d = data[:, 0]
        tensor_1d = StatisticsTensor(data=data_1d)
        assert tensor_1d.n_observables == 1

    def test_luminosity_function(self, stats_tensor):
        """Test luminosity function computation."""
        # Test basic luminosity function
        bin_centers, phi = stats_tensor.luminosity_function(
            magnitude_column=0,  # g-band
            bins=20,
            magnitude_range=(12.0, 26.0),
        )

        assert len(bin_centers) == 20
        assert len(phi) == 20
        assert torch.all(phi >= 0)  # Number density should be non-negative

        # Check that function was stored
        stored_func = stats_tensor.get_function("luminosity_function")
        assert "bin_centers" in stored_func
        assert "phi" in stored_func
        assert stored_func["method"] == "1/Vmax"

    def test_color_magnitude_diagram(self, stats_tensor):
        """Test color-magnitude diagram creation."""
        mag_centers, color_centers, density = stats_tensor.color_magnitude_diagram(
            magnitude_column=0,  # g-band
            color_column=1,  # r-band (will compute g-r color)
            bins=(25, 25),
        )

        assert len(mag_centers) == 25
        assert len(color_centers) == 25
        assert density.shape == (25, 25)
        assert torch.all(density >= 0)

        # Check stored results
        stored_cmd = stats_tensor.get_function("cmd")
        assert "density" in stored_cmd
        assert "ranges" in stored_cmd

    def test_two_point_correlation(self, stats_tensor):
        """Test two-point correlation function."""
        # Test correlation function computation
        r_centers, xi_r = stats_tensor.two_point_correlation(
            r_bins=10,
            r_range=(0.1, 10.0),
            estimator="davis_peebles",  # Use simpler estimator for testing
        )

        assert len(r_centers) == 10
        assert len(xi_r) == 10
        assert torch.all(torch.isfinite(xi_r))

        # Check stored results
        stored_xi = stats_tensor.get_function("xi_r")
        assert "r_centers" in stored_xi
        assert "xi_r" in stored_xi
        assert stored_xi["estimator"] == "davis_peebles"

    def test_bootstrap_errors(self, stats_tensor):
        """Test bootstrap error estimation."""
        # Compute a luminosity function first
        bin_centers, phi = stats_tensor.luminosity_function(
            magnitude_column=0, bins=10, function_name="test_lf"
        )
        # Ensure function is stored
        if "test_lf" not in stats_tensor.get_metadata("computed_functions", {}):
            stats_tensor.luminosity_function(magnitude_column=0, bins=10, function_name="test_lf")

        # Compute bootstrap errors
        errors = stats_tensor.bootstrap_errors("test_lf", n_bootstrap=10)

        assert "mean" in errors
        assert "std_error" in errors
        assert "lower_ci" in errors
        assert "upper_ci" in errors

        # Check that errors have correct length (same as bins)
        assert len(errors["mean"]) == len(
            bin_centers
        )  # Should match bin_centers length
        assert len(errors["std_error"]) == len(bin_centers)

        # Check that all errors are positive
        assert torch.all(errors["std_error"] >= 0)

    def test_jackknife_errors(self, stats_tensor):
        """Test jackknife error estimation."""
        # Compute a luminosity function first
        bin_centers, phi = stats_tensor.luminosity_function(
            magnitude_column=0, bins=10, function_name="test_lf_jk"
        )
        # Ensure function is stored
        if "test_lf_jk" not in stats_tensor.get_metadata("computed_functions", {}):
            stats_tensor.luminosity_function(magnitude_column=0, bins=10, function_name="test_lf_jk")

        # Compute jackknife errors
        errors = stats_tensor.jackknife_errors("test_lf_jk")

        assert "mean" in errors
        assert "std_error" in errors

        # Check that errors have correct length (same as bins)
        assert len(errors["mean"]) == len(
            bin_centers
        )  # Should match bin_centers length
        assert len(errors["std_error"]) == len(bin_centers)

        # Check that all errors are positive
        assert torch.all(errors["std_error"] >= 0)

    def test_completeness_analysis(self, stats_tensor):
        """Test completeness analysis."""
        mag_centers, completeness = stats_tensor.completeness_analysis(
            magnitude_column=0, detection_limit=24.0
        )

        assert len(mag_centers) > 0
        assert len(completeness) == len(mag_centers)
        assert torch.all(completeness >= 0)
        assert torch.all(completeness <= 1)

        # Check stored results
        stored_comp = stats_tensor.get_function("completeness")
        assert "magnitudes" in stored_comp
        assert "completeness" in stored_comp
        assert stored_comp["detection_limit"] == 24.0

    def test_function_storage_and_retrieval(self, stats_tensor):
        """Test function storage and retrieval system."""
        # Compute several functions
        stats_tensor.luminosity_function(bins=10, function_name="lf_g")
        stats_tensor.color_magnitude_diagram(bins=(15, 15), function_name="cmd_gr")

        # Test function listing
        functions = stats_tensor.list_functions()
        assert "lf_g" in functions
        assert "cmd_gr" in functions

        # Test function retrieval
        lf_results = stats_tensor.get_function("lf_g")
        assert "bin_centers" in lf_results
        assert "phi" in lf_results

        cmd_results = stats_tensor.get_function("cmd_gr")
        assert "density" in cmd_results

        # Test error for non-existent function
        with pytest.raises(ValueError):
            stats_tensor.get_function("nonexistent")

    def test_pair_counting(self, stats_tensor):
        """Test pair counting for correlation functions."""
        # Create simple test coordinates
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
        weights = torch.ones(4)
        r_edges = torch.tensor([0.5, 1.5, 2.5, 3.5])

        counts = stats_tensor._count_pairs(coords, coords, r_edges, weights, weights)

        assert len(counts) == 3
        assert torch.all(counts >= 0)

    def test_random_catalog_generation(self, stats_tensor):
        """Test random catalog generation."""
        coords = stats_tensor.coordinates
        n_random = 100

        random_coords = stats_tensor._generate_random_catalog(coords, n_random)

        assert random_coords.shape == (n_random, coords.shape[1])

        # Check that random coordinates are within bounds
        coord_min = coords.min(dim=0)[0]
        coord_max = coords.max(dim=0)[0]
        assert torch.all(random_coords >= coord_min)
        assert torch.all(random_coords <= coord_max)

    def test_different_estimators(self, stats_tensor):
        """Test different correlation function estimators."""
        estimators = ["davis_peebles"]  # Start with simple one

        for estimator in estimators:
            r_centers, xi_r = stats_tensor.two_point_correlation(
                r_bins=5, estimator=estimator, function_name=f"xi_{estimator}"
            )

            assert len(r_centers) == 5
            assert len(xi_r) == 5

            # Check that results were stored with correct estimator
            stored = stats_tensor.get_function(f"xi_{estimator}")
            assert stored["estimator"] == estimator

    def test_weighted_statistics(self, stats_tensor):
        """Test weighted statistical functions."""
        # Create weights (all equal for simplicity)
        weights = torch.ones(len(stats_tensor.data))

        # Test weighted luminosity function
        bin_centers, phi = stats_tensor.luminosity_function(
            magnitude_column=0, bins=5, weights=weights, function_name="weighted_lf"
        )

        assert len(bin_centers) == 5
        assert len(phi) == 5

        # Check that phi values are finite and non-negative
        assert torch.all(torch.isfinite(phi))
        assert torch.all(phi >= 0)

    def test_error_handling(self, sample_photometric_data):
        """Test error handling for invalid inputs."""
        data, coords, weights = sample_photometric_data

        # Test mismatched coordinates
        bad_coords = coords[:50]  # Wrong size
        with pytest.raises(ValueError):
            StatisticsTensor(data=data, coordinates=bad_coords)

        # Test mismatched weights
        bad_weights = weights[:50]  # Wrong size
        with pytest.raises(ValueError):
            StatisticsTensor(data=data, weights=bad_weights)

        # Test invalid function name
        tensor = StatisticsTensor(data=data)
        with pytest.raises(ValueError):
            tensor.bootstrap_errors("nonexistent_function")

    def test_tensor_metadata(self, stats_tensor):
        """Test tensor metadata handling."""
        assert stats_tensor.get_metadata("tensor_type") == "statistics"
        assert stats_tensor.coordinates is not None
        assert stats_tensor.weights is not None

    def test_repr(self, stats_tensor):
        """Test string representation."""
        repr_str = repr(stats_tensor)
        assert "StatisticsTensor" in repr_str
        assert "objects=200" in repr_str
        assert "observables=3" in repr_str


class TestStatisticsTensorIntegration:
    """Test StatisticsTensor integration with other components."""

    def test_survey_integration(self):
        """Test integration with survey data."""
        # Create survey-like data
        n_objects = 100

        # Magnitudes in multiple bands
        u_mag = np.random.normal(21.0, 2.0, n_objects)
        g_mag = np.random.normal(20.0, 2.0, n_objects)
        r_mag = np.random.normal(19.0, 2.0, n_objects)
        i_mag = np.random.normal(18.5, 2.0, n_objects)
        z_mag = np.random.normal(18.0, 2.0, n_objects)

        data = np.column_stack([u_mag, g_mag, r_mag, i_mag, z_mag])

        # Sky coordinates
        ra = np.random.uniform(0, 360, n_objects)
        dec = np.random.uniform(-30, 30, n_objects)
        coords = np.column_stack([ra, dec])

        tensor = StatisticsTensor(data=data, coordinates=coords)

        # Compute multi-band luminosity functions
        for i, band in enumerate(["u", "g", "r", "i", "z"]):
            bin_centers, phi = tensor.luminosity_function(
                magnitude_column=i, bins=15, function_name=f"lf_{band}"
            )
            assert len(bin_centers) == 15

    def test_time_series_statistics(self):
        """Test statistics for time series data."""
        # Create mock lightcurve data
        n_objects = 50
        n_epochs = 20

        # Time series magnitudes
        times = np.linspace(0, 100, n_epochs)
        magnitudes = []

        for i in range(n_objects):
            # Create variable lightcurve
            base_mag = np.random.uniform(18, 22)
            amplitude = np.random.uniform(0.1, 0.5)
            period = np.random.uniform(1, 10)

            mag_series = base_mag + amplitude * np.sin(2 * np.pi * times / period)
            magnitudes.append(mag_series)

        # Flatten for statistics (each epoch is an "observable")
        data = np.array(magnitudes)  # [n_objects, n_epochs]

        tensor = StatisticsTensor(data=data)

        # Compute statistics on magnitude variations
        assert tensor.n_objects == n_objects
        assert tensor.n_observables == n_epochs

    def test_large_dataset_performance(self):
        """Test performance with larger datasets."""
        # Create larger dataset
        n_objects = 1000

        data = np.random.randn(n_objects, 5)
        coords = np.random.uniform([0, -90], [360, 90], (n_objects, 2))

        tensor = StatisticsTensor(data=data, coordinates=coords)

        # Test that basic operations complete
        bin_centers, phi = tensor.luminosity_function(bins=20)
        assert len(bin_centers) == 20

        # Test correlation function (should use chunked computation)
        r_centers, xi_r = tensor.two_point_correlation(
            r_bins=5, estimator="davis_peebles"
        )
        assert len(r_centers) == 5


@pytest.mark.parametrize("n_bins", [5, 10, 20, 50])
def test_different_bin_numbers(n_bins):
    """Test luminosity function with different bin numbers."""
    data = np.random.normal(20, 2, (100, 1))
    tensor = StatisticsTensor(data=data)

    bin_centers, phi = tensor.luminosity_function(bins=n_bins)
    assert len(bin_centers) == n_bins
    assert len(phi) == n_bins


@pytest.mark.parametrize("method", ["1/Vmax", "histogram"])
def test_luminosity_function_methods(method):
    """Test different luminosity function methods."""
    data = np.random.normal(20, 2, (100, 1))
    tensor = StatisticsTensor(data=data)

    bin_centers, phi = tensor.luminosity_function(bins=10, method=method)
    assert len(bin_centers) == 10
    assert len(phi) == 10
