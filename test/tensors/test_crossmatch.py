"""
Tests for CrossMatchTensor
=========================

Test suite for catalog cross-matching operations.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import CrossMatchTensor


class TestCrossMatchTensor:
    """Test CrossMatchTensor functionality."""

    @pytest.fixture
    def sample_catalogs(self):
        """Create sample catalogs for cross-matching."""
        np.random.seed(42)

        # Catalog A: Reference catalog
        n_a = 100
        ra_a = np.random.uniform(150, 160, n_a)  # 10 degree field
        dec_a = np.random.uniform(20, 30, n_a)  # 10 degree field
        mag_a = np.random.normal(20.0, 2.0, n_a)
        catalog_a = np.column_stack([ra_a, dec_a, mag_a])

        # Catalog B: Overlapping catalog with some matches
        n_b = 80
        # Create some true matches (first 30 objects)
        n_matches = 30
        ra_b = np.zeros(n_b)
        dec_b = np.zeros(n_b)

        # True matches with small offsets
        ra_b[:n_matches] = ra_a[:n_matches] + np.random.normal(
            0, 0.5 / 3600, n_matches
        )  # 0.5" scatter
        dec_b[:n_matches] = dec_a[:n_matches] + np.random.normal(
            0, 0.5 / 3600, n_matches
        )

        # Non-matches
        ra_b[n_matches:] = np.random.uniform(150, 160, n_b - n_matches)
        dec_b[n_matches:] = np.random.uniform(20, 30, n_b - n_matches)

        mag_b = np.random.normal(19.5, 2.0, n_b)
        catalog_b = np.column_stack([ra_b, dec_b, mag_b])

        return catalog_a, catalog_b, n_matches

    @pytest.fixture
    def crossmatch_tensor(self, sample_catalogs):
        """Create CrossMatchTensor instance."""
        cat_a, cat_b, _ = sample_catalogs
        return CrossMatchTensor(
            cat_a,
            cat_b,
            catalog_names=("reference", "target"),
            coordinate_columns={"a": [0, 1], "b": [0, 1]},
        )

    def test_initialization(self, sample_catalogs):
        """Test CrossMatchTensor initialization."""
        cat_a, cat_b, _ = sample_catalogs

        # Test with both catalogs
        tensor = CrossMatchTensor(cat_a, cat_b)
        assert tensor.get_metadata("catalog_info")["catalog_a"]["n_objects"] == 100
        assert tensor.get_metadata("catalog_info")["catalog_b"]["n_objects"] == 80

        # Test with dictionary input
        cat_a_dict = {"ra": cat_a[:, 0], "dec": cat_a[:, 1], "mag": cat_a[:, 2]}
        cat_b_dict = {"ra": cat_b[:, 0], "dec": cat_b[:, 1], "mag": cat_b[:, 2]}

        tensor_dict = CrossMatchTensor(cat_a_dict, cat_b_dict)
        assert tensor_dict.get_metadata("catalog_info")["catalog_a"]["n_objects"] == 100

        # Test self-matching (single catalog)
        tensor_self = CrossMatchTensor(cat_a)
        assert tensor_self.get_metadata("catalog_info")["catalog_b"] is None

    def test_catalog_data_access(self, crossmatch_tensor):
        """Test catalog data access methods."""
        cat_a_data = crossmatch_tensor.catalog_a_data
        cat_b_data = crossmatch_tensor.catalog_b_data

        assert cat_a_data.shape[0] == 100
        assert cat_b_data.shape[0] == 80
        assert cat_a_data.shape[1] == cat_b_data.shape[1]  # Same number of columns

        assert torch.is_tensor(cat_a_data)
        assert torch.is_tensor(cat_b_data)

    def test_sky_coordinate_matching(self, crossmatch_tensor, sample_catalogs):
        """Test sky coordinate matching."""
        _, _, expected_matches = sample_catalogs

        try:
            results = crossmatch_tensor.sky_coordinate_matching(
                tolerance_arcsec=2.0,  # 2 arcsecond tolerance
                method="nearest_neighbor",
            )

            assert "matches" in results
            assert "statistics" in results
            assert isinstance(results["matches"], list)

            # Check that we found some matches
            n_matches = len(results["matches"])
            assert n_matches > 0

            # Should find most of the planted matches
            assert n_matches >= expected_matches * 0.7  # At least 70% recovery

            # Check match structure
            if n_matches > 0:
                match = results["matches"][0]
                assert "index_a" in match
                assert "index_b" in match
                assert "distance" in match

            # Check statistics
            stats = results["statistics"]
            assert "n_matches" in stats
            assert "match_rate_a" in stats
            assert "match_rate_b" in stats

        except ImportError:
            pytest.skip("sklearn not available for coordinate matching")

    def test_proper_motion_matching(self, crossmatch_tensor):
        """Test proper motion matching."""
        try:
            results = crossmatch_tensor.proper_motion_matching(
                spatial_tolerance_arcsec=1.0,
                pm_tolerance_mas_yr=5.0,
                epoch_a=2000.0,
                epoch_b=2015.0,  # 15 year difference
            )

            assert "matches" in results
            assert "n_spatial_matches" in results
            assert "n_pm_matches" in results

            # PM matching should be more restrictive
            assert results["n_pm_matches"] <= results["n_spatial_matches"]

        except ImportError:
            pytest.skip("sklearn not available for proper motion matching")

    def test_bayesian_matching(self, crossmatch_tensor):
        """Test Bayesian matching with probabilities."""
        results = crossmatch_tensor.bayesian_matching(
            prior_density=1e-6,  # 1 object per square arcsecond
            tolerance_arcsec=5.0,
        )

        assert "matches" in results
        assert isinstance(results["matches"], list)

        # Check probabilistic match structure
        if len(results["matches"]) > 0:
            match = results["matches"][0]
            assert "posterior_prob" in match
            assert "likelihood" in match
            assert "separation_arcsec" in match

            # Probabilities should be between 0 and 1
            assert 0 <= match["posterior_prob"] <= 1

    def test_multi_survey_matching(self, crossmatch_tensor):
        """Test multi-survey matching."""
        results = crossmatch_tensor.multi_survey_matching(
            surveys=["reference", "target"], tolerance_arcsec=2.0, min_detections=2
        )

        assert "object_groups" in results
        assert "n_objects" in results
        assert isinstance(results["object_groups"], dict)

    def test_angular_separation_calculation(self, crossmatch_tensor):
        """Test angular separation calculation."""
        # Test known separations
        ra1 = torch.tensor([0.0, 0.0, 0.0])
        dec1 = torch.tensor([0.0, 0.0, 0.0])
        ra2 = torch.tensor([1.0, 0.0, 1.0])  # degrees
        dec2 = torch.tensor([0.0, 1.0, 1.0])

        separations = crossmatch_tensor._angular_separation(ra1, dec1, ra2, dec2)

        assert len(separations) == 3
        assert torch.all(separations >= 0)

        # First two should be 1 degree, third should be sqrt(2) degrees
        expected = torch.tensor([1.0, 1.0, np.sqrt(2.0)])
        torch.testing.assert_close(separations, torch.deg2rad(expected), rtol=1e-4)

    def test_sky_to_cartesian_conversion(self, crossmatch_tensor):
        """Test sky coordinate to Cartesian conversion."""
        ra = torch.tensor([0.0, 90.0, 180.0, 270.0])
        dec = torch.tensor([0.0, 0.0, 0.0, 0.0])

        cartesian = crossmatch_tensor._sky_to_cartesian(ra, dec)

        assert cartesian.shape == (4, 3)

        # Check that points are on unit sphere
        distances = torch.sqrt(torch.sum(cartesian**2, dim=1))
        torch.testing.assert_close(distances, torch.ones(4), rtol=1e-6)

        # Check specific conversions
        expected = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # RA=0, Dec=0
                [0.0, 1.0, 0.0],  # RA=90, Dec=0
                [-1.0, 0.0, 0.0],  # RA=180, Dec=0
                [0.0, -1.0, 0.0],  # RA=270, Dec=0
            ]
        )
        torch.testing.assert_close(cartesian, expected, rtol=1e-6)

    def test_match_storage_and_retrieval(self, crossmatch_tensor):
        """Test match storage and retrieval system."""
        try:
            # Perform multiple matches
            crossmatch_tensor.sky_coordinate_matching(
                tolerance_arcsec=1.0, match_name="tight_match"
            )
            crossmatch_tensor.sky_coordinate_matching(
                tolerance_arcsec=3.0, match_name="loose_match"
            )

            # Test listing matches
            match_names = crossmatch_tensor.list_matches()
            assert "tight_match" in match_names
            assert "loose_match" in match_names

            # Test retrieving matches
            tight_results = crossmatch_tensor.get_matches("tight_match")
            loose_results = crossmatch_tensor.get_matches("loose_match")

            assert tight_results["tolerance_arcsec"] == 1.0
            assert loose_results["tolerance_arcsec"] == 3.0

            # Loose match should have more matches
            assert len(loose_results["matches"]) >= len(tight_results["matches"])

            # Test error for non-existent match
            with pytest.raises(ValueError):
                crossmatch_tensor.get_matches("nonexistent")

        except ImportError:
            pytest.skip("sklearn not available for match storage test")

    def test_different_matching_methods(self, crossmatch_tensor):
        """Test different matching methods."""
        methods = ["nearest_neighbor", "all_pairs"]

        for method in methods:
            try:
                results = crossmatch_tensor.sky_coordinate_matching(
                    tolerance_arcsec=2.0, method=method, match_name=f"match_{method}"
                )

                assert results["method"] == method
                assert "matches" in results

            except ImportError:
                pytest.skip(f"sklearn not available for {method} matching")

    def test_self_matching(self):
        """Test self-matching for duplicate detection."""
        # Create catalog with some duplicates
        np.random.seed(42)
        n_objects = 50

        ra = np.random.uniform(0, 10, n_objects)
        dec = np.random.uniform(0, 10, n_objects)

        # Add some duplicates
        n_duplicates = 5
        dup_indices = np.random.choice(n_objects, n_duplicates, replace=False)
        for i, idx in enumerate(dup_indices):
            if i < len(dup_indices) - 1:
                # Make next object a duplicate with small offset
                next_idx = dup_indices[i + 1]
                ra[next_idx] = ra[idx] + 0.1 / 3600  # 0.1 arcsec offset
                dec[next_idx] = dec[idx] + 0.1 / 3600

        catalog = np.column_stack([ra, dec])

        # Test self-matching
        tensor = CrossMatchTensor(catalog)

        try:
            results = tensor.multi_survey_matching(
                surveys=["self"], tolerance_arcsec=1.0, min_detections=1
            )

            assert "object_groups" in results

        except ImportError:
            pytest.skip("sklearn not available for self-matching")

    def test_match_statistics(self, crossmatch_tensor):
        """Test match statistics calculation."""
        # Create mock matches
        matches = [
            {"index_a": 0, "index_b": 5, "distance": 0.1},
            {"index_a": 1, "index_b": 10, "distance": 0.2},
            {"index_a": 2, "index_b": 15, "distance": 0.15},
        ]

        stats = crossmatch_tensor._calculate_match_statistics(matches, 100, 80)

        assert stats["n_matches"] == 3
        assert stats["n_matched_a"] == 3
        assert stats["n_matched_b"] == 3
        assert stats["match_rate_a"] == 3 / 100
        assert stats["match_rate_b"] == 3 / 80
        assert 0 <= stats["completeness"] <= 1
        assert 0 <= stats["contamination"] <= 1

    def test_coordinate_systems(self):
        """Test different coordinate systems."""
        # Create test data
        catalog = np.random.uniform([0, -90], [360, 90], (50, 2))

        systems = ["icrs", "galactic", "ecliptic"]
        for system in systems:
            tensor = CrossMatchTensor(catalog)
            # Update coordinate system in metadata
            coord_systems = tensor.get_metadata("coordinate_systems", {})
            coord_systems["a"] = system
            tensor.update_metadata(coordinate_systems=coord_systems)

            assert tensor.get_metadata("coordinate_systems")["a"] == system

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        catalog = np.random.randn(50, 3)

        # Test mismatched coordinate columns
        with pytest.raises(ValueError):
            tensor = CrossMatchTensor(
                catalog, coordinate_columns={"a": [0, 1, 2, 3]}
            )  # Too many columns

        # Test empty catalog
        empty_catalog = np.empty((0, 3))
        tensor = CrossMatchTensor(empty_catalog)
        assert tensor.get_metadata("catalog_info")["catalog_a"]["n_objects"] == 0

    def test_tensor_metadata(self, crossmatch_tensor):
        """Test tensor metadata handling."""
        assert crossmatch_tensor.get_metadata("tensor_type") == "crossmatch"

        catalog_info = crossmatch_tensor.get_metadata("catalog_info")
        assert "catalog_a" in catalog_info
        assert "catalog_b" in catalog_info
        assert catalog_info["catalog_a"]["n_objects"] == 100
        assert catalog_info["catalog_b"]["n_objects"] == 80

    def test_repr(self, crossmatch_tensor):
        """Test string representation."""
        repr_str = repr(crossmatch_tensor)
        assert "CrossMatchTensor" in repr_str
        assert "catalog_a=100" in repr_str
        assert "catalog_b=80" in repr_str


class TestCrossMatchTensorIntegration:
    """Test CrossMatchTensor integration with other components."""

    def test_large_catalog_matching(self):
        """Test matching with larger catalogs."""
        np.random.seed(42)

        # Create larger catalogs
        n_a, n_b = 1000, 800

        # Catalog A
        ra_a = np.random.uniform(0, 360, n_a)
        dec_a = np.random.uniform(-30, 30, n_a)
        catalog_a = np.column_stack([ra_a, dec_a])

        # Catalog B with some matches
        n_matches = 200
        ra_b = np.zeros(n_b)
        dec_b = np.zeros(n_b)

        # Create matches
        ra_b[:n_matches] = ra_a[:n_matches] + np.random.normal(0, 1.0 / 3600, n_matches)
        dec_b[:n_matches] = dec_a[:n_matches] + np.random.normal(
            0, 1.0 / 3600, n_matches
        )

        # Non-matches
        ra_b[n_matches:] = np.random.uniform(0, 360, n_b - n_matches)
        dec_b[n_matches:] = np.random.uniform(-30, 30, n_b - n_matches)

        catalog_b = np.column_stack([ra_b, dec_b])

        tensor = CrossMatchTensor(catalog_a, catalog_b)

        try:
            results = tensor.sky_coordinate_matching(tolerance_arcsec=3.0)

            # Should find most matches
            n_found = len(results["matches"])
            assert n_found >= n_matches * 0.8  # At least 80% recovery

        except ImportError:
            pytest.skip("sklearn not available for large catalog test")

    def test_multi_band_matching(self):
        """Test matching with multi-band photometry."""
        np.random.seed(42)
        n_objects = 100

        # Create multi-band catalog A
        ra_a = np.random.uniform(100, 110, n_objects)
        dec_a = np.random.uniform(10, 20, n_objects)
        u_a = np.random.normal(22, 1, n_objects)
        g_a = np.random.normal(21, 1, n_objects)
        r_a = np.random.normal(20, 1, n_objects)

        catalog_a = {"ra": ra_a, "dec": dec_a, "u_mag": u_a, "g_mag": g_a, "r_mag": r_a}

        # Create catalog B with different bands
        n_b = 80
        ra_b = ra_a[:n_b] + np.random.normal(0, 0.5 / 3600, n_b)  # Small offsets
        dec_b = dec_a[:n_b] + np.random.normal(0, 0.5 / 3600, n_b)
        i_b = np.random.normal(19, 1, n_b)
        z_b = np.random.normal(18, 1, n_b)

        catalog_b = {"ra": ra_b, "dec": dec_b, "i_mag": i_b, "z_mag": z_b}

        tensor = CrossMatchTensor(catalog_a, catalog_b)

        try:
            results = tensor.sky_coordinate_matching(tolerance_arcsec=2.0)

            # Should find most objects
            assert len(results["matches"]) >= n_b * 0.8

        except ImportError:
            pytest.skip("sklearn not available for multi-band test")


@pytest.mark.parametrize("tolerance", [0.5, 1.0, 2.0, 5.0])
def test_different_tolerances(tolerance):
    """Test matching with different tolerance values."""
    np.random.seed(42)

    # Create catalogs with known separations
    n_objects = 50
    ra_a = np.random.uniform(0, 10, n_objects)
    dec_a = np.random.uniform(0, 10, n_objects)
    catalog_a = np.column_stack([ra_a, dec_a])

    # Catalog B with fixed 1 arcsecond offset
    ra_b = ra_a + 1.0 / 3600  # 1 arcsecond offset
    dec_b = dec_a
    catalog_b = np.column_stack([ra_b, dec_b])

    tensor = CrossMatchTensor(catalog_a, catalog_b)

    try:
        results = tensor.sky_coordinate_matching(tolerance_arcsec=tolerance)

        n_matches = len(results["matches"])

        if tolerance >= 1.0:
            # Should find all matches
            assert n_matches >= n_objects * 0.9
        else:
            # Should find fewer matches
            assert n_matches < n_objects * 0.5

    except ImportError:
        pytest.skip(f"sklearn not available for tolerance {tolerance} test")


@pytest.mark.parametrize("method", ["nearest_neighbor", "all_pairs"])
def test_matching_methods(method):
    """Test different matching methods."""
    # Create simple test data
    catalog_a = np.array([[0, 0], [1, 1], [2, 2]])
    catalog_b = np.array([[0.1, 0.1], [1.1, 1.1], [3, 3]])

    tensor = CrossMatchTensor(catalog_a, catalog_b)

    try:
        results = tensor.sky_coordinate_matching(
            tolerance_arcsec=0.5 * 3600,  # 0.5 degree tolerance
            method=method,
        )

        assert results["method"] == method
        assert len(results["matches"]) >= 2  # Should match first two pairs

    except ImportError:
        pytest.skip(f"sklearn not available for {method} test")
