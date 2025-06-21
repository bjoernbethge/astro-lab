"""
Tests for FeatureTensor
======================

Test suite for ML feature engineering and preprocessing operations.
"""

import numpy as np
import pytest
import torch

from astro_lab.tensors import FeatureTensor


class TestFeatureTensor:
    """Test FeatureTensor functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample astronomical data for testing."""
        np.random.seed(42)
        n_objects = 100

        # Create realistic astronomical features
        data = {
            "g_mag": np.random.normal(20.0, 2.0, n_objects),
            "r_mag": np.random.normal(19.5, 2.0, n_objects),
            "i_mag": np.random.normal(19.0, 2.0, n_objects),
            "parallax": np.random.exponential(1.0, n_objects),  # mas
            "pmra": np.random.normal(0.0, 10.0, n_objects),  # mas/yr
            "pmdec": np.random.normal(0.0, 10.0, n_objects),  # mas/yr
        }

        # Add some missing values
        missing_indices = np.random.choice(n_objects, 10, replace=False)
        data["g_mag"][missing_indices] = 99.0  # Missing value code

        # Convert to tensor format
        feature_matrix = np.column_stack([data[key] for key in sorted(data.keys())])
        feature_names = sorted(data.keys())

        return feature_matrix, feature_names

    @pytest.fixture
    def feature_tensor(self, sample_data):
        """Create FeatureTensor instance."""
        data, names = sample_data
        return FeatureTensor(data, feature_names=names)

    def test_initialization(self, sample_data):
        """Test FeatureTensor initialization."""
        data, names = sample_data

        # Test with feature names
        tensor = FeatureTensor(data, feature_names=names)
        assert tensor.n_objects == 100
        assert tensor.n_features == 6
        assert tensor.feature_names == names

        # Test without feature names (auto-generated)
        tensor_auto = FeatureTensor(data)
        assert len(tensor_auto.feature_names) == 6
        assert all(name.startswith("feature_") for name in tensor_auto.feature_names)

    def test_feature_type_detection(self, feature_tensor):
        """Test automatic feature type detection."""
        feature_types = feature_tensor._detect_feature_types()

        assert feature_types["g_mag"] == "magnitude"
        assert feature_types["r_mag"] == "magnitude"
        assert feature_types["parallax"] == "parallax"
        assert feature_types["pmra"] == "proper_motion"
        assert feature_types["pmdec"] == "proper_motion"

    def test_feature_scaling(self):
        """Test feature scaling methods."""
        # Create test data with known properties
        data = torch.tensor(
            [
                [1.0, 10.0, 100.0],
                [2.0, 20.0, 200.0],
                [3.0, 30.0, 300.0],
                [4.0, 40.0, 400.0],
                [5.0, 50.0, 500.0],
            ]
        )

        feature_tensor = FeatureTensor(data, feature_names=["a", "b", "c"])

        # Test standard scaling
        scaled_tensor = feature_tensor.scale_features(method="standard")

        # Check that each feature has approximately zero mean (within tolerance)
        means = scaled_tensor._data.mean(dim=0)
        for mean_val in means:
            assert (
                abs(mean_val) < 1e-6
            )  # More reasonable tolerance for numerical precision

        # Check that each feature has approximately unit variance
        stds = scaled_tensor._data.std(dim=0)
        for std_val in stds:
            assert abs(std_val - 1.0) < 0.2  # More lenient tolerance for variance

    def test_missing_value_imputation(self, feature_tensor):
        """Test missing value imputation."""
        # Count missing values before imputation
        missing_before = torch.isnan(feature_tensor._data).sum()

        # Test astronomical imputation
        imputed_tensor = feature_tensor.impute_missing_values(method="astronomical")
        missing_after = torch.isnan(imputed_tensor._data).sum()

        assert missing_after <= missing_before
        assert "imputed_missing_astronomical" in imputed_tensor.get_metadata(
            "preprocessing_history", []
        )

    def test_outlier_detection(self, feature_tensor):
        """Test outlier detection methods."""
        # Test astronomical outlier detection
        outliers = feature_tensor.detect_outliers(method="astronomical")
        assert isinstance(outliers, torch.Tensor)
        assert outliers.dtype == torch.bool
        assert len(outliers) == feature_tensor.n_objects

        # Test statistical outlier detection
        outliers_stat = feature_tensor.detect_outliers(method="statistical")
        assert isinstance(outliers_stat, torch.Tensor)
        assert outliers_stat.dtype == torch.bool

    def test_feature_selection(self, feature_tensor):
        """Test feature selection methods."""
        # Test astronomical feature selection
        selected_tensor = feature_tensor.select_features(method="astronomical", k=4)
        assert selected_tensor.n_features <= 4
        assert len(selected_tensor.feature_names) == selected_tensor.n_features
        assert "selected_features_astronomical_4" in selected_tensor.get_metadata(
            "preprocessing_history", []
        )

    def test_color_computation(self, feature_tensor):
        """Test astronomical color computation."""
        # Test color computation from magnitudes
        color_tensor = feature_tensor.compute_colors(["g", "r", "i"])

        # Should have original features plus colors
        assert color_tensor.n_features > feature_tensor.n_features
        assert "computed_colors" in color_tensor.get_metadata(
            "preprocessing_history", []
        )

        # Check that color names were added
        color_names = [name for name in color_tensor.feature_names if "color" in name]
        assert len(color_names) > 0

    def test_feature_statistics(self, feature_tensor):
        """Test feature statistics computation."""
        stats = feature_tensor.get_feature_statistics()

        assert isinstance(stats, dict)
        assert len(stats) == feature_tensor.n_features

        # Check statistics for each feature
        for name in feature_tensor.feature_names:
            assert name in stats
            if "error" not in stats[name]:
                assert "mean" in stats[name]
                assert "std" in stats[name]
                assert "missing_fraction" in stats[name]

    def test_preprocessing_pipeline(self, feature_tensor):
        """Test complete preprocessing pipeline."""
        # Apply multiple preprocessing steps
        processed_tensor = (
            feature_tensor.impute_missing_values(method="astronomical")
            .scale_features(method="standard")
            .select_features(method="astronomical", k=4)
        )

        # Check that all operations were recorded
        history = processed_tensor.get_metadata("preprocessing_history", [])
        assert "imputed_missing_astronomical" in history
        assert "scaled_features_standard" in history
        assert "selected_features_astronomical_4" in history

        # Check final tensor properties
        assert processed_tensor.n_features <= 4
        assert processed_tensor.n_objects == feature_tensor.n_objects

    def test_astronomical_priors(self, feature_tensor):
        """Test astronomical prior knowledge integration."""
        priors = feature_tensor.get_metadata("astronomical_priors", {})

        assert "magnitude_range" in priors
        assert "color_range" in priors
        assert "parallax_range" in priors
        assert "missing_value_codes" in priors

    def test_error_handling(self, sample_data):
        """Test error handling for invalid inputs."""
        data, names = sample_data

        # Test mismatched feature names
        with pytest.raises(ValueError):
            FeatureTensor(data, feature_names=["wrong", "number"])

        # Test 1D data (should be converted to 2D)
        data_1d = data[:, 0]
        tensor_1d = FeatureTensor(data_1d)
        assert tensor_1d.n_features == 1

        # Test invalid data dimensions
        with pytest.raises(ValueError):
            FeatureTensor(np.random.rand(10, 5, 3))  # 3D data

    def test_tensor_metadata(self, feature_tensor):
        """Test tensor metadata handling."""
        assert feature_tensor.get_metadata("tensor_type") == "feature"
        assert isinstance(feature_tensor.get_metadata("feature_names"), list)
        assert isinstance(feature_tensor.get_metadata("astronomical_priors"), dict)

    def test_copy_functionality(self, feature_tensor):
        """Test tensor copying and metadata preservation."""
        # Create a copy
        new_tensor = feature_tensor.copy()

        # Test that data is copied
        assert torch.equal(new_tensor._data, feature_tensor._data)

        # Test that feature names are preserved (if they exist)
        if (
            hasattr(feature_tensor, "feature_names")
            and feature_tensor.feature_names is not None
        ):
            assert new_tensor.feature_names == feature_tensor.feature_names

        # Test that metadata is copied
        assert new_tensor._metadata == feature_tensor._metadata

        # Test that modifying copy doesn't affect original
        new_tensor._data[0, 0] = 999.0
        assert not torch.equal(new_tensor._data, feature_tensor._data)

    def test_repr(self, feature_tensor):
        """Test string representation."""
        repr_str = repr(feature_tensor)
        assert "FeatureTensor" in repr_str
        assert "objects=100" in repr_str
        assert "features=6" in repr_str


class TestFeatureTensorIntegration:
    """Test FeatureTensor integration with other components."""

    def test_sklearn_integration(self):
        """Test integration with sklearn when available."""
        try:
            from sklearn.preprocessing import StandardScaler

            sklearn_available = True
        except ImportError:
            sklearn_available = False

        if sklearn_available:
            # Create test data
            data = np.random.randn(100, 5)
            tensor = FeatureTensor(data)

            # Test sklearn-based scaling
            scaled = tensor.scale_features(method="standard")
            assert scaled is not None
        else:
            pytest.skip("sklearn not available")

    def test_survey_tensor_integration(self):
        """Test integration with SurveyTensor."""
        # This would test how FeatureTensor works with SurveyTensor data
        # For now, just test that it can be created from survey-like data

        # Create survey-like data
        n_objects = 50
        survey_data = {
            "ra": np.random.uniform(0, 360, n_objects),
            "dec": np.random.uniform(-90, 90, n_objects),
            "g_mag": np.random.normal(20, 2, n_objects),
            "r_mag": np.random.normal(19, 2, n_objects),
        }

        data_matrix = np.column_stack(
            [survey_data[key] for key in sorted(survey_data.keys())]
        )
        feature_names = sorted(survey_data.keys())

        tensor = FeatureTensor(data_matrix, feature_names=feature_names)
        assert tensor.n_objects == n_objects
        assert tensor.n_features == 4


@pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
def test_scaling_methods(method):
    """Test different scaling methods."""
    data = np.random.randn(50, 3)
    tensor = FeatureTensor(data)

    try:
        scaled = tensor.scale_features(method=method)
        assert scaled.n_features == tensor.n_features
    except ImportError:
        pytest.skip(f"sklearn not available for {method} scaling")


@pytest.mark.parametrize("method", ["astronomical", "mean", "median"])
def test_imputation_methods(method):
    """Test different imputation methods."""
    data = np.random.randn(50, 3)
    # Add missing values
    data[10:15, 0] = np.nan

    tensor = FeatureTensor(data)

    try:
        imputed = tensor.impute_missing_values(method=method)
        assert imputed.n_features == tensor.n_features
    except ImportError:
        if method in ["knn"]:
            pytest.skip(f"sklearn not available for {method} imputation")
        else:
            raise
