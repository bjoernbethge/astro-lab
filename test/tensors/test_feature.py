"""
Tests for FeatureTensor
======================

Test suite for ML feature engineering and preprocessing operations.
"""

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler

from astro_lab.tensors import FeatureTensor, SurveyTensor, PhotometricTensor


class TestFeatureTensor:
    """Test FeatureTensor functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample astronomical data for testing."""
        np.random.seed(42)
        n_objects = 100
        data = np.random.randn(n_objects, 6)
        # Add some missing values (NaNs) and non-finite codes
        data[10, 0] = np.nan
        data[20, 1] = 99.0 # Missing value code
        feature_names = ["g_mag", "r_mag", "i_mag", "parallax", "pmra", "pmdec"]
        return torch.from_numpy(data).float(), feature_names

    @pytest.fixture
    def feature_tensor(self, sample_data):
        """Create FeatureTensor instance."""
        data, names = sample_data
        return FeatureTensor(data=data, feature_names=names)

    def test_initialization(self, sample_data):
        """Test FeatureTensor initialization."""
        data, names = sample_data

        # Test with feature names
        tensor = FeatureTensor(data=data, feature_names=names)
        assert tensor.num_objects == 100
        assert tensor.num_features == 6
        assert tensor.feature_names == names

        # Test without feature names (auto-generated)
        tensor_auto = FeatureTensor(data=data)
        assert len(tensor_auto.feature_names) == 6
        assert all(name.startswith("feature_") for name in tensor_auto.feature_names)

    def test_feature_scaling(self):
        """Test feature scaling methods."""
        data = torch.randn(10, 3) * 5 + 10 # Data with non-zero mean and std != 1
        tensor = FeatureTensor(data=data)
        
        # Test standard scaling
        scaled_tensor = tensor.scale_features(method="standard")
        
        # Check that each feature has approximately zero mean and unit variance
        assert torch.allclose(scaled_tensor.data.mean(dim=0), torch.zeros(3), atol=1e-6)
        assert torch.allclose(scaled_tensor.data.std(dim=0), torch.ones(3), atol=1e-6)
        assert "scalers" in scaled_tensor.meta
        assert isinstance(scaled_tensor.meta["scalers"]["standard"], StandardScaler)

    def test_missing_value_imputation(self, sample_data):
        """Test missing value imputation."""
        data, names = sample_data
        tensor = FeatureTensor(data=data, feature_names=names)
        
        # Count missing values before imputation
        missing_before = torch.isnan(tensor.data).sum()
        assert missing_before > 0

        # Test mean imputation
        imputed_tensor = tensor.impute_missing_values(method="mean")
        missing_after = torch.isnan(imputed_tensor.data).sum()

        assert missing_after == 0
        assert "imputers" in imputed_tensor.meta

    def test_outlier_detection(self, feature_tensor):
        """Test outlier detection methods."""
        # Test isolation forest
        outliers = feature_tensor.detect_outliers(method="isolation_forest")
        assert isinstance(outliers, torch.Tensor)
        assert outliers.dtype == torch.bool
        assert len(outliers) == feature_tensor.num_objects

    def test_feature_selection(self, feature_tensor):
        """Test feature selection methods."""
        # Test variance threshold
        # Ensure there is variance difference
        feature_tensor.data[:, 0] = 1.0
        selected_tensor = feature_tensor.select_features(method="variance", threshold=0.01)
        assert selected_tensor.num_features < feature_tensor.num_features
        assert "variance_threshold" in selected_tensor.meta

    def test_color_computation(self, feature_tensor):
        """Test astronomical color computation."""
        color_tensor = feature_tensor.compute_colors(bands=["g_mag", "r_mag", "i_mag"])

        assert color_tensor.num_features > feature_tensor.num_features
        assert "g_mag-r_mag" in color_tensor.feature_names
        assert "r_mag-i_mag" in color_tensor.feature_names
        assert "history" in color_tensor.meta

    def test_feature_statistics(self, feature_tensor):
        """Test feature statistics computation."""
        stats = feature_tensor.get_feature_statistics()

        assert isinstance(stats, dict)
        assert len(stats) == feature_tensor.num_features

        # Check statistics for a feature
        first_feature_name = feature_tensor.feature_names[0]
        assert first_feature_name in stats
        assert "mean" in stats[first_feature_name]
        assert "std" in stats[first_feature_name]
        assert "min" in stats[first_feature_name]
        assert "max" in stats[first_feature_name]

    def test_preprocessing_pipeline(self, sample_data):
        """Test complete preprocessing pipeline."""
        data, names = sample_data
        tensor = FeatureTensor(data=data, feature_names=names)

        # Apply multiple preprocessing steps
        processed_tensor = (
            tensor.impute_missing_values(method="median")
            .scale_features(method="robust")
            .select_features(method="variance", threshold=0.1)
        )

        # Check that metadata reflects the changes
        assert "imputers" in processed_tensor.meta
        assert "scalers" in processed_tensor.meta
        assert "variance_threshold" in processed_tensor.meta
        assert "history" in processed_tensor.meta
        assert len(processed_tensor.meta["history"]) == 3

        assert processed_tensor.num_features <= tensor.num_features
        assert processed_tensor.num_objects == tensor.num_objects

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        data = torch.randn(10, 6)
        # Test mismatched feature names
        with pytest.raises(ValueError):
            FeatureTensor(data=data, feature_names=["wrong", "number"])

        # Test 1D data (should be converted to 2D)
        data_1d = data[:, 0]
        tensor_1d = FeatureTensor(data=data_1d)
        assert tensor_1d.num_features == 1
        assert tensor_1d.data.ndim == 2

    def test_tensor_metadata(self, feature_tensor):
        """Test tensor metadata handling."""
        feature_tensor.update_metadata(survey="SDSS")
        assert feature_tensor.get_metadata("survey") == "SDSS"
        assert isinstance(feature_tensor.feature_names, list)

    def test_copy_functionality(self, feature_tensor):
        """Test tensor copying and metadata preservation."""
        new_tensor = feature_tensor.copy()

        assert torch.equal(new_tensor.data, feature_tensor.data)
        assert new_tensor.feature_names == feature_tensor.feature_names
        assert new_tensor.meta == feature_tensor.meta

        # Test that modifying copy doesn't affect original
        new_tensor.data[0, 0] = 999.0
        assert not torch.equal(new_tensor.data, feature_tensor.data)

    def test_repr(self, feature_tensor):
        """Test string representation."""
        repr_str = repr(feature_tensor)
        assert "FeatureTensor" in repr_str
        assert f"num_objects={feature_tensor.num_objects}" in repr_str
        assert f"num_features={feature_tensor.num_features}" in repr_str


class TestFeatureTensorIntegration:
    """Test integration with other components."""

    def test_sklearn_integration(self):
        """Test integration with sklearn for custom transformations."""
        from sklearn.decomposition import PCA
        data = torch.randn(100, 10)
        tensor = FeatureTensor(data=data)
        
        pca = PCA(n_components=2)
        transformed_data = pca.fit_transform(tensor.data.numpy())
        
        new_tensor = FeatureTensor(
            data=torch.from_numpy(transformed_data),
            feature_names=["pc1", "pc2"]
        )
        new_tensor.add_history_entry("PCA", n_components=2)
        
        assert new_tensor.num_features == 2
        assert len(new_tensor.meta["history"]) == 1

    def test_survey_tensor_integration(self):
        """Test wrapping a FeatureTensor in a SurveyTensor."""
        data = torch.randn(50, 5)
        feature_tensor = FeatureTensor(data=data, feature_names=["u", "g", "r", "i", "z"])
        survey_tensor = SurveyTensor(data=feature_tensor, survey_name="SDSS")

        assert isinstance(survey_tensor.data, FeatureTensor)
        assert survey_tensor.survey_name == "SDSS"
        assert survey_tensor.data.num_features == 5


@pytest.mark.parametrize("method", ["standard", "minmax", "robust"])
def test_scaling_methods(method):
    """Test various scaling methods."""
    data = torch.randn(20, 4) * 2 + 5
    tensor = FeatureTensor(data=data)
    scaled = tensor.scale_features(method=method)
    
    assert scaled.data.shape == data.shape
    assert method in scaled.meta.get("scalers", {})
    assert "history" in scaled.meta

@pytest.mark.parametrize("method", ["mean", "median", "knn"])
def test_imputation_methods(method):
    """Test various imputation methods."""
    data = torch.randn(30, 5)
    data[5:10, 2] = float('nan') # Add some NaNs
    tensor = FeatureTensor(data=data)

    imputed = tensor.impute_missing_values(method=method)

    assert not torch.isnan(imputed.data).any()
    assert method in imputed.meta.get("imputers", {})
    assert "history" in imputed.meta
