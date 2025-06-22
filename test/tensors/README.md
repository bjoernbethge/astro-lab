# Tensor Tests

This directory contains tests for the `astro_lab.tensors` module, organized into focused test files for better maintainability.

## 📁 Test Files

- `test_base.py` - `AstroTensorBase` base class functionality
- `test_spatial_3d.py` - `Spatial3DTensor` coordinate handling
- `test_spectral.py` - `SpectralTensor` spectroscopic data
- `test_photometric.py` - `PhotometricTensor` photometric measurements
- `test_lightcurve.py` - `LightcurveTensor` time series data
- `test_orbital.py` - `OrbitTensor` orbital mechanics
- `test_interoperability.py` - Tensor interoperability and device handling
- `test_serialization.py` - Tensor serialization and deserialization
- `test_survey.py` - `SurveyTensor` and dataset integration
- `test_feature.py` - `FeatureTensor` feature extraction
- `test_statistics.py` - `StatisticsTensor` statistical analysis
- `test_clustering.py` - `ClusteringTensor` clustering algorithms
- `test_crossmatch.py` - `CrossMatchTensor` crossmatching functionality

## 🚀 Running Tests

```bash
# All tensor tests
uv run pytest test/tensors/

# Specific test file
uv run pytest test/tensors/test_base.py

# With markers
uv run pytest test/tensors/ -m "not slow"  # Skip slow tests
uv run pytest test/tensors/ -m cuda        # Only CUDA tests

# Verbose output
uv run pytest test/tensors/ -v
```

## 📊 Test Coverage

```bash
# Check coverage
uv run pytest test/tensors/ --cov=astro_lab.tensors --cov-report=html
```

## 📂 Test Data

Test data is stored in `test/tensors/data/` and includes:
- Sample astronomical catalogs
- Mock survey data
- Test coordinate systems
- Synthetic light curves 