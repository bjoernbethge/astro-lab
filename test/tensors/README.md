# Tensor Tests

This directory contains tests for the `astro_lab.tensors` module, split into focused test files for better organization and maintainability.

## Test Files

- `test_base.py` - Tests for `AstroTensorBase` base class functionality
- `test_spatial_3d.py` - Tests for `Spatial3DTensor` coordinate handling
- `test_spectral.py` - Tests for `SpectralTensor` spectroscopic data
- `test_photometric.py` - Tests for `PhotometricTensor` photometric measurements
- `test_lightcurve.py` - Tests for `LightcurveTensor` time series data
- `test_orbital.py` - Tests for `OrbitTensor` orbital mechanics
- `test_interoperability.py` - Tests for tensor interoperability and device handling
- `test_serialization.py` - Tests for tensor serialization and deserialization
- `test_survey.py` - Tests for `SurveyTensor` and dataset integration

## Running Tests

Run all tensor tests:
```bash
pytest test/tensors/
```

Run specific test file:
```bash
pytest test/tensors/test_base.py
```

Run with specific markers:
```bash
pytest test/tensors/ -m "not slow"  # Skip slow tests
pytest test/tensors/ -m cuda        # Only CUDA tests
``` 