# AstroLab Test Suite

## Overview

The test suite has been updated to work with the new preprocessing and data handling system that uses PyTorch Geometric directly instead of TensorDicts.

## Test Structure

### `test_data.py`
Tests for the data module including:
- **Preprocessors**: Test survey-specific preprocessing and quality filtering
- **Converters**: Test direct PyG graph creation from survey data
- **Cross-Matching**: Test matching between different surveys
- **DataModules**: Test Lightning integration
- **Memory Efficiency**: Test handling of large datasets

### `conftest.py`
Shared fixtures for testing:
- `sample_graph_data`: Creates sample PyG Data objects
- `sample_survey_data`: Creates sample survey data as Polars DataFrames
- `preprocessor`: Gets preprocessor for testing
- `astro_datamodule`: Creates SurveyDataModule for testing

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest test/test_data.py

# Run specific test class
pytest test/test_data.py::TestPreprocessors

# Run with coverage
pytest --cov=astro_lab test/
```

## Key Changes from Previous Version

1. **No TensorDicts**: All tests now work with PyG Data objects directly
2. **Simplified API**: Tests reflect the simpler preprocessing → graph workflow
3. **Mock Data**: Tests use mock Polars DataFrames instead of real files
4. **Cross-Matching**: New tests for survey cross-matching functionality
5. **Direct Graph Creation**: Tests for `create_graph_from_survey` function

## Writing New Tests

When adding new tests:
1. Use Polars DataFrames for input data
2. Create PyG Data objects for graph tests
3. Mock file I/O operations when possible
4. Test both preprocessing (DataFrame) and graph creation separately 