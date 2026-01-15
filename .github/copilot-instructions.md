# GitHub Copilot Instructions for AstroLab

## Project Overview

AstroLab is an Astronomical Machine Learning Laboratory focused on cosmic web analysis using Graph Neural Networks (GNNs). The project processes astronomical survey data from sources like Gaia, SDSS, and NASA archives.

## Code Style Guidelines

### Python Standards
- Use Python 3.11+ features and type hints throughout the codebase
- Follow PEP 8 style guidelines with ruff for linting and formatting
- Use type annotations for all function parameters and return values
- Prefer dataclasses and Pydantic models for structured data

### Naming Conventions
- Use `snake_case` for functions, variables, and modules
- Use `PascalCase` for classes
- Use `UPPER_SNAKE_CASE` for constants
- Prefix private methods and attributes with underscore `_`

### Documentation
- Use Google-style docstrings for all public functions and classes
- Include parameter types, return types, and examples in docstrings
- Keep docstrings concise but informative

## Architecture Patterns

### Data Processing
- Use TensorDict for efficient tensor operations with astronomical data
- Coordinate systems should use AstroPy SkyCoord for astronomical coordinates
- Spatial operations should support multiple units (parsecs, megaparsecs)

### Machine Learning
- Models should inherit from PyTorch Lightning `LightningModule`
- Use PyTorch Geometric for graph neural network operations
- Support MLflow experiment tracking for all training runs

### Visualization
- Support multiple backends: Cosmograph, PyVista, Plotly, Open3D
- Use survey-specific colors (Gold for Gaia, Blue for SDSS, etc.)
- Implement lazy loading for large datasets

## Testing Guidelines

- Write pytest tests in the `test/` directory
- Use fixtures for common test data and configurations
- Test edge cases and error conditions
- Use `pytest-cov` for coverage reporting

## Dependencies

### Package Management
- Use `uv` for dependency management (not pip directly)
- Lock dependencies with `uv.lock`
- Separate dev dependencies in `[dependency-groups]`

### Key Libraries
- PyTorch with CUDA support for GPU acceleration
- AstroPy for astronomical calculations
- scikit-learn for clustering algorithms
- Pydantic for configuration validation

## CLI Commands

The project uses a CLI interface via `astro-lab` command:
- `astro-lab process` - Data processing
- `astro-lab train` - Model training
- `astro-lab cosmic-web` - Cosmic web analysis
- `astro-lab optimize` - Hyperparameter optimization

## Security Considerations

- Never commit secrets or API keys
- Use environment variables for sensitive configuration
- Validate all user inputs, especially file paths
- Sanitize data from external astronomical archives

## Performance Tips

- Use lazy loading for large astronomical catalogs
- Leverage GPU acceleration where available
- Use Polars instead of Pandas for data processing
- Implement batch processing for large-scale analysis
