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

## Build, Test, and Validation Commands

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Install PyTorch Geometric extensions for CUDA support
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\Activate.ps1  # Windows
```

### Linting and Formatting
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix ruff issues
uv run ruff check --fix .

# Run mypy type checking
uv run mypy src/astro_lab
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Run specific test file
uv run pytest test/test_cosmic_web.py -v

# Run tests with markers
uv run pytest -m unit -v           # Unit tests only
uv run pytest -m integration -v    # Integration tests only
uv run pytest -m "not slow" -v     # Skip slow tests

# Run tests with coverage
uv run pytest --cov=astro_lab --cov-report=html
```

### Documentation
```bash
# Generate/update documentation
python docs/generate_docs.py update

# Serve documentation locally (http://127.0.0.1:8000)
python docs/generate_docs.py serve

# Or use mkdocs directly
uv run mkdocs serve
```

### Running the Application
```bash
# Show CLI help
astro-lab --help

# Process astronomical data
astro-lab preprocess gaia --max-samples 10000

# Analyze cosmic web structures
astro-lab cosmic-web gaia --max-samples 100000 --clustering-scales 5 10 25

# Train a model
astro-lab train --survey gaia --model gcn --epochs 50

# Start interactive UI
marimo run src/astro_lab/ui/app.py
```

## Project Structure

```
astro-lab/
├── .github/                    # GitHub configuration
│   ├── copilot-instructions.md # This file
│   ├── agents/                 # Custom agent definitions
│   └── workflows/              # CI/CD workflows
├── src/astro_lab/              # Main source code
│   ├── cli/                    # Command-line interface
│   ├── data/                   # Data processing and datasets
│   │   ├── cosmic_web.py       # Core cosmic web analysis
│   │   └── datasets/           # Survey-specific datasets
│   ├── models/                 # GNN models and architectures
│   │   ├── core/               # Core model implementations
│   │   └── components/         # Model building blocks
│   ├── tensors/                # Tensor operations for astronomy
│   │   └── tensordict_astro.py # Spatial tensor operations
│   ├── training/               # Training framework and utilities
│   ├── widgets/                # Visualization widgets
│   │   ├── cosmograph_bridge.py # Interactive 3D visualization
│   │   └── plotly_bridge.py    # Plotly integration
│   └── ui/                     # Interactive UI modules
├── test/                       # Test suite
├── configs/                    # Configuration files
├── docs/                       # Documentation
└── scripts/                    # Utility scripts
```

## Restrictions and What NOT to Do

### Files and Directories to Never Modify
- **DO NOT** modify files in `.venv/` or `__pycache__/` directories
- **DO NOT** change `uv.lock` directly - use `uv sync` or `uv add` instead
- **DO NOT** modify `.github/agents/` files unless specifically asked - these contain specialized agent instructions
- **DO NOT** delete or modify existing tests without understanding their purpose
- **DO NOT** commit large data files or model checkpoints to the repository
- **DO NOT** modify the PyTorch Geometric installation commands in documentation

### Code Practices to Avoid
- **DO NOT** use `pip` directly - always use `uv` for package management
- **DO NOT** use `pandas` when `polars` is available for data processing
- **DO NOT** add `any` type hints - always specify concrete types
- **DO NOT** remove type hints from existing code
- **DO NOT** ignore linting errors - fix them or add proper ignore comments with justification
- **DO NOT** use hardcoded file paths - use Path objects and make paths configurable
- **DO NOT** commit API keys, tokens, or credentials
- **DO NOT** bypass the pre-commit hooks

### Testing Practices to Avoid
- **DO NOT** skip writing tests for new functionality
- **DO NOT** remove existing tests to make your changes pass
- **DO NOT** use `pytest.skip()` without a clear reason
- **DO NOT** create tests that depend on specific ordering or external state

## Development Workflow

### Adding a New Feature
1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Write the implementation following code style guidelines
3. Add tests in the `test/` directory
4. Run linter: `uv run ruff check --fix .`
5. Run type checker: `uv run mypy src/astro_lab`
6. Run tests: `uv run pytest -v`
7. Update documentation if needed
8. Commit with descriptive message: `git commit -m "feat: add your feature"`
9. Create a pull request

### Fixing a Bug
1. Create a bugfix branch: `git checkout -b fix/bug-description`
2. Write a failing test that reproduces the bug
3. Fix the bug in the source code
4. Verify the test now passes
5. Run full test suite to ensure no regressions
6. Commit with descriptive message: `git commit -m "fix: resolve bug description"`

### Adding a New Survey Dataset
1. Create a new dataset class in `src/astro_lab/data/datasets/`
2. Inherit from appropriate base class
3. Implement required methods: `download()`, `preprocess()`, `load()`
4. Add survey configuration in `configs/`
5. Add tests in `test/test_data.py`
6. Update documentation with survey information

## Common Patterns and Examples

### Creating a Spatial Tensor
```python
from astro_lab.tensors import SpatialTensorDict
import torch

# Create spatial tensor with coordinate system
coordinates = torch.rand(1000, 3) * 100  # 1000 points in 100 pc cube
spatial = SpatialTensorDict(
    coordinates,
    coordinate_system="icrs",
    unit="parsec"
)

# Perform cosmic web clustering
labels = spatial.cosmic_web_clustering(eps_pc=10.0, min_samples=5)
```

### Creating a GNN Model
```python
from astro_lab.models.astro_model import create_astro_graph_gnn

# Create a GNN for cosmic web analysis
model = create_astro_graph_gnn(
    num_features=16,
    num_classes=3,
    hidden_channels=64,
    num_layers=3,
    dropout=0.2
)
```

### Visualizing Cosmic Web Results
```python
from astro_lab.widgets.cosmograph_bridge import CosmographBridge
from astro_lab.data.cosmic_web import analyze_gaia_cosmic_web

# Analyze and visualize
results = analyze_gaia_cosmic_web(max_samples=10000)
bridge = CosmographBridge()
widget = bridge.from_cosmic_web_results(results, survey_name="gaia")
widget.show()  # Interactive 3D visualization with gold points
```

### Error Handling Pattern
```python
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def process_catalog(catalog_path: Path, output_dir: Optional[Path] = None) -> dict:
    """Process astronomical catalog with proper error handling.
    
    Args:
        catalog_path: Path to input catalog file
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary containing processing results
        
    Raises:
        FileNotFoundError: If catalog file does not exist
        ValueError: If catalog format is invalid
    """
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    
    try:
        # Process the catalog
        results = _do_processing(catalog_path)
        logger.info(f"Processed {len(results)} objects from {catalog_path}")
        return results
    except Exception as e:
        logger.error(f"Failed to process catalog: {e}")
        raise
```

## Integration with Custom Agents

This repository has specialized custom agents in `.github/agents/` for specific tasks:
- **astrophysics-expert**: Domain knowledge and scientific validation
- **cli-developer**: CLI development and configuration
- **cosmic-web-analyst**: Cosmic web analysis and algorithms
- **data-scientist**: ML pipelines and data processing
- **gnn-architect**: GNN architecture design
- **performance-optimizer**: Performance and GPU optimization
- **test-engineer**: Testing strategies and quality assurance
- **visualization-expert**: 3D visualization and graphics

When working on tasks in these domains, the relevant agent may be automatically engaged to provide specialized assistance.
