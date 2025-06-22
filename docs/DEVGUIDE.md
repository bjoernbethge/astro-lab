# 🔧 Development Guide

Comprehensive guide for contributing to AstroLab development and setting up your development environment.

## 📋 Table of Contents

- [🚀 Quick Setup](#-quick-setup)
- [🏗️ Project Architecture](#️-project-architecture)
- [🧪 Development Workflow](#-development-workflow)
- [📝 Contributing Guidelines](#-contributing-guidelines)
- [🔬 Adding New Features](#-adding-new-features)
- [🐛 Debugging & Testing](#-debugging--testing)
- [⚡ Performance Optimization](#-performance-optimization)
- [📚 Documentation Standards](#-documentation-standards)
- [🚀 Release Process](#-release-process)
- [📚 Related Documentation](#-related-documentation)

## 🚀 Quick Setup

### Prerequisites
- **Python 3.11+**: Modern Python with type hints support
- **UV**: Fast Python package manager
- **Git**: Version control
- **CUDA** (optional): GPU acceleration for deep learning

### Installation
```bash
# Clone repository
git clone https://github.com/bjoernbethge/astro-lab.git
cd astro-lab

# Install dependencies
uv sync

# Install PyTorch Geometric
uv pip install pyg torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Verify installation
uv run pytest -v
```

## 🏗️ Project Architecture

### Core Components
```
src/astro_lab/
├── cli/           # Command-line interface
│   ├── train.py   # Training workflows
│   └── data.py    # Data management
├── data/          # Data loading and processing
│   ├── core.py    # Core data structures
│   └── transforms.py # Data transformations
├── models/        # Neural network architectures
│   ├── factory.py # Model factory
│   ├── astro.py   # Astronomical models
│   └── base_gnn.py # Base graph neural networks
├── training/      # Training framework
│   ├── trainer.py # AstroTrainer
│   └── lightning_module.py # Lightning integration
├── tensors/       # Specialized tensor types
│   ├── base.py    # Base tensor classes
│   └── spatial_3d.py # 3D spatial tensors
└── utils/         # Utilities and visualization
    ├── viz/       # Visualization tools
    └── blender/   # Blender integration
```

### Key Design Principles
- **Modularity**: Each component is self-contained
- **Type Safety**: Comprehensive type hints throughout
- **GPU Acceleration**: CUDA-optimized operations
- **Reproducibility**: MLflow experiment tracking
- **Extensibility**: Easy to add new surveys and models

## 🧪 Development Workflow

### 1. Environment Setup
```bash
# Activate virtual environment
uv shell

# Install development dependencies
uv pip install pytest pytest-cov black isort mypy pre-commit

# Setup pre-commit hooks
pre-commit install
```

### 2. Code Quality
```bash
# Format code
uv run black src/ test/
uv run isort src/ test/

# Type checking
uv run mypy src/

# Linting
uv run flake8 src/
```

### 3. Testing
```bash
# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=src/astro_lab --cov-report=html

# Run specific test categories
uv run pytest test/models/ -v
uv run pytest test/tensors/ -v
```

### 4. Interactive Development
```bash
# Start Marimo reactive notebook
uv run marimo edit

# Start Jupyter Lab
uv run jupyter lab

# Launch MLflow UI
uv run mlflow ui --backend-store-uri ./data/experiments
```

## 📝 Contributing Guidelines

### Code Style
- **Python**: Follow PEP 8 with Black formatting
- **Type Hints**: Use comprehensive type annotations
- **Docstrings**: Google-style docstrings for all functions
- **Comments**: Clear, concise comments for complex logic

### Commit Messages
Use conventional commit format:
```bash
feat: add new stellar classification model
fix: resolve tensor memory leak in cosmic web analysis
docs: update data loaders documentation
test: add comprehensive model tests
refactor: simplify CLI argument parsing
```

### Pull Request Process
1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with tests
4. **Run** the test suite: `uv run pytest -v`
5. **Commit** your changes: `git commit -m 'feat: add amazing feature'`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

## 🔬 Adding New Features

### New Survey Support
1. **Data Loading**: Add survey-specific loader in `src/astro_lab/data/`
2. **Configuration**: Create survey config in `configs/surveys/`
3. **Tests**: Add comprehensive tests in `test/data/`
4. **Documentation**: Update relevant documentation

### New Model Architecture
1. **Model Implementation**: Add model in `src/astro_lab/models/`
2. **Factory Registration**: Register in `src/astro_lab/models/factory.py`
3. **Configuration**: Add model config support
4. **Tests**: Add model tests in `test/models/`

### New Tensor Type
1. **Tensor Implementation**: Add tensor in `src/astro_lab/tensors/`
2. **Base Class**: Inherit from appropriate base tensor
3. **Operations**: Implement required tensor operations
4. **Tests**: Add comprehensive tensor tests

## 🐛 Debugging & Testing

### Common Debugging Commands
```bash
# CUDA issues
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Memory issues
uv run python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"

# Import issues
uv run python -c "import astro_lab; print('Import successful')"
```

### Debug Mode
```bash
# Run with debug logging
uv run python -m astro_lab.cli train -c config.yaml --verbose

# Run tests with debug output
uv run pytest -v -s --tb=short
```

### Performance Profiling
```python
import cProfile
import pstats

# Profile specific function
profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### End-to-End Testing
```bash
# Test complete training pipeline
uv run python -m astro_lab.cli train -c configs/gaia_optimization.yaml --epochs 1

# Test data loading pipeline
uv run python -c "from astro_lab.data.core import create_cosmic_web_loader; create_cosmic_web_loader('gaia', max_samples=100)"
```

## ⚡ Performance Optimization

### GPU Optimization
```python
# Use mixed precision training
from astro_lab.training import AstroTrainer

trainer = AstroTrainer(
    precision="16-mixed",
    accelerator="gpu",
    devices=1
)

# Optimize batch size
from astro_lab.data import create_astro_datamodule

datamodule = create_astro_datamodule(
    dataset="gaia",
    batch_size=64,  # Adjust based on GPU memory
    num_workers=4   # Parallel data loading
)
```

### Memory Management
```python
# Use gradient checkpointing for large models
from astro_lab.models import AstroSurveyGNN

model = AstroSurveyGNN(
    hidden_dim=512,
    num_layers=6,
    gradient_checkpointing=True
)

# Optimize tensor operations
from astro_lab.tensors import optimize_memory_usage
optimized_data = optimize_memory_usage(data, precision="float16")
```

## 📚 Documentation Standards

### Code Documentation
- **Module Docstrings**: Overview of module functionality
- **Function Docstrings**: Google-style with type hints
- **Class Docstrings**: Description of class purpose and usage
- **Inline Comments**: For complex algorithms

### API Documentation
- **Type Hints**: Comprehensive type annotations
- **Examples**: Practical usage examples
- **Error Handling**: Document expected exceptions
- **Performance Notes**: Memory and time complexity

### User Documentation
- **Quick Start**: Simple examples for common use cases
- **Tutorials**: Step-by-step guides for complex workflows
- **Reference**: Complete API reference
- **Troubleshooting**: Common issues and solutions

## 🚀 Release Process

### Version Management
```bash
# Bump version
uv run bump2version patch  # or minor/major

# Create release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Release notes written
- [ ] PyPI package built (if applicable)

## 📚 Related Documentation

### Core Documentation
- **[Data Loaders](DATA_LOADERS.md)** - Data processing and loading
- **[Cosmic Web Analysis](COSMIC_WEB_ANALYSIS.md)** - Complete analysis framework
- **[Cosmograph Integration](COSMOGRAPH_INTEGRATION.md)** - Interactive visualization

### Survey-Specific Guides
- **[Gaia Cosmic Web](GAIA_COSMIC_WEB.md)** - Stellar structure analysis
- **[SDSS/NSA Analysis](NSA_COSMIC_WEB.md)** - Galaxy survey analysis
- **[Exoplanet Pipeline](EXOPLANET_PIPELINE.md)** - Exoplanet detection workflows

### Main Documentation
- **[Main README](../README.md)** - Complete framework overview
- **[Examples](../examples/README.md)** - Ready-to-run examples

## 🤝 Community Guidelines

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and ideas
- **Code Review**: Be constructive and helpful in PR reviews

### Code of Conduct
- **Respect**: Treat all contributors with respect
- **Inclusion**: Welcome contributors from all backgrounds
- **Collaboration**: Work together to improve the project

---

**Ready to contribute?** Start with a [good first issue](https://github.com/bjoernbethge/astro-lab/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) or explore the [Data Loaders Guide](DATA_LOADERS.md)! 