# AstroLab Development Guide

## 🚀 Project Overview

AstroLab is a comprehensive Python framework for astronomical data analysis, machine learning, and visualization. The project combines modern ML tools with specialized astronomy libraries.

## 📦 Dependency Architecture

### Core Package Structure

The project is divided into several specialized packages:

```
astro-lab/
├── astro-viz/          # 3D Visualization & Blender Integration
├── astro-torch/        # PyTorch & ML for Astronomy
├── astro-pack/         # Astronomy Libraries & Data Access
├── astro-lab/          # Main Framework
└── astro-lab-ml/       # ML-specific Extensions
```

### 🎯 marimo-flow v0.1.1 Dependencies

**marimo-flow** brings two main components:

#### Marimo v0.14.0 (Interactive Notebook System)

**Core Features:**
- `click v8.2.1` - CLI Interface + `colorama v0.4.6`
- `docutils v0.21.2` - Documentation
- `itsdangerous v2.2.0` - Security
- `jedi v0.19.2` - Code Intelligence + `parso v0.8.4`
- `loro v1.5.1` - Collaborative editing
- `markdown v3.8.1` - Markdown Support
- `narwhals v1.42.1` - DataFrame API
- `packaging v24.2` - Package Management
- `psutil v7.0.0` - System Monitoring
- `pygments v2.19.1` - Syntax Highlighting
- `pymdown-extensions v10.15` - Extended Markdown Features
- `pyyaml v6.0.2` - YAML Support
- `starlette v0.46.2` - Web Framework + `anyio v4.9.0`
- `tomlkit v0.13.3` - TOML Support
- `uvicorn v0.34.3` - ASGI Server + `h11 v0.16.0`
- `websockets v15.0.1` - WebSocket Support

**LSP Support:**
- `python-lsp-ruff v2.2.2` + `python-lsp-server v1.12.2` + `ruff v0.12.0`
- `cattrs v25.1.1` + `lsprotocol v2025.0.0`

**Recommended Features:**
- `altair v5.5.0` - Visualization + `jsonschema v4.24.0`
- `duckdb v1.3.1` - In-Memory Database
- `nbformat v5.10.4` - Jupyter Notebook Format + `jupyter-core v5.8.1`
- `openai v1.88.0` - AI Integration + `httpx v0.28.1` + `pydantic v2.11.7`
- `polars[pyarrow] v1.31.0` - DataFrame Library + `pyarrow v20.0.0`
- `sqlglot v26.29.0` - SQL Parser/Transpiler

#### MLflow v3.1.0 (ML Experiment Tracking)

**Core MLflow:**
- `alembic v1.16.2` - Database Migration + `sqlalchemy v2.0.41`
- `docker v7.1.0` - Container Support + `pywin32 v310`
- `flask v3.1.1` - Web UI + `werkzeug v3.1.3` + `jinja2 v3.1.6`
- `graphene v3.4.3` - GraphQL API + `graphql-core v3.2.6`
- `mlflow-skinny v3.1.0` - Core MLflow + `fastapi v0.115.13` + `databricks-sdk v0.57.0`

**Data Science Stack:**
- `matplotlib v3.10.3` - Plotting + `numpy v1.26.4` + `pillow v11.2.1`
- `pandas v2.3.0` - DataFrames + `pytz v2025.2`
- `pyarrow v20.0.0` - Columnar Data
- `scikit-learn v1.7.0` - ML Library + `scipy v1.15.3` + `joblib v1.5.1`
- `waitress v3.0.2` - WSGI Server

### 🔬 Astronomy-specific Packages

#### astro-torch v0.1.0
- `astroml v1.0.2.post1` - ML for Astronomy + `scikit-learn v1.7.0`
- `astropy v7.1.0` - Astronomy Core Library
- `polars v1.31.0` - DataFrame Library
- `torch v2.7.1+cu128` - PyTorch with CUDA

#### astro-pack v0.1.0
- `astroml v1.0.2.post1` - ML for Astronomy
- `astrophot v0.16.13` - Photometry + PyTorch + Pyro
- `astropy v7.1.0` - Astronomy Core
- `astroquery v0.4.10` - Database Queries
- `poliastro v0.7.0` - Orbital Mechanics + Numba
- `sdss-access v3.0.8` - SDSS Data Access
- `sgp4 v2.24` - Satellite Tracking

#### astro-viz v0.1.0
- `astropy v7.1.0` - Astronomy Core
- `bpy v4.4.0` - Blender Python API
- `pyvista v0.45.2` - 3D Visualization + VTK

### 🧠 ML/Training Packages
- `lightning v2.5.1.post0` - PyTorch Lightning
- `optuna v4.4.0` - Hyperparameter Optimization
- `torch-geometric v2.6.1` - Graph Neural Networks
- `numba v0.61.2` - JIT Compilation

### 📊 Interactive Tools
- `cosmograph v0.0.47` - Graph Visualization
- `anywidget v0.9.18` - Jupyter Widgets
- `jupyter v1.1.1` - Jupyter Ecosystem

## 🛠️ Development Setup

### Installation

```bash
# Clone Repository
git clone <repository-url>
cd astro-lab

# Install with uv
uv sync

# Verify Installation
uv run pytest -v
```

### Key Dependencies

**Core Scientific Stack:**
- `numpy v1.26.4` - Numerical Computing
- `scipy v1.15.3` - Scientific Computing
- `pandas v2.3.0` - DataFrames
- `matplotlib v3.10.3` - Plotting
- `scikit-learn v1.7.0` - Machine Learning

**Astronomy-specific:**
- `astropy v7.1.0` - Astronomy Core
- `astroml v1.0.2.post1` - ML for Astronomy
- `astrophot v0.16.13` - Photometry
- `astroquery v0.4.10` - Data Access

**Deep Learning:**
- `torch v2.7.1+cu128` - PyTorch with CUDA
- `lightning v2.5.1.post0` - Training Framework
- `torch-geometric v2.6.1` - Graph Neural Networks

**Interactive Development:**
- `marimo v0.14.0` - Reactive Notebooks
- `jupyter v1.1.1` - Jupyter Ecosystem
- `mlflow v3.1.0` - Experiment Tracking

## 🔧 Development Commands

```bash
# Testing
uv run pytest -v                    # All Tests
uv run pytest test/test_*.py -v     # Specific Tests

# Dependency Management
uv tree                             # Dependency Tree
uv tree --package marimo-flow       # Specific Package

# Interactive Development
uv run marimo edit                  # Marimo Notebook
uv run jupyter lab                  # Jupyter Lab
uv run mlflow ui                    # MLflow UI

# CLI Tools
uv run python -m astro_lab.cli.preprocessing --help
uv run python -m astro_lab.cli.train --help
```

## 📁 Project Structure

```
src/astro_lab/
├── cli/                    # Command Line Interface
├── data/                   # Data Processing & Loading
├── models/                 # ML Models & Architectures
├── tensors/                # Specialized Tensor Types
├── training/               # Training Utilities
├── utils/                  # Utility Functions
└── simulation/             # Simulation Tools
```

## 🎯 Key Features

- **Reactive Notebooks**: Marimo for interactive development
- **ML Experiment Tracking**: MLflow Integration
- **Specialized Tensors**: Astronomy-specific data types
- **Graph Neural Networks**: For spatial data structures
- **3D Visualization**: Blender & PyVista Integration
- **CUDA Support**: GPU-accelerated computations

## 🚀 Getting Started

1. **Setup Environment**: `uv sync`
2. **Run Tests**: `uv run pytest -v`
3. **Start Marimo**: `uv run marimo edit`
4. **Explore Data**: Check `data/` directory
5. **Train Models**: Use CLI tools or notebooks

## 📚 Additional Resources

- [Data Loaders Documentation](DATA_LOADERS.md)
- [Exoplanet Pipeline](EXOPLANET_PIPELINE.md)
- Example notebooks in `examples/` 