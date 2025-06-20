# UV Dependency Management Commands

## Basic Dependency Management

```bash
# Add dependencies
uv add torch>=2.0.0
uv add "numpy>=1.24.0,<2.0.0"

# Development dependencies
uv add --dev pytest ruff mypy

# Optional dependencies to groups
uv add --optional jupyter "jupyter>=1.0.0"
uv add --group test pytest-cov

# Remove dependencies
uv remove torch
uv remove --dev pytest
```

## Environment Management

```bash
# Synchronize environment
uv sync                    # All dependencies
uv sync --dev             # With dev dependencies
uv sync --no-dev          # Without dev dependencies
uv sync --group test      # Only test group

# List dependencies
uv pip list
uv pip freeze
uv pip freeze > requirements.txt

# Update lock file
uv lock
uv lock --upgrade         # Upgrade all packages
uv lock --upgrade-package torch  # Only upgrade torch
```

## MLflow Integration

```bash
# Export dependencies for MLflow
uv pip freeze | grep -E "(torch|numpy|pandas|mlflow)" > mlflow_requirements.txt

# Or use directly in Python:
uv run python -c "
import subprocess
result = subprocess.run(['uv', 'pip', 'freeze'], capture_output=True, text=True)
print(result.stdout)
"
```

## Build and Distribution

```bash
# Build package
uv build
uv build --wheel
uv build --sdist

# Test with different Python versions
uv run --python 3.11 python -m pytest
uv run --python 3.12 python -m pytest
```

## Using Dependency Groups

```bash
# Install all groups
uv sync --all-groups

# Specific groups
uv sync --group lint --group test

# Exclude groups
uv sync --no-group docs
```

## Troubleshooting

```bash
# Clear cache
uv cache clean

# Resolve dependency conflicts
uv lock --resolution lowest-direct
uv lock --resolution highest

# Verbose output for debugging
uv sync -v
uv add torch -v
``` 