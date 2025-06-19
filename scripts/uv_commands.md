# UV Dependency Management Commands

## Grundlegende Dependency-Verwaltung

```bash
# Dependencies hinzufügen
uv add torch>=2.0.0
uv add "numpy>=1.24.0,<2.0.0"

# Development dependencies
uv add --dev pytest ruff mypy

# Optional dependencies zu Gruppen
uv add --optional jupyter "jupyter>=1.0.0"
uv add --group test pytest-cov

# Dependencies entfernen
uv remove torch
uv remove --dev pytest
```

## Environment Management

```bash
# Environment synchronisieren
uv sync                    # Alle dependencies
uv sync --dev             # Mit dev dependencies
uv sync --no-dev          # Ohne dev dependencies
uv sync --group test      # Nur test group

# Dependencies auflisten
uv pip list
uv pip freeze
uv pip freeze > requirements.txt

# Lock file aktualisieren
uv lock
uv lock --upgrade         # Alle packages upgraden
uv lock --upgrade-package torch  # Nur torch upgraden
```

## MLflow Integration

```bash
# Dependencies für MLflow exportieren
uv pip freeze | grep -E "(torch|numpy|pandas|mlflow)" > mlflow_requirements.txt

# Oder direkt in Python verwenden:
uv run python -c "
import subprocess
result = subprocess.run(['uv', 'pip', 'freeze'], capture_output=True, text=True)
print(result.stdout)
"
```

## Build und Distribution

```bash
# Package bauen
uv build
uv build --wheel
uv build --sdist

# Mit verschiedenen Python Versionen testen
uv run --python 3.11 python -m pytest
uv run --python 3.12 python -m pytest
```

## Dependency-Gruppen verwenden

```bash
# Alle Gruppen installieren
uv sync --all-groups

# Spezifische Gruppen
uv sync --group lint --group test

# Gruppen ausschließen
uv sync --no-group docs
```

## Troubleshooting

```bash
# Cache leeren
uv cache clean

# Dependency-Konflikte lösen
uv lock --resolution lowest-direct
uv lock --resolution highest

# Verbose output für Debugging
uv sync -v
uv add torch -v
``` 