# Documentation

This directory contains the AstroLab documentation configuration and generation scripts.

## Quick Start

```bash
# Update documentation
python docs/generate_docs.py update

# Serve documentation locally
python docs/generate_docs.py serve

# Deploy to GitHub Pages
python docs/generate_docs.py deploy
```

## Structure

- `generate_docs.py` - Automated documentation generation and management
- `api/` - Auto-generated API reference documentation
- `assets/` - Documentation assets (CSS, images)
- `index.md` - Main documentation page
- `api.md` - API reference overview

## Automatic Generation

The documentation is automatically generated based on the actual project structure in `src/astro_lab/`. When modules are added, removed, or renamed, running `generate_docs.py update` will reflect these changes in the documentation.

## Configuration

Documentation settings are configured in `mkdocs.yml` at the project root. The script automatically updates the navigation structure based on discovered modules. 