# GitHub Actions Workflow Validation

This document describes the workflow validation system for AstroLab.

## Overview

All GitHub Actions workflows in the `.github/workflows/` directory are automatically validated using:
- **actionlint**: Validates workflow syntax, action versions, and common mistakes
- **yamllint**: Checks YAML syntax

## Validation Workflow

The `validate-workflows.yml` workflow runs on:
- Push to `main` branch (when workflow files change)
- Pull requests to `main` branch (when workflow files change)
- Manual trigger via `workflow_dispatch`

## Action Versions

All workflows use the following validated action versions:

| Action | Version | Notes |
|--------|---------|-------|
| `actions/checkout` | `v4` | Latest stable version |
| `actions/setup-python` | `v5` | Latest stable version |
| `actions/cache` | `v4` | Latest stable version |
| `actions/upload-artifact` | `v4` | Latest stable version |
| `astral-sh/setup-uv` | `v4` | Latest stable version for uv package manager |
| `github/codeql-action/*` | `v4` | CodeQL analysis actions |
| `peaceiris/actions-gh-pages` | `v4` | GitHub Pages deployment |
| `peter-evans/create-pull-request` | `v7` | PR creation automation |
| `pypa/gh-action-pypi-publish` | `release/v1` | PyPI publishing with trusted publishing |
| `codecov/codecov-action` | `v5` | Code coverage reporting |

## Local Validation

To validate workflows locally before pushing:

```bash
# Install actionlint
curl -sSL https://raw.githubusercontent.com/rhysd/actionlint/main/scripts/download-actionlint.bash | bash -s -- latest /usr/local/bin

# Validate all workflows
actionlint .github/workflows/*.yml

# Install yamllint
pip install yamllint

# Check YAML syntax
yamllint -d relaxed .github/workflows/
```

## Common Issues Fixed

### 1. Non-existent Action Versions
**Problem**: Workflows used `v6` and `v7` versions that don't exist for many actions.

**Solution**: Updated to correct stable versions:
- `actions/checkout@v4` (not v6)
- `actions/setup-python@v5` (not v6)
- `astral-sh/setup-uv@v4` (not v7)
- `actions/cache@v4` (not v5 for cache@v5 doesn't exist in some contexts)
- `actions/upload-artifact@v4` (not v6)
- `peter-evans/create-pull-request@v7` (not v8)

### 2. Codecov Action Input
**Problem**: Used `file` parameter which doesn't exist.

**Solution**: Changed to `files` parameter (plural).

```yaml
# Before
- uses: codecov/codecov-action@v5
  with:
    file: ./coverage.xml

# After
- uses: codecov/codecov-action@v5
  with:
    files: ./coverage.xml
```

### 3. Workflow Input Reference
**Problem**: Incorrect reference to workflow inputs in conditions.

**Solution**: Fixed conditional expression for workflow_call inputs.

```yaml
# Before
if: github.event.inputs.alert_count > 0 || github.event_name == 'workflow_dispatch'

# After  
if: github.event_name == 'workflow_dispatch' || (github.event_name == 'workflow_call' && inputs.alert_count > 0)
```

## Workflow Best Practices

All workflows follow these best practices:

1. **Timeout Settings**: All jobs have `timeout-minutes` to prevent hanging
2. **Concurrency Control**: Workflows use `concurrency.cancel-in-progress: true` to avoid duplicate runs
3. **Minimal Permissions**: Each workflow specifies only required permissions
4. **Caching**: Dependencies are cached for faster runs
5. **Frozen Dependencies**: Use `uv sync --frozen` in CI for reproducibility
6. **Error Handling**: Steps use `continue-on-error` or `if: always()` where appropriate

## Continuous Improvement

The validation workflow helps maintain:
- ✅ Correct action versions
- ✅ Valid workflow syntax
- ✅ Proper input/output definitions
- ✅ Consistent coding style
- ✅ Security best practices

## Resources

- [actionlint documentation](https://github.com/rhysd/actionlint)
- [GitHub Actions documentation](https://docs.github.com/en/actions)
- [Workflow syntax reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
