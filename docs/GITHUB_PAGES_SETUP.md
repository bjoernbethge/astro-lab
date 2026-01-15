# GitHub Pages Setup Guide

This document describes the GitHub Pages configuration for AstroLab documentation.

## Current Configuration

### Automated Deployment

The documentation is automatically built and deployed via GitHub Actions (`.github/workflows/docs.yml`):

- **Trigger**: On push to `main` branch or manual workflow dispatch
- **Build**: MkDocs builds documentation from markdown files
- **Deploy**: `peaceiris/actions-gh-pages@v4` deploys to `gh-pages` branch
- **URL**: https://bjoernbethge.github.io/astro-lab/

### What the Workflow Does

1. Checks out the repository
2. Installs Python 3.11 and uv package manager
3. Runs `docs/generate_docs.py update` to generate API documentation
4. Builds documentation with `mkdocs build --clean`
5. Deploys the `site/` directory to the `gh-pages` branch
6. Creates a deployment summary

## Verifying GitHub Pages is Working

### 1. Check GitHub Repository Settings

Go to: https://github.com/bjoernbethge/astro-lab/settings/pages

Ensure:
- **Source**: Deploy from a branch
- **Branch**: `gh-pages` (root)
- **Custom domain**: (optional)

### 2. Check Recent Workflow Runs

Go to: https://github.com/bjoernbethge/astro-lab/actions/workflows/docs.yml

Look for:
- Green checkmarks indicating successful builds
- Red X marks indicate failures (check logs)

### 3. Verify gh-pages Branch Exists

```bash
git fetch origin
git branch -a | grep gh-pages
```

If missing, the first workflow run will create it.

### 4. Test the Documentation URL

Visit: https://bjoernbethge.github.io/astro-lab/

If you get a 404:
- Check that the `gh-pages` branch exists
- Verify GitHub Pages is enabled in repository settings
- Wait a few minutes after the first deployment
- Clear your browser cache

## Troubleshooting Common Issues

### 404 Not Found

**Possible causes:**
1. GitHub Pages is not enabled in repository settings
2. The `gh-pages` branch doesn't exist yet
3. The workflow hasn't run successfully
4. Repository is private (GitHub Pages requires public repos or GitHub Pro)

**Solutions:**
1. Enable GitHub Pages in Settings → Pages → Source → gh-pages branch
2. Manually trigger the workflow: Actions → Generate and Deploy Documentation → Run workflow
3. Check workflow logs for errors
4. Make the repository public or upgrade to GitHub Pro

### Build Failures

**Check the workflow logs:**
```bash
# View recent workflow runs
gh run list --workflow=docs.yml

# View specific run details
gh run view <run-id>
```

**Common issues:**
- Missing dependencies: Fixed by `uv sync --frozen`
- Documentation generation errors: Check `docs/generate_docs.py`
- MkDocs build errors: Check `mkdocs.yml` configuration

### Out of Date Documentation

The documentation updates automatically on push to `main`. To manually trigger:

```bash
# Via GitHub CLI
gh workflow run docs.yml

# Via GitHub web interface
Actions → Generate and Deploy Documentation → Run workflow
```

## Manual Deployment (for testing)

If you want to manually deploy documentation:

```bash
# Install dependencies
uv sync --frozen

# Generate API docs
uv run python docs/generate_docs.py update

# Build documentation
uv run mkdocs build --clean

# Deploy to GitHub Pages
uv run mkdocs gh-deploy --force
```

## Adding Screenshots to Documentation

1. Add screenshots to `docs/images/screenshots/`
2. Reference them in markdown files:
   ```markdown
   ![Description](images/screenshots/example.png)
   ```
3. Commit and push to `main`
4. The workflow will automatically include them in the deployed site

## Monitoring

- **GitHub Actions Usage**: Monitored by `.github/workflows/monitor-actions-usage.yml`
- **Deployment Status**: Check the Actions tab for workflow runs
- **Documentation Status**: Visit the live site to verify updates

## Contact

For issues with GitHub Pages deployment, contact the repository maintainer or open an issue.
