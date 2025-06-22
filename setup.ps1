#!/usr/bin/env pwsh

Write-Host "🚀 AstroLab Setup for Windows" -ForegroundColor Green

# Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Update PATH for current session
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
}
else {
    Write-Host "✅ uv is already installed" -ForegroundColor Green
}

# uv sync
Write-Host "🔄 Running uv sync..." -ForegroundColor Yellow
uv sync

# Install PyG Extensions
Write-Host "🔧 Installing PyG Extensions..." -ForegroundColor Yellow
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate virtual environment
Write-Host "🔌 Activating virtual environment..." -ForegroundColor Yellow
& ".\.venv\Scripts\Activate.ps1"

Write-Host "✅ Setup completed!" -ForegroundColor Green
Write-Host "🎯 Virtual environment is now active. You can now use AstroLab!" -ForegroundColor Cyan
Write-Host "💡 To activate the environment later, run: .\.venv\Scripts\Activate.ps1" -ForegroundColor Blue 