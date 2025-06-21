#!/usr/bin/env pwsh

Write-Host "🚀 AstroLab Setup für Windows" -ForegroundColor Green

# Prüfe ob uv installiert ist
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 uv wird installiert..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    # Aktualisiere PATH für aktuelle Session
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
} else {
    Write-Host "✅ uv ist bereits installiert" -ForegroundColor Green
}

# uv sync
Write-Host "🔄 Führe uv sync aus..." -ForegroundColor Yellow
uv sync

# PyG Extensions installieren
Write-Host "🔧 Installiere PyG Extensions..." -ForegroundColor Yellow
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

Write-Host "✅ Setup abgeschlossen!" -ForegroundColor Green
Write-Host "Du kannst jetzt AstroLab verwenden!" -ForegroundColor Cyan 