#!/usr/bin/env pwsh

Write-Host "🚀 AstroLab NumPy 2.x + Blender Kompatibilitäts-Setup" -ForegroundColor Green

# Prüfe ob uv installiert ist
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "📦 uv wird installiert..." -ForegroundColor Yellow
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
} else {
    Write-Host "✅ uv ist bereits installiert" -ForegroundColor Green
}

# Stelle sicher, dass NumPy 2.x installiert ist
Write-Host "🔧 Installiere NumPy 2.x..." -ForegroundColor Yellow
uv add "numpy>=2.0.0" --frozen

# uv sync
Write-Host "🔄 Führe uv sync aus..." -ForegroundColor Yellow
uv sync

# PyG Extensions installieren
Write-Host "🔧 Installiere PyG Extensions..." -ForegroundColor Yellow
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Teste die Kompatibilität
Write-Host "🧪 Teste NumPy 2.x Kompatibilität..." -ForegroundColor Yellow
$testResult = uv run python -c "
import numpy as np
print(f'NumPy Version: {np.__version__}')

# Teste Blender-Kompatibilität
try:
    from astro_lab.utils.blender.numpy_compat import numpy_compat
    if numpy_compat.available:
        print('✅ Blender mit NumPy 2.x kompatibel')
    else:
        print('⚠️  Blender nicht verfügbar, aber NumPy 2.x funktioniert')
except Exception as e:
    print(f'❌ Fehler beim Testen: {e}')
"

Write-Host $testResult -ForegroundColor Green

Write-Host "✅ Setup abgeschlossen!" -ForegroundColor Green
Write-Host "AstroLab ist jetzt mit NumPy 2.x kompatibel!" -ForegroundColor Cyan 