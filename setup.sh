#!/bin/bash

echo "🚀 AstroLab Setup für Linux"

# Prüfe ob uv installiert ist
if ! command -v uv &> /dev/null; then
    echo "📦 uv wird installiert..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Aktualisiere PATH für aktuelle Session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv ist bereits installiert"
fi

# uv sync
echo "🔄 Führe uv sync aus..."
uv sync

# PyG Extensions installieren
echo "🔧 Installiere PyG Extensions..."
uv pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

echo "✅ Setup abgeschlossen!"
echo "Du kannst jetzt AstroLab verwenden!" 