#!/bin/bash

echo "🚀 AstroLab Setup for Linux"

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Update PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
else
    echo "✅ uv is already installed"
fi

# uv sync
echo "🔄 Running uv sync..."
uv sync

# Install PyG Extensions
echo "🔧 Installing PyG Extensions..."
uv pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

echo "✅ Setup completed!"
echo "🎯 Virtual environment is now active. You can now use AstroLab!"
echo "💡 To activate the environment later, run: source .venv/bin/activate" 