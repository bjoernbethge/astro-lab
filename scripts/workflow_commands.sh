#!/bin/bash
# Astro-Lab UV Workflow Commands

echo "🚀 Astro-Lab UV Dependency Management"
echo "======================================"

# Environment setup
setup_env() {
    echo "📦 Setting up environment..."
    uv sync --all-groups
    echo "✅ Environment ready!"
}

# Development workflow
dev_workflow() {
    echo "🔧 Development workflow..."
    uv sync --group dev --group test
    uv run ruff check .
    uv run mypy src/
    echo "✅ Development environment ready!"
}

# Training workflow
train_workflow() {
    echo "🎯 Training workflow..."
    uv sync --no-dev  # Production dependencies only
    uv run python train_with_mlflow.py --epochs 5
    echo "✅ Training completed!"
}

# MLflow server
start_mlflow() {
    echo "📊 Starting MLflow server..."
    uv run mlflow ui --backend-store-uri file:./mlruns --host 127.0.0.1 --port 5001
}

# Dependency updates
update_deps() {
    echo "⬆️ Updating dependencies..."
    uv lock --upgrade
    uv sync
    echo "✅ Dependencies updated!"
}

# Clean environment
clean_env() {
    echo "🧹 Cleaning environment..."
    uv cache clean
    rm -rf .venv
    uv sync
    echo "✅ Environment cleaned!"
}

# Show help
show_help() {
    echo "Available commands:"
    echo "  setup     - Setup complete environment"
    echo "  dev       - Setup development environment"
    echo "  train     - Run training workflow"
    echo "  mlflow    - Start MLflow server"
    echo "  update    - Update all dependencies"
    echo "  clean     - Clean and rebuild environment"
    echo "  help      - Show this help"
}

# Main command dispatcher
case "$1" in
    setup)   setup_env ;;
    dev)     dev_workflow ;;
    train)   train_workflow ;;
    mlflow)  start_mlflow ;;
    update)  update_deps ;;
    clean)   clean_env ;;
    help|*)  show_help ;;
esac 