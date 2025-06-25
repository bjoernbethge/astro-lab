"""
AstroLab CLI Optimization Module
================================

Unified command-line interface for hyperparameter optimization and visualization.
Uses Optuna for optimization and MLflow for tracking and visualization.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from astro_lab.training.trainer import AstroTrainer
from astro_lab.utils.config.loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(description="AstroLab Optimize CLI")
    parser.add_argument("--config", required=True, help="Configuration file path")
    parser.add_argument(
        "--trials", type=int, default=10, help="Number of optimization trials"
    )
    parser.add_argument(
        "--ui", action="store_true", help="Start MLflow UI after optimization"
    )
    parser.add_argument("--port", type=int, default=5000, help="Port for MLflow UI")
    args = parser.parse_args()

    # Load config using ConfigLoader
    config_loader = ConfigLoader(args.config)
    config_loader.load_config()

    # Get config sections
    training_dict = config_loader.get_training_config()
    model_dict = config_loader.get_model_config()

    # Create proper config objects
    from astro_lab.models.config import ModelConfig
    from astro_lab.training.config import TrainingConfig

    model_config = ModelConfig(**model_dict)
    training_config = TrainingConfig(
        name=training_dict.get("name", "optimization_training"),
        model=model_config,
        **{k: v for k, v in training_dict.items() if k != "name" and k != "model"},
    )

    # Create DataModule
    from astro_lab.data.datamodule import AstroDataModule

    survey = model_dict.get("name", "gaia")
    datamodule = AstroDataModule(
        survey=survey,
        batch_size=training_dict.get("data", {}).get("batch_size", 32),
        max_samples=training_dict.get("data", {}).get("max_samples", 1000),
    )
    datamodule.setup()

    trainer = AstroTrainer(training_config=training_config)

    # Run optimization
    print(f"🚀 Starting hyperparameter optimization with {args.trials} trials...")
    results = trainer.optimize_hyperparameters(
        train_dataloader=datamodule.train_dataloader(),  # type: ignore
        val_dataloader=datamodule.val_dataloader(),  # type: ignore
        n_trials=args.trials,
    )

    print("✅ Optimization complete!")
    print(f"   Best value: {results['best_value']:.4f}")
    print(f"   Best parameters: {results['best_params']}")

    # Start MLflow UI if requested
    if args.ui:
        print(f"🌐 Starting MLflow UI on port {args.port}...")
        print(f"   Open browser at: http://localhost:{args.port}")
        print("   Press Ctrl+C to stop the UI")

        try:
            # Start MLflow UI
            subprocess.run(
                [
                    "mlflow",
                    "ui",
                    "--port",
                    str(args.port),
                    "--backend-store-uri",
                    "file:./data/experiments/mlruns",
                ]
            )
        except KeyboardInterrupt:
            print("\n👋 MLflow UI stopped")
        except Exception as e:
            print(f"❌ Failed to start MLflow UI: {e}")
            print("   You can start it manually with:")
            print("   mlflow ui --backend-store-uri file:./data/experiments/mlruns")


if __name__ == "__main__":
    main()
