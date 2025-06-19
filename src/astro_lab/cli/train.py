"""
AstroLab Training CLI

Modern training interface with YAML configuration support.
Uses the new training module with Lightning + MLflow + Optuna.
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from torch.utils.data import DataLoader

from astro_lab.data import (
    AstroDataManager,
    create_gaia_dataloader,
    create_nsa_dataloader,
    create_sdss_spectral_dataloader,
)
from astro_lab.models.utils import (
    create_gaia_classifier,
    create_lsst_transient_detector,
    create_sdss_galaxy_classifier,
)
from astro_lab.training import AstroTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if config is None:
        raise ValueError(f"Empty configuration file: {config_path}")
    return config


def create_model_from_config(config: Dict[str, Any]):
    """Create model from configuration."""
    model_type = config["model"]["type"]
    model_params = config["model"].get("params", {})

    if model_type == "gaia_classifier":
        return create_gaia_classifier(**model_params)
    elif model_type == "sdss_galaxy_classifier":
        return create_sdss_galaxy_classifier(**model_params)
    elif model_type == "lsst_transient_detector":
        return create_lsst_transient_detector(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_dataloaders_from_config(config: Dict[str, Any]) -> tuple:
    """Create dataloaders from configuration."""
    data_config = config["data"]
    dataset_name = data_config["dataset"]

    # Get common dataloader parameters
    batch_size = data_config.get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    shuffle = data_config.get("shuffle", True)

    if dataset_name == "gaia":
        # Use PyTorch Geometric DataLoader for Gaia
        train_loader = create_gaia_dataloader(
            magnitude_limit=data_config.get("magnitude_limit", 15.0),
            k_neighbors=data_config.get("k_neighbors", 8),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        # For validation/test, we'll use different parameters
        val_loader = create_gaia_dataloader(
            magnitude_limit=data_config.get("magnitude_limit", 15.0),
            k_neighbors=data_config.get("k_neighbors", 8),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = None  # Single dataset for now

    elif dataset_name == "sdss":
        # Use SDSS spectral dataloader
        train_loader = create_sdss_spectral_dataloader(
            max_spectra=data_config.get("sample_size", 1000),
            k_neighbors=data_config.get("k_neighbors", 5),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = create_sdss_spectral_dataloader(
            max_spectra=data_config.get("sample_size", 1000) // 5,
            k_neighbors=data_config.get("k_neighbors", 5),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = None

    elif dataset_name == "nsa":
        # Use NSA galaxy dataloader
        train_loader = create_nsa_dataloader(
            max_galaxies=data_config.get("sample_size", 10000),
            k_neighbors=data_config.get("k_neighbors", 8),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = create_nsa_dataloader(
            max_galaxies=data_config.get("sample_size", 10000) // 5,
            k_neighbors=data_config.get("k_neighbors", 8),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_loader = None

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return train_loader, val_loader, test_loader


def train_from_config(config_path: str):
    """Train model from configuration file."""
    logger.info(f"Loading configuration from: {config_path}")
    config = load_config(config_path)

    # Create model
    logger.info("Creating model...")
    model = create_model_from_config(config)
    logger.info(f"Created model: {model.__class__.__name__}")

    # Create dataloaders
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = create_dataloaders_from_config(config)
    logger.info(
        f"Created dataloaders - Train batches: {len(train_loader)}, "
        f"Val batches: {len(val_loader) if val_loader else 0}"
    )

    # Create trainer
    trainer_config = config["training"]
    trainer = AstroTrainer(
        model=model,
        task_type=trainer_config.get("task_type", "classification"),
        experiment_name=trainer_config.get("experiment_name", "astrolab_experiment"),
        max_epochs=trainer_config.get("max_epochs", 100),
        patience=trainer_config.get("patience", 10),
        learning_rate=trainer_config.get("learning_rate", 1e-3),
        weight_decay=trainer_config.get("weight_decay", 1e-4),
        scheduler=trainer_config.get("scheduler", "cosine"),
        accelerator=trainer_config.get("accelerator", "auto"),
        devices=trainer_config.get("devices", "auto"),
        precision=trainer_config.get("precision", "16-mixed"),
    )

    logger.info("Starting training...")

    # Train model
    trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
    )

    logger.info("Training completed!")
    logger.info(f"Best model path: {trainer.best_model_path}")

    # Final metrics
    metrics = trainer.get_metrics()
    logger.info("Final metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")

    return trainer


def optimize_from_config(config_path: str):
    """Run hyperparameter optimization from configuration."""
    logger.info(f"Loading optimization configuration from: {config_path}")
    config = load_config(config_path)

    # Get model factory
    model_type = config["model"]["type"]

    if model_type == "gaia_classifier":
        model_factory = create_gaia_classifier
    elif model_type == "sdss_galaxy_classifier":
        model_factory = create_sdss_galaxy_classifier
    elif model_type == "lsst_transient_detector":
        model_factory = create_lsst_transient_detector
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders_from_config(config)

    # Create dummy trainer for optimization
    trainer_config = config["training"]
    trainer = AstroTrainer(
        model=model_factory(),  # Dummy model
        experiment_name=trainer_config.get("experiment_name", "astrolab_optimization"),
    )

    # Run optimization
    optuna_config = config.get("optimization", {})
    study = trainer.optimize_hyperparameters(
        model_factory=model_factory,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        n_trials=optuna_config.get("n_trials", 50),
        timeout=optuna_config.get("timeout"),
    )

    logger.info("Optimization completed!")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value: {study.best_value}")

    return study


def create_default_config(output_path: str):
    """Create default configuration file."""
    default_config = {
        "model": {
            "type": "gaia_classifier",
            "params": {
                "hidden_dim": 128,
                "num_classes": 7,
                "dropout": 0.1,
            },
        },
        "data": {
            "dataset": "gaia",
            "batch_size": 32,
            "num_workers": 4,
            "magnitude_limit": 15.0,
            "k_neighbors": 8,
            "sample_size": 10000,
            "shuffle": True,
        },
        "training": {
            "task_type": "classification",
            "experiment_name": "gaia_stellar_classification",
            "max_epochs": 100,
            "patience": 10,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "scheduler": "cosine",
            "accelerator": "auto",
            "devices": "auto",
            "precision": "16-mixed",
        },
        "optimization": {
            "n_trials": 50,
            "timeout": 3600,  # 1 hour
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(default_config, f, default_flow_style=False, indent=2)

    logger.info(f"Created default configuration: {output_path}")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="AstroLab Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create default config
  python train_cli.py create-config --output config.yaml
  
  # Train model
  python train_cli.py train --config config.yaml
  
  # Optimize hyperparameters
  python train_cli.py optimize --config config.yaml
  
  # Quick train with defaults
  python train_cli.py train --dataset gaia --model gaia_classifier --epochs 50
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Create config command
    config_parser = subparsers.add_parser(
        "create-config", help="Create default configuration file"
    )
    config_parser.add_argument(
        "--output", "-o", default="config.yaml", help="Output configuration file"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--config", "-c", help="Configuration file path")

    # Quick train options (without config file)
    train_parser.add_argument(
        "--dataset", choices=["gaia", "sdss", "nsa"], help="Dataset to use"
    )
    train_parser.add_argument(
        "--model",
        choices=[
            "gaia_classifier",
            "sdss_galaxy_classifier",
            "lsst_transient_detector",
        ],
        help="Model type",
    )
    train_parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs"
    )
    train_parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    train_parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    train_parser.add_argument(
        "--experiment-name", default="quick_train", help="Experiment name"
    )

    # Optimize command
    optimize_parser = subparsers.add_parser("optimize", help="Optimize hyperparameters")
    optimize_parser.add_argument(
        "--config", "-c", required=True, help="Configuration file path"
    )

    args = parser.parse_args()

    if args.command == "create-config":
        create_default_config(args.output)

    elif args.command == "train":
        if args.config:
            # Train from config file
            train_from_config(args.config)
        else:
            # Quick train mode
            if not args.dataset or not args.model:
                logger.error("For quick train, --dataset and --model are required")
                return

            # Create temporary config
            quick_config = {
                "model": {"type": args.model, "params": {}},
                "data": {
                    "dataset": args.dataset,
                    "batch_size": args.batch_size,
                    "num_workers": 4,
                },
                "training": {
                    "max_epochs": args.epochs,
                    "learning_rate": args.learning_rate,
                    "experiment_name": args.experiment_name,
                },
            }

            # Save temporary config
            temp_config_path = "temp_config.yaml"
            with open(temp_config_path, "w") as f:
                yaml.dump(quick_config, f)

            try:
                train_from_config(temp_config_path)
            finally:
                Path(temp_config_path).unlink(missing_ok=True)

    elif args.command == "optimize":
        optimize_from_config(args.config)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
