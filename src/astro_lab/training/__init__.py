"""
Training Module - Neural Network Training Infrastructure
======================================================

Provides training infrastructure for neural network models including
Lightning modules, MLflow logging, and Optuna optimization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core dependencies - should always be available
import lightning
import mlflow
import numpy as np
import optuna
import torch
import torch.nn as nn

from astro_lab.data.datamodule import AstroDataModule
from astro_lab.models.config import ModelConfig
from astro_lab.utils.config.loader import ConfigLoader

from .config import TrainingConfig
from .lightning_module import AstroLightningModule
from .mlflow_logger import AstroMLflowLogger

# Import training components
from .trainer import AstroTrainer

# OptunaTrainer removed - functionality integrated into AstroTrainer
OptunaTrainer = None

# Setup logger
logger = logging.getLogger(__name__)


def run_training(args):
    """Run training based on CLI arguments."""
    if args.config:
        # Load config and create TrainingConfig object
        config_loader = ConfigLoader(args.config)
        config_loader.load_config()

        # Get config sections
        training_dict = config_loader.get_training_config()
        model_dict = config_loader.get_model_config()

        # Create proper config objects
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(
            name=training_dict.get("name", "config_training"),
            model=model_config,
            **{k: v for k, v in training_dict.items() if k != "name" and k != "model"},
        )

        # Create DataModule with survey from config
        survey = model_dict.get("name", "gaia")  # Default to gaia if not specified
        datamodule = AstroDataModule(
            survey=survey,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

        # Setup datamodule to detect classes
        datamodule.setup()

        # Update model config with detected classes if needed
        if datamodule.num_classes:
            model_config.output_dim = datamodule.num_classes
            logger.info(f"Detected {datamodule.num_classes} classes from data")

        trainer = AstroTrainer(training_config=training_config)
        trainer.fit(datamodule=datamodule)

    elif args.dataset and args.model:
        # Quick training with minimal config
        # Create DataModule first to detect classes
        datamodule = AstroDataModule(
            survey=args.dataset,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
        )

        # Setup datamodule to detect classes
        datamodule.setup()

        if not datamodule.num_classes:
            raise ValueError(
                f"Could not detect number of classes from {args.dataset} dataset. "
                "Please check your data."
            )

        logger.info(
            f"Detected {datamodule.num_classes} classes from {args.dataset} data"
        )

        # IMPORTANT: For factory models like gaia_classifier, we need special handling
        if args.model in ["gaia_classifier", "lsst_transient", "lightcurve_classifier"]:
            # These are classification models that need num_classes
            model_config = ModelConfig(
                name=args.model,
                output_dim=datamodule.num_classes,
                task="classification",
            )
        elif args.model in ["sdss_galaxy", "galaxy_modeler"]:
            # These are regression models - use default output dims
            model_config = ModelConfig(
                name=args.model,
                task="regression",
                # output_dim will be set by the factory based on the specific model
            )
        else:
            # Generic model
            model_config = ModelConfig(
                name=args.model,
                output_dim=datamodule.num_classes,
            )

        training_config = TrainingConfig(
            name="quick_training",
            model=model_config,
            scheduler={"max_epochs": args.epochs},
            hardware={"devices": args.devices, "precision": args.precision},
        )

        trainer = AstroTrainer(training_config=training_config)

        # Final validation before training
        logger.info(
            f"Starting training with model={args.model}, num_classes={datamodule.num_classes}"
        )

        trainer.fit(datamodule=datamodule)
    else:
        print("Error: Either --config or both --dataset and --model must be specified")
        return 1

    return 0


def train_from_config(config_path: Union[str, Path], **kwargs) -> int:
    """
    Train a model from configuration file.
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments to override config
    Returns:
        Exit code (0 for success)
    """
    from astro_lab.data import create_astro_datamodule
    from astro_lab.models.config import ModelConfig
    from astro_lab.utils.config.loader import ConfigLoader

    try:
        config_loader = ConfigLoader(config_path)
        config_loader.load_config()
        training_dict = config_loader.get_training_config()
        model_dict = config_loader.get_model_config()
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(
            name=training_dict.get("name", "config_training"),
            model=model_config,
            **{k: v for k, v in training_dict.items() if k not in ["name", "model"]},
        )
        # Override with kwargs
        if "epochs" in kwargs and kwargs["epochs"] is not None:
            training_config.scheduler["max_epochs"] = kwargs["epochs"]
        if "learning_rate" in kwargs and kwargs["learning_rate"] is not None:
            training_config.optimizer["lr"] = kwargs["learning_rate"]
        if "devices" in kwargs and kwargs["devices"] is not None:
            training_config.hardware["devices"] = kwargs["devices"]
        if "precision" in kwargs and kwargs["precision"] is not None:
            training_config.hardware["precision"] = kwargs["precision"]
        survey = model_dict.get("name", "gaia")
        datamodule = create_astro_datamodule(
            survey=survey,
            batch_size=kwargs.get("batch_size", 32),
            max_samples=kwargs.get("max_samples", 1000),
        )
        datamodule.setup()
        trainer = AstroTrainer(training_config=training_config)
        trainer.fit(datamodule=datamodule)
        return 0
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def train_quick(dataset: str, model: str, **kwargs) -> int:
    """
    Quick training with dataset and model names.
    Args:
        dataset: Dataset name (e.g., 'gaia', 'sdss')
        model: Model preset name
        **kwargs: Training parameters
    Returns:
        Exit code (0 for success)
    """
    from astro_lab.data import create_astro_datamodule
    from astro_lab.models.config import ModelConfig

    MODEL_PRESETS = {
        "gaia_classifier": {"task": "classification"},
        "lsst_transient": {"task": "classification"},
        "lightcurve_classifier": {"task": "classification"},
        "sdss_galaxy": {"task": "regression"},
        "galaxy_modeler": {"task": "regression"},
    }
    try:
        if model not in MODEL_PRESETS:
            available = ", ".join(MODEL_PRESETS.keys())
            logger.error(f"Unknown model '{model}'. Available models: {available}")
            return 1
        datamodule = create_astro_datamodule(
            survey=dataset,
            batch_size=kwargs.get("batch_size", 32),
            max_samples=kwargs.get("max_samples", 1000),
        )
        datamodule.setup()
        num_classes = getattr(datamodule, "num_classes", None)
        if MODEL_PRESETS[model]["task"] == "classification" and not num_classes:
            logger.error(
                f"Could not detect number of classes for {dataset} dataset. Please check your data."
            )
            return 1
        model_config = ModelConfig(
            name=model,
            task=MODEL_PRESETS[model]["task"],
            output_dim=num_classes if num_classes else 1,
        )
        training_config = TrainingConfig(
            name=f"quick_{dataset}_{model}",
            model=model_config,
            scheduler={
                "max_epochs": kwargs.get("epochs", 10),
                "learning_rate": kwargs.get("learning_rate", 0.001),
            },
            hardware={
                "devices": kwargs.get("devices", 1),
                "precision": kwargs.get("precision", "16-mixed"),
                "strategy": kwargs.get("strategy", "auto"),
            },
        )
        trainer = AstroTrainer(training_config=training_config)
        trainer.fit(datamodule=datamodule)
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def optimize_hyperparameters(
    config_path: Union[str, Path], n_trials: int = 10, **kwargs
) -> int:
    """
    Run hyperparameter optimization.
    Args:
        config_path: Path to configuration file
        n_trials: Number of optimization trials
        **kwargs: Additional arguments
    Returns:
        Exit code (0 for success)
    """
    from astro_lab.data import create_astro_datamodule
    from astro_lab.models.config import ModelConfig
    from astro_lab.utils.config.loader import ConfigLoader

    try:
        config_loader = ConfigLoader(config_path)
        config_loader.load_config()
        training_dict = config_loader.get_training_config()
        model_dict = config_loader.get_model_config()
        model_config = ModelConfig(**model_dict)
        training_config = TrainingConfig(
            name=training_dict.get("name", "optimization_training"),
            model=model_config,
            **{k: v for k, v in training_dict.items() if k not in ["name", "model"]},
        )
        survey = model_dict.get("name", "gaia")
        datamodule = create_astro_datamodule(
            survey=survey,
            batch_size=training_dict.get("data", {}).get("batch_size", 32),
            max_samples=training_dict.get("data", {}).get("max_samples", 1000),
        )
        datamodule.setup()
        trainer = AstroTrainer(training_config=training_config)
        trainer.optimize_hyperparameters(
            train_dataloader=datamodule.train_dataloader(),
            val_dataloader=datamodule.val_dataloader(),
            n_trials=n_trials,
        )
        return 0
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return 1


__all__ = [
    "AstroTrainer",
    "AstroLightningModule",
    "AstroMLflowLogger",
    "TrainingConfig",
    "run_training",
    "train_from_config",
    "train_quick",
    "optimize_hyperparameters",
]
