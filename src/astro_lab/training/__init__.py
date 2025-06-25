"""
Training Module - Neural Network Training Infrastructure
======================================================

Provides training infrastructure for neural network models including
Lightning modules, MLflow logging, and Optuna optimization.
"""

import logging
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


__all__ = [
    "AstroTrainer",
    "AstroLightningModule",
    "AstroMLflowLogger",
    "TrainingConfig",
    "run_training",
]
